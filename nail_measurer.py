"""
nail_measurer_v6.py
-------------------
Fully automatic per-finger nail measurement using ONE top photo.

What's new in v6
----------------
  - Single photo only (top photo) — side photo no longer needed
  - C-curve estimated from nail fold brightness analysis
    (brightness drop at nail edges vs centre correlates with curvature)
  - Auto-scales to any camera resolution

Usage (single finger):
    python nail_measurer_v6.py --top index1.jpg
                               --finger index --aruco-size 20 --output results/

Usage (all 5 fingers batch):
    python nail_measurer_v6.py --batch
        --fingers thumb index middle ring pinky
        --tops   thumb.jpg index.jpg middle.jpg ring.jpg pinky.jpg
        --aruco-size 20 --output results/

Photography requirements:
    - Finger pointing UP on dark (navy/black) background
    - ArUco marker placed beside finger on same surface
    - Even lighting, no harsh shadows
    - Camera directly above, 30-40cm distance
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

WL_STANDARD = {
    "thumb":  {"ratio": 0.91, "std_dev": 0.08},
    "index":  {"ratio": 0.91, "std_dev": 0.07},
    "middle": {"ratio": 0.91, "std_dev": 0.06},
    "ring":   {"ratio": 0.91, "std_dev": 0.06},
    "pinky":  {"ratio": 0.90, "std_dev": 0.07},
}

NAIL_COLORS = {
    "thumb":  (255,  80,  80),
    "index":  ( 80, 200,  80),
    "middle": ( 80,  80, 255),
    "ring":   (200,  80, 200),
    "pinky":  ( 80, 200, 200),
}

ARUCO_DICTS = {
    "4x4_50":  cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "5x5_50":  cv2.aruco.DICT_5X5_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "6x6_50":  cv2.aruco.DICT_6X6_50,
    "6x6_100": cv2.aruco.DICT_6X6_100,
}


# ─────────────────────────────────────────────────────────────
# 1. ArUco detection
# ─────────────────────────────────────────────────────────────

def detect_aruco(image: np.ndarray, aruco_size_mm: float):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for name, did in ARUCO_DICTS.items():
        d   = cv2.aruco.getPredefinedDictionary(did)
        det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
        corners, ids, _ = det.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            c     = corners[0][0]
            sides = [np.linalg.norm(c[i] - c[(i+1) % 4]) for i in range(4)]
            avg   = float(np.mean(sides))
            mpp   = aruco_size_mm / avg
            print(f"  [ArUco] dict={name}  id={int(ids[0][0])}  "
                  f"avg_side={avg:.1f}px  →  {mpp:.5f} mm/px")
            return mpp, c, int(ids[0][0])
    raise RuntimeError(
        "ArUco marker not detected.\n"
        "  → Ensure marker is fully visible, sharp, and well-lit.\n"
        "  → Generate a fresh marker: python generate_aruco.py"
    )


# ─────────────────────────────────────────────────────────────
# 2. Finger segmentation (resolution-aware)
# ─────────────────────────────────────────────────────────────

def segment_finger(image: np.ndarray):
    H, W = image.shape[:2]
    scale    = max(W, H) / 2000.0
    ks_large = max(9,  int(9  * scale) | 1)
    ks_small = max(5,  int(5  * scale) | 1)
    print(f"  [Segment] {W}x{H}  scale={scale:.2f}  "
          f"kernels={ks_large},{ks_small}")

    L = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)[:,:,0]
    _, skin = cv2.threshold(L, 130, 255, cv2.THRESH_BINARY)
    kL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks_large, ks_large))
    kS = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks_small, ks_small))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kL, iterations=3)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN,  kS, iterations=2)
    cnts, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError(
            "No finger detected.\n"
            "  → Use a dark background (navy, black, dark green).\n"
            "  → Ensure finger is well-lit and fully in frame."
        )
    finger_cnt  = max(cnts, key=cv2.contourArea)
    finger_mask = np.zeros((H, W), np.uint8)
    cv2.drawContours(finger_mask, [finger_cnt], -1, 255, -1)
    bbox = cv2.boundingRect(finger_cnt)
    print(f"  [Segment] bbox: x={bbox[0]} y={bbox[1]} "
          f"w={bbox[2]} h={bbox[3]}")
    return finger_mask, finger_cnt, bbox


# ─────────────────────────────────────────────────────────────
# 3. Row scan helpers
# ─────────────────────────────────────────────────────────────

def row_scan(finger_mask, bbox, mpp, H):
    fx, fy, fw, fh = bbox
    max_scan = min(int(30 / mpp), fh)
    widths, ledges, redges = [], [], []
    empty_streak = 0
    for row in range(fy, fy + max_scan):
        if row >= H: break
        cols = np.where(finger_mask[row, fx:fx+fw] > 0)[0]
        if len(cols) < 3:
            empty_streak += 1
            if empty_streak > 5: break
            continue
        empty_streak = 0
        widths.append(int(cols[-1] - cols[0]))
        ledges.append(int(cols[0]  + fx))
        redges.append(int(cols[-1] + fx))
    return widths, ledges, redges


# ─────────────────────────────────────────────────────────────
# 4. C-curve from nail fold brightness (top photo only)
# ─────────────────────────────────────────────────────────────

def estimate_ccurve_from_nailfold(image: np.ndarray,
                                   finger_mask: np.ndarray,
                                   tip_y: int, cuticle_y: int,
                                   tip_x: int, nail_half: float,
                                   mpp: float) -> dict:
    """
    Estimate c-curve from brightness drop at nail edges vs centre.

    Principle:
      - A flat nail has uniform brightness across its width
      - A curved nail casts shadow at the edges (nail fold overlap)
      - Brightness drop (centre - edge) correlates with curvature

    Scans at 3 positions (30%, 50%, 70% of nail length)
    and takes the median.

    Empirical model (calibrated):
      c_curve ≈ brightness_drop * 0.08 + 0.8
    """
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    gray_enh = clahe.apply(gray)

    length_px = cuticle_y - tip_y
    x_left  = int(tip_x - nail_half)
    x_right = int(tip_x + nail_half)

    scan_fracs = [0.30, 0.50, 0.70]
    c_estimates = []
    scan_debug  = []

    for frac in scan_fracs:
        row_center = int(tip_y + length_px * frac)
        row_start  = max(0, row_center - int(2 / mpp))
        row_end    = min(image.shape[0], row_center + int(2 / mpp))

        strip  = gray_enh[row_start:row_end, x_left:x_right].astype(float)
        mask_s = finger_mask[row_start:row_end, x_left:x_right]
        strip[mask_s == 0] = np.nan

        with np.errstate(all='ignore'):
            profile = np.nanmean(strip, axis=0)

        profile_smooth = uniform_filter1d(profile, size=7)
        nc = len(profile_smooth)
        if nc < 6:
            continue

        centre_b = float(profile_smooth[nc//3 : 2*nc//3].mean())
        left_b   = float(profile_smooth[:nc//6].mean())
        right_b  = float(profile_smooth[-nc//6:].mean())
        edge_b   = (left_b + right_b) / 2.0
        drop     = centre_b - edge_b

        c_est = float(np.clip(round(drop * 0.08 + 0.8, 2), 0.3, 5.0))
        c_estimates.append(c_est)
        scan_debug.append({
            "position_pct": int(frac * 100),
            "row": row_center,
            "centre_brightness": round(centre_b, 1),
            "edge_brightness":   round(edge_b, 1),
            "brightness_drop":   round(drop, 1),
            "c_estimate_mm":     c_est,
        })

    if not c_estimates:
        c_final = 2.0   # safe fallback
    else:
        c_final = round(float(np.median(c_estimates)), 2)

    # Arc radius from final c-curve
    w_mm  = nail_half * 2 * mpp
    arc_r = round((w_mm**2 / (8 * c_final)) + (c_final / 2), 2) if c_final > 0.1 else None
    thick = round(max(0.25, min(c_final * 1.5, 0.85)), 2)

    print(f"  [C-curve]  scans: {[s['c_estimate_mm'] for s in scan_debug]}  "
          f"median={c_final}mm  R={arc_r}mm")

    return {
        "c_curve_mm":    c_final,
        "arc_radius_mm": arc_r,
        "thickness_mm":  thick,
        "_ccurve_debug": scan_debug,
    }


# ─────────────────────────────────────────────────────────────
# 5. TOP photo full measurement
# ─────────────────────────────────────────────────────────────

def measure_top(image: np.ndarray, mpp: float,
                finger_mask: np.ndarray, bbox: tuple) -> dict:
    H, W  = image.shape[:2]
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fx, fy, fw, fh = bbox
    tip_y = fy

    # ── Row scan ─────────────────────────────────────────────
    widths, ledges, redges = row_scan(finger_mask, bbox, mpp, H)

    if len(widths) < 5:
        raise RuntimeError("Not enough finger rows detected in top photo.")

    ws  = uniform_filter1d(np.array(widths, float), size=7)
    dW  = np.gradient(ws)

    plate_start = 5
    for i in range(5, len(dW)):
        if abs(dW[i]) < 1.5:
            plate_start = i
            break

    stable   = ws[plate_start:plate_start+60]
    width_px = float(np.percentile(stable, 75)) if len(stable) > 5 else ws[plate_start]

    tip_x = int(np.mean([
        (ledges[i]+redges[i])//2
        for i in range(plate_start, min(plate_start+40, len(ledges)))
    ]))

    margin    = 0.08
    nail_half = width_px * (0.5 - margin)

    # ── Cuticle detection ─────────────────────────────────────
    hw       = int(width_px * 0.38)
    x1, x2   = max(0, tip_x - hw), min(W, tip_x + hw)
    max_srch = min(int(20 / mpp), len(widths)-1)
    roi      = gray[tip_y:tip_y+max_srch, x1:x2].copy()
    roi_m    = finger_mask[tip_y:tip_y+max_srch, x1:x2]
    roi[roi_m == 0] = 0
    clahe    = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4,4))
    enh      = clahe.apply(roi)
    sob      = cv2.Sobel(enh.astype(float), cv2.CV_64F, 0, 1, ksize=5)
    rstr     = uniform_filter1d(np.maximum(sob, 0).sum(axis=1), size=3)
    min_d    = max(int(4 / mpp), plate_start + 2)
    prom_min = max(rstr[min_d:].max() * 0.10, 0.5) if len(rstr) > min_d else 0.5
    pks, _   = find_peaks(rstr[min_d:], distance=6, prominence=prom_min)
    cuticle_y = tip_y + min_d + int(pks[0]) if len(pks) > 0 else tip_y + int(11/mpp)

    cut_idx   = min(cuticle_y - tip_y, len(widths)-1)
    length_px = float(cuticle_y - tip_y)

    # ── C-curve from nail fold ────────────────────────────────
    cc_data = estimate_ccurve_from_nailfold(
        image, finger_mask,
        tip_y, cuticle_y,
        tip_x, nail_half, mpp
    )

    # ── Build nail polygon ────────────────────────────────────
    n_pts = cut_idx + 1
    left_pts, right_pts = [], []
    for i in range(n_pts):
        cx = (ledges[i]+redges[i])//2
        left_pts.append( [cx - int(nail_half), tip_y+i])
        right_pts.append([cx + int(nail_half), tip_y+i])

    # Tip arc
    tip_bot    = tip_y + plate_start
    arc_half_w = nail_half
    arc_half_h = float(plate_start)
    tip_arc    = []
    for i in range(51):
        t  = np.pi * i / 50
        ax = tip_x - arc_half_w * np.cos(t)
        ay = tip_bot - arc_half_h * abs(np.sin(t))
        tip_arc.append([int(ax), int(ay)])

    # Cuticle arc
    arc_w       = nail_half
    arc_h       = nail_half * 0.28
    cuticle_arc = []
    for i in range(41):
        angle = np.pi * i / 40
        ax = tip_x - arc_w * np.cos(angle)
        ay = cuticle_y - arc_h * np.sin(angle)
        cuticle_arc.append([int(ax), int(ay)])

    full_poly    = (tip_arc +
                    right_pts[plate_start+1:] +
                    cuticle_arc +
                    list(reversed(left_pts[plate_start+1:])))
    nail_polygon = np.array(full_poly, np.int32)

    # Smooth spline
    pts = nail_polygon.astype(float)
    try:
        diff = np.diff(pts, axis=0)
        keep = np.concatenate([[True], np.any(diff != 0, axis=1)])
        pts  = pts[keep]
        tck, _ = splprep(
            [np.append(pts[:,0], pts[0,0]),
             np.append(pts[:,1], pts[0,1])],
            s=len(pts)*2.5, per=True, k=3)
        xs, ys = splev(np.linspace(0, 1, 400), tck)
        smooth = np.column_stack([xs, ys]).astype(np.int32)
    except Exception:
        smooth = pts.astype(np.int32)

    # ── Skin tone ─────────────────────────────────────────────
    nail_mask_full = np.zeros((H, W), np.uint8)
    cv2.fillPoly(nail_mask_full, [nail_polygon], 255)
    k25  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
    ring = cv2.bitwise_and(
        cv2.dilate(nail_mask_full, k25),
        cv2.bitwise_not(nail_mask_full))
    ring = cv2.bitwise_and(ring, finger_mask)
    pixels = image[ring > 0]
    if len(pixels):
        b, g, r = [int(np.median(pixels[:,i])) for i in range(3)]
        hex_color = f"#{r:02X}{g:02X}{b:02X}"
    else:
        hex_color = "#FFFFFF"

    w_mm = round(nail_half * 2 * mpp, 2)
    l_mm = round(length_px * mpp, 2)

    return {
        "width_mm":        w_mm,
        "length_mm":       l_mm,
        "skin_tone_hex":   hex_color,
        "nail_polygon_px": smooth.tolist(),
        **cc_data,
        "_nail_half":      nail_half,
        "_tip_x":          tip_x,
        "_tip_y":          tip_y,
        "_cuticle_y":      cuticle_y,
        "_plate_start":    plate_start,
    }


# ─────────────────────────────────────────────────────────────
# 6. W/L correction
# ─────────────────────────────────────────────────────────────

def apply_wl_correction(finger: str, width_mm: float, length_mm: float) -> dict:
    ref         = WL_STANDARD.get(finger, {"ratio": 0.91, "std_dev": 0.07})
    std_wl      = ref["ratio"]
    std_sd      = ref["std_dev"]
    measured_wl = round(width_mm / length_mm, 3) if length_mm else 0.0
    wl_diff     = round(measured_wl - std_wl, 3)
    within_1sig = abs(wl_diff) <= std_sd
    corr_length = round(width_mm / std_wl, 2)
    return {
        "corrected_length_mm": corr_length,
        "wl_ratio_check": {
            "source":         "Jung et al. (2015)",
            "measured_wl":    measured_wl,
            "standard_wl":    std_wl,
            "wl_std_dev":     std_sd,
            "wl_diff":        wl_diff,
            "within_1_sigma": within_1sig,
            "flag":           "ok" if within_1sig else "length_suspect",
        },
    }


# ─────────────────────────────────────────────────────────────
# 7. Visualisation
# ─────────────────────────────────────────────────────────────

def save_annotated(image, data, aruco_corners, finger, save_path):
    vis   = image.copy()
    color = NAIL_COLORS.get(finger, (200,200,200))

    if aruco_corners is not None:
        cv2.polylines(vis, [aruco_corners.astype(int)], True, (0,255,255), 3)

    smooth    = np.array(data["nail_polygon_px"], np.int32)
    ov        = vis.copy()
    cv2.fillPoly(ov, [smooth.reshape(-1,1,2)], color)
    cv2.addWeighted(ov, 0.35, vis, 0.65, 0, vis)
    cv2.polylines(vis, [smooth.reshape(-1,1,2)], True, color, 3)

    tip_x     = data["_tip_x"]
    tip_y     = data["_tip_y"]
    cuticle_y = data["_cuticle_y"]
    nail_half = data["_nail_half"]

    # Cuticle line
    cv2.line(vis,
             (tip_x-int(nail_half), cuticle_y),
             (tip_x+int(nail_half), cuticle_y),
             (0,165,255), 2)

    # C-curve scan lines
    length_px = cuticle_y - tip_y
    scan_colors = [(255,100,0),(0,255,100),(255,0,255)]
    for frac, col in zip([0.30, 0.50, 0.70], scan_colors):
        row = int(tip_y + length_px * frac)
        cv2.line(vis,
                 (tip_x-int(nail_half), row),
                 (tip_x+int(nail_half), row),
                 col, 1)

    # Labels
    lx = tip_x + int(nail_half) + 20
    for txt, dy, col in [
        (f"W:    {data['width_mm']}mm",              40,  color),
        (f"L:    {data['length_mm']}mm",             100, color),
        (f"C:    {data['c_curve_mm']}mm",            160, color),
        (f"R:    {data['arc_radius_mm']}mm",         220, color),
        (f"Skin: {data['skin_tone_hex']}",           280, (0,200,255)),
    ]:
        cv2.putText(vis, txt, (lx, tip_y+dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 5)
        cv2.putText(vis, txt, (lx, tip_y+dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)

    scale = 900 / vis.shape[0]
    cv2.imwrite(save_path, cv2.resize(vis, (int(vis.shape[1]*scale), 900)))
    print(f"  [Saved] {save_path}")


# ─────────────────────────────────────────────────────────────
# 8. Single finger pipeline
# ─────────────────────────────────────────────────────────────

def measure_finger(top_path: str, finger: str,
                   aruco_size_mm: float, output_dir: str) -> dict:

    print(f"\n{'='*55}")
    print(f"  Measuring: {finger.upper()}")
    print(f"{'='*55}")

    top_img = cv2.imread(top_path)
    if top_img is None:
        sys.exit(f"ERROR: Cannot open '{top_path}'")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[1/3] ArUco + finger segmentation …")
    mpp, aruco_corners, marker_id = detect_aruco(top_img, aruco_size_mm)
    finger_mask, _, bbox          = segment_finger(top_img)

    print(f"\n[2/3] Nail measurement + C-curve …")
    data = measure_top(top_img, mpp, finger_mask, bbox)
    print(f"  width={data['width_mm']}mm  "
          f"length={data['length_mm']}mm  "
          f"c-curve={data['c_curve_mm']}mm  "
          f"skin={data['skin_tone_hex']}")

    print(f"\n[3/3] W/L correction + save …")
    wl = apply_wl_correction(finger, data["width_mm"], data["length_mm"])
    data.update(wl)
    data["aspect_ratio"] = round(
        data["width_mm"] / data["length_mm"], 3
    ) if data["length_mm"] else 0.0

    save_annotated(top_img, data, aruco_corners, finger,
                   os.path.join(output_dir, f"{finger}_annotated.jpg"))

    print(f"\n  ┌─ {finger.upper()} ──────────────────────────────────")
    print(f"  │  Width           : {data['width_mm']} mm")
    print(f"  │  Length          : {data['length_mm']} mm")
    print(f"  │  Corrected L     : {data['corrected_length_mm']} mm")
    print(f"  │  C-curve         : {data['c_curve_mm']} mm")
    print(f"  │  Arc radius      : {data['arc_radius_mm']} mm")
    print(f"  │  Thickness (est) : {data['thickness_mm']} mm")
    print(f"  │  Skin tone       : {data['skin_tone_hex']}")
    print(f"  │  W/L flag        : {data['wl_ratio_check']['flag']}")
    print(f"  └────────────────────────────────────────────")

    return {"finger": finger, **data}


# ─────────────────────────────────────────────────────────────
# 9. JSON builder
# ─────────────────────────────────────────────────────────────

def build_payload(results: list, aruco_size_mm: float) -> dict:
    FID   = {"thumb":0,"index":1,"middle":2,"ring":3,"pinky":4}
    clean = sorted(results, key=lambda r: FID.get(r["finger"], 9))

    def strip(r):
        return {k: v for k, v in r.items() if not k.startswith("_")}

    STANDARD_LENGTH = {
        "thumb":14.5,"index":12.5,"middle":13.5,"ring":12.5,"pinky":10.5,
    }
    votes = []
    for r in results:
        std = STANDARD_LENGTH.get(r["finger"])
        cl  = r.get("corrected_length_mm")
        if std and cl:
            votes.append("long" if cl >= std else "short")
    nail_length = "long" if votes.count("long") > votes.count("short") else "short"

    return {
        "nail_length": nail_length,
        "meta": {
            "aruco_physical_size_mm": aruco_size_mm,
            "nails_detected":         len(clean),
            "measurement_method":     "single top photo + nail fold brightness",
            "notes": {
                "width_mm":            "75th pct of stable row-scan nail plate width",
                "length_mm":           "Tip to cuticle via Sobel edge detection",
                "corrected_length_mm": "width / Jung et al. (2015) W/L ratio",
                "c_curve_mm":          "Estimated from nail fold brightness drop at 30/50/70% of nail",
                "arc_radius_mm":       "R = w²/(8h) + h/2",
                "skin_tone_hex":       "Median BGR of skin ring around nail",
            },
        },
        "nails": [strip(r) for r in clean],
        "by_finger": {r["finger"]: strip(r) for r in clean},
        "mesh_params": {
            r["finger"]: {
                "bounding_box_mm": {
                    "x": r["width_mm"],
                    "y": r["thickness_mm"],
                    "z": r["length_mm"],
                },
                "curvature": {
                    "c_curve_sagitta_mm": r["c_curve_mm"],
                    "arc_radius_mm":      r["arc_radius_mm"],
                },
                "nail_polygon_px": r.get("nail_polygon_px", []),
                "skin_tone_hex":   r.get("skin_tone_hex", "#FFFFFF"),
            }
            for r in clean
        },
    }


# ─────────────────────────────────────────────────────────────
# 10. CLI
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Automatic nail measurement from single top photo")
    p.add_argument("--top",        help="Top photo path")
    p.add_argument("--finger",     default="index", choices=FINGER_NAMES)
    p.add_argument("--batch",      action="store_true")
    p.add_argument("--fingers",    nargs="+", default=FINGER_NAMES)
    p.add_argument("--tops",       nargs="+")
    p.add_argument("--aruco-size", type=float, default=20.0)
    p.add_argument("--output",     default="nail_results_v6")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    results = []

    if args.batch:
        if not args.tops:
            sys.exit("ERROR: --batch requires --tops")
        if len(args.tops) != len(args.fingers):
            sys.exit("ERROR: --tops and --fingers must have equal length")
        for finger, top in zip(args.fingers, args.tops):
            r = measure_finger(top, finger, args.aruco_size, args.output)
            results.append(r)
    else:
        if not args.top:
            sys.exit("ERROR: Provide --top photo path, or use --batch")
        r = measure_finger(args.top, args.finger,
                           args.aruco_size, args.output)
        results.append(r)

    payload   = build_payload(results, args.aruco_size)
    json_path = os.path.join(args.output, "nail_measurements.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'='*55}")
    print(f"✅ Saved → {json_path}")
    print(f"   Fingers: {[r['finger'] for r in results]}")
    print(f"\n🎉 Next: python nail_exact_stl.py --input {json_path}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()