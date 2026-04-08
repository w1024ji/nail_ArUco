"""
nail_measurer_v6.py
-------------------
Fully automatic per-finger nail measurement using TWO photos:
  1. TOP photo   — finger pointing up, ArUco marker visible
                   → width, length, nail contour, cuticle shape, skin tone
  2. SIDE photo  — finger held upright from the side, ArUco visible
                   → nail height profile → c-curve sagitta + arc radius

Usage (single finger):
    python nail_measurer_v6.py --top index1.jpg --side index2.jpg
                               --finger index --aruco-size 20 --output results/

Usage (all 5 fingers batch):
    python nail_measurer_v6.py --batch
        --fingers thumb index middle ring pinky
        --tops   thumb_top.jpg index_top.jpg middle_top.jpg ring_top.jpg pinky_top.jpg
        --sides  thumb_side.jpg index_side.jpg middle_side.jpg ring_side.jpg pinky_side.jpg
        --aruco-size 20 --output results/

Photography requirements:
    TOP photo  : finger pointing UP, flat on dark (navy/black) background
                 ArUco marker placed beside finger on same surface
    SIDE photo : finger pointing UP, side face toward camera
                 ArUco marker beside finger (same surface level)
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
# 2. Finger segmentation
# ─────────────────────────────────────────────────────────────

def segment_finger(image: np.ndarray):
    H, W = image.shape[:2]
    L = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)[:,:,0]
    _, skin = cv2.threshold(L, 130, 255, cv2.THRESH_BINARY)
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, k9, iterations=3)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN,  k5, iterations=2)
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
    return finger_mask, finger_cnt, bbox


# ─────────────────────────────────────────────────────────────
# 3. TOP photo measurement
# ─────────────────────────────────────────────────────────────

def measure_top(image: np.ndarray, mpp: float,
                finger_mask: np.ndarray, bbox: tuple) -> dict:
    H, W  = image.shape[:2]
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fx, fy, fw, fh = bbox
    tip_y = fy

    # Row scan
    max_scan = min(int(30 / mpp), fh)
    widths, ledges, redges = [], [], []
    for row in range(tip_y, tip_y + max_scan):
        if row >= H: break
        cols = np.where(finger_mask[row, fx:fx+fw] > 0)[0]
        if len(cols) < 5: break
        widths.append(int(cols[-1] - cols[0]))
        ledges.append(int(cols[0]  + fx))
        redges.append(int(cols[-1] + fx))

    if len(widths) < 10:
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

    # Cuticle detection
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

    # Build nail polygon
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

    full_poly = (tip_arc +
                 right_pts[plate_start+1:] +
                 cuticle_arc +
                 list(reversed(left_pts[plate_start+1:])))
    nail_polygon = np.array(full_poly, np.int32)

    # Smooth spline
    pts = nail_polygon.astype(float)
    try:
        tck, _ = splprep(
            [np.append(pts[:,0], pts[0,0]),
             np.append(pts[:,1], pts[0,1])],
            s=len(pts)*2.5, per=True, k=3)
        xs, ys = splev(np.linspace(0, 1, 400), tck)
        smooth = np.column_stack([xs, ys]).astype(np.int32)
    except Exception:
        smooth = pts.astype(np.int32)

    # Skin tone
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

    return {
        "width_mm":       round(nail_half * 2 * mpp, 2),
        "length_mm":      round(length_px * mpp, 2),
        "skin_tone_hex":  hex_color,
        "nail_polygon_px": smooth.tolist(),
        "_nail_half":     nail_half,
        "_tip_x":         tip_x,
        "_tip_y":         tip_y,
        "_cuticle_y":     cuticle_y,
        "_plate_start":   plate_start,
    }


# ─────────────────────────────────────────────────────────────
# 4. SIDE photo measurement (C-curve)
# ─────────────────────────────────────────────────────────────

def measure_side(image: np.ndarray, mpp: float,
                 finger_mask: np.ndarray, bbox: tuple,
                 width_mm: float) -> dict:
    fx, fy, fw, fh = bbox

    widths_arr = []
    for row in range(fy, fy + fh):
        cols = np.where(finger_mask[row, fx:fx+fw] > 0)[0]
        widths_arr.append(int(cols[-1]-cols[0]) if len(cols) > 3 else 0)

    ws = uniform_filter1d(np.array(widths_arr, float), size=5)

    tip_rows = ws[:int(3/mpp)]
    tip_rows = tip_rows[tip_rows > 5]
    edge_w   = float(np.percentile(tip_rows, 10)) if len(tip_rows) > 0 else ws[2]

    mid_rows = ws[int(3/mpp):int(8/mpp)]
    mid_w    = float(np.percentile(mid_rows, 90)) if len(mid_rows) > 0 else ws[20]

    h_px  = (mid_w - edge_w) / 2
    h_mm  = round(max(h_px * mpp, 0.1), 2)

    arc_r = round((width_mm**2 / (8*h_mm)) + (h_mm/2), 2)
    thick = round(max(0.25, min(h_mm * 1.5, 0.85)), 2)

    print(f"  [C-curve]  edge={edge_w*mpp:.2f}mm  mid={mid_w*mpp:.2f}mm  "
          f"h={h_mm}mm  R={arc_r}mm")

    return {
        "c_curve_mm":    h_mm,
        "arc_radius_mm": arc_r,
        "thickness_mm":  thick,
        "_edge_row_mpp": mpp,
        "_bbox_side":    list(bbox),
    }


# ─────────────────────────────────────────────────────────────
# 5. W/L correction
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
# 6. Visualisation
# ─────────────────────────────────────────────────────────────

def save_annotated_top(image, top_data, aruco_corners, finger, save_path):
    vis   = image.copy()
    color = NAIL_COLORS.get(finger, (200,200,200))
    if aruco_corners is not None:
        cv2.polylines(vis, [aruco_corners.astype(int)], True, (0,255,255), 3)
    smooth = np.array(top_data["nail_polygon_px"], np.int32)
    ov = vis.copy()
    cv2.fillPoly(ov, [smooth.reshape(-1,1,2)], color)
    cv2.addWeighted(ov, 0.35, vis, 0.65, 0, vis)
    cv2.polylines(vis, [smooth.reshape(-1,1,2)], True, color, 3)
    tip_x     = top_data["_tip_x"]
    cuticle_y = top_data["_cuticle_y"]
    nail_half = top_data["_nail_half"]
    tip_y     = top_data["_tip_y"]
    cv2.line(vis,
             (tip_x-int(nail_half), cuticle_y),
             (tip_x+int(nail_half), cuticle_y),
             (0,165,255), 2)
    lx = tip_x + int(nail_half) + 30
    for txt, dy, col in [
        (f"W: {top_data['width_mm']}mm",               40,  color),
        (f"L: {top_data['length_mm']}mm",              100, color),
        (f"C: {top_data.get('c_curve_mm','?')}mm",     160, color),
        (f"R: {top_data.get('arc_radius_mm','?')}mm",  220, color),
        (f"Skin: {top_data['skin_tone_hex']}",         280, (0,200,255)),
    ]:
        cv2.putText(vis, txt, (lx, tip_y+dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 5)
        cv2.putText(vis, txt, (lx, tip_y+dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
    scale = 900 / vis.shape[0]
    cv2.imwrite(save_path, cv2.resize(vis, (int(vis.shape[1]*scale), 900)))
    print(f"  [Saved] {save_path}")


def save_annotated_side(image, side_data, finger_mask, bbox,
                        aruco_corners, finger, mpp, save_path):
    vis   = image.copy()
    color = NAIL_COLORS.get(finger, (200,200,200))
    fx, fy, fw, fh = bbox
    if aruco_corners is not None:
        cv2.polylines(vis, [aruco_corners.astype(int)], True, (0,255,255), 3)
    for i in range(0, int(15/mpp), 3):
        row = fy + i
        if row >= image.shape[0]: break
        cols = np.where(finger_mask[row, fx:fx+fw] > 0)[0]
        if len(cols) > 3:
            cv2.line(vis, (int(cols[0]+fx), row),
                          (int(cols[-1]+fx), row), (100,100,255), 1)
    edge_row = fy + int(1/mpp)
    mid_row  = fy + int(5/mpp)
    cv2.line(vis, (fx, edge_row), (fx+fw, edge_row), (0,255,0), 2)
    cv2.line(vis, (fx, mid_row),  (fx+fw, mid_row),  (0,165,255), 2)
    cv2.putText(vis, "tip edge", (fx+fw+10, edge_row),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    cv2.putText(vis, "mid arch", (fx+fw+10, mid_row),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,165,255), 2)
    lx, ly = fx+fw+10, fy+200
    for txt, col in [
        (f"C-curve: {side_data['c_curve_mm']}mm", color),
        (f"Arc R:   {side_data['arc_radius_mm']}mm", (0,200,255)),
    ]:
        cv2.putText(vis, txt, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 5)
        cv2.putText(vis, txt, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
        ly += 60
    scale = 900 / vis.shape[0]
    cv2.imwrite(save_path, cv2.resize(vis, (int(vis.shape[1]*scale), 900)))
    print(f"  [Saved] {save_path}")


# ─────────────────────────────────────────────────────────────
# 7. Single finger pipeline
# ─────────────────────────────────────────────────────────────

def measure_finger(top_path: str, side_path: str, finger: str,
                   aruco_size_mm: float, output_dir: str) -> dict:

    print(f"\n{'='*55}")
    print(f"  Measuring: {finger.upper()}")
    print(f"{'='*55}")

    top_img  = cv2.imread(top_path)
    side_img = cv2.imread(side_path)
    if top_img is None:
        sys.exit(f"ERROR: Cannot open top photo '{top_path}'")
    if side_img is None:
        sys.exit(f"ERROR: Cannot open side photo '{side_path}'")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[1/4] TOP photo — ArUco + segmentation …")
    mpp_top, aruco_top, marker_id = detect_aruco(top_img, aruco_size_mm)
    finger_mask_top, _, bbox_top  = segment_finger(top_img)

    print(f"\n[2/4] TOP photo — nail measurement …")
    top_data = measure_top(top_img, mpp_top, finger_mask_top, bbox_top)
    print(f"  width={top_data['width_mm']}mm  "
          f"length={top_data['length_mm']}mm  "
          f"skin={top_data['skin_tone_hex']}")

    print(f"\n[3/4] SIDE photo — ArUco + segmentation …")
    try:
        mpp_side, aruco_side, _ = detect_aruco(side_img, aruco_size_mm)
    except RuntimeError:
        print("  ⚠  ArUco not found in side photo — reusing TOP scale")
        mpp_side   = mpp_top
        aruco_side = None
    finger_mask_side, _, bbox_side = segment_finger(side_img)

    print(f"\n[4/4] SIDE photo — c-curve …")
    side_data = measure_side(side_img, mpp_side, finger_mask_side,
                             bbox_side, top_data["width_mm"])

    # Merge
    top_data.update(side_data)
    wl = apply_wl_correction(finger, top_data["width_mm"], top_data["length_mm"])
    top_data.update(wl)
    top_data["aspect_ratio"] = round(
        top_data["width_mm"] / top_data["length_mm"], 3
    ) if top_data["length_mm"] else 0.0

    # Save images
    save_annotated_top(top_img, top_data, aruco_top, finger,
                       os.path.join(output_dir, f"{finger}_top.jpg"))
    save_annotated_side(side_img, side_data, finger_mask_side, bbox_side,
                        aruco_side, finger, mpp_side,
                        os.path.join(output_dir, f"{finger}_side.jpg"))

    print(f"\n  ┌─ {finger.upper()} ──────────────────────────────────")
    print(f"  │  Width           : {top_data['width_mm']} mm")
    print(f"  │  Length          : {top_data['length_mm']} mm")
    print(f"  │  Corrected L     : {top_data['corrected_length_mm']} mm")
    print(f"  │  C-curve         : {top_data['c_curve_mm']} mm")
    print(f"  │  Arc radius      : {top_data['arc_radius_mm']} mm")
    print(f"  │  Thickness (est) : {top_data['thickness_mm']} mm")
    print(f"  │  Skin tone       : {top_data['skin_tone_hex']}")
    print(f"  │  W/L flag        : {top_data['wl_ratio_check']['flag']}")
    print(f"  └────────────────────────────────────────────")

    return {"finger": finger, **top_data}


# ─────────────────────────────────────────────────────────────
# 8. JSON builder
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
            "measurement_method":     "per-finger: top photo + side photo",
            "notes": {
                "width_mm":            "75th pct of stable row-scan nail plate width",
                "length_mm":           "Tip to cuticle via Sobel edge detection",
                "corrected_length_mm": "width / Jung et al. (2015) W/L ratio",
                "c_curve_mm":          "Nail arch sagitta from side photo width profile",
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
# 9. CLI
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Automatic per-finger nail measurement (top + side photo)")
    p.add_argument("--top",        help="Top photo path")
    p.add_argument("--side",       help="Side photo path")
    p.add_argument("--finger",     default="index", choices=FINGER_NAMES)
    p.add_argument("--batch",      action="store_true")
    p.add_argument("--fingers",    nargs="+", default=FINGER_NAMES)
    p.add_argument("--tops",       nargs="+")
    p.add_argument("--sides",      nargs="+")
    p.add_argument("--aruco-size", type=float, default=20.0)
    p.add_argument("--output",     default="nail_results_v6")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    results = []

    if args.batch:
        if not args.tops or not args.sides:
            sys.exit("ERROR: --batch requires --tops and --sides")
        if len(args.tops) != len(args.fingers) or len(args.sides) != len(args.fingers):
            sys.exit("ERROR: --tops, --sides, --fingers must have equal length")
        for finger, top, side in zip(args.fingers, args.tops, args.sides):
            r = measure_finger(top, side, finger, args.aruco_size, args.output)
            results.append(r)
    else:
        if not args.top or not args.side:
            sys.exit("ERROR: Provide --top and --side, or use --batch")
        r = measure_finger(args.top, args.side, args.finger,
                           args.aruco_size, args.output)
        results.append(r)

    payload   = build_payload(results, args.aruco_size)
    json_path = os.path.join(args.output, "nail_measurements.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'='*55}")
    print(f"✅ Saved → {json_path}")
    print(f"   Fingers: {[r['finger'] for r in results]}")
    print(f"\n🎉 Next: python nail_tip_generator.py --input {json_path}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()