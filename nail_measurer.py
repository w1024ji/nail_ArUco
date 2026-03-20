"""
nail_measurer.py  ·  v5  ·  Fully Automatic
--------------------------------------------
Measures all 5 fingernails from a single top-down photo with an ArUco marker.

What's new in v5
----------------
  • W/L ratio validation  — each nail is checked against the Jung et al. (2015)
                            Width-to-Length ratio standard for women (W/L ≈ 0.91
                            for all fingers, ±0.06–0.08).
  • corrected_length_mm   — added to every nail entry. Derived as width / W/L_std.
                            More reliable when the Sobel cuticle detection is uncertain.
  • wl_ratio_check        — added to every nail entry:
                              measured_wl      : width / length_mm (raw)
                              standard_wl      : 0.91 (per Jung et al.)
                              wl_std_dev       : ±1σ from study
                              within_1_sigma   : true/false
                              flag             : "ok" or "length_suspect"
                            If within_1_sigma is false, corrected_length_mm
                            is the more reliable value to use downstream.

What's new in v4
----------------
  • Width:  row-scan 92nd-percentile (stable plate width, not bounding-box shrinkage)
  • Length: positive Sobel cuticle detection (finds the nail-plate bottom edge)
  • Polygon: built from actual left/right skin edges, not Canny fill

Requirements
------------
  pip install opencv-python opencv-contrib-python numpy scipy

Usage
-----
  python nail_measurer.py --image hand.jpg --aruco-size 20 --output results/

Photography tips
----------------
  ✅ Dark background (navy, black, dark green) — critical for hand segmentation
  ✅ Shoot straight down from ~35 cm
  ✅ All 5 fingers flat and fully in frame, fingers pointing upward
  ✅ ArUco marker printed at known size, on the same surface next to the hand
  ✅ Even diffuse lighting, no harsh shadows
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
# 1.  ArUco scale detection
# ─────────────────────────────────────────────────────────────

ARUCO_DICTS = {
    "6x6_50":  cv2.aruco.DICT_6X6_50,
    "6x6_100": cv2.aruco.DICT_6X6_100,
    "4x4_50":  cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "5x5_50":  cv2.aruco.DICT_5X5_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
}


# ─────────────────────────────────────────────────────────────
# W/L ratio reference — Jung et al. (2015)
# "Fingernail Configuration" anthropometric study, women
# W/L = nail width / nail length
# ─────────────────────────────────────────────────────────────

WL_STANDARD = {
    "thumb":  {"ratio": 0.91, "std_dev": 0.08},
    "index":  {"ratio": 0.91, "std_dev": 0.07},
    "middle": {"ratio": 0.91, "std_dev": 0.06},
    "ring":   {"ratio": 0.91, "std_dev": 0.06},
    "pinky":  {"ratio": 0.90, "std_dev": 0.07},
}


def detect_aruco(image: np.ndarray, aruco_size_mm: float):
    """
    Try every common ArUco dictionary and return (mm_per_pixel, corners, id).
    Raises RuntimeError if no marker is found.
    """
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
        "  → Ensure the marker is fully visible, sharp, and well-lit.\n"
        "  → Generate a fresh marker with: python generate_aruco.py"
    )


# ─────────────────────────────────────────────────────────────
# 2.  Hand segmentation (brightness-based, dark background)
# ─────────────────────────────────────────────────────────────

def build_hand_mask(image: np.ndarray) -> tuple:
    """
    Segment hand from a dark background using the L channel of Lab colour space.
    Skin (L > 120) is much brighter than a dark navy/black background (L < 100).
    Returns (hand_mask uint8, bounding_rect (x,y,w,h)).
    """
    L = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)[:, :, 0]

    # Auto-tune threshold: valley between dark bg and bright skin
    hist = cv2.calcHist([L], [0], None, [256], [0, 256]).flatten()
    best_t, min_v = 120, float("inf")
    for t in range(80, 165):
        if hist[t] < min_v:
            min_v, best_t = hist[t], t
    thresh = max(110, min(best_t, 160))

    skin  = (L > thresh).astype(np.uint8) * 255
    k9    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k5    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin  = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, k9, iterations=4)
    skin  = cv2.morphologyEx(skin, cv2.MORPH_OPEN,  k5, iterations=2)

    cnts, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError(
            "No hand detected.\n"
            "  → Use a darker background (navy, black, dark green).\n"
            "  → Ensure the hand is well-lit and fully in frame."
        )

    hand_cnt  = max(cnts, key=cv2.contourArea)
    hand_mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(hand_mask, [hand_cnt], -1, 255, -1)
    bbox = cv2.boundingRect(hand_cnt)
    print(f"  [Hand]  threshold L>{thresh}  area={cv2.contourArea(hand_cnt):.0f}px²")
    return hand_mask, bbox


# ─────────────────────────────────────────────────────────────
# 3.  Fingertip detection
# ─────────────────────────────────────────────────────────────

def find_fingertips(hand_mask: np.ndarray, bbox: tuple) -> list:
    """
    Scan each image column for the topmost skin pixel, smooth the profile,
    then find up to 5 local minima (= fingertips).
    Returns list of dicts: {tip_x, tip_y, left_x, right_x}.
    """
    hx, hy, hw, hh = bbox
    h, w = hand_mask.shape

    top_row = np.full(w, float(h))
    for x in range(hx, hx + hw):
        rows = np.where(hand_mask[:, x] > 0)[0]
        if len(rows):
            top_row[x] = float(rows.min())

    profile  = top_row[hx:hx + hw]
    smoothed = uniform_filter1d(profile, size=25)

    peaks, props = find_peaks(-smoothed, distance=55, prominence=8)
    if len(peaks) > 5:
        order = np.argsort(props["prominences"])[::-1][:5]
        peaks = np.sort(peaks[order])

    fingers = []
    for p_off in peaks:
        tip_x = int(p_off + hx)
        tip_y = int(smoothed[p_off])
        valley_thresh = smoothed[p_off] + 35

        lx = hx
        for dx in range(int(p_off) - 1, -1, -1):
            if smoothed[dx] > valley_thresh:
                lx = hx + dx
                break

        rx = hx + hw
        for dx in range(int(p_off) + 1, len(smoothed)):
            if smoothed[dx] > valley_thresh:
                rx = hx + dx
                break

        fingers.append({"tip_x": tip_x, "tip_y": tip_y, "left_x": lx, "right_x": rx})

    return fingers


# ─────────────────────────────────────────────────────────────
# 4.  Per-nail measurement
# ─────────────────────────────────────────────────────────────

def measure_nail(hand_mask: np.ndarray, gray: np.ndarray,
                 finger: dict, mpp: float, clahe) -> dict | None:
    """
    Measure one nail using:
      • Width  — 92nd-percentile of row-scan widths once the plate stabilises
      • Length — tip-to-cuticle using positive Sobel (light→dark boundary)
      • C-curve — PCA sagitta on the nail outline polygon
    Returns a measurement dict, or None on failure.
    """
    h, w   = hand_mask.shape
    tip_x  = finger["tip_x"]
    tip_y  = finger["tip_y"]
    lx     = finger["left_x"]
    rx     = finger["right_x"]

    # ── 4a. Row-scan: widths + left/right edges ─────────────
    widths, ledges, redges = [], [], []
    for y in range(tip_y, tip_y + 400):
        if y >= h:
            break
        cols = np.where(hand_mask[y, lx:rx] > 0)[0]
        if len(cols) < 5:
            break
        widths.append(int(cols[-1] - cols[0]))
        ledges.append(int(cols[0]  + lx))
        redges.append(int(cols[-1] + lx))

    if len(widths) < 10:
        return None

    ws  = uniform_filter1d(np.array(widths, float), size=7)
    dW  = np.gradient(ws)

    # ── 4b. Plate start: where width growth slows ───────────
    plate_start = 5
    for i in range(5, len(dW)):
        if abs(dW[i]) < 1.2:
            plate_start = i
            break

    # ── 4c. TRUE WIDTH via 92nd-pct of stable region ────────
    stable_ws = ws[plate_start : plate_start + 60]
    width_px  = float(np.percentile(stable_ws, 92)) if len(stable_ws) > 5 else ws[plate_start]

    # ── 4d. CUTICLE position via positive Sobel ─────────────
    #  The cuticle shows as a light→dark transition (positive Sobel in y).
    #  Search within the central nail-width band, 5–24 mm below tip.
    hw_nail  = int(width_px * 0.42)
    x1_      = max(0, tip_x - hw_nail)
    x2_      = min(w, tip_x + hw_nail)
    max_srch = min(int(24 / mpp), len(widths) - 1)
    roi      = gray[tip_y : tip_y + max_srch, x1_ : x2_].copy()
    roi_mask = hand_mask[tip_y : tip_y + max_srch, x1_ : x2_]
    roi[roi_mask == 0] = 0

    enh  = clahe.apply(roi)
    sob  = cv2.Sobel(enh.astype(float), cv2.CV_64F, 0, 1, ksize=5)
    rstr = uniform_filter1d(np.maximum(sob, 0).sum(axis=1), size=5)

    min_d    = max(int(5 / mpp), plate_start + 3)
    prom_min = max(rstr[min_d:].max() * 0.18, 1.0) if len(rstr) > min_d else 1.0
    pks, _   = find_peaks(rstr[min_d:], distance=8, prominence=prom_min)

    if len(pks) > 0:
        cuticle_y = tip_y + min_d + int(pks[0])
    else:
        cuticle_y = tip_y + int(13 / mpp)   # fallback ≈ 13 mm (short nail default)

    cut_idx   = min(cuticle_y - tip_y, len(widths) - 1)
    length_px = float(cuticle_y - tip_y)

    # ── 4e. Nail polygon from actual skin edges ──────────────
    poly = ([[ledges[i], tip_y + i] for i in range(cut_idx + 1)] +
            [[redges[i], tip_y + i] for i in reversed(range(cut_idx + 1))])
    nail_polygon = np.array(poly, np.int32)

    # ── 4f. C-curve via PCA sagitta ─────────────────────────
    pts = nail_polygon.astype(float)
    try:
        tck, _ = splprep(
            [np.append(pts[:, 0], pts[0, 0]),
             np.append(pts[:, 1], pts[0, 1])],
            s=len(pts) * 2, per=True)
        xs, ys  = splev(np.linspace(0, 1, 400), tck)
        smooth  = np.column_stack([xs, ys])
    except Exception:
        smooth = pts

    mean = smooth.mean(0); cen = smooth - mean
    try:
        _, ev  = np.linalg.eigh(np.cov(cen.T))
        w_ax   = ev[:, 0]; proj = cen @ w_ax
        pm     = smooth[np.argmin(proj)]; pm2 = smooth[np.argmax(proj)]
        chord  = pm2 - pm; cl = float(np.linalg.norm(chord))
        if cl > 0:
            cu    = chord / cl
            dists = (smooth - pm) @ np.array([-cu[1], cu[0]])
            c_px  = float(max(dists.max(), -dists.min()))
        else:
            c_px, cl = 0.0, 0.0
    except Exception:
        c_px, cl = 0.0, 0.0

    # ── 4g. Final mm values ──────────────────────────────────
    w_mm  = round(width_px  * mpp, 2)
    l_mm  = round(length_px * mpp, 2)
    c_mm  = round(c_px      * mpp, 2)
    arc_r = round((cl**2 / (8 * c_px) + c_px / 2) * mpp, 2) if c_px > 1 else None
    thick = round(max(0.25, min(c_mm * 1.5, 0.85)), 2)
    ar    = round(w_mm / l_mm, 3) if l_mm else 0.0

    return dict(
        width_mm=w_mm, length_mm=l_mm, c_curve_mm=c_mm,
        arc_radius_mm=arc_r, thickness_mm=thick, aspect_ratio=ar,
        _cuticle_y=cuticle_y, _width_px=width_px,
        nail_polygon_px=nail_polygon.tolist(),
    )


# ─────────────────────────────────────────────────────────────
# 5.  W/L ratio validation & length correction
# ─────────────────────────────────────────────────────────────

def apply_wl_correction(finger: str, width_mm: float, length_mm: float) -> dict:
    """
    Validate the measured W/L ratio against Jung et al. (2015) and compute
    a corrected length derived purely from width and the standard W/L ratio.

    Returns a dict with:
      corrected_length_mm  — width / standard_wl  (reliable even if cuticle
                             detection was imprecise)
      wl_ratio_check       — full validation block for downstream consumers
    """
    ref = WL_STANDARD.get(finger, {"ratio": 0.91, "std_dev": 0.07})
    std_wl = ref["ratio"]
    std_sd = ref["std_dev"]

    measured_wl      = round(width_mm / length_mm, 3) if length_mm else 0.0
    wl_diff          = round(measured_wl - std_wl, 3)
    within_1_sigma   = abs(wl_diff) <= std_sd
    corrected_length = round(width_mm / std_wl, 2)

    return {
        "corrected_length_mm": corrected_length,
        "wl_ratio_check": {
            "source":          "Jung et al. (2015) — Fingernail Configuration",
            "measured_wl":     measured_wl,
            "standard_wl":     std_wl,
            "wl_std_dev":      std_sd,
            "wl_diff":         wl_diff,
            "within_1_sigma":  within_1_sigma,
            "flag":            "ok" if within_1_sigma else "length_suspect",
            "note": (
                "length_mm is reliable"
                if within_1_sigma else
                "measured W/L is outside ±1σ — corrected_length_mm "
                "is the recommended value for downstream use"
            ),
        },
    }


# ─────────────────────────────────────────────────────────────
# 6.  Finger naming
# ─────────────────────────────────────────────────────────────

def assign_names(fingers: list) -> list:
    """
    Name the detected fingers.
    Thumb = the one with the largest tip_y (lowest in image) among the two edge fingers.
    Remaining 4 = pinky→ring→middle→index (or reversed) by x-position.
    """
    n = len(fingers)
    if n == 0:
        return []

    by_x   = sorted(range(n), key=lambda i: fingers[i]["tip_x"])
    edges  = [by_x[0], by_x[-1]]
    thumb  = max(edges, key=lambda i: fingers[i]["tip_y"])
    rest   = [i for i in by_x if i != thumb]

    # Thumb on right → right hand palm-down → left-to-right: pinky,ring,middle,index
    thumb_on_right = fingers[thumb]["tip_x"] > fingers[rest[len(rest)//2]]["tip_x"]
    seq = ["pinky", "ring", "middle", "index"] if thumb_on_right else ["index", "middle", "ring", "pinky"]

    names = [""] * n
    names[thumb] = "thumb"
    for rank, fi in enumerate(rest):
        names[fi] = seq[rank] if rank < len(seq) else f"finger_{rank}"
    return names


# ─────────────────────────────────────────────────────────────
# 7.  Visualisation
# ─────────────────────────────────────────────────────────────

NAIL_COLORS = {
    "thumb":  (255,  80,  80),
    "index":  ( 80, 200,  80),
    "middle": ( 80,  80, 255),
    "ring":   (200,  80, 200),
    "pinky":  ( 80, 200, 200),
}


def draw_results(image: np.ndarray, results: list,
                 aruco_corners: np.ndarray, save_path: str) -> np.ndarray:
    vis = image.copy()
    cv2.polylines(vis, [aruco_corners.astype(int)], True, (0, 255, 255), 2)

    for r in results:
        name = r["finger"]
        col  = NAIL_COLORS.get(name, (200, 200, 200))
        cnt  = np.array(r["nail_polygon_px"], np.int32).reshape(-1, 1, 2)
        tx, ty = r["tip_x"], r["tip_y"]

        # Semi-transparent nail fill
        ov = vis.copy()
        cv2.fillPoly(ov, [cnt], col)
        cv2.addWeighted(ov, 0.25, vis, 0.75, 0, vis)
        cv2.polylines(vis, [cnt], True, col, 2)

        # Width bar (cyan) at plate level
        hw = int(r["_width_px"] / 2)
        plate_y   = ty + 22
        cuticle_y = r["_cuticle_y"]
        cv2.line(vis, (tx - hw, plate_y),   (tx + hw, plate_y),   (0, 230, 230), 2)
        cv2.line(vis, (tx - hw, cuticle_y), (tx + hw, cuticle_y), (0, 230, 230), 2)

        # Labels
        for txt, dy in [(name.upper(), -56),
                        (f"W: {r['width_mm']} mm",  -40),
                        (f"L: {r['length_mm']} mm", -24),
                        (f"C: {r['c_curve_mm']} mm",  -8)]:
            cv2.putText(vis, txt, (tx - 40, ty + dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,  0,  0), 3)
            cv2.putText(vis, txt, (tx - 40, ty + dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col,          1)

    if save_path:
        cv2.imwrite(save_path, vis)
    return vis


def print_table(results: list):
    FID = {"thumb":0,"index":1,"middle":2,"ring":3,"pinky":4}
    rows = sorted(results, key=lambda r: FID.get(r["finger"], 9))
    print("\n" + "=" * 80)
    print(f"{'Finger':<8} {'Width':>9} {'Length':>9} {'CorrL':>8} {'W/L':>6} {'Flag':<16} {'C-curve':>9}")
    print("-" * 80)
    for r in rows:
        flag = r["wl_ratio_check"]["flag"]
        wl   = r["wl_ratio_check"]["measured_wl"]
        print(f"{r['finger']:<8} {r['width_mm']:>8.2f}mm {r['length_mm']:>8.2f}mm "
              f"{r['corrected_length_mm']:>7.2f}mm {wl:>6.3f} {flag:<16} "
              f"{r['c_curve_mm']:>8.2f}mm")
    print("=" * 80)


# ─────────────────────────────────────────────────────────────
# 8.  Overall nail length classification
# ─────────────────────────────────────────────────────────────

# Standard lengths from STANDARD_NAILS (Asian women, Lee et al. 2019)
# Used as the threshold: corrected_length_mm vs standard_length_mm per finger.
# Each finger votes "long" or "short"; majority wins.
# Tie (e.g. 2 long, 2 short, 1 unknown) → "short" (conservative default).

def nail_length_category(results: list) -> str:
    """
    Returns "long" or "short" for the whole hand.

    Method:
      1. For each finger, compare corrected_length_mm to the Asian women
         standard length (Lee et al. 2019).
      2. If corrected_length_mm >= standard → vote "long", else → vote "short".
      3. Majority of votes decides the final label.
         Tie → "short" (conservative default).
    """
    STANDARD_LENGTH = {
        "thumb":  14.5,
        "index":  12.5,
        "middle": 13.5,
        "ring":   12.5,
        "pinky":  10.5,
    }

    votes = []
    for r in results:
        finger   = r["finger"]
        std_len  = STANDARD_LENGTH.get(finger)
        corr_len = r.get("corrected_length_mm")
        if std_len is None or corr_len is None:
            continue
        votes.append("long" if corr_len >= std_len else "short")

    if not votes:
        return "short"   # fallback if no fingers measured

    long_count  = votes.count("long")
    short_count = votes.count("short")
    return "long" if long_count > short_count else "short"


# ─────────────────────────────────────────────────────────────
# 9.  Main pipeline
# ─────────────────────────────────────────────────────────────

def run(image_path: str, aruco_size_mm: float, output_dir: str):

    print(f"\n[1/5] Loading  {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        sys.exit(f"ERROR: Cannot open '{image_path}'")
    print(f"      {image.shape[1]} × {image.shape[0]} px")

    print(f"\n[2/5] Detecting ArUco ({aruco_size_mm} mm) …")
    mpp, aruco_corners, marker_id = detect_aruco(image, aruco_size_mm)

    print(f"\n[3/5] Segmenting hand …")
    gray      = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hand_mask, bbox = build_hand_mask(image)

    print(f"\n[4/5] Finding fingertips & measuring …")
    fingers = find_fingertips(hand_mask, bbox)
    print(f"      {len(fingers)} fingertip(s) detected")
    if not fingers:
        sys.exit("  ⚠  No fingertips found. Use a darker background or try manual_selector.py")

    names  = assign_names(fingers)
    clahe  = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    results = []

    for i, (finger, name) in enumerate(zip(fingers, names)):
        m = measure_nail(hand_mask, gray, finger, mpp, clahe)
        if m is None:
            print(f"      ⚠  {name}: segmentation failed, skipping")
            continue
        wl = apply_wl_correction(name, m["width_mm"], m["length_mm"])
        row = {
            "finger":    name,
            "finger_id": i,
            "tip_x":     finger["tip_x"],
            "tip_y":     finger["tip_y"],
            **{k: v for k, v in m.items()},
            **wl,
        }
        results.append(row)
        ar_s  = f"{m['arc_radius_mm']:.1f}" if m["arc_radius_mm"] else "—"
        flag  = wl["wl_ratio_check"]["flag"]
        flag_s = "" if flag == "ok" else "  ⚠  length_suspect"
        print(f"      {name:<8}  W={m['width_mm']:5.2f}mm  L={m['length_mm']:5.2f}mm  "
              f"corrL={wl['corrected_length_mm']:5.2f}mm  W/L={wl['wl_ratio_check']['measured_wl']:.3f}"
              f"{flag_s}")

    if not results:
        sys.exit("  ⚠  No nails measured. Try manual_selector.py")

    print_table(results)

    # Overall nail length classification (majority vote across all fingers)
    nail_length = nail_length_category(results)
    print(f"  → Overall nail length : {nail_length.upper()}\n")
    os.makedirs(output_dir, exist_ok=True)

    draw_results(image, results, aruco_corners,
                 save_path=os.path.join(output_dir, "annotated.jpg"))
    print("      annotated.jpg")

    # Strip internal keys before serialising
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    FID   = {"thumb":0,"index":1,"middle":2,"ring":3,"pinky":4}

    payload = {
        "nail_length": nail_length,
        "meta": {
            "source_image":           os.path.basename(image_path),
            "aruco_marker_id":        marker_id,
            "aruco_physical_size_mm": aruco_size_mm,
            "mm_per_pixel":           round(mpp, 6),
            "nails_detected":         len(clean),
            "measurement_notes": {
                "width_mm":             "92nd-pct of stable row-scan widths across nail plate",
                "length_mm":            "Tip to cuticle, detected via positive Sobel edge",
                "corrected_length_mm":  "width / standard W/L ratio (Jung et al. 2015) — "
                                        "use this when wl_ratio_check.flag is 'length_suspect'",
                "wl_ratio_check":       "Validates measured W/L against Jung et al. (2015); "
                                        "flags nails where length detection may be inaccurate",
                "c_curve_mm":           "Arc depth (sagitta) — PCA on nail outline polygon",
                "arc_radius_mm":        "R = width²/(8·c_curve) + c_curve/2",
                "thickness_mm":         "Geometric estimate: c_curve × 1.5, clamped 0.25–0.85 mm",
            },
        },
        "nails": sorted(clean, key=lambda r: FID.get(r["finger"], 9)),
        "by_finger": {
            r["finger"]: {k: v for k, v in r.items()
                          if k not in ("finger","finger_id","tip_x","tip_y","nail_polygon_px")}
            for r in clean
        },
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
            }
            for r in clean
        },
    }

    json_path = os.path.join(output_dir, "nail_measurements.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"      nail_measurements.json\n")
    return results, payload


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Automatically measure 5 fingernails using an ArUco scale marker")
    p.add_argument("--image",      required=True,
                   help="Path to hand photo (dark background required)")
    p.add_argument("--aruco-size", type=float, default=20.0,
                   help="Physical side length of the printed ArUco inner square in mm (default: 20)")
    p.add_argument("--output",     default="nail_results",
                   help="Output folder (default: nail_results/)")
    args = p.parse_args()
    run(args.image, args.aruco_size, args.output)