"""
nail_measurer.py  ·  v3  ·  Fully Automatic
--------------------------------------------
Measures all 5 fingernails from a single top-down photo with an ArUco marker.

Requirements
------------
  pip install opencv-python opencv-contrib-python numpy scipy

Usage
-----
  python nail_measurer.py --image hand.jpg --aruco-size 20 --output results/

Photography tips for best results
----------------------------------
  ✅ Use a DARK background (navy, black, dark green)
  ✅ Shoot straight down from ~35 cm
  ✅ All 5 fingers flat and fully in frame
  ✅ Marker on same surface next to the hand
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
# ArUco detection
# ─────────────────────────────────────────────────────────────

ARUCO_DICTS = {
    "6x6_50":  cv2.aruco.DICT_6X6_50,
    "6x6_100": cv2.aruco.DICT_6X6_100,
    "4x4_50":  cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "5x5_50":  cv2.aruco.DICT_5X5_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
}


def detect_aruco(image: np.ndarray, aruco_size_mm: float):
    """Returns (mm_per_pixel, corners_4x2, marker_id). Raises RuntimeError if not found."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for name, did in ARUCO_DICTS.items():
        d   = cv2.aruco.getPredefinedDictionary(did)
        det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
        corners, ids, _ = det.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            c    = corners[0][0]
            sides = [np.linalg.norm(c[i] - c[(i+1)%4]) for i in range(4)]
            avg  = float(np.mean(sides))
            mpp  = aruco_size_mm / avg
            print(f"  [ArUco] dict={name}  id={ids[0][0]}  {mpp:.5f} mm/px")
            return mpp, c, int(ids[0][0])
    raise RuntimeError(
        "ArUco marker not detected.\n"
        "  → Make sure the marker is fully visible, sharp, and well-lit.\n"
        "  → Try printing a new marker with generate_aruco.py"
    )


# ─────────────────────────────────────────────────────────────
# Hand segmentation
# ─────────────────────────────────────────────────────────────

def build_hand_mask(image: np.ndarray, l_threshold: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment the hand from the background using brightness in Lab space.
    Works best with a DARK background (L < 100) and normal skin (L > 130).

    Returns (hand_mask, hand_contour).
    """
    lab  = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L    = lab[:,:,0]

    # Auto-tune the threshold: find the valley between dark bg and bright skin
    hist = cv2.calcHist([L], [0], None, [256], [0,256]).flatten()
    # Look for a threshold between 80 and 160
    best_thresh = l_threshold
    min_valley  = float('inf')
    for t in range(80, 160):
        valley = float(hist[t])
        if valley < min_valley:
            min_valley  = valley
            best_thresh = t

    # Use the auto-tuned threshold (but clamp it)
    thresh = max(110, min(best_thresh, 160))

    skin  = (L > thresh).astype(np.uint8) * 255
    k9    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    k5    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    skin  = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, k9, iterations=4)
    skin  = cv2.morphologyEx(skin, cv2.MORPH_OPEN,  k5, iterations=2)

    cnts, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("No hand detected. Check lighting and background contrast.")

    # Hand = largest blob (exclude small noise)
    h_cnt     = max(cnts, key=cv2.contourArea)
    hand_mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(hand_mask, [h_cnt], -1, 255, -1)
    print(f"  [Hand] threshold=L>{thresh}  area={cv2.contourArea(h_cnt):.0f}px²")
    return hand_mask, h_cnt


# ─────────────────────────────────────────────────────────────
# Fingertip detection
# ─────────────────────────────────────────────────────────────

def find_fingertips(hand_mask: np.ndarray) -> list[dict]:
    """
    Finds up to 5 fingertip x-positions by scanning the column-wise top profile.
    Returns list of dicts with keys: tip_x, tip_y, left_x, right_x, f_top, f_bot
    sorted left → right.
    """
    h_img, w_img = hand_mask.shape
    hx, hy, hw, hh = cv2.boundingRect(
        max(cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
            key=cv2.contourArea))

    # Per-column topmost skin pixel
    top_row = np.full(w_img, float(h_img))
    for x in range(hx, hx+hw):
        rows = np.where(hand_mask[:,x] > 0)[0]
        if len(rows):
            top_row[x] = float(rows.min())

    profile  = top_row[hx:hx+hw]
    smoothed = uniform_filter1d(profile, size=25)

    # Local minima in y = local maxima in height = fingertips
    peaks, props = find_peaks(-smoothed, distance=55, prominence=8)

    # Keep best 5 by prominence
    if len(peaks) > 5:
        order = np.argsort(props['prominences'])[::-1][:5]
        peaks = np.sort(peaks[order])

    fingers = []
    for p_off in peaks:
        tip_x = int(p_off + hx)
        tip_y = int(smoothed[p_off])

        # x-extent: walk left/right until profile rises ≥ 35px above the tip
        threshold = smoothed[p_off] + 35
        lx = hx
        for dx in range(int(p_off)-1, -1, -1):
            if smoothed[dx] > threshold:
                lx = hx + dx
                break
        rx = hx + hw
        for dx in range(int(p_off)+1, len(smoothed)):
            if smoothed[dx] > threshold:
                rx = hx + dx
                break

        # Full finger extent (top to bottom) within this x-strip
        col_mask = hand_mask.copy()
        col_mask[:, :lx]  = 0
        col_mask[:, rx:]  = 0
        f_rows = np.where(col_mask.sum(axis=1) > 0)[0]
        if len(f_rows) == 0:
            continue
        fingers.append({
            "tip_x": tip_x, "tip_y": tip_y,
            "left_x": lx,   "right_x": rx,
            "f_top": int(f_rows.min()), "f_bot": int(f_rows.max()),
        })

    return fingers


# ─────────────────────────────────────────────────────────────
# Nail segmentation within fingertip ROI
# ─────────────────────────────────────────────────────────────

def segment_nail(image: np.ndarray, hand_mask: np.ndarray, finger: dict) -> np.ndarray | None:
    """
    Returns a nail contour (OpenCV format, image coordinates) for one finger.
    Uses CLAHE + Canny inside the fingertip ROI, falling back to the full
    fingertip mask if edge detection yields nothing useful.
    """
    h_img, w_img = image.shape[:2]
    lab   = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L     = lab[:,:,0]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))

    tip_x  = finger["tip_x"]
    lx, rx = finger["left_x"], finger["right_x"]
    f_top  = finger["f_top"]
    f_h    = finger["f_bot"] - f_top

    # Nail region = top 30% of finger height
    nail_bot = f_top + int(f_h * 0.30)
    x1, x2  = max(0, lx), min(w_img, rx)
    y1, y2  = max(0, f_top), min(h_img, nail_bot)

    if x2 - x1 < 5 or y2 - y1 < 5:
        return None

    # Hand mask restricted to nail ROI
    nail_roi_mask = hand_mask[y1:y2, x1:x2]

    # CLAHE on L channel → Canny edges
    L_crop   = L[y1:y2, x1:x2]
    L_enh    = clahe.apply(L_crop)
    edges    = cv2.Canny(L_enh, 18, 55)

    # Fill edges to closed nail shape
    k5    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    filled = cv2.dilate(edges, k5, iterations=2)
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, k5, iterations=6)
    filled = cv2.bitwise_and(filled, nail_roi_mask)

    cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    good    = [c for c in cnts if cv2.contourArea(c) > 400]

    if good:
        nail_cnt = max(good, key=cv2.contourArea)
    else:
        # Fallback: use the entire fingertip ROI mask
        fb_cnts, _ = cv2.findContours(nail_roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not fb_cnts:
            return None
        nail_cnt = max(fb_cnts, key=cv2.contourArea)
        # Already in ROI coords — shift below
    
    # Shift contour from ROI coordinates to full image coordinates
    shifted = nail_cnt.copy()
    shifted[:,:,0] += x1
    shifted[:,:,1] += y1
    return shifted


# ─────────────────────────────────────────────────────────────
# Measurement
# ─────────────────────────────────────────────────────────────

def compute_measurements(nail_cnt: np.ndarray, mm_per_pixel: float) -> dict:
    """Compute length, width, c-curve, arc radius from a nail contour."""
    rect       = cv2.minAreaRect(nail_cnt)
    bw_r, bh_r = sorted(rect[1])   # bw ≤ bh; bh = length axis

    pts = nail_cnt[:,0,:].astype(float)
    if len(pts) >= 6:
        try:
            tck, _ = splprep(
                [np.append(pts[:,0], pts[0,0]),
                 np.append(pts[:,1], pts[0,1])],
                s=len(pts)*2, per=True)
            xs, ys = splev(np.linspace(0,1,400), tck)
            smooth = np.column_stack([xs,ys])
        except Exception:
            smooth = pts
    else:
        smooth = pts

    # C-curve via PCA on the smoothed contour
    mean = smooth.mean(0); cen = smooth - mean
    try:
        _, evecs = np.linalg.eigh(np.cov(cen.T))
        w_ax     = evecs[:,0]; proj = cen @ w_ax
        pm       = smooth[np.argmin(proj)]; pm2 = smooth[np.argmax(proj)]
        chord    = pm2 - pm; chord_len = float(np.linalg.norm(chord))
        if chord_len > 0:
            cu     = chord / chord_len
            perp   = np.array([-cu[1], cu[0]])
            dists  = (smooth - pm) @ perp
            c_px   = float(max(dists.max(), -dists.min()))
        else:
            c_px, chord_len = 0.0, 0.0
    except Exception:
        c_px, chord_len = 0.0, 0.0

    l_mm   = round(bh_r * mm_per_pixel, 2)
    w_mm   = round(bw_r * mm_per_pixel, 2)
    c_mm   = round(c_px * mm_per_pixel, 2)
    arc_r  = round((chord_len**2/(8*c_px) + c_px/2) * mm_per_pixel, 2) if c_px > 1 else None
    thick  = round(max(0.25, min(c_mm * 1.5, 0.85)), 2)
    ar     = round(w_mm / l_mm, 3) if l_mm else 0.0

    return dict(length_mm=l_mm, width_mm=w_mm, c_curve_mm=c_mm,
                arc_radius_mm=arc_r, thickness_mm=thick, aspect_ratio=ar)


# ─────────────────────────────────────────────────────────────
# Finger naming
# ─────────────────────────────────────────────────────────────

def assign_finger_names(fingers: list[dict]) -> list[str]:
    """
    Assign finger names based on x-position and relative tip height.
    The thumb is usually an outlier: lower (larger tip_y) and may be on either edge.
    Remaining fingers go pinky→ring→middle→index from left or right depending on hand.
    """
    if not fingers:
        return []

    n = len(fingers)
    # Sort by x
    sorted_by_x = sorted(range(n), key=lambda i: fingers[i]["tip_x"])

    # Thumb heuristic: the one with the largest tip_y (lowest in image) among edge fingers
    # Check leftmost and rightmost
    edge_candidates = [sorted_by_x[0], sorted_by_x[-1]]
    thumb_idx       = max(edge_candidates, key=lambda i: fingers[i]["tip_y"])

    # Remaining fingers sorted by x — assign pinky→ring→middle→index
    rest = [i for i in sorted_by_x if i != thumb_idx]

    # Determine hand side: if thumb is on the right → right hand (fingers go l→r: pinky,ring,mid,idx)
    #                       if thumb is on the left  → left hand  (fingers go l→r: idx,mid,ring,pinky)
    thumb_on_right = (fingers[thumb_idx]["tip_x"] > fingers[rest[len(rest)//2]]["tip_x"])

    if thumb_on_right:  # right hand, palm down
        names_for_rest = ["pinky","ring","middle","index"]
    else:               # left hand, palm down
        names_for_rest = ["index","middle","ring","pinky"]

    result = [""] * n
    result[thumb_idx] = "thumb"
    for rank, fi in enumerate(rest):
        if rank < len(names_for_rest):
            result[fi] = names_for_rest[rank]
        else:
            result[fi] = f"finger_{rank}"

    return result


# ─────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────

NAIL_COLORS = {
    "thumb":  (255,  80,  80),
    "index":  ( 80, 200,  80),
    "middle": ( 80,  80, 255),
    "ring":   (200,  80, 200),
    "pinky":  ( 80, 200, 200),
}

def draw_results(image: np.ndarray, nail_results: list[dict],
                 aruco_corners: np.ndarray, save_path: str) -> np.ndarray:
    vis = image.copy()

    # ArUco outline
    cv2.polylines(vis, [aruco_corners.astype(int)], True, (0,255,255), 2)

    for r in nail_results:
        name  = r["finger"]
        col   = NAIL_COLORS.get(name, (200,200,200))
        cnt   = np.array(r["nail_polygon_px"], np.int32).reshape(-1,1,2)
        tip_x = r["tip_x"]; tip_y = r["tip_y"]

        # Semi-transparent fill
        overlay = vis.copy()
        cv2.fillPoly(overlay, [cnt], col)
        cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
        cv2.polylines(vis, [cnt], True, col, 2)

        # Bounding box
        rect = cv2.minAreaRect(cnt)
        box  = cv2.boxPoints(rect).astype(int)
        cv2.drawContours(vis, [box], -1, (255,255,0), 1)

        # Labels
        labels = [
            name.upper(),
            f"L: {r['length_mm']} mm",
            f"W: {r['width_mm']} mm",
            f"C: {r['c_curve_mm']} mm",
        ]
        for j, txt in enumerate(labels):
            dy = -55 + j*16
            cv2.putText(vis, txt, (tip_x-40, tip_y+dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3)
            cv2.putText(vis, txt, (tip_x-40, tip_y+dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    if save_path:
        cv2.imwrite(save_path, vis)
    return vis


def print_table(nail_results: list[dict]):
    h = f"{'Finger':<8} {'Length':>9} {'Width':>8} {'C-curve':>9} {'ArcR':>8} {'Thick':>7} {'AR':>6}"
    print("\n" + "="*60)
    print(h)
    print("-"*60)
    for r in sorted(nail_results, key=lambda x: x["finger_id"]):
        ar_s = f"{r['arc_radius_mm']:.1f}" if r["arc_radius_mm"] else "—"
        print(f"{r['finger']:<8} {r['length_mm']:>8.2f}mm {r['width_mm']:>7.2f}mm "
              f"{r['c_curve_mm']:>8.2f}mm {ar_s:>7}mm {r['thickness_mm']:>6.2f}mm "
              f"{r['aspect_ratio']:>6.3f}")
    print("="*60)


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def run(image_path: str, aruco_size_mm: float, output_dir: str):
    print(f"\n[1/5] Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        sys.exit(f"ERROR: Cannot open '{image_path}'")
    print(f"      {image.shape[1]} × {image.shape[0]} px")

    print(f"\n[2/5] Detecting ArUco marker ({aruco_size_mm} mm) …")
    mm_per_pixel, aruco_corners, marker_id = detect_aruco(image, aruco_size_mm)

    print("\n[3/5] Segmenting hand …")
    hand_mask, _ = build_hand_mask(image)

    print("\n[4/5] Finding fingertips & measuring nails …")
    fingers = find_fingertips(hand_mask)
    print(f"      {len(fingers)} fingertip(s) detected")

    if len(fingers) == 0:
        print("\n  ⚠  No fingertips found.")
        print("     → Use a darker background for better contrast.")
        print("     → Or run manual_selector.py for manual outlining.")
        sys.exit(1)

    names = assign_finger_names(fingers)

    nail_results = []
    for i, (finger, name) in enumerate(zip(fingers, names)):
        nail_cnt = segment_nail(image, hand_mask, finger)
        if nail_cnt is None:
            print(f"  ⚠  Could not segment nail for {name}")
            continue
        m = compute_measurements(nail_cnt, mm_per_pixel)
        nail_results.append({
            "finger":     name,
            "finger_id":  i,
            "tip_x":      finger["tip_x"],
            "tip_y":      finger["tip_y"],
            **m,
            "pixel_scale_mm_per_px": round(mm_per_pixel, 6),
            "nail_polygon_px": nail_cnt[:,0,:].tolist(),
        })
        print(f"      {name:<8}  L={m['length_mm']:5.2f}mm  W={m['width_mm']:5.2f}mm  "
              f"C={m['c_curve_mm']:5.2f}mm  Thick={m['thickness_mm']}mm")

    if not nail_results:
        print("\n  ⚠  No nails measured. Try manual_selector.py.")
        sys.exit(1)

    print_table(nail_results)

    print(f"\n[5/5] Saving to {output_dir}/ …")
    os.makedirs(output_dir, exist_ok=True)

    # Annotated image
    draw_results(image, nail_results, aruco_corners,
                 save_path=os.path.join(output_dir, "annotated.jpg"))
    print("      annotated.jpg")

    # JSON
    payload = {
        "meta": {
            "source_image":          os.path.basename(image_path),
            "aruco_marker_id":       marker_id,
            "aruco_physical_size_mm": aruco_size_mm,
            "mm_per_pixel":          round(mm_per_pixel, 6),
            "nails_detected":        len(nail_results),
            "measurement_notes": {
                "length_mm":     "Base-to-tip along the nail long axis",
                "width_mm":      "Widest point side-to-side",
                "c_curve_mm":    "Arc depth (sagitta) across the width",
                "arc_radius_mm": "Radius of curvature: R = w²/(8h) + h/2",
                "thickness_mm":  "Estimated from geometry (c_curve × 1.5, clamped 0.25–0.85 mm)",
            }
        },
        "nails":      nail_results,
        "by_finger":  {r["finger"]: {k:v for k,v in r.items()
                        if k not in ("finger","finger_id","tip_x","tip_y","nail_polygon_px")}
                       for r in nail_results},
        "mesh_params": {r["finger"]: {
            "bounding_box_mm": {"x": r["width_mm"], "y": r["thickness_mm"], "z": r["length_mm"]},
            "curvature":       {"c_curve_sagitta_mm": r["c_curve_mm"], "arc_radius_mm": r["arc_radius_mm"]},
        } for r in nail_results},
    }
    json_path = os.path.join(output_dir, "nail_measurements.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"      nail_measurements.json\n")
    return nail_results, payload


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Automatically measure 5 fingernails using an ArUco marker")
    p.add_argument("--image",      required=True,
                   help="Path to hand photo (use dark background for best results)")
    p.add_argument("--aruco-size", type=float, default=20.0,
                   help="Physical side length of the printed ArUco marker in mm (default: 20)")
    p.add_argument("--output",     default="nail_results",
                   help="Output folder (default: nail_results/)")
    args = p.parse_args()
    run(args.image, args.aruco_size, args.output)