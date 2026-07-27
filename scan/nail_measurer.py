"""
nail_measurer_v6.py
-------------------
Fully automatic per-finger nail measurement using ONE top photo.
Optionally accepts an end-on (C-curve) photo per finger for accurate
sagitta measurement without an ArUco marker.

C-curve methods (in priority order)
-------------------------------------
  1. End-on photo  (--ccurve-top / --ccurve-tops):
       Detects the nail plate as the brightest surface region,
       measures sagitta directly in pixels, scales using the known
       nail width from the top photo.  Most accurate.
  2. Nail-fold brightness fallback (no end-on photo):
       Estimates curvature from the brightness drop at nail edges
       vs. centre in the top photo.  Less accurate but needs no
       extra photo.

Usage (single finger):
    python nail_measurer.py --top index.jpg
                            --finger index --aruco-size 20 --output results/
    # with end-on C-curve photo:
    python nail_measurer.py --top index.jpg --ccurve-top index_ccurve.jpg
                            --finger index --aruco-size 20 --output results/

Usage (batch — ccurve-tops is optional; use "" to skip per finger):
    python nail_measurer.py --batch
        --fingers thumb index middle ring pinky
        --tops   thumb.jpg index.jpg middle.jpg ring.jpg pinky.jpg
        --ccurve-tops thumb_ccurve.jpg "" middle_ccurve.jpg "" ""
        --aruco-size 20 --output results/

Photography requirements (top photo):
    - Finger pointing UP on dark (navy/black) background
    - ArUco marker placed beside finger on same surface
    - Even lighting, no harsh shadows
    - Camera directly above, 30-40cm distance

Photography requirements (end-on C-curve photo):
    - Finger tip pointing toward camera
    - Any background (dark preferred for cleaner segmentation)
    - Camera at the same height as the fingertip
    - No ArUco marker needed
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, peak_prominences

# ── End-on C-curve measurement (optional, replaces brightness fallback) ──
# Import from measure_ccurve.py which lives in the same directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from measure_ccurve import measure_ccurve as _endOn_ccurve
    _ENDON_AVAILABLE = True
except ImportError:
    _ENDON_AVAILABLE = False


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

# Reference dimensions for per-finger size classification (Asian women,
# Jung et al. 2015 + derived values — same source as nail_tip_generator.py).
STANDARD_NAILS = {
    "thumb":  {"width_mm": 12.1, "length_mm": 11.3},
    "index":  {"width_mm":  9.1, "length_mm":  9.8},
    "middle": {"width_mm":  9.6, "length_mm": 10.5},
    "ring":   {"width_mm":  8.3, "length_mm":  9.8},
    "pinky":  {"width_mm":  7.0, "length_mm":  8.2},
}

_SIZE_THRESHOLDS = [
    (-2.0, "much_smaller"), (-1.0, "smaller"),
    ( 1.0, "average"),      ( 2.0, "larger"), (float("inf"), "much_larger"),
]

def _size_category(diff_mm: float) -> str:
    for threshold, label in _SIZE_THRESHOLDS:
        if diff_mm < threshold:
            return label
    return "much_larger"

def _overall_size(w_cat: str, l_cat: str) -> str:
    rank = {"much_smaller": 0, "smaller": 1, "average": 2,
            "larger": 3, "much_larger": 4}
    return list(rank.keys())[round((rank[w_cat] + rank[l_cat]) / 2.0)]

ARUCO_DICTS = {
    "4x4_50":  cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "5x5_50":  cv2.aruco.DICT_5X5_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "6x6_50":  cv2.aruco.DICT_6X6_50,
    "6x6_100": cv2.aruco.DICT_6X6_100,
}

# Minimum groove ridge response accepted as a nail fold (see
# detect_lateral_edges).  Kept low because the LIT side of the finger responds
# far more weakly than the shadow side; the silhouette is excluded structurally
# rather than by threshold, so a low value here is safe.
LATERAL_THR = 2.5


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

def segment_finger(image: np.ndarray, aruco_corners: np.ndarray = None):
    H, W = image.shape[:2]
    scale    = max(W, H) / 2000.0
    ks_large = max(9,  int(9  * scale) | 1)
    ks_small = max(5,  int(5  * scale) | 1)
    print(f"  [Segment] {W}x{H}  scale={scale:.2f}  "
          f"kernels={ks_large},{ks_small}")

    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L = lab_img[:,:,0].copy()
    # a* channel: skin/nail is warm/reddish (a* > 128 in OpenCV = positive),
    # blue/cool backgrounds are near or below 128.  Adding this condition
    # cleanly rejects blue/teal mats even when their brightness (L) is high.
    A = lab_img[:,:,1]

    # Blank out the ArUco marker region before thresholding so the white
    # marker paper is never mistaken for a finger (especially for small
    # fingers like the pinky where the marker may be larger).
    if aruco_corners is not None:
        padding = int(15 * scale)
        pts = aruco_corners.astype(np.int32)
        rect = cv2.boundingRect(pts)
        x, y, rw, rh = rect
        x  = max(0, x  - padding)
        y  = max(0, y  - padding)
        rw = min(W - x, rw + 2 * padding)
        rh = min(H - y, rh + 2 * padding)
        L[y:y+rh, x:x+rw] = 0

    _, skin_L = cv2.threshold(L, 130, 255, cv2.THRESH_BINARY)
    skin_A    = (A > 131).astype(np.uint8) * 255   # a* > 3: warm/reddish tone
    skin      = cv2.bitwise_and(skin_L, skin_A)
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
    # Filter out tiny noise contours (< 0.1% of image area)
    min_area = H * W * 0.001
    valid_cnts = [c for c in cnts if cv2.contourArea(c) > min_area]
    if not valid_cnts:
        valid_cnts = cnts
    # Reject warm-toned BACKGROUND (floor, furniture, shoes) that the a*/L
    # threshold also passes.  The marker is by protocol placed right beside the
    # finger on the same surface, so the finger blob is horizontally adjacent to
    # it while stray background is far away — measured in marker widths this is
    # a wide margin (finger ~0.0, background ~2.9-3.5 on the side-lit photos).
    # Without this, a warm strip at the frame edge that reaches HIGHER than the
    # fingertip beats the finger under the topmost rule below.
    if aruco_corners is not None and len(valid_cnts) > 1:
        mc_x = float(aruco_corners[:, 0].mean())
        m_side = max(cv2.boundingRect(aruco_corners.astype(np.int32))[2], 1)
        def _gap(c):
            bx, _, bw, _ = cv2.boundingRect(c)
            return max(bx - mc_x, mc_x - (bx + bw), 0.0) / m_side
        near = [c for c in valid_cnts if _gap(c) < 2.0]
        if near:
            valid_cnts = near
    # Pick the contour whose topmost pixel is highest in the image.
    # The target finger's nail tip is always at the TOP; any other skin
    # regions (wrist, adjacent fingers, holding hand) appear lower down.
    finger_cnt  = min(valid_cnts, key=lambda c: cv2.boundingRect(c)[1])
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
            if empty_streak > 10: break
            continue
        empty_streak = 0
        widths.append(int(cols[-1] - cols[0]))
        ledges.append(int(cols[0]  + fx))
        redges.append(int(cols[-1] + fx))
    return widths, ledges, redges


# ─────────────────────────────────────────────────────────────
# 3b. Free-edge (white / translucent nail tip) silhouette
# ─────────────────────────────────────────────────────────────

def detect_free_edge(image: np.ndarray, finger_mask: np.ndarray,
                     bbox: tuple, mpp: float,
                     aruco_corners: np.ndarray = None) -> dict:
    """
    Detect the true silhouette of the nail free edge — the white / semi-
    transparent nail tip that extends BEYOND the finger flesh, over the
    background.

    Why this is needed
    ------------------
    The warm-tone (a*) skin filter used in segment_finger() rejects the free
    edge because it is neutral / desaturated (no skin beneath it).  The old
    approach extended the tip with a plain L>135 brightness scan and then drew
    a *constant-width* flat top, which lost the almond/stiletto taper and
    stopped short of the true tip on dim, translucent edges.

    Method
    ------
    Above the finger flesh the free edge is the ONLY object in front of the
    background, so we segment it by BACKGROUND SUBTRACTION rather than by skin
    colour:
      1. Sample the background colour from patches on both sides of the finger
         (adaptive — works for navy/black/blue mats, not hard-coded to blue).
      2. Build a background mask via normalized Lab distance; invert it.
      3. Blank the ArUco marker so its white paper is never mistaken for nail.
      4. Starting at the fingertip centre, walk upward row by row, following
         the connected not-background run that contains the tracked centre.
         Each row yields the true left/right edge of the tip → the real taper.

    Returns
    -------
    dict with:
      tip_y      : int  — topmost detected free-edge row (== fy if none found)
      edges      : dict {row: (l, r)} for rows in [tip_y, fy)
    """
    H, W = image.shape[:2]
    fx, fy, fw, fh = bbox
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab).astype(np.int16)

    # Nail centre at the finger's top row (handles slightly angled fingers).
    top_row = min(fy + 2, H - 1)
    top_cols = np.where(finger_mask[top_row] > 0)[0]
    cx = int((top_cols[0] + top_cols[-1]) // 2) if len(top_cols) else fx + fw // 2

    # ── 1. Adaptive background colour (patches beside the finger) ──
    marker_box = None
    if aruco_corners is not None:
        marker_box = cv2.boundingRect(aruco_corners.astype(np.int32))
    samples = []
    for sy in range(max(0, fy - int(20 / mpp)), min(H, fy + int(20 / mpp)),
                    max(8, int(3 / mpp))):
        for sx in (fx - int(12 / mpp), fx - int(7 / mpp),
                   fx + fw + int(7 / mpp), fx + fw + int(12 / mpp)):
            if not (0 <= sx < W and 0 <= sy < H):
                continue
            if finger_mask[sy, sx] != 0:
                continue
            if marker_box is not None:
                mx, my, mw, mh = marker_box
                if mx - 20 <= sx <= mx + mw + 20 and my - 20 <= sy <= my + mh + 20:
                    continue
            samples.append(lab[sy, sx])
    if len(samples) < 8:
        # Not enough clean background — cannot separate the tip safely.
        return {"tip_y": fy, "edges": {}}
    samples = np.array(samples, dtype=np.float32)
    bg_mean = samples.mean(0)
    bg_std  = samples.std(0)

    # ── 2. Background mask via normalized Lab distance ──
    dist = np.sqrt(
        ((lab.astype(np.float32) - bg_mean) ** 2 / (bg_std ** 2 + 25)).sum(axis=2))
    notbg = (dist >= 3.0).astype(np.uint8) * 255

    # A CAST SHADOW is also "not background" — very much so, since it is far
    # darker than the mat — so plain background subtraction swallows it and
    # pushes the tip up into it.  Side-lighting guarantees such a shadow exists,
    # and when it falls just above the fingertip this cost ~1.4mm of length on
    # both light-box photos (mat L~110, shadow L~58-78, nail L~170+).  The free
    # edge is bright and translucent, so require candidates to be no darker than
    # the background rather than merely different from it.
    notbg[lab[:, :, 0].astype(np.float32) < bg_mean[0] * 0.85] = 0

    # ── 3. Blank the ArUco marker region ──
    if marker_box is not None:
        mx, my, mw, mh = marker_box
        pad = int(2 / mpp)
        notbg[max(0, my - pad):my + mh + pad, max(0, mx - pad):mx + mw + pad] = 0
    notbg = cv2.morphologyEx(
        notbg, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # ── 4. Grow the free edge upward from the fingertip centre ──
    edges = {}
    cur_c = cx
    tip_y = fy
    miss  = 0
    max_up   = min(fy, int(20 / mpp))          # scan up to 20 mm
    max_miss = max(3, int(1.0 / mpp))          # tolerate ~1 mm gaps
    for dy in range(1, max_up):
        row = fy - dy
        seg = notbg[row]
        if seg[cur_c] == 0:
            win = int(2.5 / mpp)
            lo  = max(0, cur_c - win)
            near = np.where(seg[lo:cur_c + win] > 0)[0]
            if len(near) == 0:
                miss += 1
                if miss > max_miss:
                    break
                continue
            cur_c = lo + int(near[len(near) // 2])
        l = cur_c
        while l > 0 and seg[l - 1] > 0:
            l -= 1
        r = cur_c
        while r < W - 1 and seg[r + 1] > 0:
            r += 1
        if (r - l) > fw * 1.35:            # leaked into background/marker
            miss += 1
            if miss > max_miss:
                break
            continue
        miss = 0
        edges[row] = (l, r)
        tip_y = row
        cur_c = (l + r) // 2

    if edges:
        print(f"  [Free edge] detected {len(edges)} rows, "
              f"extended {fy - tip_y}px up → tip_y={tip_y} "
              f"({(fy - tip_y) * mpp:.1f}mm)")
    else:
        print(f"  [Free edge] none detected (tip flush with flesh)")
    return {"tip_y": tip_y, "edges": edges}


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
        c_final = round(float(np.nanmedian(c_estimates)), 2)

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
# 4b. Nail axis refinement (correct off-centre nails)
# ─────────────────────────────────────────────────────────────

def refine_nail_centers(A_full: np.ndarray, finger_mask: np.ndarray,
                        ledges: list, redges: list,
                        fy: int, cuticle_y: int,
                        nail_half: float, mpp: float) -> dict:
    """
    Per-row horizontal centre of the NAIL PLATE (not the finger).

    The nail is often not centred on the finger — it can sit a few mm to one
    side, so a body polygon centred on the finger silhouette spills onto the
    lateral skin.  The nail bed is the reddest core of the finger, so within a
    band around the finger centre we take the centroid of the highest-a* (most
    reddish) pixels per row.  When the nail *is* centred this centroid coincides
    with the finger centre, so the correction is self-limiting and safe.

    Returns {row: nail_center_x} for rows in [fy, cuticle_y].
    """
    A = A_full.astype(np.float32)
    W = A.shape[1]
    rows = list(range(fy, cuticle_y + 1))
    raw  = np.full(len(rows), np.nan)
    band = max(int(nail_half * 1.3), int(3 / mpp))
    for i, row in enumerate(rows):
        si = row - fy
        fc = (ledges[si] + redges[si]) // 2 if 0 <= si < len(ledges) else None
        if fc is None:
            continue
        xl = max(0, fc - band)
        xr = min(W, fc + band)
        seg_a = A[row, xl:xr]
        seg_m = finger_mask[row, xl:xr] > 0
        vals  = seg_a[seg_m]
        if len(vals) < 5:
            continue
        thr  = np.percentile(vals, 65)          # reddest ~35 % = nail-bed core
        core = seg_m & (seg_a >= thr)
        xs   = np.where(core)[0]
        if len(xs) < 3:
            continue
        raw[i] = xl + float(xs.mean())

    valid = ~np.isnan(raw)
    if valid.sum() < 3:
        # Fall back to finger centre everywhere.
        return {row: ((ledges[row-fy] + redges[row-fy]) // 2)
                for row in rows if 0 <= row - fy < len(ledges)}
    idx = np.arange(len(rows))
    # The nail axis is essentially straight, so fit a ROBUST LINE rather than a
    # per-row centroid (which wobbles into the finger pad below the cuticle and
    # produces an hourglass outline).  One iteration of residual rejection drops
    # outlier rows (specular highlights, pad bleed) before the final fit.
    vy, vx = idx[valid].astype(float), raw[valid]
    coef = np.polyfit(vy, vx, 1)
    resid = np.abs(vx - np.polyval(coef, vy))
    keep = resid <= (2.0 * resid.std() + 1e-6)
    if keep.sum() >= 3:
        coef = np.polyfit(vy[keep], vx[keep], 1)
    axis = np.polyval(coef, idx)
    return {row: int(round(axis[i])) for i, row in enumerate(rows)}


def _fit_edge_robust(pts, deg=2, iters=6, min_pts=8):
    """Robust polynomial x=f(y) fit with iterative outlier rejection."""
    if len(pts) < min_pts:
        return None
    p = np.asarray(pts, float)
    xs, ys = p[:, 0], p[:, 1]
    keep = np.ones(len(p), bool)
    for _ in range(iters):
        coef = np.polyfit(ys[keep], xs[keep], deg)
        resid = np.abs(xs - np.polyval(coef, ys))
        new_keep = resid < max(np.median(resid) * 2.5, 2.0)
        if new_keep.sum() < min_pts:
            break
        keep = new_keep
    return coef, ys[keep].min(), ys[keep].max(), int(keep.sum())


def detect_cuticle_by_color(image: np.ndarray, cx: int, fy: int,
                            width_px: float, mpp: float):
    """
    Cuticle row from COLOUR, independent of lighting direction.

    The shadow-ridge trick that finds the lateral folds cannot find the cuticle:
    a groove only shadows when the light rakes ACROSS it, and the cuticle runs
    perpendicular to the lateral folds, so a single side light can reveal one or
    the other but never both (measured: lateral ridge +8.1, cuticle ridge -1.0 on
    the same photo).  This uses pigment instead, which does not care where the
    light is:

      • a* reaches a MINIMUM at the cuticle — the pale lunula sits there, with
        the redder nail bed above it and skin below.
      • b* is flat across the nail plate then CLIMBS below the cuticle, because
        skin is markedly yellower than the plate.

    Neither alone is reliable (each failed on a different one of the four test
    photos: a* snaps to a specular highlight on the plate, b* onset drifts when
    its baseline is contaminated).  Scoring a* minima by the b* rise that follows
    them fixes both failures, and the winning candidate wins by a wide margin
    rather than a hair.

    The search window comes from the measured WIDTH, not from the 0.91 W/L prior
    — that prior is badly wrong for some nails (true ratios of 0.64 and 0.86 have
    been measured on this user's hand) and a window built on it can exclude the
    real cuticle entirely.

    Returns the cuticle row, or None.
    """
    H, W = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    A, B = lab[:, :, 1], lab[:, :, 2]
    hw = max(3, int(2.5 / mpp))
    x0, x1 = max(0, cx - hw), min(W, cx + hw)
    if x1 - x0 < 3:
        return None
    n = max(3, int(0.6 / mpp) | 1)
    sm = lambda v: np.convolve(v, np.ones(n) / n, mode="same")
    a, b = sm(A[:, x0:x1].mean(1)), sm(B[:, x0:x1].mean(1))

    # Upper bound 1.7, not 2.2: the transverse skin creases below the cuticle
    # produce the same a*-minimum + b*-rise signature the cuticle does, and with
    # the window open to 2.2 a crease can outscore the real cuticle outright
    # (measured on a live frame: crease at 1.81 scored 5.18 while the true
    # cuticle scored 0.33, giving L=18.6mm instead of ~12.8mm).  Plate length in
    # units of the measured width is 1.072-1.453 across all seven reference
    # photos, so 1.7 keeps every validated case with ~17% margin while putting
    # the 1.81 crease out of reach.  The 0.7 floor is left alone - nothing has
    # ever failed against it.
    lo = fy + int(width_px * 0.7)
    hi = min(H - 3, fy + int(width_px * 1.7))
    if hi - lo < 5:
        return None
    mins = [y for y in range(lo + 2, hi - 2)
            if a[y] <= a[y - 1] and a[y] <= a[y + 1]]
    if not mins:
        return None
    rise = max(2, int(2.0 / mpp))
    best, best_s = None, -1e9
    for y in mins:
        below = float(np.mean(b[y:y + rise]))
        above = float(np.mean(b[max(lo, y - rise):y])) if y > lo else below
        depth = float(np.mean(a[max(lo, y - rise):y])) - a[y] if y > lo else 0.0
        s = (below - above) + depth
        if s > best_s:
            best, best_s = y, s
    if best_s < 1.0:                    # no convincing transition anywhere
        return None
    return best


def detect_lateral_edges(image: np.ndarray, finger_mask: np.ndarray,
                         axis_hint: int, fy: int, cuticle_y: int,
                         nail_half: float, mpp: float) -> dict:
    """
    True per-row LEFT/RIGHT nail-fold edges.

    No colour THRESHOLD separates nail plate from lateral finger skin — both are
    reddish — which is why the body used to fall back to a constant-width
    polygon.  But the fold between them is a narrow shadowed crease, and that is
    a RIDGE, detectable where a threshold is not:

        L dips  while  a* AND saturation both spike

    so the fold is a ridge in  S/12 + a*/4 - L/12  along each row.

    SIDE-LIGHTING (a light raking across the finger) makes this signature far
    stronger and is what the method was developed on — it typically yields
    150-250 inlier rows per side, versus 15-75 under flat lighting, where the
    fit rests on much thinner evidence.  Flat photos are not rejected outright
    (the measured width is usually still better than the skin-mask estimate,
    which spills onto the lateral skin) but they are less trustworthy.

    Two things this must guard against:
      • The side facing AWAY from the light gives a strong groove; the LIT side
        has no shadow in the groove (just a specular highlight) and responds
        far more weakly.  Hence the low threshold — the feature is the same on
        both sides, only the amplitude differs.
      • The finger SILHOUETTE against the dark mat is a much stronger ridge and
        will out-rank the real groove.  Eroding the finger mask to hide it does
        NOT work: on tilted fingers (e.g. thumb) the fold can sit only ~8 px
        inside the silhouette, so any erosion big enough to suppress the
        silhouette also destroys the real edge.  Instead we use the structural
        difference — a groove has SKIN ON BOTH SIDES, a silhouette has
        background outside it — and require skin to continue outward past each
        candidate peak.

    Returns {row: (left_x, right_x)} for rows in [fy, cuticle_y], or {} when the
    groove is not detectable, so the caller can fall back to constant width.
    """
    if "--no-lateral" in sys.argv:
        return {}
    H, W = finger_mask.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    L, A, S = lab[:, :, 0], lab[:, :, 1], hsv[:, :, 1]
    # "Is there still finger out there?" — used below to tell a fold (skin on
    # both sides) from the silhouette (background outside).  Derive it from the
    # finger mask rather than a fixed a* cut: a hard-coded threshold is tied to
    # one white balance and silently fails under different lighting (the LED box
    # photos read a*≈127-138 where the room-lamp photos read 140-150, so an
    # a*>136 test rejected the entire nail and killed one whole side).
    skin = cv2.dilate(finger_mask, np.ones((5, 5), np.uint8)) > 0

    # Horizontal top-hat isolates the narrow groove from the broad shading
    # gradient the side light necessarily creates across the finger.
    kx = max(21, (int(3.0 / mpp) | 1))          # ≈3 mm — wider than any groove
    th = lambda v, s: s * (v - cv2.blur(v, (kx, 1)))
    resp = th(S, 1) / 12.0 + th(A, 1) / 4.0 + th(L, -1) / 12.0
    # Vertical blur enforces row-to-row continuity (a real fold is a long line).
    resp = cv2.GaussianBlur(resp, (1, 15), 0)
    resp[cv2.erode(finger_mask, np.ones((3, 3), np.uint8)) == 0] = -99.0

    d_min = max(2, int(nail_half * 0.30))
    d_max = max(d_min + 4, int(nail_half * 1.60))
    out_lo, out_hi = max(2, int(0.15 / mpp)), max(6, int(0.5 / mpp))

    def scan_row(r, sgn, cx, lo, hi):
        row = resp[r]
        best, best_v = None, LATERAL_THR
        for d in range(lo, hi):
            i = int(cx) + sgn * d
            if i < 2 or i >= W - 2:
                break
            if row[i] <= best_v:
                continue
            if not (row[i] >= row[i - 1] and row[i] >= row[i + 1]):
                continue
            # Reject the finger silhouette: skin must continue OUTWARD.
            o0, o1 = ((i + out_lo, i + out_hi) if sgn > 0
                      else (i - out_hi, i - out_lo))
            o0, o1 = max(0, o0), min(W, o1)
            if o1 <= o0 or skin[r, o0:o1].mean() < 0.75:
                continue
            best, best_v = i, row[i]
        return best

    # Scan BEYOND the detected cuticle: the fold runs on past it, and the extra
    # rows stabilise the fit.  Results are clipped back to [fy, cuticle_y] later.
    r_lo = max(0, fy)
    r_hi = min(H, int(cuticle_y + 0.4 * max(cuticle_y - fy, 1)) + 1)

    # Pass 1 — the per-row axis from refine_nail_centers can carry a spurious
    # slope (it is fitted on the finger silhouette, not the nail), which drags
    # the search window off the fold on one side.  So anchor pass 1 on a STRAIGHT
    # vertical axis at the median centre and search wide.
    axis0 = int(axis_hint)
    wide_hi = max(d_min + 4, int(nail_half * 2.20))
    lpts, rpts = [], []
    for r in range(r_lo, r_hi):
        i = scan_row(r, -1, axis0, d_min, wide_hi)
        if i is not None:
            lpts.append((i, r))
        i = scan_row(r, 1, axis0, d_min, wide_hi)
        if i is not None:
            rpts.append((i, r))

    _dbg = "--debug" in sys.argv
    if _dbg:
        print(f"  [dbg] fy={fy} cut={cuticle_y} half={nail_half:.1f} axis0={axis0} "
              f"rows={r_lo}-{r_hi} pass1 L={len(lpts)} R={len(rpts)}")

    lf, rf = _fit_edge_robust(lpts), _fit_edge_robust(rpts)

    # One fold readable, the other not.  Normal on THUMBS, whose nail is broad
    # and flat with one genuinely shallow fold (the weak side measured 1.74 /
    # 2.45 / 1.88 against a 2.5 threshold across three separate photos, while the
    # lit side read 8.04 / 5.28 / 5.87).  Since the light sits on one side, the
    # fold it rakes across is always the strong one.
    #
    # MIRROR the good edge about the finger silhouette CENTRE.  Validated on the
    # three photos where both folds WERE found, by hiding one and predicting it:
    # left-edge RMS error 0.40 / 0.27 / 0.88 mm.  It works because the nail is
    # very nearly centred on the finger (true axis sits only 0.11-0.44mm off the
    # silhouette centre) and barely tapers (0.18-0.49mm over its length).
    #
    # NB mirroring about the CENTRE is not the same as matching the INSET from
    # each silhouette edge — that variant was tried and gave 10.31mm on a 12mm
    # thumb, worse than no reconstruction at all.  Use the centre.
    if (lf is None) != (rf is None):
        good, sgn = (rf, +1) if lf is None else (lf, -1)
        gcoef, gy0, gy1 = good[0], good[1], good[2]
        mirrored = []
        for r in range(int(gy0), int(gy1) + 1):
            xs = np.nonzero(finger_mask[r])[0]
            if len(xs) < 2:
                continue
            axis = (int(xs.min()) + int(xs.max())) / 2.0
            mirrored.append((2.0 * axis - np.polyval(gcoef, r), r))
        made = _fit_edge_robust(mirrored)
        if made is not None:
            if lf is None:
                lf = made
            else:
                rf = made
            print("  [Lateral] one fold only; opposite edge mirrored about "
                  "the finger centre (approx, ~0.3-0.9mm)")

    if lf is None or rf is None:
        if _dbg:
            print(f"  [dbg] pass1 fit failed L={lf is not None} R={rf is not None}")
        return {}

    # Pass 2 — re-scan in a tight window around the pass-1 curves.  This both
    # recovers rows the wide search lost to competing ridges and lets the fit
    # follow a genuinely tilted nail.
    for _ in range(2):
        lco, rco = lf[0], rf[0]
        half = max(4, int(nail_half * 0.35))
        lpts, rpts = [], []
        for r in range(r_lo, r_hi):
            mid = (np.polyval(lco, r) + np.polyval(rco, r)) / 2.0
            for sgn, coef, store in ((-1, lco, lpts), (1, rco, rpts)):
                d0 = abs(np.polyval(coef, r) - mid)
                lo, hi = max(2, int(d0 - half)), int(d0 + half)
                i = scan_row(r, sgn, mid, lo, hi)
                if i is not None:
                    store.append((i, r))
        nl, nr = _fit_edge_robust(lpts), _fit_edge_robust(rpts)
        if nl is None or nr is None:
            break
        lf, rf = nl, nr
    if _dbg:
        print(f"  [dbg] pass2 L={len(lpts)} R={len(rpts)}")

    lcoef, ly0, ly1, lin = lf
    rcoef, ry0, ry1, rin = rf

    # Only trust rows where BOTH sides had inlier support; extrapolating a
    # quadratic past its data swings wildly.
    y0, y1 = int(max(ly0, ry0)), int(min(ly1, ry1))
    y0, y1 = max(y0, fy), min(y1, cuticle_y)      # clip back to the nail plate
    if y1 - y0 < max(5, int(1.5 / mpp)):
        if _dbg:
            print(f"  [dbg] span too short {y0}-{y1}")
        return {}

    edges = {}
    for r in range(y0, y1 + 1):
        lx, rx = float(np.polyval(lcoef, r)), float(np.polyval(rcoef, r))
        if rx - lx < nail_half * 0.8 or rx - lx > nail_half * 3.0:
            return {}                     # implausible width → distrust wholesale
        edges[r] = (lx, rx)
    print(f"  [Lateral] side-lit fold edges: {lin}L/{rin}R inliers, "
          f"rows {y0}-{y1}, width {np.median([e[1]-e[0] for e in edges.values()])*mpp:.2f}mm")
    return edges


# ─────────────────────────────────────────────────────────────
# 5. TOP photo full measurement
# ─────────────────────────────────────────────────────────────

def measure_top(image: np.ndarray, mpp: float,
                finger_mask: np.ndarray, bbox: tuple,
                nail_plate_mask: np.ndarray = None,
                aruco_corners: np.ndarray = None) -> dict:
    """
    nail_plate_mask : optional uint8 binary mask (255=nail plate).
        When provided, width is measured from the nail plate boundary instead
        of the full finger skin boundary, and the polygon margin is set to 0
        so the outline follows the detected nail plate edges exactly.
        finger_mask is still used for skin-tone sampling.
    """
    H, W  = image.shape[:2]
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fx, fy, fw, fh = bbox
    tip_y = fy

    # ── Detect the free-edge (white/translucent nail tip) silhouette ─
    # The a*-channel skin filter excludes the free edge (neutral hue, no skin
    # beneath it).  detect_free_edge() recovers it by BACKGROUND SUBTRACTION
    # and returns the TRUE per-row left/right edges, so the almond/stiletto
    # taper up to the real tip is preserved (instead of a constant-width top).
    _lab   = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L_full = _lab[:, :, 0]
    A_full = _lab[:, :, 1]   # a* channel: warm >128, cool <128 (OpenCV scale)

    fe          = detect_free_edge(image, finger_mask, bbox, mpp, aruco_corners)
    tip_y       = fe["tip_y"]
    free_edges  = fe["edges"]          # {row: (l, r)} for rows in [tip_y, fy)

    # ── Row scan ─────────────────────────────────────────────
    # Use nail plate mask for width/edge scan when available (more accurate).
    width_mask = nail_plate_mask if nail_plate_mask is not None else finger_mask
    widths, ledges, redges = row_scan(width_mask, bbox, mpp, H)

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

    # When the nail plate boundary is known exactly, use it directly (margin=0).
    # For finger-skin masks the boundary includes surrounding skin, so shrink by 8%.
    margin    = 0.0 if nail_plate_mask is not None else 0.08
    nail_half = width_px * (0.5 - margin)

    # ── Build per-row nail center from skin mask; width stays fixed ──
    # Outward expansion from the skin-mask edge is intentionally NOT done:
    # the finger skin just outside the mask also passes a warm-tone threshold,
    # causing the polygon to widen into the lateral finger skin.
    # Instead:
    #  • nail_half (measured in the stable zone near the tip) is used as a
    #    constant half-width for the nail body polygon — prevents widening at
    #    the cuticle where the finger is physically broader.
    #  • Per-row center (cx) still tracks the ledges/redges so angled fingers
    #    are followed correctly.
    #  • Free-edge zone uses a sinusoidal taper from nail_half → 0 at the tip.
    if nail_plate_mask is None:
        nail_le = list(ledges)
        nail_re = list(redges)

        # Prepend the DETECTED free-edge silhouette (rows tip_y … fy-1).
        # detect_free_edge() returns the true per-row left/right edges of the
        # white/translucent tip, so the almond/stiletto taper up to the real
        # point is preserved.  Rows the detector missed (small gaps) are filled
        # by linear interpolation between the nearest detected rows, anchored at
        # the bottom to the flesh boundary (ledges[0]/redges[0] at row fy) for
        # a seamless join into the nail body.
        n_free = fy - tip_y            # number of free-edge rows
        if n_free > 0 and free_edges:
            idx   = np.arange(n_free)          # index i → row tip_y + i
            lvals = np.full(n_free, np.nan)
            rvals = np.full(n_free, np.nan)
            for i in range(n_free):
                row = tip_y + i
                if row in free_edges:
                    lvals[i], rvals[i] = free_edges[row]
            known = ~np.isnan(lvals)
            # Append a virtual bottom anchor at row fy (index n_free).
            k_idx = np.concatenate([idx[known], [n_free]])
            k_l   = np.concatenate([lvals[known], [float(ledges[0])]])
            k_r   = np.concatenate([rvals[known], [float(redges[0])]])
            free_le = np.interp(idx, k_idx, k_l).astype(int).tolist()
            free_re = np.interp(idx, k_idx, k_r).astype(int).tolist()
            nail_le = free_le + nail_le
            nail_re = free_re + nail_re
        # else: no free edge detected — nail starts flush at fy (nail_le/re unchanged)
    else:
        nail_le = ledges
        nail_re = redges

    # ── Cuticle detection ─────────────────────────────────────
    hw       = int(width_px * 0.38)
    x1, x2   = max(0, tip_x - hw), min(W, tip_x + hw)
    w_mm_est = width_px * (1.0 - 2 * margin) * mpp

    # On side-lit photos the fold edges give the TRUE width, which the cuticle
    # estimate below depends on (cuticle ≈ fy + width/0.91).  Run the detector
    # here on a provisional row range so the W/L baseline is built on a measured
    # width rather than the skin-mask approximation — a mask-derived width that
    # is a few mm small drags the cuticle several mm up the nail.
    _prov_cut = min(H - 1, fy + int(nail_half * 3.0))
    _lat_pre = detect_lateral_edges(
        image, finger_mask, tip_x, fy, _prov_cut, nail_half, mpp)
    if _lat_pre:
        w_mm_est = max(rx - lx for lx, rx in _lat_pre.values()) * mpp
        print(f"  [Cuticle] using measured fold width {w_mm_est:.2f}mm")

    # Colour-based cuticle first: it is lighting-independent and measured within
    # ±0.81mm on all four reference photos, versus -3.6mm for the W/L fallback
    # when the L-gradient finds nothing.
    _cut_color = detect_cuticle_by_color(
        image, tip_x, fy, w_mm_est / mpp, mpp)

    # W/L ratio baseline (Jung et al. 2015, ~0.91).
    # NOTE: W/L applies to the NAIL PLATE only (from fy, the natural fingertip
    # boundary, to the cuticle).  tip_y may have been extended upward to include
    # the white free edge — that extension must NOT be added to the W/L estimate.
    wl_len_px  = int((w_mm_est / 0.91) / mpp)
    wl_cuticle = fy + wl_len_px   # cuticle position relative to natural nail start

    # Gradient-based cuticle detection:
    # The W/L estimate (Jung et al. plate ratio 0.91) is a strong, accurate
    # prior for the cuticle: the NAIL PLATE (smile-line→cuticle) follows ~0.91
    # even on long nails — it is the free edge that makes them long, not the
    # plate — so wl_cuticle lands within ~1 mm of the true cuticle.  We use the
    # brightness gradient only to REFINE it locally: within a tight ±3.5 mm
    # window we snap to the strong transition (|grad| ≥ 0.5, either sign)
    # CLOSEST to wl_cuticle.  "Closest" beats "first" (which undershoots onto a
    # plate highlight) and "strongest" (which overshoots onto the bright finger
    # pad below the cuticle) — both seen across fingers.
    L_img = L_full.astype(np.float32)   # reuse already-computed L channel
    # Tight window centred on the W/L estimate, but never above fy+8 mm (which
    # would risk the free-edge/smile transition).
    search_top = max(fy + int(8 / mpp), wl_cuticle - int(3.5 / mpp))
    search_bot = min(H, wl_cuticle + int(3.5 / mpp))
    cuticle_y  = wl_cuticle   # default fallback

    if search_bot > search_top + 10:
        # Per-row brightness: track the finger center dynamically using the
        # left/right edges already computed by row_scan (handles angled fingers).
        # Use a band = ±50 % of nail_half centred on the tracked nail axis.
        hw_band    = max(int(nail_half * 0.5), int(2 / mpp))
        n_rows     = search_bot - search_top
        row_means  = np.full(n_rows, np.nan)
        for r in range(n_rows):
            row    = search_top + r
            scan_i = row - fy          # index into ledges / redges arrays
            if 0 <= scan_i < len(ledges):
                c_x = (ledges[scan_i] + redges[scan_i]) // 2
            else:
                c_x = tip_x
            xl  = max(0, c_x - hw_band)
            xr  = min(W, c_x + hw_band)
            px  = L_img[row, xl:xr][finger_mask[row, xl:xr] > 0]
            if len(px) > 3:
                row_means[r] = float(np.mean(px))

        valid = ~np.isnan(row_means)
        if np.sum(valid) > 10:
            row_means_sm = uniform_filter1d(
                np.where(valid, row_means, 0), size=max(3, int(1.5 / mpp)))
            grad = np.gradient(row_means_sm)
            grad[~valid] = 0
            # Among strong transitions (|grad| >= 0.5, drop or rise) pick the
            # one CLOSEST to the W/L estimate.  If none qualify, keep wl_cuticle.
            wl_local = wl_cuticle - search_top     # W/L estimate as a local idx
            cands = [i for i in range(len(grad))
                     if valid[i] and abs(grad[i]) >= 0.5]
            if cands:
                best_idx = min(cands, key=lambda i: abs(i - wl_local))
                cuticle_y = search_top + best_idx
                sign = "drop" if grad[best_idx] < 0 else "rise"
                print(f"  [Cuticle] gradient: y={cuticle_y} "
                      f"(L {sign}={grad[best_idx]:+.2f}, "
                      f"length={(cuticle_y-tip_y)*mpp:.1f}mm, "
                      f"W/L baseline={(wl_cuticle-tip_y)*mpp:.1f}mm)")
            else:
                print(f"  [Cuticle] W/L fallback (no strong transition): "
                      f"y={wl_cuticle}  length={(wl_cuticle-tip_y)*mpp:.1f}mm")
        else:
            print(f"  [Cuticle] W/L fallback (sparse): "
                  f"{w_mm_est:.1f}mm / 0.91 = {w_mm_est/0.91:.1f}mm")
    else:
        print(f"  [Cuticle] W/L estimate: {w_mm_est:.1f}mm / 0.91 "
              f"= {w_mm_est/0.91:.1f}mm")

    # Colour (a* minimum + b* rise) overrides the L-gradient / W-L result.  It
    # does not depend on light direction, and unlike the 0.91 prior it does not
    # assume a population-average nail shape.
    if _cut_color is not None:
        print(f"  [Cuticle] colour: y={_cut_color} "
              f"(length={(_cut_color-tip_y)*mpp:.1f}mm) "
              f"[was y={cuticle_y}, {(cuticle_y-tip_y)*mpp:.1f}mm]")
        cuticle_y = _cut_color

    cut_idx   = min(cuticle_y - tip_y, len(widths)-1)
    length_px = float(cuticle_y - tip_y)

    # ── C-curve from nail fold ────────────────────────────────
    cc_data = estimate_ccurve_from_nailfold(
        image, finger_mask,
        tip_y, cuticle_y,
        tip_x, nail_half, mpp
    )

    # ── Refine the nail axis (correct off-centre nails) ───────
    # The nail often sits off the finger centre; centre the body on the actual
    # nail-bed core so the polygon doesn't spill onto the lateral skin.
    nail_centers = refine_nail_centers(
        A_full, finger_mask, ledges, redges, fy, cuticle_y, nail_half, mpp)

    # ── True lateral fold edges ───────────────────────────────
    # Returns {} when the fold is not detectable, in which case the body keeps
    # the constant-width fallback below.
    _axis = int(np.median(list(nail_centers.values()))) if nail_centers else tip_x
    lateral = detect_lateral_edges(
        image, finger_mask, _axis, fy, cuticle_y, nail_half, mpp)

    # Count the GENUINE fold detections before the hold-extension below fills
    # the gaps, because that count is the honest measure of how much evidence
    # the width rests on.  Side-lit photos give 52-73% of rows; the flat-lit
    # light box gives 5-13%, and a width fitted on a handful of rows is the
    # likeliest reason the reading jumps between two values.
    _lateral_rows = len(lateral)
    _lateral_span = max(cuticle_y - fy, 1)

    # The fold is only detectable over part of the nail (typically not right up
    # at the tip, where it runs under the free edge).  Rows outside that span
    # would otherwise fall back to the constant-width branch below, which is
    # centred differently — producing a visible STEP where the two meet.  Extend
    # the measured edges over the remaining rows by holding the end value, so the
    # outline stays continuous.
    if lateral:
        _lrows = sorted(lateral)
        _first, _last = _lrows[0], _lrows[-1]
        for _r in range(fy, _first):
            lateral[_r] = lateral[_first]
        for _r in range(_last + 1, cuticle_y + 1):
            lateral[_r] = lateral[_last]

    # ── Build nail polygon ────────────────────────────────────
    # When a nail plate mask is available (e.g. from GrabCut), use its
    # actual contour — this follows the true nail plate boundary including
    # natural curvature, instead of the synthetic arc+straight construction.
    _use_contour = False
    if nail_plate_mask is not None:
        # Clip mask to [tip_y, cuticle_y] — sides come from GrabCut,
        # bottom boundary comes from gradient-based cuticle detection.
        _nm = nail_plate_mask.copy()
        _nm[:tip_y, :] = 0
        _nm[cuticle_y:, :] = 0
        _cnts, _ = cv2.findContours(
            _nm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if _cnts:
            _cnt = max(_cnts, key=cv2.contourArea)
            if cv2.contourArea(_cnt) > 50:
                nail_polygon = _cnt.reshape(-1, 2).astype(np.int32)
                _use_contour = True

    if not _use_contour:
        # nail_le/nail_re span from tip_y to cuticle_y (index i → y = tip_y + i);
        # indices 0..n_free-1 are the detected free-edge silhouette, the rest is
        # the nail body.
        n_pts  = cut_idx + 1
        n_free = fy - tip_y

        # ── Width & centre continuity at the free-edge / body join ──
        # The nail does not change width where the free edge (over background)
        # meets the body (over flesh).  So take the body half-width and centre
        # from the free-edge silhouette JUST above the join — not an independent
        # stable-zone measurement — which removes the step/kink there.  Rows are
        # sampled a little above fy to avoid the bottom rows that blend into the
        # (wider) finger-flesh boundary.
        if n_free > 3 and free_edges:
            j1 = n_free - 1
            j0 = max(0, j1 - max(3, int(2.0 / mpp)))
            join_hw = float(np.median(
                [(nail_re[i] - nail_le[i]) / 2.0 for i in range(j0, j1 + 1)]))
            join_cx = float(np.median(
                [(nail_re[i] + nail_le[i]) / 2.0 for i in range(j0, j1 + 1)]))
            body_half = int(round(max(join_hw, nail_half * 0.6)))
        else:
            join_cx   = float(nail_centers.get(fy, tip_x))
            body_half = int(nail_half)

        # Re-anchor the refined nail axis so it is continuous with the free-edge
        # centre at the join (keeps the axis slope, kills the sideways step).
        axis_shift = join_cx - float(nail_centers.get(fy, join_cx))

        # ── Per-row side edges for the whole nail ──
        left_arr, right_arr = [], []
        for i in range(n_pts):
            y_pos = tip_y + i
            if i < n_free and i < len(nail_le):
                left_arr.append(float(nail_le[i]))
                right_arr.append(float(nail_re[i]))
            elif y_pos in lateral:
                # Measured fold edges — true per-row width, not a constant.
                lx, rx = lateral[y_pos]
                left_arr.append(lx)
                right_arr.append(rx)
            else:
                if y_pos in nail_centers:
                    cx = nail_centers[y_pos] + axis_shift
                elif i < len(nail_le):
                    cx = (nail_le[i] + nail_re[i]) / 2.0
                else:
                    cx = tip_x
                left_arr.append(cx - body_half)
                right_arr.append(cx + body_half)
        left_arr  = np.array(left_arr, float)
        right_arr = np.array(right_arr, float)

        # Smooth the side edges to remove pixel jaggedness from the background-
        # subtracted free edge AND to blend the free-edge/body join into one
        # continuous curve — but PROTECT the tip (top rows where the two sides
        # converge) so the almond point stays sharp.
        tip_protect = min(n_pts, max(2, int(2.5 / mpp)))
        win = max(3, (int(2.0 / mpp) | 1))
        if n_pts > tip_protect + 3:
            left_arr[tip_protect:]  = uniform_filter1d(left_arr[tip_protect:],  win)
            right_arr[tip_protect:] = uniform_filter1d(right_arr[tip_protect:], win)

        left_pts  = [[int(round(left_arr[i])),  tip_y + i] for i in range(n_pts)]
        right_pts = [[int(round(right_arr[i])), tip_y + i] for i in range(n_pts)]

        # Cuticle arc — centred on the (re-anchored) nail axis at the cuticle row
        # Anchor it to the measured fold edges there when we have them, so the
        # arc meets the sides instead of stepping in/out from them.
        if cuticle_y in lateral:
            _lx, _rx = lateral[cuticle_y]
            cuticle_cx = int(round((_lx + _rx) / 2.0))
            arc_w      = (_rx - _lx) / 2.0
        else:
            cuticle_cx = int(round(nail_centers.get(cuticle_y, tip_x) + axis_shift))
            arc_w      = body_half
        arc_h       = arc_w * 0.28
        cuticle_arc = []
        for i in range(41):
            angle = np.pi * i / 40
            ax = cuticle_cx - arc_w * np.cos(angle)
            ay = cuticle_y  - arc_h * np.sin(angle)
            cuticle_arc.append([int(ax), int(ay)])

        # No separate tip_arc: per-row edges already taper to a point at tip_y.
        # Build the full outline from right side → cuticle arc → left side (reversed).
        full_poly    = (right_pts +
                        cuticle_arc +
                        list(reversed(left_pts)))
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
    # Sample a band of skin on the finger below the cuticle, well away
    # from the nail plate and any nail polish.
    offset_px    = int(5  / mpp)   # skip 5 mm below the cuticle fold
    band_px      = int(15 / mpp)   # then sample a 15 mm tall band
    sample_top   = min(cuticle_y + offset_px, H - 1)
    sample_bot   = min(cuticle_y + offset_px + band_px, H)
    skin_mask    = np.zeros((H, W), np.uint8)
    skin_mask[sample_top:sample_bot, :] = 255
    skin_mask    = cv2.bitwise_and(skin_mask, finger_mask)
    pixels = image[skin_mask > 0]
    if len(pixels):
        b, g, r = [int(np.median(pixels[:,i])) for i in range(3)]
        hex_color = f"#{r:02X}{g:02X}{b:02X}"
    else:
        hex_color = "#FFFFFF"

    # Prefer the MEASURED fold-to-fold width (side-lit photos).  The nail is
    # widest near the free edge and narrows toward the cuticle, so report the
    # widest measured row — that is the dimension a tip has to cover.
    if lateral:
        half_px = max(rx - lx for lx, rx in lateral.values()) / 2.0
    else:
        half_px = float(nail_half)
    w_mm = round(half_px * 2 * float(mpp), 2)
    l_mm = round(float(length_px) * float(mpp), 2)

    return {
        "width_mm":        w_mm,
        "width_source":    "lateral_folds" if lateral else "constant_half",
        "length_mm":       l_mm,
        "skin_tone_hex":   hex_color,
        "nail_polygon_px": smooth.tolist(),
        **cc_data,
        "_nail_half":      nail_half,
        "_tip_x":          tip_x,
        "_tip_y":          tip_y,
        "_cuticle_y":      cuticle_y,
        "_plate_start":    plate_start,
        "_lateral_rows":   _lateral_rows,
        "_lateral_span":   _lateral_span,
    }


# ─────────────────────────────────────────────────────────────
# 6. W/L correction
# ─────────────────────────────────────────────────────────────

def apply_wl_correction(finger: str, width_mm: float, length_mm: float) -> dict:
    ref         = WL_STANDARD.get(finger, {"ratio": 0.91, "std_dev": 0.07})
    std_wl      = ref["ratio"]
    std_sd      = ref["std_dev"]
    measured_wl = round(float(width_mm) / float(length_mm), 3) if length_mm else 0.0
    wl_diff     = round(float(measured_wl) - float(std_wl), 3)
    within_1sig = bool(abs(wl_diff) <= std_sd)
    corr_length = round(float(width_mm) / float(std_wl), 2)
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

def draw_annotated(image, data, aruco_corners, finger):
    """Draw the measurement overlay and return it (full resolution).

    Split out of save_annotated so the live camera preview
    (nail_live.py) can render the same overlay without touching disk.
    """
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

    return vis


def save_annotated(image, data, aruco_corners, finger, save_path):
    vis   = draw_annotated(image, data, aruco_corners, finger)
    scale = 900 / vis.shape[0]
    cv2.imwrite(save_path, cv2.resize(vis, (int(vis.shape[1]*scale), 900)))
    print(f"  [Saved] {save_path}")


# ─────────────────────────────────────────────────────────────
# 8. Single finger pipeline
# ─────────────────────────────────────────────────────────────

def measure_finger(top_path: str, finger: str,
                   aruco_size_mm: float, output_dir: str,
                   ccurve_path: str = None) -> dict:
    """
    ccurve_path : optional path to an end-on (tip-facing) photo.
        When provided, C-curve is measured directly from that image
        using the nail width as scale reference (no ArUco needed).
        When omitted, the brightness-drop fallback is used instead.
    """
    print(f"\n{'='*55}")
    print(f"  Measuring: {finger.upper()}")
    print(f"{'='*55}")

    top_img = cv2.imread(top_path)
    if top_img is None:
        sys.exit(f"ERROR: Cannot open '{top_path}'")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[1/3] ArUco + finger segmentation …")
    mpp, aruco_corners, marker_id = detect_aruco(top_img, aruco_size_mm)
    finger_mask, _, bbox          = segment_finger(top_img, aruco_corners)

    print(f"\n[2/3] Nail measurement + C-curve …")
    data = measure_top(top_img, mpp, finger_mask, bbox,
                       aruco_corners=aruco_corners)

    # ── Override C-curve with end-on measurement if photo is provided ──
    use_endon = (ccurve_path and os.path.isfile(ccurve_path)
                 and _ENDON_AVAILABLE)
    if use_endon:
        print(f"\n  [C-curve] end-on photo: {ccurve_path}")
        debug_path = os.path.join(output_dir, f"{finger}_ccurve_debug.jpg")
        try:
            cc = _endOn_ccurve(ccurve_path,
                               width_mm=data["width_mm"],
                               debug_out=debug_path)
            data["c_curve_mm"]    = cc["c_curve_mm"]
            data["arc_radius_mm"] = cc["arc_radius_mm"]
            data["_ccurve_method"] = "end-on photo"
            print(f"  [C-curve] OK end-on  "
                  f"h={cc['c_curve_mm']}mm  R={cc['arc_radius_mm']}mm  "
                  f"(debug -> {debug_path})")
        except Exception as e:
            print(f"  [C-curve] WARN end-on failed ({e}), "
                  f"keeping brightness fallback")
            data["_ccurve_method"] = "brightness fallback (end-on error)"
    else:
        data["_ccurve_method"] = "brightness fallback"
        if ccurve_path and not os.path.isfile(ccurve_path):
            print(f"  [C-curve] WARN ccurve photo not found: {ccurve_path}")

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
    print(f"  │  C-curve         : {data['c_curve_mm']} mm  "
          f"[{data.get('_ccurve_method', '?')}]")
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
# 10. Profile builder (size classification + skin tone)
# ─────────────────────────────────────────────────────────────

def build_profile(results: list) -> list:
    """
    For each measured finger return a small profile dict with:
      - finger name
      - nail_size: per-finger size classification relative to Asian women
                   standard (much_smaller / smaller / average / larger /
                   much_larger)
      - skin_tone: hex colour sampled from the skin ring around the nail
    """
    profiles = []
    for r in results:
        finger = r["finger"]
        std    = STANDARD_NAILS.get(finger)
        if std is None:
            continue
        w      = r["width_mm"]
        l      = r.get("corrected_length_mm") or r["length_mm"]
        w_cat  = _size_category(w - std["width_mm"])
        l_cat  = _size_category(l - std["length_mm"])
        profiles.append({
            "finger":    finger,
            "nail_size": _overall_size(w_cat, l_cat),
            "skin_tone": r.get("skin_tone_hex", "#FFFFFF"),
        })
    return profiles


# ─────────────────────────────────────────────────────────────
# 11. CLI
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Automatic nail measurement from single top photo")
    p.add_argument("--top",          help="Top photo path")
    p.add_argument("--ccurve-top",   default=None,
                   help="End-on C-curve photo path (optional, single finger)")
    p.add_argument("--finger",       default="index", choices=FINGER_NAMES)
    p.add_argument("--batch",        action="store_true")
    p.add_argument("--fingers",      nargs="+", default=FINGER_NAMES)
    p.add_argument("--tops",         nargs="+")
    p.add_argument("--ccurve-tops",  nargs="+", default=None,
                   help="End-on C-curve photos (batch, same count as --tops; "
                        'use "" or "none" to skip per finger)')
    p.add_argument("--aruco-size",   type=float, default=20.0)
    p.add_argument("--debug", action="store_true", help="verbose detector diagnostics")
    p.add_argument("--no-lateral", action="store_true",
                   help="disable side-lit lateral fold-edge detection")
    p.add_argument("--output",       default="nail_results_v6")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    results = []

    if args.batch:
        if not args.tops:
            sys.exit("ERROR: --batch requires --tops")
        if len(args.tops) != len(args.fingers):
            sys.exit("ERROR: --tops and --fingers must have equal length")

        # Build per-finger ccurve path list (None when not provided / "none")
        ccurve_map = {}
        if args.ccurve_tops:
            if len(args.ccurve_tops) != len(args.fingers):
                sys.exit("ERROR: --ccurve-tops must have the same count as --tops")
            for finger, cp in zip(args.fingers, args.ccurve_tops):
                if cp and cp.lower() not in ("", "none"):
                    ccurve_map[finger] = cp

        for finger, top in zip(args.fingers, args.tops):
            r = measure_finger(top, finger, args.aruco_size, args.output,
                               ccurve_path=ccurve_map.get(finger))
            results.append(r)
    else:
        if not args.top:
            sys.exit("ERROR: Provide --top photo path, or use --batch")
        r = measure_finger(args.top, args.finger,
                           args.aruco_size, args.output,
                           ccurve_path=args.ccurve_top)
        results.append(r)

    payload   = build_payload(results, args.aruco_size)
    json_path = os.path.join(args.output, "nail_measurements.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    profiles      = build_profile(results)
    profile_path  = os.path.join(args.output, "profile.json")
    # Single finger → save the dict directly; batch → save as a list.
    profile_data  = profiles[0] if len(profiles) == 1 else profiles
    with open(profile_path, "w") as f:
        json.dump(profile_data, f, indent=2)

    print(f"\n{'='*55}")
    print(f"[OK] Saved -> {json_path}")
    print(f"   Fingers: {[r['finger'] for r in results]}")
    print(f"\nNext: python nail_exact_stl.py --input {json_path}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()