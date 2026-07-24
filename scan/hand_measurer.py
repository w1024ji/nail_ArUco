"""
hand_measurer.py
----------------
Full-hand nail measurement from a single photo.

The photo should show a hand (palm down, fingers spread) on a dark background
with an ArUco marker visible in the frame.  All four fingers (index, middle,
ring, pinky) plus the thumb are measured in one pass.

Usage:
    python hand_measurer.py --image wonji.jpg --aruco-size 20 --output hand_results/
"""

import argparse
import json
import os
import sys
import warnings

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

# Import shared helpers from nail_measurer — do NOT modify that file.
from nail_measurer import (
    detect_aruco,
    measure_top,
    apply_wl_correction,
    build_payload,
    build_profile,
    save_annotated,
    NAIL_COLORS,
)


# ─────────────────────────────────────────────────────────────
# 1. Hand segmentation
# ─────────────────────────────────────────────────────────────

def segment_hand(image: np.ndarray, aruco_corners: np.ndarray = None):
    """
    Segment the entire hand blob (all fingers + palm) from the image.

    Uses the same L-channel threshold approach as segment_finger() in
    nail_measurer.py, but keeps ALL skin pixels (not just the largest
    connected component) so that fingers joined at the palm are not split.

    Returns
    -------
    hand_mask    : uint8 binary mask, 255 = skin
    hand_contour : largest contour (the hand outline)
    bbox         : cv2.boundingRect of hand_contour
    """
    H, W = image.shape[:2]
    scale    = max(W, H) / 2000.0
    ks_large = max(9, int(9  * scale) | 1)
    ks_small = max(5, int(5  * scale) | 1)
    print(f"  [Segment-Hand] {W}x{H}  scale={scale:.2f}  "
          f"kernels={ks_large},{ks_small}")

    L = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)[:, :, 0]

    # Blank out the ArUco marker region so it is not mistaken for skin.
    if aruco_corners is not None:
        padding = int(15 * scale)
        pts  = aruco_corners.astype(np.int32)
        rect = cv2.boundingRect(pts)
        x, y, rw, rh = rect
        x  = max(0, x  - padding)
        y  = max(0, y  - padding)
        rw = min(W - x, rw + 2 * padding)
        rh = min(H - y, rh + 2 * padding)
        L[y:y + rh, x:x + rw] = 0

    _, skin = cv2.threshold(L, 130, 255, cv2.THRESH_BINARY)
    kL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks_large, ks_large))
    kS = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks_small, ks_small))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kL, iterations=3)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN,  kS, iterations=2)

    cnts, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError(
            "No hand detected.\n"
            "  -> Use a dark background (navy, black, dark green).\n"
            "  -> Ensure the hand is well-lit and fully in frame."
        )

    # Include all skin blobs large enough to be a finger.
    # The pinky can sometimes be disconnected from the main hand blob when the
    # inter-finger gap is too wide to bridge with morphological closing.
    # Threshold: at least 3% of the largest blob's area.
    max_area = max(cv2.contourArea(c) for c in cnts)
    min_area_px = max_area * 0.03
    sig_cnts = [c for c in cnts if cv2.contourArea(c) >= min_area_px]
    if not sig_cnts:
        sig_cnts = [max(cnts, key=cv2.contourArea)]

    hand_mask = np.zeros((H, W), np.uint8)
    for c in sig_cnts:
        cv2.drawContours(hand_mask, [c], -1, 255, -1)

    # Union bounding box across all significant blobs.
    all_pts  = np.concatenate([c.reshape(-1, 2) for c in sig_cnts])
    ux, uy, uw, uh = cv2.boundingRect(all_pts)
    bbox = (ux, uy, uw, uh)

    hand_contour = max(sig_cnts, key=cv2.contourArea)

    print(f"  [Segment-Hand] {len(sig_cnts)} blob(s), "
          f"union bbox: x={ux} y={uy} w={uw} h={uh}")
    return hand_mask, hand_contour, bbox


# ─────────────────────────────────────────────────────────────
# 2. Fingertip detection
# ─────────────────────────────────────────────────────────────

def find_fingertips(hand_mask: np.ndarray, hand_bbox: tuple, mpp: float):
    """
    Detect fingertip locations from the hand silhouette.

    Strategy:
      - For each column inside the hand bounding box, record the topmost
        (lowest y value) skin pixel.
      - Smooth the resulting 1-D profile.
      - Find local minima (= tips that protrude upward, i.e. have small y).

    Parameters
    ----------
    hand_mask : binary mask of the segmented hand
    hand_bbox : (x, y, w, h) bounding rect of the hand
    mpp       : mm per pixel

    Returns
    -------
    List of (tip_x, tip_y) tuples sorted left-to-right.
    """
    hx, hy, hw, hh = hand_bbox
    H = hand_mask.shape[0]

    # Build per-column topmost-skin-pixel profile.
    col_xs = []
    top_ys = []
    for col in range(hx, hx + hw):
        rows = np.where(hand_mask[:, col] > 0)[0]
        if len(rows) == 0:
            continue
        col_xs.append(col)
        top_ys.append(int(rows[0]))

    if len(top_ys) < 10:
        raise RuntimeError("Not enough skin columns found - hand segmentation may have failed.")

    col_xs = np.array(col_xs, dtype=int)
    top_ys = np.array(top_ys, dtype=float)

    # Try progressively relaxed peak-detection thresholds until ≥4 main tips
    # are found.  "Main tips" = in the top 40 % of the hand bounding box
    # (fingers pointing upward); deeper tips are thumb / palm artefacts.
    # Each tuple: (smooth_mm, distance_mm, prominence_mm)
    # Try stricter passes first; fall back to looser ones for close-together fingers.
    PASSES = [
        (5, 8, 10),   # original strict pass
        (3, 5,  5),
        (3, 3,  2),
        (2, 3,  2),   # smaller smooth window for fingers that aren't well spread
    ]
    # For full-hand images, the outermost fingers (index, pinky) legitimately
    # sit near the bbox edges.  The padding artefact is at most 1-2 pixels, so
    # a fixed 10-pixel margin is plenty.
    edge_margin = 10
    tips = []
    for smooth_mm, dist_mm, prom_mm in PASSES:
        win = max(3, int(smooth_mm / mpp))
        smoothed_p = uniform_filter1d(top_ys, size=win)
        pad_p = win * 2
        padded_p = np.concatenate([
            np.full(pad_p, -float(H)),
            -smoothed_p,
            np.full(pad_p, -float(H)),
        ])
        raw, _ = find_peaks(
            padded_p,
            distance=max(1, int(dist_mm / mpp)),
            prominence=max(1, int(prom_mm / mpp)),
        )
        cands = [
            p - pad_p for p in raw
            if 0 <= p - pad_p < len(col_xs)
            and (p - pad_p) >= edge_margin
            and (p - pad_p) <= len(col_xs) - edge_margin
        ]
        # Use the ACTUAL topmost skin pixel (top_ys[p]), not the smoothed
        # value.  The smoothed y may land slightly above the real skin edge,
        # causing measure_finger_strip to miss the tip in the re-segmented mask.
        cand_tips = [(int(col_xs[p]), int(top_ys[p])) for p in cands]
        cand_tips.sort(key=lambda t: t[0])

        # Count tips inside top-40 % of hand bbox (= actual fingertips, not palm)
        top_cutoff = hy + int(hh * 0.40)
        main_count = sum(1 for _, ty in cand_tips if ty <= top_cutoff)

        if main_count >= 4:
            tips = cand_tips
            break
        # Keep the best attempt so far (most tips overall)
        if len(cand_tips) > len(tips):
            tips = cand_tips

    # ── Anatomy-based pinky fallback ─────────────────────────
    # If fewer than 4 main tips (top-40% of hand bbox) found after all
    # passes, the pinky is likely too close to the ring to form a distinct
    # peak.  Search the rightmost portion of the hand bbox (right of the
    # rightmost detected tip) for the highest skin pixel — that is the
    # pinky tip.
    top_cutoff_final = hy + int(hh * 0.40)
    main_count_final = sum(1 for _, ty in tips if ty <= top_cutoff_final)
    if main_count_final < 4 and len(tips) >= 2:
        rightmost_x = max(t[0] for t in tips)
        min_gap_px  = max(1, int(10 / mpp))  # pinky must be ≥10mm to the right
        right_cols  = col_xs > (rightmost_x + min_gap_px)
        if np.any(right_cols):
            rc  = col_xs[right_cols]
            ry  = top_ys[right_cols]
            # Smooth to avoid single-pixel noise peaks
            sm  = uniform_filter1d(ry.astype(float), size=max(3, int(3 / mpp)))
            best_idx = int(np.argmin(sm))
            extra_tip = (int(rc[best_idx]), int(ry[best_idx]))
            # Must be in the fingertip zone (top-40% of hand bbox) and not
            # too close to an existing tip in x.
            in_top40 = extra_tip[1] <= top_cutoff_final
            far_enough = all(abs(extra_tip[0] - t[0]) > min_gap_px for t in tips)
            # Y-proximity check: the fallback tip must be within 15 mm of the
            # lowest (highest y) existing main tip.  Wrist/palm artifacts sit
            # much further down the image even when top-40% allows them.
            # If the pinky is genuinely absent/folded, the fallback candidate
            # will fail this check and be correctly rejected.
            main_tip_ys = [ty for _, ty in tips if ty <= top_cutoff_final]
            if main_tip_ys:
                worst_main_y  = max(main_tip_ys)  # lowest-on-image main tip
                max_y_allowed = worst_main_y + int(15 / mpp)
                y_plausible   = extra_tip[1] <= max_y_allowed
            else:
                y_plausible = in_top40
            if in_top40 and far_enough and y_plausible:
                tips.append(extra_tip)
                print(f"  [Fingertips] Anatomy fallback: added tip at "
                      f"({extra_tip[0]},{extra_tip[1]})")
            else:
                print(f"  [Fingertips] Anatomy fallback candidate "
                      f"({extra_tip[0]},{extra_tip[1]}) rejected "
                      f"(in_top40={in_top40}, far_enough={far_enough}, "
                      f"y_plausible={y_plausible})")

    if len(tips) == 0:
        raise RuntimeError(
            "No fingertips detected.\n"
            "  -> Check that fingers are spread and pointing upward.\n"
            "  -> Try increasing the ArUco size or improving lighting."
        )

    tips.sort(key=lambda t: t[0])   # left-to-right

    print(f"  [Fingertips] Found {len(tips)} tips: "
          + ", ".join(f"({tx},{ty})" for tx, ty in tips))
    return tips


# ─────────────────────────────────────────────────────────────
# 3. Finger assignment
# ─────────────────────────────────────────────────────────────

def assign_fingers(tips: list, hand_bbox: tuple, mpp: float) -> dict:
    """
    Assign finger names to detected tips.

    The thumb is the tip that sits significantly lower (higher y value) and
    to the left of the hand centre.  The remaining 4 tips are assigned
    index/middle/ring/pinky left-to-right.

    Returns
    -------
    dict: {finger_name: (tip_x, tip_y)}
    """
    if not tips:
        return {}

    hx, hy, hw, hh = hand_bbox
    hand_center_x   = hx + hw / 2
    hand_height     = hh

    ys = [t[1] for t in tips]
    median_y = float(np.median(ys))
    thumb_threshold_y = median_y + 0.15 * hand_height

    # Separate thumb candidate from main fingers.
    thumb_candidates = [
        t for t in tips
        if t[1] > thumb_threshold_y and t[0] < hand_center_x
    ]
    main_tips = [t for t in tips if t not in thumb_candidates]

    assigned = {}

    if thumb_candidates:
        # Pick the lowest-and-leftmost as the thumb.
        thumb = max(thumb_candidates, key=lambda t: t[1])
        assigned["thumb"] = thumb
        print(f"  [Assign] thumb  -> tip=({thumb[0]},{thumb[1]})")
    else:
        print("  [Assign] No thumb candidate detected; skipping thumb.")

    # Assign main fingers left-to-right.
    main_tips.sort(key=lambda t: t[0])
    main_names = ["index", "middle", "ring", "pinky"]
    for name, tip in zip(main_names, main_tips):
        assigned[name] = tip
        print(f"  [Assign] {name:6s} -> tip=({tip[0]},{tip[1]})")

    if len(main_tips) < 4:
        detected = [main_names[i] for i in range(len(main_tips))]
        missing  = main_names[len(main_tips):]
        warnings.warn(
            f"Only {len(main_tips)} main tips found (expected 4). "
            f"Missing: {missing}. Detected: {detected}."
        )

    return assigned


# ─────────────────────────────────────────────────────────────
# 3b. Auto-orientation helper
# ─────────────────────────────────────────────────────────────

def _count_fingertip_peaks(mask: np.ndarray, mpp: float) -> int:
    """Return the number of fingertip peaks detected assuming fingers point UP.

    Uses loose thresholds (3 mm prominence / 5 mm min distance).  Peaks are
    only counted if they are NOT at the top image border (y > 3% of H) — this
    rejects wrist/edge artifacts that appear when the hand is upside-down or
    sideways after a wrong rotation.

    Stores total prominence score on _count_fingertip_peaks._last_score so
    _auto_orient can use it for tie-breaking.
    """
    h, w = mask.shape[:2]
    top_ys = []
    for col in range(w):
        rows = np.where(mask[:, col] > 0)[0]
        if len(rows):
            top_ys.append(int(rows[0]))
    if len(top_ys) < 10:
        _count_fingertip_peaks._last_score = 0.0
        return 0

    arr = np.array(top_ys, float)
    smoothed = uniform_filter1d(arr, size=max(5, int(5 / mpp)))
    pad = max(5, int(5 / mpp)) * 2
    padded = np.concatenate([
        np.full(pad, -float(h)),
        -smoothed,
        np.full(pad, -float(h)),
    ])
    peaks, props = find_peaks(
        padded,
        distance=max(1, int(5 / mpp)),
        prominence=max(1, int(3 / mpp)),
    )
    # Reject peaks whose topmost skin pixel is within the top 3% of image height
    # — those are border artifacts caused by an incorrect rotation orientation.
    min_y_border = h * 0.03
    valid = [
        i for i, p in enumerate(peaks)
        if 0 <= p - pad < len(arr) and arr[p - pad] > min_y_border
    ]
    proms = props["prominences"][valid]
    score = float(np.sum(proms))
    _count_fingertip_peaks._last_score = score
    return len(valid)


def _auto_orient(image: np.ndarray, hand_mask: np.ndarray,
                 mpp: float, aruco_corners=None):
    """Rotate image/mask in 90-degree steps so fingers point upward.

    Tries 0 / 90 / 180 / 270 CCW rotations and picks the one that yields the
    most fingertip peaks (tie-broken by total prominence score).
    Returns (image, mask, aruco_corners, k) where k is the number of 90-degree
    CCW rotations applied (0-3).
    """
    best_k, best_n, best_score = 0, -1, -1.0
    for k in range(4):
        rotated = np.rot90(hand_mask, k)
        n = _count_fingertip_peaks(rotated, mpp)
        score = getattr(_count_fingertip_peaks, "_last_score", 0.0)
        print(f"  [Orient] k={k} ({k*90}° CCW): {n} peaks, score={score:.0f}")
        if n > best_n or (n == best_n and score > best_score):
            best_n, best_score, best_k = n, score, k

    if best_k == 0:
        return image, hand_mask, aruco_corners, 0

    print(f"  [Orient] rotating {best_k * 90} deg CCW to align fingers upward")
    rot_img  = np.rot90(image,     best_k)
    rot_mask = np.rot90(hand_mask, best_k)

    # Rotate ArUco corners to match.
    rot_corners = None
    if aruco_corners is not None:
        h, w = image.shape[:2]
        rot_corners = aruco_corners.copy().astype(float)
        for _ in range(best_k):
            ch, cw = rot_corners.shape  # (N, 2)
            rot_corners = np.column_stack([
                rot_corners[:, 1],
                h - 1 - rot_corners[:, 0],
            ])
            h, w = w, h   # swap after each 90-deg CCW

    return rot_img, rot_mask, rot_corners, best_k


# ─────────────────────────────────────────────────────────────
# 3c. Finger rotation helpers (kept for future use)
# ─────────────────────────────────────────────────────────────

def _compute_finger_angle(finger_mask: np.ndarray,
                          tip_xi: int, tip_yi: int) -> float:
    """Return the angle (degrees) the finger is tilted from vertical.

    Positive = finger leans right, negative = leans left.
    Compares the centroid of the top quarter to the centroid of the
    second quarter of the finger mask.  This gives two points along
    the finger axis, unaffected by palm pixels at the bottom.
    """
    ys, xs = np.where(finger_mask > 0)
    if len(xs) < 100:
        return 0.0
    y_min, y_max = int(ys.min()), int(ys.max())
    y_range = y_max - y_min
    if y_range < 20:
        return 0.0
    # Quarter boundaries from the top.
    q1_bot = y_min + int(y_range * 0.25)   # top quarter
    q2_top = q1_bot
    q2_bot = y_min + int(y_range * 0.50)   # second quarter

    in_q1 = ys <= q1_bot
    in_q2 = (ys > q2_top) & (ys <= q2_bot)

    if int(np.sum(in_q1)) < 30 or int(np.sum(in_q2)) < 30:
        return 0.0

    cx1, cy1 = float(np.mean(xs[in_q1])), float(np.mean(ys[in_q1]))
    cx2, cy2 = float(np.mean(xs[in_q2])), float(np.mean(ys[in_q2]))

    # Vector from Q2 (lower) to Q1 (upper) = finger axis toward the tip.
    dx = cx1 - cx2
    dy = cy1 - cy2     # negative (Q1 is above Q2)
    return float(np.degrees(np.arctan2(dx, -dy)))


def _rotate_image(img: np.ndarray, angle_deg: float):
    """Rotate *img* by *angle_deg* around its centre, expanding the canvas.

    Returns (rotated_image, forward_matrix).
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    flags = cv2.INTER_LINEAR
    if img.dtype == np.uint8 and img.ndim == 2:
        flags = cv2.INTER_NEAREST          # binary masks
    rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=flags)
    return rotated, M


def _inverse_matrix(M_fwd: np.ndarray,
                    src_w: int, src_h: int,
                    dst_w: int, dst_h: int,
                    angle_deg: float) -> np.ndarray:
    """Return the affine matrix that undoes _rotate_image's transform."""
    ncx, ncy = dst_w / 2.0, dst_h / 2.0
    M_inv = cv2.getRotationMatrix2D((ncx, ncy), -angle_deg, 1.0)
    M_inv[0, 2] += (src_w - dst_w) / 2
    M_inv[1, 2] += (src_h - dst_h) / 2
    return M_inv


def _xform_pt(M: np.ndarray, x: float, y: float):
    """Apply 2x3 affine *M* to a single point -> (x', y')."""
    p = M @ np.array([x, y, 1.0])
    return float(p[0]), float(p[1])


def _xform_poly(M: np.ndarray, poly_list: list) -> list:
    """Apply affine *M* to a list of [x, y] points."""
    pts = np.array(poly_list, dtype=np.float64)
    ones = np.ones((len(pts), 1), dtype=np.float64)
    out = (M @ np.hstack([pts, ones]).T).T[:, :2]
    return out.astype(np.int32).tolist()


# ─────────────────────────────────────────────────────────────
# 4. Nail plate detector helper
# ─────────────────────────────────────────────────────────────

def _detect_nail_plate(
    strip_img: np.ndarray,
    finger_mask: np.ndarray,
    tip_xi: int,
    tip_yi: int,
    sH: int,
    sW: int,
    mpp: float,
) -> np.ndarray | None:
    """
    Two-phase nail plate detection.

    Phase 1 -- P10 column brightness
        Compute per-column average brightness in the nail zone and find
        the contiguous region above the 10th-percentile threshold.  This
        gives a robust width estimate (~65-75 % of finger width).

    Phase 2 -- GrabCut refinement
        Initialise GrabCut with the P10 column range as probable-foreground
        and the remaining finger skin as probable-background.  GrabCut
        refines the contour shape (curved sides, natural tip/cuticle arcs)
        within the P10 width constraint.

    The final mask is clipped to the P10 column range so that GrabCut
    cannot over-expand beyond the brightness-derived boundary.
    """
    # ── Phase 1: P10 column brightness ────────────────────────
    L = cv2.cvtColor(strip_img, cv2.COLOR_BGR2Lab)[:, :, 0].astype(np.float32)
    nail_end = min(sH, tip_yi + int(15 / mpp))

    col_bright = np.full(sW, np.nan)
    for c in range(sW):
        rows_in = np.where(finger_mask[tip_yi:nail_end, c] > 0)[0]
        if len(rows_in) > 2:
            col_bright[c] = float(L[tip_yi + rows_in, c].mean())

    in_finger = np.where(~np.isnan(col_bright))[0]
    if len(in_finger) < 10:
        print("  [NailPlate] Too few finger columns - skipping")
        return None

    fl, fr = int(in_finger[0]), int(in_finger[-1])
    finger_width = fr - fl
    edge_margin  = max(2, int(finger_width * 0.12))

    interior    = col_bright[fl + edge_margin : fr - edge_margin + 1]
    interior_sm = uniform_filter1d(
        np.where(np.isnan(interior), 0, interior),
        size=max(3, int(3 / mpp)),
    )
    # Adaptive threshold: stop expanding from the nail center when brightness
    # drops to (peak − 1.5 × std) relative to the interior profile.
    # This detects the lateral nail groove (where nail meets skin fold) without
    # needing high absolute contrast — important for natural (unpolished) nails.
    # Hard floor at P10 so we never exclude the nail on high-contrast (polished)
    # nails where std is large and the adaptive threshold might fall below the
    # actual nail brightness.
    center     = len(interior_sm) // 2
    peak_b     = float(np.nanmax(interior_sm))
    std_b      = max(2.0, float(np.nanstd(interior_sm)))
    adaptive_t = peak_b - 1.5 * std_b          # stops at ~1.5-sigma drop from peak
    floor_t    = float(np.nanpercentile(interior_sm, 10))
    threshold  = max(adaptive_t, floor_t)       # use whichever is STRICTER (higher)

    print(f"  [NailPlate] brightness: peak={peak_b:.1f} std={std_b:.1f} "
          f"thr={threshold:.1f} (adaptive={adaptive_t:.1f} P10={floor_t:.1f})")

    above  = interior_sm >= threshold
    l_idx  = center
    while l_idx > 0 and above[l_idx - 1]:
        l_idx -= 1
    r_idx = center
    while r_idx < len(above) - 1 and above[r_idx + 1]:
        r_idx += 1

    if r_idx <= l_idx:
        print("  [NailPlate] No bright interior region - skipping")
        return None

    nail_col_l = fl + edge_margin + l_idx
    nail_col_r = fl + edge_margin + r_idx
    nail_width = nail_col_r - nail_col_l
    ratio      = nail_width / max(finger_width, 1)
    if ratio < 0.30 or ratio > 0.90:
        print(f"  [NailPlate] P10 ratio {ratio:.2f} out of range - skipping")
        return None

    # ── Phase 2: GrabCut with P10-guided initialisation ───────
    # Two vertical zones for GrabCut:
    #   - nail_init_bottom (~12mm): expected nail plate -> PR_FGD
    #   - clip_bottom (~25mm): safety clip -> everything beyond is 0
    # The gap between 12mm and 25mm stays PR_BGD (skin), so GrabCut
    # decides where the actual nail-skin transition (cuticle) is.
    nail_init_bottom = min(sH, tip_yi + int(12 / mpp))
    clip_bottom      = min(sH, tip_yi + int(25 / mpp))
    gc_mask = np.full((sH, sW), cv2.GC_BGD, dtype=np.uint8)

    # All finger = probable background (skin).
    gc_mask[finger_mask > 0] = cv2.GC_PR_BGD

    # P10 columns inside the expected nail zone (tip to ~12mm) = PR_FGD.
    nail_zone_cols = np.zeros((sH, sW), dtype=bool)
    nail_zone_cols[max(0, tip_yi):nail_init_bottom,
                   nail_col_l:nail_col_r + 1] = True
    gc_mask[(finger_mask > 0) & nail_zone_cols] = cv2.GC_PR_FGD

    # Centre seed = definite foreground.
    seed_hw = max(4, int(2   / mpp))
    seed_hh = max(4, int(2.5 / mpp))
    seed_cy = tip_yi + max(2, int(3 / mpp))
    y0 = max(0,  seed_cy - seed_hh)
    y1 = min(sH, seed_cy + seed_hh)
    x0 = max(0,  tip_xi  - seed_hw)
    x1 = min(sW, tip_xi  + seed_hw)
    gc_mask[y0:y1, x0:x1][finger_mask[y0:y1, x0:x1] > 0] = cv2.GC_FGD

    n_fgd    = int(np.sum(gc_mask == cv2.GC_FGD))
    n_pr_bgd = int(np.sum(gc_mask == cv2.GC_PR_BGD))
    if n_fgd < 30 or n_pr_bgd < 50:
        # Not enough samples -- fall back to P10 column-clipped mask.
        nail_plate = np.zeros((sH, sW), np.uint8)
        nail_plate[:, nail_col_l:nail_col_r + 1] = \
            finger_mask[:, nail_col_l:nail_col_r + 1]
        print(f"  [NailPlate] P10 only (not enough GC samples): "
              f"w={nail_width*mpp:.1f}mm ratio={ratio:.2f}")
        return nail_plate

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(strip_img, gc_mask, None,
                    bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        nail_plate = np.zeros((sH, sW), np.uint8)
        nail_plate[:, nail_col_l:nail_col_r + 1] = \
            finger_mask[:, nail_col_l:nail_col_r + 1]
        print(f"  [NailPlate] P10 fallback (GrabCut error): "
              f"w={nail_width*mpp:.1f}mm ratio={ratio:.2f}")
        return nail_plate

    fg = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255, 0,
    ).astype(np.uint8)

    # ── Phase 3: Constrain to P10 width + nail zone ───────────
    fg[:, :nail_col_l]      = 0
    fg[:, nail_col_r + 1:]  = 0
    fg[:max(0, tip_yi - int(1 / mpp)), :] = 0
    fg[clip_bottom:, :]     = 0

    kN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kN, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  kN, iterations=1)

    # Keep the component at the fingertip.
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    tip_label = int(lbl[tip_yi, tip_xi])
    if tip_label == 0:
        win = max(1, int(3 / mpp))
        region = lbl[
            max(0, tip_yi):min(sH, tip_yi + 2 * win),
            max(0, tip_xi - win):min(sW, tip_xi + win),
        ]
        nz = region[region > 0]
        if len(nz) == 0:
            nail_plate = np.zeros((sH, sW), np.uint8)
            nail_plate[:, nail_col_l:nail_col_r + 1] = \
                finger_mask[:, nail_col_l:nail_col_r + 1]
            print(f"  [NailPlate] P10 fallback (no GC tip): "
                  f"w={nail_width*mpp:.1f}mm ratio={ratio:.2f}")
            return nail_plate
        tip_label = int(np.bincount(nz.ravel()).argmax())

    nail_plate = (lbl == tip_label).astype(np.uint8) * 255

    plate_area  = int(np.sum(nail_plate > 0))
    finger_area = int(np.sum(finger_mask > 0))
    gc_ratio    = plate_area / max(finger_area, 1)

    print(f"  [NailPlate] P10+GrabCut: {plate_area}px "
          f"({100*gc_ratio:.0f}% of finger), "
          f"P10 w={nail_width*mpp:.1f}mm ratio={ratio:.2f}")
    return nail_plate


# ─────────────────────────────────────────────────────────────
# 5. Per-finger strip measurement  (uses _detect_nail_plate above)
# ─────────────────────────────────────────────────────────────

def measure_finger_strip(
    image: np.ndarray,
    hand_mask: np.ndarray,
    finger_name: str,
    col_left: int,
    col_right: int,
    tip_x: int,
    tip_y: int,
    mpp: float,
    output_dir: str,
) -> dict:
    """
    Measure one finger from a 2-D crop of the full-hand image.

    The crop is bounded horizontally by [col_left, col_right] and vertically
    by [tip_y - 5mm, tip_y + 40mm].  This prevents the palm (which sits far
    below the fingertip) from contaminating the width/cuticle detection,
    especially for angled fingers like the pinky.

    Steps
    -----
    1. Extract image crop and mask crop.
    2. Find the largest contour in the mask crop -> per-finger mask + bbox.
    3. Call measure_top() (from nail_measurer) on the crop.
    4. Apply W/L correction, add aspect_ratio, save annotated image.

    Returns
    -------
    dict with all measurement fields plus "finger" key.
    """
    H, W = image.shape[:2]
    col_left  = max(0, col_left)
    col_right = min(W, col_right)

    # Reject tips that are too close to the image edge — reliable measurement
    # is impossible when the finger is partially out of frame.
    edge_mm = 4.0
    edge_px = int(edge_mm / mpp)
    if tip_x > W - edge_px:
        raise RuntimeError(
            f"'{finger_name}' tip too close to right image edge "
            f"(tip_x={tip_x}, image_w={W}, margin={edge_mm}mm). "
            "Retake photo with the full hand in frame."
        )
    if tip_x < edge_px:
        raise RuntimeError(
            f"'{finger_name}' tip too close to left image edge "
            f"(tip_x={tip_x}, image_w={W}, margin={edge_mm}mm). "
            "Retake photo with the full hand in frame."
        )

    # Vertical bounds: start a few mm above the tip, end 40 mm below.
    row_top    = max(0, tip_y - int(5 / mpp))
    row_bottom = min(H, tip_y + int(40 / mpp))

    strip_img  = image[row_top:row_bottom, col_left:col_right]
    strip_mask = hand_mask[row_top:row_bottom, col_left:col_right]

    sH, sW = strip_img.shape[:2]

    # Re-segment skin directly in the crop.
    # Using the pre-filled hand_mask can bridge inter-finger gaps (the filled
    # contour is solid across the whole hand silhouette).  Re-thresholding in
    # the small crop gives per-finger blobs without that artefact.
    L_crop = cv2.cvtColor(strip_img, cv2.COLOR_BGR2Lab)[:, :, 0]
    _, raw_skin = cv2.threshold(L_crop, 130, 255, cv2.THRESH_BINARY)
    kS = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    raw_skin = cv2.morphologyEx(raw_skin, cv2.MORPH_CLOSE, kS, iterations=2)
    raw_skin = cv2.morphologyEx(raw_skin, cv2.MORPH_OPEN,  kS, iterations=1)

    # Find the connected component that contains the fingertip.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_skin, connectivity=8)

    # Tip position in strip coordinates.
    tip_xi = int(tip_x - col_left)
    tip_yi = int(tip_y - row_top)
    tip_xi = max(0, min(sW - 1, tip_xi))
    tip_yi = max(0, min(sH - 1, tip_yi))

    label_at_tip  = int(labels[tip_yi, tip_xi])
    used_fallback = False  # track whether we needed a lower-threshold fallback

    if label_at_tip == 0:
        # L > 130 re-segmentation missed the tip.
        # 1st fallback: search in a 3 mm window around the tip.
        win = max(1, int(3 / mpp))
        region = labels[
            max(0, tip_yi - win):min(sH, tip_yi + win),
            max(0, tip_xi - win):min(sW, tip_xi + win),
        ]
        nonzero = region[region > 0]
        if len(nonzero) > 0:
            label_at_tip = int(np.bincount(nonzero.ravel()).argmax())
        else:
            # 2nd fallback: lower L threshold (90) intersected with the
            # original hand_mask — catches dark-toned nails / shadowed fingers
            # while staying within the morphologically-reliable hand boundary.
            used_fallback = True
            _, raw_skin_lo = cv2.threshold(L_crop, 90, 255, cv2.THRESH_BINARY)
            raw_skin_lo = cv2.bitwise_and(raw_skin_lo, strip_mask)
            raw_skin_lo = cv2.morphologyEx(raw_skin_lo, cv2.MORPH_CLOSE, kS, iterations=2)
            raw_skin_lo = cv2.morphologyEx(raw_skin_lo, cv2.MORPH_OPEN,  kS, iterations=1)
            _, labels_lo, _, _ = cv2.connectedComponentsWithStats(
                raw_skin_lo, connectivity=8)
            lab_lo = int(labels_lo[tip_yi, tip_xi])
            if lab_lo > 0:
                labels       = labels_lo
                label_at_tip = lab_lo
            else:
                # Search in window with low-threshold labels
                region_lo = labels_lo[
                    max(0, tip_yi - win):min(sH, tip_yi + win),
                    max(0, tip_xi - win):min(sW, tip_xi + win),
                ]
                nz_lo = region_lo[region_lo > 0]
                if len(nz_lo) == 0:
                    raise RuntimeError(
                        f"Fingertip for '{finger_name}' not found in skin mask - "
                        "check tip_y detection or background contrast."
                    )
                labels       = labels_lo
                label_at_tip = int(np.bincount(nz_lo.ravel()).argmax())

    finger_mask_strip = (labels == label_at_tip).astype(np.uint8) * 255

    if used_fallback:
        # The fallback (L>90 ∩ hand_mask) blob may include palm skin that
        # enters the column range just a few mm below the fingertip.
        # Restrict to the actual narrow finger body by probing 2 mm of the
        # hand_mask at the tip level to find the true column extent.
        probe_bot = min(sH, tip_yi + max(1, int(2 / mpp)))
        probe_cols = np.where(
            strip_mask[tip_yi:probe_bot, :].any(axis=0))[0]
        if len(probe_cols) >= 3:
            margin_px   = max(2, int(2 / mpp))
            tight_left  = max(0,  int(probe_cols[0])  - margin_px)
            tight_right = min(sW, int(probe_cols[-1]) + margin_px)
            col_restrict = np.zeros_like(finger_mask_strip)
            col_restrict[:, tight_left:tight_right] = 255
            finger_mask_strip = cv2.bitwise_and(finger_mask_strip, col_restrict)

    # ── Per-finger de-rotation ────────────────────────────────
    # Measure the finger tilt angle and straighten the strip so that
    # measure_top() always operates on a vertically-aligned finger.
    # Without this, a tilted finger (e.g. pinky leaning right) causes the
    # horizontal row-scan to overestimate width and miss the C-curve.
    tilt = _compute_finger_angle(finger_mask_strip, tip_xi, tip_yi)
    if abs(tilt) > 5.0:
        print(f"  [Strip] {finger_name}: tilt={tilt:.1f}°, de-rotating to vertical")
        strip_img_r, M_fwd = _rotate_image(strip_img, tilt)
        fmask_r,     _     = _rotate_image(finger_mask_strip, tilt)
        _, fmask_r = cv2.threshold(fmask_r, 127, 255, cv2.THRESH_BINARY)
        sH_r, sW_r = strip_img_r.shape[:2]
        # Transform tip position into rotated space.
        tip_xi_r, tip_yi_r = _xform_pt(M_fwd, float(tip_xi), float(tip_yi))
        tip_xi_r = max(0, min(sW_r - 1, int(round(tip_xi_r))))
        tip_yi_r = max(0, min(sH_r - 1, int(round(tip_yi_r))))
        # Inverse matrix to bring results back to original strip coords.
        M_inv_r = _inverse_matrix(M_fwd, sW, sH, sW_r, sH_r, tilt)
        # Use rotated versions downstream.
        _img_m, _mask_m = strip_img_r, fmask_r
        _txi, _tyi, _sH_m, _sW_m = tip_xi_r, tip_yi_r, sH_r, sW_r
    else:
        tilt = 0.0
        M_inv_r = None
        _img_m, _mask_m = strip_img, finger_mask_strip
        _txi, _tyi, _sH_m, _sW_m = tip_xi, tip_yi, sH, sW

    cnts, _ = cv2.findContours(_mask_m, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError(f"No contour found for finger '{finger_name}'.")
    finger_cnt = max(cnts, key=cv2.contourArea)
    bbox = cv2.boundingRect(finger_cnt)

    nail_plate_mask = _detect_nail_plate(
        _img_m, _mask_m, _txi, _tyi, _sH_m, _sW_m, mpp)
    try:
        data = measure_top(_img_m, mpp, _mask_m, bbox, nail_plate_mask)
    except RuntimeError as _e:
        if "Not enough finger rows" in str(_e) and nail_plate_mask is not None:
            # The nail plate mask may not extend to the very tip (GrabCut seed
            # starts a few mm below the tip), so row_scan finds 0 valid rows.
            # Fall back to using finger_mask_strip directly (wider, but reliable).
            print(f"  [Strip] Nail plate caused row-scan failure; "
                  f"retrying without nail_plate_mask")
            nail_plate_mask = None
            data = measure_top(_img_m, mpp, _mask_m, bbox, None)
        else:
            raise

    # Replace the schematic oval polygon (from measure_top) with the actual
    # GrabCut nail plate contour.  The schematic is built from a constant
    # nail_half and never matches the true nail shape.  The GrabCut mask
    # accurately follows the lateral grooves and tip arc.
    # We clip the mask at the cuticle line so the polygon does not extend
    # into the skin below the cuticle (GrabCut clip_bottom=25mm is generous).
    if nail_plate_mask is not None:
        cuticle_row = max(0, int(data["_cuticle_y"]))
        plate_clipped = nail_plate_mask.copy()
        plate_clipped[cuticle_row:, :] = 0   # remove everything below cuticle

        # Light morphological smoothing to remove single-pixel jaggies.
        kSmooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        plate_smooth = cv2.morphologyEx(plate_clipped, cv2.MORPH_CLOSE,
                                        kSmooth, iterations=1)
        plate_smooth = cv2.morphologyEx(plate_smooth, cv2.MORPH_OPEN,
                                        kSmooth, iterations=1)
        np_cnts, _ = cv2.findContours(plate_smooth, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_TC89_KCOS)
        if np_cnts:
            np_cnt   = max(np_cnts, key=cv2.contourArea)
            perim    = cv2.arcLength(np_cnt, True)
            approx   = cv2.approxPolyDP(np_cnt,
                                        max(1.0, perim * 0.008), True)
            data["nail_polygon_px"] = approx.reshape(-1, 2).tolist()
            print(f"  [Strip] Using GrabCut contour as polygon "
                  f"({len(data['nail_polygon_px'])} pts)")

    # Save per-finger annotated (with the de-rotated image if applicable).
    finger_out = os.path.join(output_dir, finger_name)
    os.makedirs(finger_out, exist_ok=True)
    ann_path = os.path.join(finger_out, f"{finger_name}_annotated.jpg")
    save_annotated(_img_m, data, None, finger_name, ann_path)

    # If the strip was rotated, transform polygon + key points back to
    # original strip coordinates so the hand_annotated overlay is correct.
    if M_inv_r is not None:
        data["nail_polygon_px"] = _xform_poly(M_inv_r, data["nail_polygon_px"])
        tx0 = float(data["_tip_x"])
        ty0 = float(data["_tip_y"])
        cy0 = float(data["_cuticle_y"])
        tip_b  = M_inv_r @ np.array([tx0, ty0, 1.0])
        cut_b  = M_inv_r @ np.array([tx0, cy0, 1.0])
        data["_tip_x"]    = float(tip_b[0])
        data["_tip_y"]    = float(tip_b[1])
        data["_cuticle_y"] = float(cut_b[1])

    data["finger"]    = finger_name
    data["_col_left"] = col_left
    data["_row_top"]  = row_top

    # W/L correction.
    wl = apply_wl_correction(finger_name, data["width_mm"], data["length_mm"])
    data.update(wl)

    # Re-sample skin using the morphologically reliable hand_mask (strip_mask).
    # The per-strip L-threshold re-segmentation (finger_mask_strip) can include
    # bright background pixels bridged to the finger by morphological CLOSE, so
    # we prefer strip_mask here.  We try several bands (cuticle+5mm → cuticle →
    # cuticle-5mm → nail body 2-8mm below tip) until we find a warm-toned
    # (skin-plausible) sample: R > G ≥ B and R > 80.
    cut_y_s = int(data["_cuticle_y"])
    off_px  = int(5  / mpp)
    bnd_px  = int(15 / mpp)
    bands = [
        (cut_y_s + off_px,              cut_y_s + off_px + bnd_px),
        (cut_y_s,                        cut_y_s + bnd_px),
        (cut_y_s - off_px,              cut_y_s - off_px + bnd_px),
        (tip_yi  + int(2 / mpp),        tip_yi  + int(8 / mpp)),   # nail body
    ]
    skin_found = False
    for st, sb in bands:
        st = max(0, min(st, sH - 1))
        sb = max(0, min(sb, sH))
        if sb <= st:
            continue
        sm  = np.zeros((sH, sW), np.uint8)
        sm[st:sb, :] = 255
        sm  = cv2.bitwise_and(sm, strip_mask)
        pix = strip_img[sm > 0]
        if len(pix) > 10:
            b, g, r = [int(np.median(pix[:, i])) for i in range(3)]
            if r > 80 and r >= g and r >= b:   # skin is warm (R dominant)
                data["skin_tone_hex"] = f"#{r:02X}{g:02X}{b:02X}"
                skin_found = True
                break
    if not skin_found:
        # No warm-toned band found — background bleed or unusual lighting.
        # Keep whatever measure_top() computed but mark it as unreliable.
        print(f"  [SkinTone] {finger_name}: no warm band found; "
              f"falling back to measure_top value: {data['skin_tone_hex']}")

    # Aspect ratio.
    data["aspect_ratio"] = (
        round(data["width_mm"] / data["length_mm"], 3)
        if data["length_mm"] else 0.0
    )

    return data


# ─────────────────────────────────────────────────────────────
# 6. Main orchestration
# ─────────────────────────────────────────────────────────────

def measure_hand(image_path: str, aruco_size_mm: float, output_dir: str):
    """
    Full pipeline for a single full-hand photo.

    Steps
    -----
    1. Load image, detect ArUco -> mpp
    2. Segment hand -> hand_mask, hand_bbox
    3. Find fingertips -> assign to finger names
    4. Compute column boundaries for each finger
    5. Measure each finger strip
    6. Save hand_annotated.jpg with tip markers
    7. Save nail_measurements.json and profile.json

    Returns
    -------
    dict: the full nail_measurements payload
    """
    print(f"\n{'='*60}")
    print(f"  Hand Measurer")
    print(f"  Image : {image_path}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    image = cv2.imread(image_path)
    if image is None:
        sys.exit(f"ERROR: Cannot open image '{image_path}'")

    H, W = image.shape[:2]
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: ArUco ─────────────────────────────────────────
    print("\n[1/8] Detecting ArUco marker …")
    mpp, aruco_corners, marker_id = detect_aruco(image, aruco_size_mm)

    # ── Step 2: Segment hand ──────────────────────────────────
    print("\n[2/8] Segmenting hand …")
    hand_mask, hand_contour, hand_bbox = segment_hand(image, aruco_corners)

    # ── Step 2b: Auto-orient (rotate so fingers point up) ────
    image, hand_mask, aruco_corners, _orient_k = _auto_orient(
        image, hand_mask, mpp, aruco_corners)
    H, W = image.shape[:2]
    # Recompute hand bbox on (possibly rotated) mask.
    # Use the UNION of all significant contours — the pinky / thumb may be in
    # a blob that is separate from the main hand blob.
    cnts_h, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    if cnts_h:
        all_pts_h = np.concatenate([c.reshape(-1, 2) for c in cnts_h])
        ux, uy, uw, uh = cv2.boundingRect(all_pts_h)
        hand_bbox = (ux, uy, uw, uh)

    # ── Step 3: Fingertips ────────────────────────────────────
    print("\n[3/8] Finding fingertips …")
    tips     = find_fingertips(hand_mask, hand_bbox, mpp)
    assigned = assign_fingers(tips, hand_bbox, mpp)

    if not assigned:
        sys.exit("ERROR: No fingers could be assigned from detected tips.")

    # ── Step 4: Column boundaries ─────────────────────────────
    print("\n[4/8] Computing finger column boundaries …")

    MAIN_FINGER_ORDER = ["index", "middle", "ring", "pinky"]
    main_assigned = {
        f: assigned[f] for f in MAIN_FINGER_ORDER if f in assigned
    }
    main_fingers_sorted = sorted(main_assigned.items(), key=lambda kv: kv[1][0])

    pad_px = int(20 / mpp)

    # Build column boundaries for main fingers.
    col_bounds = {}  # finger_name -> (col_left, col_right)
    for i, (fname, (tx, ty)) in enumerate(main_fingers_sorted):
        if i == 0:
            left = max(0, tx - pad_px)
        else:
            prev_tx = main_fingers_sorted[i - 1][1][0]
            left    = (prev_tx + tx) // 2

        if i == len(main_fingers_sorted) - 1:
            right = min(W, tx + pad_px)
        else:
            next_tx = main_fingers_sorted[i + 1][1][0]
            right   = (tx + next_tx) // 2

        col_bounds[fname] = (left, right)
        print(f"  {fname:6s} cols: [{left}, {right}]")

    # Thumb column boundary.
    if "thumb" in assigned:
        tx, ty = assigned["thumb"]
        col_bounds["thumb"] = (max(0, tx - pad_px), min(W, tx + pad_px))
        print(f"  thumb  cols: [{col_bounds['thumb'][0]}, {col_bounds['thumb'][1]}]")

    # ── Step 5: Measure each finger ───────────────────────────
    print("\n[5/8] Measuring each finger …")
    results = []
    for finger_name in list(main_assigned.keys()):  # thumb excluded
        col_left, col_right = col_bounds[finger_name]
        print(f"\n  --- {finger_name.upper()} (cols {col_left}-{col_right}) ---")
        tip_x, tip_y = assigned[finger_name]
        try:
            data = measure_finger_strip(
                image, hand_mask, finger_name,
                col_left, col_right, tip_x, tip_y, mpp, output_dir,
            )
            results.append(data)
            print(f"  {finger_name}: W={data['width_mm']}mm  "
                  f"L={data['length_mm']}mm  "
                  f"C={data['c_curve_mm']}mm  "
                  f"skin={data['skin_tone_hex']}")
        except Exception as exc:
            print(f"  WARNING: {finger_name} failed -> {exc}")

    if not results:
        sys.exit("ERROR: All finger measurements failed.")

    # ── Step 6: Save hand_annotated.jpg ──────────────────────
    print("\n[6/8] Saving hand annotation …")
    vis = image.copy()
    if aruco_corners is not None:
        cv2.polylines(vis, [aruco_corners.astype(int)], True, (0, 255, 255), 3)

    # Typography sizing scaled to the full-resolution image.
    # At mpp≈0.048 the image is ~4000px wide; we need font ~2.5× bigger than
    # the per-finger strip fonts (which are sized for ~750px-wide strips).
    font_scale = max(1.5, round(1.0 / mpp * 0.13, 1))
    lbl_thick_bg = max(6, int(font_scale * 3))
    lbl_thick_fg = max(2, int(font_scale * 1.2))
    line_h = int(6 / mpp)   # ~6 mm per label line

    for data in results:
        fname    = data["finger"]
        col_left = data["_col_left"]
        row_top  = data["_row_top"]
        color    = NAIL_COLORS.get(fname, (200, 200, 200))

        # ── Nail polygon (strip → global coords) ─────────────
        poly = np.array(data["nail_polygon_px"], dtype=np.int32)
        poly_g = poly + np.array([[col_left, row_top]], dtype=np.int32)

        ov = vis.copy()
        cv2.fillPoly(ov, [poly_g.reshape(-1, 1, 2)], color)
        cv2.addWeighted(ov, 0.35, vis, 0.65, 0, vis)
        cv2.polylines(vis, [poly_g.reshape(-1, 1, 2)], True, color, 3)

        # ── Key points in global coords ───────────────────────
        tip_x_g  = int(data["_tip_x"])    + col_left
        tip_y_g  = int(data["_tip_y"])    + row_top
        cut_y_g  = int(data["_cuticle_y"]) + row_top
        half     = int(data["_nail_half"])
        length_g = cut_y_g - tip_y_g

        # ── Cuticle line ──────────────────────────────────────
        cv2.line(vis,
                 (tip_x_g - half, cut_y_g),
                 (tip_x_g + half, cut_y_g),
                 (0, 165, 255), max(2, int(1 / mpp)))

        # ── C-curve scan lines at 30 / 50 / 70 % ─────────────
        for frac, sc in zip([0.30, 0.50, 0.70],
                            [(255, 100, 0), (0, 255, 100), (255, 0, 255)]):
            row = int(tip_y_g + length_g * frac)
            cv2.line(vis,
                     (tip_x_g - half, row),
                     (tip_x_g + half, row),
                     sc, max(1, int(0.5 / mpp)))

        # ── Measurement labels ────────────────────────────────
        lx = tip_x_g + half + int(2 / mpp)
        labels_txt = [
            (fname.upper(),                       color),
            (f"W: {data['width_mm']}mm",          color),
            (f"L: {data['length_mm']}mm",         color),
            (f"C: {data['c_curve_mm']}mm",        color),
        ]
        for i, (txt, txt_color) in enumerate(labels_txt):
            pt = (lx, tip_y_g + int(2 / mpp) + i * line_h)
            cv2.putText(vis, txt, pt,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 0, 0), lbl_thick_bg)
            cv2.putText(vis, txt, pt,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        txt_color, lbl_thick_fg)

    ann_path = os.path.join(output_dir, "hand_annotated.jpg")
    scale    = 900 / vis.shape[0]
    cv2.imwrite(
        ann_path,
        cv2.resize(vis, (int(vis.shape[1] * scale), 900)),
    )
    print(f"  [Saved] {ann_path}")

    # ── Step 7: nail_measurements.json ───────────────────────
    print("\n[7/8] Building nail_measurements.json …")
    payload   = build_payload(results, aruco_size_mm)
    json_path = os.path.join(output_dir, "nail_measurements.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  [Saved] {json_path}")

    # ── Step 8: profile.json ──────────────────────────────────
    print("\n[8/8] Building profile.json …")
    profiles      = build_profile(results)
    profile_path  = os.path.join(output_dir, "profile.json")
    with open(profile_path, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"  [Saved] {profile_path}")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  HAND MEASUREMENT SUMMARY")
    print(f"{'='*60}")
    for r in results:
        fn = r["finger"]
        print(f"  {fn:6s}  W={r['width_mm']:5.2f}mm  "
              f"L={r['length_mm']:5.2f}mm  "
              f"C={r['c_curve_mm']:4.2f}mm  "
              f"skin={r['skin_tone_hex']}")
    print(f"\n  nail_measurements.json -> {json_path}")
    print(f"  profile.json           -> {profile_path}")
    print(f"{'='*60}")

    return payload


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Measure all nails from a single full-hand photo."
    )
    p.add_argument("--image",       required=True,
                   help="Path to full-hand photo (palm down, ArUco visible)")
    p.add_argument("--aruco-size",  type=float, default=20.0,
                   help="Physical ArUco marker side length in mm (default: 20)")
    p.add_argument("--output",      default="hand_results",
                   help="Output directory for results (default: hand_results)")
    args = p.parse_args()

    measure_hand(args.image, args.aruco_size, args.output)


if __name__ == "__main__":
    main()
