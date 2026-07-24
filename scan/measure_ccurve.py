"""
measure_ccurve.py
-----------------
End-on finger photo → accurate C-curve (sagitta) measurement.

Scale reference: known nail WIDTH (mm) from top-photo measurement.
No ArUco marker required.

Algorithm (v2 — adaptive, no hard-coded colour thresholds)
----------------------------------------------------------
1. Background colour from the image border (median b*).  The finger is the
   only warm-ish object, so its b* sits far above a blue/dark background.
   Threshold = centre of the largest empty gap in the b* histogram
   (Otsu fallback).  This survives any white-balance / colour cast.
2. Finger = most central large contour of that mask.
3. Nail vs finger pulp inside the finger: Otsu split on a*
   (nail plate + free edge are neutral/grey, pulp is redder).
4. FREE-EDGE BAND = nail pixels adjacent to the pulp (within ~1.6× the
   nail thickness).  This isolates the tip cross-section and rejects the
   nail-plate dome behind it and shadowed skin wrapping the pulp sides.
5. Robust circle fit (Kåsa, iterative outlier rejection) to the band's
   top boundary.  Hook tips = extreme-x band pixels lying on the fitted
   circle (annulus filter kills stray blobs like box-felt spikes).
6. Chord = Euclidean distance between hook tips (tilt-proof) ≡ the
   top-view width →  scale = W_mm / chord_px.
   Sagitta from the fit:  h = R − sqrt(R² − (chord/2)²)   — this includes
   the curled-down hook portions the per-column trace can miss.
   arc_R  = W_mm² / (8·h_mm) + h_mm / 2   (consistency form)
"""

import argparse, json, os, sys
import cv2
import numpy as np


def _adaptive_bg_split(B: np.ndarray, border: np.ndarray):
    """Threshold separating background b* (border median) from the finger.

    Returns the b* value above which a pixel is 'not background'.
    Uses the centre of the largest empty histogram gap; Otsu fallback.
    """
    bg_b = float(np.median(B[border]))
    hi = float(B.max())
    if hi - bg_b < 4:
        raise RuntimeError("No b* contrast between border and image content.")
    edges = np.arange(np.floor(bg_b), np.ceil(hi) + 1, 1.0)
    hist, _ = np.histogram(B, bins=edges)
    # ignore tiny counts (noise) when hunting for the gap
    empty = hist <= max(5, B.size * 2e-6)
    best_len, best_start, run, start = 0, None, 0, 0
    for i, e in enumerate(empty):
        if e:
            if run == 0:
                start = i
            run += 1
            if run > best_len:
                best_len, best_start = run, start
        else:
            run = 0
    if best_len >= 3:
        thr = edges[best_start] + best_len / 2.0
        return thr, bg_b
    # fallback: Otsu on normalised b*
    b8 = np.clip((B - B.min()) / (np.ptp(B) + 1e-6) * 255, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(b8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(B.min() + t / 255.0 * np.ptp(B)), bg_b


def _fit_circle_robust(xs: np.ndarray, ys: np.ndarray, iters: int = 8):
    """Kåsa circle fit with iterative outlier rejection.

    Returns (cx, cy, r, inlier_mask).
    """
    keep = np.ones(len(xs), bool)
    cx = cy = r = None
    for _ in range(iters):
        x, y = xs[keep], ys[keep]
        M = np.column_stack([x, y, np.ones(len(x))])
        b = x ** 2 + y ** 2
        sol, *_ = np.linalg.lstsq(M, b, rcond=None)
        cx, cy = sol[0] / 2, sol[1] / 2
        r = np.sqrt(sol[2] + cx ** 2 + cy ** 2)
        res = np.abs(np.hypot(xs - cx, ys - cy) - r)
        s = max(np.median(res[keep]) * 2.5, 1.5)
        new_keep = res < s
        if new_keep.sum() < 8 or (new_keep == keep).all():
            break
        keep = new_keep
    return cx, cy, r, keep


def measure_ccurve(image_path: str, width_mm: float,
                   debug_out: str = None,
                   thickness_mm: float = 0.85) -> dict:

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open: {image_path}")

    H, W_img = img.shape[:2]
    scale_factor = max(H, W_img) / 2000.0          # for adaptive kernel sizes
    print(f"  [Image] {W_img}×{H}  scale_factor={scale_factor:.2f}")

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    A = lab[:, :, 1].astype(np.float32) - 128
    B = lab[:, :, 2].astype(np.float32) - 128

    # ── 1. Background split on b* (adaptive) ─────────────────
    bw = max(20, int(0.02 * max(H, W_img)))
    border = np.zeros((H, W_img), bool)
    border[:bw, :] = border[-bw:, :] = True
    border[:, :bw] = border[:, -bw:] = True
    thr_b, bg_b = _adaptive_bg_split(B, border)
    print(f"  [BG] border b*={bg_b:.1f}  →  finger threshold b*>{thr_b:.1f}")

    mask = (B > thr_b).astype(np.uint8) * 255
    ks = max(5, int(9 * scale_factor) | 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)

    # ── 2. Finger = most central large contour ───────────────
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_area = max(1500, H * W_img * 1e-4)
    big = [c for c in cnts if cv2.contourArea(c) > min_area]
    if not big:
        raise RuntimeError("No finger-sized warm region found — "
                           "check that the fingertip is in frame.")

    def centrality(c):
        x, y, w, h = cv2.boundingRect(c)
        return np.hypot(x + w / 2 - W_img / 2, y + h / 2 - H / 2)

    finger_cnt = min(big, key=centrality)
    fmask = np.zeros((H, W_img), np.uint8)
    cv2.drawContours(fmask, [finger_cnt], -1, 255, -1)
    fx, fy, fw, fh = cv2.boundingRect(finger_cnt)
    print(f"  [Finger] bbox x={fx} y={fy} w={fw} h={fh}")

    # ── 3. Nail vs pulp: Otsu on a* inside the finger ────────
    vals = A[fmask > 0]
    a8 = np.clip((A - vals.min()) / (np.ptp(vals) + 1e-6) * 255,
                 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(a8[fmask > 0], 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr_a = vals.min() + t / 255.0 * np.ptp(vals)
    print(f"  [Nail/pulp] a* Otsu split at {thr_a:.1f}")

    kn = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(3, ks // 2), max(3, ks // 2)))
    nail_bin = ((A < thr_a) & (fmask > 0)).astype(np.uint8) * 255
    pulp_bin = ((A >= thr_a) & (fmask > 0)).astype(np.uint8) * 255
    for m in (nail_bin, pulp_bin):
        tmp = cv2.morphologyEx(m, cv2.MORPH_OPEN, kn)
        m[:] = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, kn, iterations=2)

    pcnts, _ = cv2.findContours(pulp_bin, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)
    if not pcnts:
        raise RuntimeError("Could not find the finger pulp below the nail.")
    pulp_cnt = max(pcnts, key=cv2.contourArea)
    pulp_mask = np.zeros((H, W_img), np.uint8)
    cv2.drawContours(pulp_mask, [pulp_cnt], -1, 255, -1)

    # ── 4. Free-edge band: nail pixels hugging the pulp ──────
    ncols = np.where(nail_bin.any(axis=0))[0]
    if len(ncols) < 10:
        raise RuntimeError("Nail region too small.")
    rough_scale = width_mm / float(ncols[-1] - ncols[0])
    thick_px = thickness_mm / rough_scale
    rad = max(5, int(1.6 * thick_px)) | 1
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad, rad))
    band = cv2.bitwise_and(cv2.dilate(pulp_mask, kd), nail_bin)
    bcnts, _ = cv2.findContours(band, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)
    if not bcnts:
        raise RuntimeError("Could not isolate the free-edge band.")
    band_cnt = max(bcnts, key=cv2.contourArea)
    band_mask = np.zeros((H, W_img), np.uint8)
    cv2.drawContours(band_mask, [band_cnt], -1, 255, -1)
    print(f"  [Band] thickness~{thick_px:.0f}px  dilate r={rad}px  "
          f"bbox={cv2.boundingRect(band_cnt)}")

    # ── 5. Circle fit to band top boundary, hook-tip chord ───
    cols = np.where(band_mask.any(axis=0))[0]
    top_y = np.array([np.argmax(band_mask[:, c] > 0) for c in cols], float)
    xs, ys = cols.astype(float), top_y
    cx, cy, r_px, keep = _fit_circle_robust(xs, ys)
    res_med = float(np.median(np.abs(
        np.hypot(xs[keep] - cx, ys[keep] - cy) - r_px)))
    print(f"  [Circle fit] centre=({cx:.0f},{cy:.0f})  R={r_px:.1f}px  "
          f"inliers={keep.sum()}/{len(xs)}  med.res={res_med:.1f}px")

    # hook tips: extreme-x band pixels lying on the fitted circle
    bys, bxs = np.nonzero(band_mask)
    on_circle = np.abs(np.hypot(bxs - cx, bys - cy) - r_px) < \
        max(3.0, 2.5 * res_med)
    if on_circle.sum() < 8:
        raise RuntimeError("Fitted circle does not match the band.")
    obx, oby = bxs[on_circle], bys[on_circle]
    iL, iR = np.argmin(obx), np.argmax(obx)
    x_L, y_L = float(obx[iL]), float(oby[iL])
    x_R, y_R = float(obx[iR]), float(oby[iR])
    chord_px = float(np.hypot(x_R - x_L, y_R - y_L))
    x_P, y_P = float(xs[keep][np.argmin(ys[keep])]), float(ys[keep].min())

    if chord_px < 10:
        raise RuntimeError(
            f"Degenerate arc: chord={chord_px:.0f}px — "
            "check that the nail is clearly visible in the photo.")

    # ── 6. Scale & final values ───────────────────────────────
    scale_mm_per_px = width_mm / chord_px
    half_c = chord_px / 2.0
    if r_px <= half_c:
        sagitta_px = r_px          # ≥ half circle; clamp
    else:
        sagitta_px = r_px - np.sqrt(r_px ** 2 - half_c ** 2)
    h_mm = round(sagitta_px * scale_mm_per_px, 2)
    arc_R = round(width_mm ** 2 / (8 * h_mm) + h_mm / 2, 2)
    fit_R_mm = round(r_px * scale_mm_per_px, 2)
    # arc length over the curve (what a flexible ruler measures)
    half = min(1.0, chord_px / (2 * r_px))
    arc_len_mm = round(2 * r_px * np.arcsin(half) * scale_mm_per_px, 2)

    print(f"  [Nail arc]  L=({x_L:.0f},{y_L:.0f})  R=({x_R:.0f},{y_R:.0f})  "
          f"peak=({x_P:.0f},{y_P:.0f})")
    print(f"  [Scale]  chord={chord_px:.1f}px  W_mm={width_mm}mm  "
          f"→  {scale_mm_per_px:.5f} mm/px")
    print(f"  [C-curve]  sagitta={sagitta_px:.1f}px  →  h={h_mm}mm")
    print(f"  [Arc R]  chord formula R={arc_R}mm   circle-fit R={fit_R_mm}mm")
    print(f"  [Arc length] over-the-curve width ~ {arc_len_mm}mm")

    # ── 7. Debug visualisation ────────────────────────────────
    if debug_out:
        vis = img.copy()
        ov = vis.copy()
        ov[pulp_mask > 0] = (200, 0, 200)
        ov[band_mask > 0] = (0, 200, 255)
        cv2.addWeighted(ov, 0.35, vis, 0.65, 0, vis)
        cv2.drawContours(vis, [finger_cnt], -1, (0, 255, 0), 2)

        for xa, ya, kp in zip(xs, ys, keep):
            cv2.circle(vis, (int(xa), int(ya)), 2,
                       (0, 255, 255) if kp else (0, 0, 255), -1)
        # fitted circle arc
        th = np.linspace(0, 2 * np.pi, 720)
        for t_ in th:
            px_, py_ = int(cx + r_px * np.cos(t_)), int(cy + r_px * np.sin(t_))
            if 0 <= px_ < W_img and 0 <= py_ < H:
                vis[py_, px_] = (255, 0, 255)

        cv2.circle(vis, (int(x_L), int(y_L)), 6, (0, 200, 0), 2)
        cv2.circle(vis, (int(x_R), int(y_R)), 6, (0, 200, 0), 2)
        cv2.circle(vis, (int(x_P), int(y_P)), 6, (0, 0, 255), -1)
        cv2.line(vis, (int(x_L), int(y_L)), (int(x_R), int(y_R)),
                 (255, 200, 0), 2)

        # zoomed crop around the fingertip with labels
        m = 80
        x0, y0 = max(fx - m, 0), max(fy - m, 0)
        x1, y1 = min(fx + fw + m, W_img), min(fy + fh + m, H)
        crop = vis[y0:y1, x0:x1]
        zoom = max(1, int(900 / max(crop.shape[:2])))
        crop = cv2.resize(crop, None, fx=zoom, fy=zoom,
                          interpolation=cv2.INTER_NEAREST)
        pad = np.zeros((crop.shape[0] + 160, crop.shape[1], 3), np.uint8)
        pad[:crop.shape[0]] = crop
        for i, txt in enumerate([
                f"W={width_mm}mm  chord={chord_px:.0f}px  "
                f"scale={scale_mm_per_px:.4f}mm/px",
                f"C-curve h={h_mm}mm   R={arc_R}mm (fit {fit_R_mm}mm)",
                f"over-the-curve width={arc_len_mm}mm"]):
            cv2.putText(pad, txt, (12, crop.shape[0] + 40 + 45 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imwrite(debug_out, pad)
        print(f"  [Debug] saved → {debug_out}")

    return {
        "c_curve_mm":    h_mm,
        "arc_radius_mm": arc_R,
        "arc_radius_fit_mm": fit_R_mm,
        "arc_length_mm": arc_len_mm,
        "arc_width_px":  round(chord_px, 1),
        "sagitta_px":    round(sagitta_px, 1),
        "scale_mm_per_px": round(scale_mm_per_px, 5),
        "nail_endpoints": {
            "left":  [int(x_L), int(y_L)],
            "right": [int(x_R), int(y_R)],
            "peak":  [int(x_P), int(y_P)],
        },
    }


def main():
    p = argparse.ArgumentParser(
        description="C-curve from end-on finger photo (no ArUco needed)")
    p.add_argument("--image",     required=True, help="End-on photo path")
    p.add_argument("--width-mm",  type=float, required=True,
                   help="Known nail width in mm (from top-photo measurement)")
    p.add_argument("--debug-out", default=None,
                   help="Path to save annotated debug image")
    p.add_argument("--json-out",  default=None,
                   help="Path to save result JSON")
    args = p.parse_args()

    print(f"\nC-curve measurement: {args.image}")
    print(f"  Known nail width: {args.width_mm}mm\n")

    result = measure_ccurve(args.image, args.width_mm, args.debug_out)

    print(f"\n  ┌─ C-CURVE RESULT ─────────────────────────────")
    print(f"  │  Sagitta (C-curve)  : {result['c_curve_mm']} mm")
    print(f"  │  Arc radius (R)     : {result['arc_radius_mm']} mm")
    print(f"  │  Scale used         : {result['scale_mm_per_px']} mm/px")
    print(f"  └───────────────────────────────────────────────")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  [JSON] saved → {args.json_out}")


if __name__ == "__main__":
    main()
