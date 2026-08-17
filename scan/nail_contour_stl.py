"""
nail_contour_stl.py — hybrid: real contour body + designed tip shape

Body/sides: the ACTUAL captured nail_polygon_px outline (nail_measurer.py's
traced footprint) — the person's real, asymmetric nail silhouette.

Cuticle end and tip end: nail_measurer.py's traced polygon includes its own
synthetic cuticle-arc placeholder near the back (not measured), and the real
free-edge trace at the tip is just whatever that person's fingertip happened
to look like in the photo — neither is meant to define the final product
shape. So both ends are replaced with nail_exact_stl.py's proven analytic
formulas:
  - Cuticle: the single-circular-arc cuticle line (closes to a point).
  - Tip: the SAME shape catalogue as nail_exact_stl.py (round/oval/almond/
    square/stiletto/ballerina), including its usual tip extension for
    wearability — so picking --shape square gives the real per-finger body
    silhouette with a designed, comfortable square tip grafted on.

Both replacements are anchored to the REAL measured width right at the
boundary row, so there's no visible jump where real data hands off to the
designed geometry.

How the two boundaries are found: nail_measurer.py builds the traced polygon
row-by-row (this is what makes horizontal-slicing it back well-posed — see
slice_polygon_at_y). The real per-row width plateaus in the body and falls
off near both ends; the plateau value (85th percentile) is used as a
reference, and the first row (scanning in from each end) that reaches 92% of
it marks that end's real/designed boundary.
"""
import argparse, json, os, sys
import numpy as np
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nail_exact_stl import (write_binary_stl, arc_z, x_extent,
                            SHAPES, TIP_HEIGHT_FACTOR, FLAT_TIP_SHAPES, LONG_SHAPES)


def polygon_px_to_mm(nail):
    poly = np.asarray(nail["nail_polygon_px"], dtype=float)
    xs, ys = poly[:, 0], poly[:, 1]

    mpp = nail.get("mpp_mm_per_px")
    if mpp:
        # Exact per-photo ArUco calibration (mm/px), stored by nail_measurer.py.
        # Isotropic camera -> a single scale is correct for both axes.
        scale = scale_x = scale_y = float(mpp)
        source = "measured (ArUco mpp_mm_per_px)"
    else:
        # Older captures without mpp_mm_per_px: approximate the scale from
        # the polygon's own pixel bounding box vs. the reported width/length.
        px_w = xs.max() - xs.min()
        px_h = ys.max() - ys.min()
        scale_x = nail["width_mm"] / px_w
        scale_y = nail["length_mm"] / px_h
        scale   = (scale_x + scale_y) / 2.0
        source  = "approximated (bbox ratio, no stored mpp)"

    # Cuticle sits at the larger pixel row (bottom of the traced region);
    # tip sits at the smaller pixel row. Flip/shift so mm_y=0 at the cuticle
    # edge and mm_y increases toward the tip (matches nail_exact_stl.py's
    # y convention: cuticle near 0, tip at larger y).
    y_cuticle_px = ys.max()
    mm_x = (xs - xs.min()) * scale
    mm_y = (y_cuticle_px - ys) * scale
    return np.column_stack([mm_x, mm_y]), scale, scale_x, scale_y, source


def slice_polygon_at_y(poly_mm, y_val):
    """Scanline horizontal slice: all x-crossings of the closed polygon at y_val."""
    n = len(poly_mm)
    xs_hit = []
    for i in range(n):
        x1, y1 = poly_mm[i]
        x2, y2 = poly_mm[(i + 1) % n]
        if (y1 <= y_val < y2) or (y2 <= y_val < y1):
            t = (y_val - y1) / (y2 - y1)
            xs_hit.append(x1 + t * (x2 - x1))
    if not xs_hit:
        return None
    return min(xs_hit), max(xs_hit)


def cuticle_arc_extent(depth_from_boundary, W, cut_depth, x_cen):
    """
    Single circular-arc cuticle line — identical formula to
    nail_exact_stl.py's x_extent() cuticle-arc branch. Passes through the two
    side corners (depth=0, full width W) and closes to a point at
    depth=cut_depth (the very back). depth_from_boundary is the distance
    back from the real/arc boundary (0 at boundary -> cut_depth at the back).
    """
    W2  = W / 2.0
    y_c = (W2 * W2 - cut_depth * cut_depth) / (2.0 * cut_depth)
    R_c = cut_depth + y_c
    y_val = -depth_from_boundary   # nail_exact_stl.py convention: 0 at boundary, -cut_depth at the point
    half_w = np.sqrt(np.maximum(R_c * R_c - (y_val - y_c) ** 2, 0.0))
    return x_cen - half_w, x_cen + half_w


def generate_contour_stl(nail, output_path, shape="round", thickness_mm=0.6,
                         tip_extension_mm=None, edge_round_mm=None,
                         corner_round_mm=None, taper_mm=None, nx=50, ny=140):
    C     = float(nail["c_curve_mm"])
    arc_r = float(nail["arc_radius_mm"])
    poly_mm, scale, sx, sy, scale_source = polygon_px_to_mm(nail)

    y_min, y_max = poly_mm[:, 1].min(), poly_mm[:, 1].max()
    eps = (y_max - y_min) * 1e-3
    ys_real = np.linspace(y_min + eps, y_max - eps, ny)

    left  = np.zeros(ny)
    right = np.zeros(ny)
    for i, y in enumerate(ys_real):
        hit = slice_polygon_at_y(poly_mm, y)
        if hit is None:
            hit = (left[i - 1], right[i - 1]) if i > 0 else (0.0, 0.0)
        left[i], right[i] = hit

    # Light smoothing: the traced polygon (photo edge-detection + spline fit)
    # carries small-scale noise — local blips over ~0.1-0.2mm that read as a
    # visible bump/crease on the printed surface even though they're tiny in
    # absolute size, because flat per-facet shading exaggerates any sudden
    # normal-direction change. A small uniform filter removes that noise
    # while leaving the real nail's actual (much larger-scale) shape intact.
    smooth_win = max(3, (int(round(ny / (ys_real[-1] - ys_real[0]) * 0.4)) | 1))
    left  = uniform_filter1d(left,  size=smooth_win, mode="nearest")
    right = uniform_filter1d(right, size=smooth_win, mode="nearest")

    width = right - left
    plateau_w = float(np.percentile(width, 85))

    # ── Cuticle boundary: first row (from the back) reaching the plateau ──
    i_cut = int(np.argmax(width >= 0.92 * plateau_w))
    cut_depth  = max(ys_real[i_cut] - ys_real[0], 0.3)
    W_cut      = width[i_cut]
    x_cen_cut  = (left[i_cut] + right[i_cut]) / 2.0
    y_cut_boundary = ys_real[i_cut]

    # ── Tip/body boundary: first row (from the tip, scanning backward) ──
    # reaching the plateau — this is where the real trace stops being
    # representative of "generic body width" and becomes tip-specific.
    rev = width[::-1]
    i_tip = ny - 1 - int(np.argmax(rev >= 0.92 * plateau_w))
    y_body_tip = max(ys_real[i_tip], y_cut_boundary + 0.3)   # keep ordering safe
    W_body_tip    = width[i_tip]
    left_body_tip = left[i_tip]

    # ── Designed tip parameters (mirrors nail_exact_stl.py's generate_stl) ──
    ext_default = 7.0 if shape in LONG_SHAPES else 3.0
    L_ext = float(tip_extension_mm) if tip_extension_mm is not None else ext_default
    CORNER_R = float(corner_round_mm) if corner_round_mm is not None else (1.5 if shape == "square" else 0.0)
    TAPER_MM = float(taper_mm) if taper_mm is not None else (1.0 if shape == "square" else 0.0)

    L_total_local = y_body_tip + L_ext   # stands in for nail_exact_stl.py's L_total = L + L_ext
    if shape == "stiletto":
        tip_h = L_ext + y_body_tip * 0.30
    elif shape == "ballerina":
        tip_h = L_total_local * 0.5
    elif shape == "square":
        # The taper+corner-round region must stay entirely within the
        # extension (L_ext) — the real traced nail (up to y_body_tip) has to
        # be fully covered at full width first, and only narrow after that.
        # Capped at L_ext so y_taper_start never dips below y_body_tip.
        tip_h = min(L_ext, max(CORNER_R, W_body_tip * TIP_HEIGHT_FACTOR.get(shape, 0.45)))
        CORNER_R = min(CORNER_R, tip_h)
    else:
        tip_h = W_body_tip * TIP_HEIGHT_FACTOR.get(shape, 0.50)

    y_taper_start  = L_total_local - tip_h
    L_total_final  = L_total_local
    y_real_end     = min(y_body_tip, y_taper_start)
    # Straight shelf only exists when the extension is longer than the taper
    # zone (y_taper_start > y_body_tip, e.g. round/oval/square with a short
    # tip_h). When tip_h is long (stiletto/ballerina), the taper starts
    # BEFORE y_body_tip and eats into the real-data region instead — no
    # shelf, real data just gets cut a bit shorter.
    has_shelf = y_taper_start > y_body_tip

    # Anchor for the analytic tip formula: the REAL measured width right at
    # the actual hand-off row (y_real_end), not W_body_tip/left_body_tip —
    # those describe row i_tip, a DIFFERENT (later) row whenever the taper
    # starts early (tip_h > L_ext, e.g. oval/almond/stiletto/ballerina) and
    # eats into the real data before reaching i_tip. Anchoring to the wrong
    # row leaves a visible step where real data hands off to the formula.
    if y_real_end < y_body_tip:
        left_anchor  = float(np.interp(y_real_end, ys_real, left))
        right_anchor = float(np.interp(y_real_end, ys_real, right))
    else:
        left_anchor, right_anchor = left_body_tip, left_body_tip + W_body_tip
    W_anchor = right_anchor - left_anchor
    center_anchor = (left_anchor + right_anchor) / 2.0

    # Real fingers (thumb especially) often have a nail bed that isn't
    # dead-straight — the whole plate's centreline drifts sideways along its
    # length. Freezing the centreline at the hand-off row and tapering
    # symmetrically from there ignores that ongoing drift, so the edge that
    # was "supposed" to keep drifting suddenly bends the other way right at
    # the seam. Estimate the drift rate from the real data just before the
    # hand-off and keep extrapolating it through the designed tip, so only
    # the WIDTH follows the chosen shape while the axis stays consistent.
    idx_anchor = int(np.argmin(np.abs(ys_real - y_real_end)))
    idx_prev   = max(0, idx_anchor - max(2, smooth_win))
    if idx_anchor > idx_prev:
        cen_now  = (left[idx_anchor] + right[idx_anchor]) / 2.0
        cen_prev = (left[idx_prev] + right[idx_prev]) / 2.0
        dy = ys_real[idx_anchor] - ys_real[idx_prev]
        center_slope = (cen_now - cen_prev) / dy if dy > 1e-6 else 0.0
    else:
        center_slope = 0.0

    # ── Build the final row grid: cuticle point -> designed tip apex ──
    ys = np.linspace(ys_real[0], L_total_final, ny)
    left_f, right_f = np.zeros(ny), np.zeros(ny)
    for i, y in enumerate(ys):
        if y <= y_cut_boundary:
            l, r = cuticle_arc_extent(y_cut_boundary - y, W_cut, cut_depth, x_cen_cut)
        elif y <= y_real_end:
            # Sample the SMOOTHED real-data arrays (not a fresh raw slice —
            # that would reintroduce the photo-trace noise the filter above
            # just removed).
            l, r = float(np.interp(y, ys_real, left)), float(np.interp(y, ys_real, right))
        elif has_shelf and y <= y_taper_start:
            # Straight shelf: real body plateau extended out to where the
            # designed taper begins, following the real axis drift.
            center_y = center_anchor + center_slope * (y - y_real_end)
            l, r = center_y - W_body_tip / 2.0, center_y + W_body_tip / 2.0
        else:
            xl, xr = x_extent(y - y_taper_start, W_anchor, tip_h, tip_h,
                              0.0, shape, tip_r=CORNER_R, taper_mm=TAPER_MM)
            half_w = (xr - xl) / 2.0
            center_y = center_anchor + center_slope * (y - y_real_end)
            l, r = center_y - half_w, center_y + half_w
        left_f[i], right_f[i] = l, r
    left, right = left_f, right_f

    grid_x = left[:, None] + np.linspace(0, 1, nx)[None, :] * (right - left)[:, None]
    grid_y = np.repeat(ys[:, None], nx, axis=1)

    x_cen_row = ((left + right) / 2.0)[:, None]     # per-row local centre
    arc_off   = arc_z(grid_x, x_cen_row, C, arc_r)   # 0 (centre) -> C (edges)
    z_bot     = C - arc_off                          # dome: C (centre) -> 0 (edges)
    z_top     = z_bot + thickness_mm

    # ── Top perimeter edge rounding (mirrors nail_exact_stl.py exactly, ──
    # gated on the cuticle boundary instead of a literal y>=0).
    EDGE_R = float(edge_round_mm) if edge_round_mm is not None else \
        (1.0 if shape in {"almond", "ballerina", "square", "stiletto"} else 0.0)
    if EDGE_R > 0:
        r = min(EDGE_R, thickness_mm * 0.95)
        x_left_row  = grid_x[:, 0:1]
        x_right_row = grid_x[:, -1:]
        d_lat = np.minimum(grid_x - x_left_row, x_right_row - grid_x)

        if shape not in FLAT_TIP_SHAPES:
            dz_lat = np.where(d_lat < r, r - np.sqrt(np.maximum(r**2 - (r - d_lat)**2, 0.0)), 0.0)
            dz_lat = np.where(grid_y >= y_cut_boundary, dz_lat, 0.0)
            # Pointed shapes (almond, stiletto) also get a tip-apex fillet.
            # Combine via max (not sequential subtraction): applying both
            # reductions one after another can, near the tip corner where
            # both distances are small, subtract more than the full shell
            # thickness and clamp z_top down to z_bot over a whole patch —
            # collapsing the shell to zero thickness there and making the
            # top/bottom faces coincide (breaks watertightness).
            if shape not in ("round", "oval"):
                d_tip_pt = np.maximum(L_total_final - grid_y, 0.0)
                dz_tip_pt = np.where(d_tip_pt < r,
                                     r - np.sqrt(np.maximum(r**2 - (r - d_tip_pt)**2, 0.0)), 0.0)
                dz_lat = np.maximum(dz_lat, dz_tip_pt)
            z_top  = np.maximum(z_top - dz_lat, z_bot)
        elif shape == "square":
            d_tip_arr = np.maximum(L_total_final - grid_y, 0.0)
            in_body   = grid_y >= y_cut_boundary
            dz_tip = np.where((d_tip_arr < r) & in_body,
                              r - np.sqrt(np.maximum(r**2 - (r - d_tip_arr)**2, 0.0)), 0.0)
            in_corner = (d_lat < r) & (d_tip_arr < r) & in_body
            under_c   = r**2 - (r - d_lat)**2 - (r - d_tip_arr)**2
            dz_corner = np.where(in_corner,
                                 np.where(under_c >= 0.0, r - np.sqrt(np.maximum(under_c, 0.0)), r), 0.0)
            z_top = np.maximum(z_top - np.maximum(dz_tip, dz_corner), z_bot)
        else:
            d_tip_arr = np.maximum(L_total_final - grid_y, 0.0)
            in_corner = (d_lat < r) & (d_tip_arr < r)
            under_c   = r**2 - (r - d_lat)**2 - (r - d_tip_arr)**2
            dz_corner = np.where(in_corner,
                                 np.where(under_c >= 0.0, r - np.sqrt(np.maximum(under_c, 0.0)), r), 0.0)
            z_top = np.maximum(z_top - dz_corner, z_bot)

    # Round to 1e-6mm before building any triangles. Points that are
    # mathematically identical (e.g. every column at the collapsed tip/back
    # row) can otherwise differ by a few 1e-7mm due to floating-point noise
    # from being computed via different formulas/columns — far below
    # print resolution, but enough that trimesh's 1e-8mm vertex-merge
    # tolerance won't weld them, leaving a pinhole gap at the closure point.
    grid_x = np.round(grid_x, 6)
    grid_y = np.round(grid_y, 6)
    z_top  = np.round(z_top, 6)
    z_bot  = np.round(z_bot, 6)

    top3d = np.stack([grid_x, grid_y, z_top], axis=2)
    bot3d = np.stack([grid_x, grid_y, z_bot], axis=2)

    tris = []
    NX = nx - 1

    # Top face (+Z)
    for i in range(ny - 1):
        for j in range(NX):
            tris.append([top3d[i, j],   top3d[i, j+1], top3d[i+1, j]])
            tris.append([top3d[i, j+1], top3d[i+1, j+1], top3d[i+1, j]])

    # Bottom face (-Z)
    for i in range(ny - 1):
        for j in range(NX):
            tris.append([bot3d[i, j+1], bot3d[i, j],   bot3d[i+1, j]])
            tris.append([bot3d[i+1, j+1], bot3d[i, j+1], bot3d[i+1, j]])

    # Left wall (-X)
    for i in range(ny - 1):
        T0, T1 = top3d[i, 0], top3d[i+1, 0]
        B0, B1 = bot3d[i, 0], bot3d[i+1, 0]
        tris.append([T0, T1, B0])
        tris.append([T1, B1, B0])

    # Right wall (+X)
    for i in range(ny - 1):
        T0, T1 = top3d[i, NX], top3d[i+1, NX]
        B0, B1 = bot3d[i, NX], bot3d[i+1, NX]
        tris.append([T0, B1, T1])
        tris.append([T0, B0, B1])

    # Back cap (-Y) at row 0. Degenerate (~zero-area) whenever the cuticle
    # arc has already converged to a point; kept as a safety net for the
    # eps-short grid start, dropped by the filter below when unneeded.
    for j in range(NX):
        t0, t1 = top3d[0, j], top3d[0, j+1]
        b0, b1 = bot3d[0, j], bot3d[0, j+1]
        tris.append([b0, t1, t0])
        tris.append([b0, b1, t1])

    # Tip cap (+Y) at row ny-1 — needed for flat-tip shapes (square,
    # ballerina); degenerate and dropped for pointed shapes.
    M = ny - 1
    for j in range(NX):
        t0, t1 = top3d[M, j], top3d[M, j+1]
        b0, b1 = bot3d[M, j], bot3d[M, j+1]
        tris.append([t0, t1, b0])
        tris.append([t1, b1, b0])

    EPS = 1e-4
    clean = []
    for tri in tris:
        v0, v1, v2 = (np.asarray(tri[0]), np.asarray(tri[1]), np.asarray(tri[2]))
        if (np.linalg.norm(v0 - v1) > EPS and
                np.linalg.norm(v1 - v2) > EPS and
                np.linalg.norm(v0 - v2) > EPS):
            clean.append(tri)

    # Orient outward: signed volume via divergence theorem should be positive
    # for a correctly-wound closed mesh. Flip everything if it comes out
    # negative (globally inside-out) rather than trying to reason about
    # winding by hand for every face type above.
    vol = 0.0
    for tri in clean:
        v0, v1, v2 = tri
        vol += np.dot(v0, np.cross(v1, v2))
    vol /= 6.0
    if vol < 0:
        clean = [[t[0], t[2], t[1]] for t in clean]

    n_tris = write_binary_stl(output_path, clean)
    kb = round(os.path.getsize(output_path) / 1024, 1)
    print(f"  [CONTOUR-STL] shape={shape}  {n_tris} tris  {kb} KB  vol={abs(vol):.1f}mm3  -> {output_path}")
    print(f"  px->mm scale: x={sx:.5f} y={sy:.5f} (avg {scale:.5f})  [{scale_source}]")
    print(f"  cuticle arc: replaced back {cut_depth:.2f}mm  |  "
          f"body/tip boundary at y={y_body_tip:.2f}mm (W={W_body_tip:.2f}mm)  |  "
          f"designed tip: +{L_ext:.1f}mm ext, tip_h={tip_h:.2f}mm, "
          f"total length={L_total_final:.2f}mm")
    return n_tris, abs(vol)


def main():
    p = argparse.ArgumentParser(
        description="Hybrid nail STL: real traced body + designed tip shape")
    p.add_argument("--input", required=True)
    p.add_argument("--finger", default=None)
    p.add_argument("--output", default="nail_contour_stl")
    p.add_argument("--thickness", type=float, default=0.6)
    p.add_argument("--shape", default="round", choices=SHAPES,
                   help="Tip shape: round | oval | almond | square | "
                        "stiletto | ballerina  (default: round)")
    p.add_argument("--tip-extension", type=float, default=None,
                   help="Extra mm beyond the real/designed body boundary "
                        "(default: 7mm for almond/stiletto/ballerina, 3mm otherwise)")
    p.add_argument("--edge-round", type=float, default=None,
                   help="Fillet radius (mm) for the top perimeter edge "
                        "(default: 1.0 for almond/ballerina/square/stiletto, 0 for round/oval)")
    p.add_argument("--corner-round", type=float, default=None,
                   help="Plan-view fillet radius (mm) for square's tip corners "
                        "(default: 1.5 for square, ignored by other shapes)")
    p.add_argument("--taper", type=float, default=None,
                   help="Total mm square's plate narrows by leading into the "
                        "tip corners (default: 1.0 for square, ignored otherwise)")
    args = p.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    os.makedirs(args.output, exist_ok=True)

    nails = data.get("nails", [])
    if args.finger:
        nails = [n for n in nails if n["finger"] == args.finger]
    if not nails:
        sys.exit("ERROR: No matching nails in JSON")

    for nail in nails:
        finger = nail["finger"]
        print(f"\n  [{finger.upper()}]  W={nail['width_mm']}mm  L={nail['length_mm']}mm  "
              f"C={nail['c_curve_mm']}mm  R={nail['arc_radius_mm']}mm  "
              f"poly_pts={len(nail['nail_polygon_px'])}")
        out = os.path.join(args.output, f"nail_{finger}_{args.shape}.stl")
        generate_contour_stl(nail, out, shape=args.shape, thickness_mm=args.thickness,
                             tip_extension_mm=args.tip_extension,
                             edge_round_mm=args.edge_round,
                             corner_round_mm=args.corner_round,
                             taper_mm=args.taper)


if __name__ == "__main__":
    main()
