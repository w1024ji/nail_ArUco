"""
nail_exact_stl.py  v15
Exact-fit nail STL: structured grid, correct c-curve orientation, natural
cuticle arch, watertight manifold.

Changes vs v14:
  - FIXED cuticle arch shape.
      v14 used a cosine-smoothstep with a W/5 floor width; the flat floor
      at y = -cuticle_depth appeared as a visible bump/shelf at the centre
      of the cuticle edge.
      v15 replaces this with a single circular arc whose two endpoints are
      the side corners (0, 0) and (W, 0) and whose deepest point is exactly
      (W/2, -cuticle_depth).  One smooth arc, no flat sections, no bump.
      The arch-bottom cap face (needed only for the flat floor) is removed.

Changes vs v13 (retained from v14):
  - FIXED c-curve orientation on both surfaces.
      v13 had z_bot = arc_z (bowl: centre low, edges high) → wrong;
      the inner surface didn't match the convex real nail.
      v15: z_bot = C - arc_z  (dome: centre=C, edges=0) — seats flush.
           z_top = z_bot + THICK — uniform shell, also dome-shaped.

Winding (all analytically verified):
    Top face       : [t[i,j],   t[i,j+1], t[i+1,j]  ]  +Z
                     [t[i,j+1], t[i+1,j+1], t[i+1,j]]  +Z
    Bottom face    : [b[i,j+1], b[i,j],   b[i+1,j]  ]  -Z
                     [b[i+1,j+1], b[i,j+1], b[i+1,j]]  -Z
    Left wall      : [T[i,0],   T[i+1,0], B[i,0]  ]    -X
                     [T[i+1,0], B[i+1,0], B[i,0]  ]    -X
    Right wall     : [T[i,N], B[i+1,N], T[i+1,N]]      +X
                     [T[i,N], B[i,N],   B[i+1,N]]      +X
    Arch-bottom cap: [t[0,j], b[0,j], t[0,j+1]]        -Y
                     [b[0,j], b[0,j+1], t[0,j+1]]      -Y
    Tip cap        : [t[M,j], t[M,j+1], b[M,j]]        +Y  (flat-tip shapes only)
                     [t[M,j+1], b[M,j+1], b[M,j]]      +Y
  Every shared edge is traversed in opposite directions → manifold.
"""
import argparse, json, os, struct, sys
import numpy as np


# ─────────────────────────────────────────────────────────────
# STL writer
# ─────────────────────────────────────────────────────────────

def write_binary_stl(filepath, triangles):
    tris = np.asarray(triangles, dtype=np.float32)
    n = len(tris)
    with open(filepath, "wb") as f:
        f.write(b"nail_exact_stl_v15" + b"\x00" * 62)
        f.write(struct.pack("<I", n))
        for tri in tris:
            v0, v1, v2 = tri[0], tri[1], tri[2]
            nrm = np.cross(v1 - v0, v2 - v0)
            ln  = float(np.linalg.norm(nrm))
            if ln > 0:
                nrm = (nrm / ln).astype(np.float32)
            else:
                nrm = np.zeros(3, dtype=np.float32)
            f.write(struct.pack("<3f", *nrm))
            f.write(struct.pack("<3f", *v0.astype(np.float32)))
            f.write(struct.pack("<3f", *v1.astype(np.float32)))
            f.write(struct.pack("<3f", *v2.astype(np.float32)))
            f.write(b"\x00\x00")
    return n


# ─────────────────────────────────────────────────────────────
# Tip shape catalogue
# ─────────────────────────────────────────────────────────────

SHAPES = ("round", "oval", "almond", "square", "stiletto", "ballerina")

# Height of the tip taper region as a fraction of nail width W.
# Shapes whose tips taper to a point (round, almond, stiletto) are closed
# by the degenerate-triangle filter.  Flat-ended shapes (square, ballerina)
# need an explicit tip-cap face and a non-zero tip_h so the cap
# has a well-defined cross-section.
TIP_HEIGHT_FACTOR = {
    "round":     0.50,  # semi-ellipse (tip_h = 0.5·W) → perfect semi-circle, short and wide
    "oval":      0.65,  # semi-ellipse (tip_h = 0.65·W) → taller ellipse, more elongated than round
    "almond":    0.85,  # elongated ellipse — tapered sides, fully rounded tip
    "square":    0.45,  # flat tip, but with a subtle width taper leading into it (see taper_mm)
    "stiletto":  1.50,  # linear taper to a sharp point (longer, more dramatic)
    "ballerina": 1.00,  # linear taper to flat tip ~40 % of nail width
}

# Shapes whose tip is a flat face (need an explicit cap triangle strip).
FLAT_TIP_SHAPES = {"square", "ballerina"}

# Shapes worn long — default tip extension 7 mm instead of 3 mm.
LONG_SHAPES = {"almond", "stiletto", "ballerina"}

# 폭 보정(fit margin): 측정한 평평한 폭을 C-curve로 구부리면 실제 손가락을
# 감싸는 현(chord) 폭이 측정값보다 작아져서, 프린트한 손톱이 실제보다 작게
# 나옴.  이를 보정하기 위해 STL 생성 전에 측정 폭에 이 값을 더한다.
# (--exact 모드는 검증용 실측 복제이므로 적용하지 않음.)
WIDTH_FIT_MARGIN_MM = 1.5


# ─────────────────────────────────────────────────────────────
# Nail shape: analytic cross-section width at each Y
# ─────────────────────────────────────────────────────────────

def x_extent(y_val, W, L_total, tip_h, cuticle_depth, shape="round", tip_r=0.0, taper_mm=0.0):
    """
    Return (x_left, x_right) for the nail footprint at height y_val.

    Three regions:
      y in [-cuticle_depth, 0]        : circular-arc cuticle arch
      y in [0,  L_total - tip_h]      : straight sides, full width W
      y in [L_total-tip_h, L_total]   : tip region — shape-dependent
    """
    y_side_top = L_total - tip_h

    if y_val >= y_side_top:
        # ── Tip region ────────────────────────────────────────
        if shape == "square":
            # Real square nails aren't perfectly parallel-sided: the plate
            # narrows a little (a few tenths of a mm per side) from the body
            # toward the free edge before the flat tip, THEN the last tip_r
            # mm of that already-slightly-narrower edge gets the circular-arc
            # corner rounding. t=0 at the start of this region (full W),
            # t=1 at the tip (W - taper_mm).
            t = min((y_val - y_side_top) / tip_h, 1.0) if tip_h > 0 else 1.0
            half_reduction = (taper_mm / 2.0) * t
            xl0, xr0 = half_reduction, float(W) - half_reduction
            if tip_r > 0 and y_val >= (L_total - tip_r):
                d_y   = y_val - (L_total - tip_r)          # 0→tip_r
                inset = tip_r - float(np.sqrt(max(tip_r ** 2 - d_y ** 2, 0.0)))
                return xl0 + inset, xr0 - inset
            return xl0, xr0

        # Normalised position within tip: 0 at base, 1 at tip end.
        t = min((y_val - y_side_top) / tip_h, 1.0) if tip_h > 0 else 1.0

        if shape in ("round", "oval"):
            # Ellipse: x²/a² + y²/b² = 1 → half_w = (W/2)·√(1−t²)
            # Both use the same smooth ellipse curve; oval just has a taller
            # tip region (TIP_HEIGHT_FACTOR 0.65 vs 0.50) → more elongated.
            half_w = W / 2 * float(np.sqrt(max(1.0 - t * t, 0.0)))

        elif shape == "almond":
            # Single smooth ellipse — same formula as round/oval.
            # The taller TIP_HEIGHT_FACTOR (0.85 vs oval's 0.65) gives the
            # characteristic almond silhouette: more tapered sides, rounded tip.
            half_w = W / 2 * float(np.sqrt(max(1.0 - t * t, 0.0)))

        elif shape == "stiletto":
            # Convex taper to a sharp point.
            # Power < 1 makes sides bow outward (convex) near the taper base
            # then converge smoothly to the tip — matches the elegant stiletto
            # silhouette in the reference (vs. a harsh straight-line taper).
            # w(t) = W·(1−t)^0.65
            half_w = W / 2 * (1.0 - t) ** 0.65

        elif shape == "ballerina":
            # Linear taper from full width to ~40 % of W at the flat top.
            # w(t) = W·(1 − 0.6·t)  →  0.4·W at t=1
            half_w = W / 2 * (1.0 - 0.6 * t)

        else:
            # Fallback: round
            half_w = W / 2 * float(np.sqrt(max(1.0 - t * t, 0.0)))

        return W / 2 - half_w, W / 2 + half_w

    elif y_val >= 0.0:
        # Straight sides
        return 0.0, float(W)

    elif y_val >= -cuticle_depth:
        # Single circular arc cuticle line.
        # The arc passes through (0, 0) and (W, 0) — the two side corners —
        # and reaches its deepest point at (W/2, -cuticle_depth).
        # Arc centre sits at (W/2, y_c); radius R_c from the two constraints:
        #   corner on arc : (W/2)² + y_c² = R_c²
        #   bottom of arc : R_c = cuticle_depth + y_c
        # → y_c = ((W/2)² − cuticle_depth²) / (2·cuticle_depth)
        W2    = W / 2.0
        y_c   = (W2 * W2 - cuticle_depth * cuticle_depth) / (2.0 * cuticle_depth)
        R_c   = cuticle_depth + y_c
        half_w = float(np.sqrt(max(R_c * R_c - (y_val - y_c) ** 2, 0.0)))
        return W2 - half_w, W2 + half_w

    else:
        return W / 2, W / 2   # below arch


# ─────────────────────────────────────────────────────────────
# C-curve Z offset (sagitta of circular arc)
# ─────────────────────────────────────────────────────────────

def arc_z(x_arr, x_cen, c_mm, arc_r):
    """
    Vectorised sagitta: z = arc_r - sqrt(arc_r^2 - (x - x_cen)^2).
    Returns 0 at the nail centre, rises to c_mm at the edges.
    """
    if c_mm < 0.05:
        return np.zeros_like(np.asarray(x_arr, dtype=float))
    dx = np.asarray(x_arr, dtype=float) - x_cen
    return arc_r - np.sqrt(np.maximum(arc_r ** 2 - dx ** 2, 0.0))


# ─────────────────────────────────────────────────────────────
# STL generator — structured grid mesh
# ─────────────────────────────────────────────────────────────

def generate_stl(params, output_path):
    C         = float(params["c_curve_mm"])
    arc_r     = float(params["arc_radius_mm"])
    THICK     = float(params.get("thickness_mm", 0.6))
    EXACT     = bool(params.get("exact"))
    CUT_DEPTH = float(params.get("cuticle_depth_mm", 1.5))
    # Extra dome height (mm) at the cuticle end, blending to 0 at the free
    # edge — makes the inner cavity rounder where the finger flesh is rounder.
    # Forced to 0 in exact mode (validation replica must stay true).
    # Reverted to the previous uniform C-curve dome: the cuticle-side extra
    # roundness (cuticle_curve) is now OFF by default (0.0).  It made prints
    # come out wrong, so the inner dome again uses the measured C-curve at
    # every row.  Pass --cuticle-curve >0 to re-enable the blend if wanted.
    CUT_CURVE = 0.0 if params.get("exact") else \
        float(params.get("cuticle_curve_mm", 0.0))
    _shape_tmp = params.get("shape", "round")
    if EXACT:
        # Exact-replica mode (validation prints): true measured dimensions,
        # no comfort shrink, no tip extension, no W/L length "correction".
        # Centreline length incl. the cuticle arch equals the measured length.
        W     = float(params["width_mm"])
        L     = max(float(params["length_mm"]) - CUT_DEPTH, 1.0)
        L_ext = 0.0
    else:
        # 폭 보정: C-curve로 구부리면 현(chord) 폭이 줄어 실제보다 작게
        # 프린트되므로, 측정 폭에 fit margin(+WIDTH_FIT_MARGIN_MM)을 더한다.
        W     = float(params["width_mm"]) + WIDTH_FIT_MARGIN_MM
        # Length: use the MEASURED length (Sobel tip→cuticle).  The old code
        # preferred corrected_length_mm = width / 0.91 (Jung et al. W/L prior),
        # but ruler ground truth showed 0.91 is wrong (true ~0.64–0.86), which
        # made prints ~4 mm too short.  corrected_length_mm is now only a
        # fallback when no measured length exists.
        L     = float(params.get("length_mm")
                      or params.get("corrected_length_mm") or 0.0)
        # Shape-specific default extensions: stiletto/coffin get 7 mm,
        # all other shapes default to 3 mm.  An explicit --tip-extension
        # value always wins (passed as a non-None entry in params).
        _ext_default = 7.0 if _shape_tmp in LONG_SHAPES else 3.0
        L_ext = float(params.get("tip_extension_mm") or _ext_default)
    x_cen     = W / 2.0

    shape   = params.get("shape", "round")
    EDGE_R  = float(params.get("edge_round_mm", 0.0))
    L_total = L + L_ext

    # Square's plan-view (XY) tip corners: rounded off by CORNER_R mm so the
    # free-edge corners don't scratch skin (a quarter-circle inset, see the
    # "square" branch of x_extent()). 0 = sharp 90° corners.
    CORNER_R = float(params.get("corner_round_mm", 0.0) or 0.0)
    # Square's subtle width taper leading into the tip (see x_extent()) —
    # total mm the plate narrows by, split evenly across both sides.
    TAPER_MM = float(params.get("taper_mm", 0.0) or 0.0)

    # Stiletto: taper covers the extension plus the top 30 % of the natural nail,
    # so the sides start narrowing before the free edge — matches the smooth
    # elongated silhouette in the reference (not just an abrupt spike at the tip).
    # Coffin: taper over extension only (flat tip keeps full width longer).
    if shape == "stiletto":
        tip_h = L_ext + L * 0.30
    elif shape == "coffin":
        tip_h = L_ext
    elif shape == "ballerina":
        # Straight sides from cuticle to midpoint, taper from midpoint to tip.
        tip_h = L_total * 0.5
    elif shape == "square":
        # The taper+corner-round region must stay entirely within the
        # extension (L_ext) — the real measured nail (length L) has to be
        # fully covered at full width first, and only narrow after that.
        # Capped at L_ext so y_side_top = L_total-tip_h never dips below L.
        tip_h = min(L_ext, max(CORNER_R, W * TIP_HEIGHT_FACTOR.get(shape, 0.45)))
        CORNER_R = min(CORNER_R, tip_h)
    else:
        tip_h = W * TIP_HEIGHT_FACTOR.get(shape, 0.50)

    # ── Build structured grid ─────────────────────────────────
    nx = 50   # columns across width
    ny = 80   # rows along length

    ys = np.linspace(-CUT_DEPTH, L_total, ny)

    # grid_x[i, j], grid_y[i, j] = XY position of grid point (i, j)
    grid_x = np.zeros((ny, nx))
    grid_y = np.zeros((ny, nx))
    for i, y in enumerate(ys):
        xl, xr = x_extent(y, W, L_total, tip_h, CUT_DEPTH, shape,
                          tip_r=CORNER_R, taper_mm=TAPER_MM)
        grid_x[i] = xl + np.linspace(0, 1, nx) * (xr - xl)
        grid_y[i] = y

    # ── Z values ──────────────────────────────────────────────
    # arc_z gives 0 at the nail centre, rising to C at the edges (bowl shape).
    # Inverting it gives a dome: C at centre, 0 at edges — this matches the
    # convex top surface of the real nail so the inner surface seats flush.
    # Adding THICK gives the outer surface, also dome-shaped (natural look).
    #
    # Per-row curvature blend (fit fix): the fingertip flesh under the nail
    # bed is ROUNDER near the cuticle than the nail is at its free edge, so
    # a uniform C-curve tube pinches at the base and is hard to put on.
    # Blend the dome height from (C + CUT_CURVE) at the cuticle arch down to
    # the measured C at the free edge (y = L); the tip extension keeps the
    # measured C.  Each row's arc radius is recomputed holding the measured
    # arc's CHORD constant  (chord^2 = 8*C*(R - C/2)),  so at t=0 the row
    # reproduces the measured (C, arc_r) pair exactly.
    if C < 0.05:
        z_bot = np.zeros_like(grid_x)              # flat nail — unchanged
    elif CUT_CURVE > 1e-6:
        chord_sq = 8.0 * C * (arc_r - C / 2.0)     # measured-arc chord^2
        yr    = np.clip((ys + CUT_DEPTH) / max(L + CUT_DEPTH, 1e-6), 0.0, 1.0)
        t     = 1.0 - (yr * yr * (3.0 - 2.0 * yr))   # smoothstep: 1@cuticle→0@free edge
        C_row = C + CUT_CURVE * t                    # (ny,)
        R_row = chord_sq / (8.0 * C_row) + C_row / 2.0
        dx    = grid_x - x_cen                       # (ny, nx)
        sag   = R_row[:, None] - np.sqrt(
            np.maximum(R_row[:, None] ** 2 - dx ** 2, 0.0))
        z_bot = C_row[:, None] - sag                 # rounder dome at base
    else:
        arc_off = arc_z(grid_x, x_cen, C, arc_r)   # (ny, nx)  0→C  bowl
        z_bot   = C - arc_off                        # (ny, nx)  C→0  dome (inner)
    z_top = z_bot + THICK                          # (ny, nx)  uniform shell

    # ── Top perimeter edge rounding ───────────────────────────
    # Applies a quarter-circle fillet where the top surface meets the side
    # walls, eliminating the sharp 90° corner.  Only in the body/tip region
    # (y ≥ 0) — the cuticle arch is left untouched per design intent.
    if EDGE_R > 0:
        r = min(EDGE_R, THICK * 0.95)

        # Lateral fillet: distance from left / right edge of each row's footprint
        x_left  = grid_x[:, 0:1]                        # (ny, 1)
        x_right = grid_x[:, -1:]                         # (ny, 1)
        d_lat   = np.minimum(grid_x - x_left,
                             x_right - grid_x)           # (ny, nx)
        dz_lat  = np.where(d_lat < r,
                           r - np.sqrt(np.maximum(r**2 - (r - d_lat)**2, 0.0)),
                           0.0)
        # Non-flat shapes: lateral fillet along the body, skip cuticle arch.
        # Flat-tip shapes (square, ballerina): apply a *spherical* corner
        # fillet ONLY at the two tip corners (where side wall + tip cap +
        # top surface all meet) — NOT the full edge line.
        if shape not in FLAT_TIP_SHAPES:
            dz_lat = np.where(grid_y >= 0.0, dz_lat, 0.0)
            z_top  = np.maximum(z_top - dz_lat, z_bot)
        elif shape == "square":
            # Square: full-width tip-edge fillet across the entire front face.
            # This gives the "bent A4 paper, edges smoothed" look:
            # the C-curve dome transitions smoothly into the vertical front face
            # along the whole width, not just at the two corners.
            d_tip_arr = np.maximum(L_total - grid_y, 0.0)
            in_body   = grid_y >= 0.0
            # Quarter-circle fillet where top surface meets front face (full width)
            dz_tip = np.where(
                (d_tip_arr < r) & in_body,
                r - np.sqrt(np.maximum(r**2 - (r - d_tip_arr)**2, 0.0)),
                0.0
            )
            # Spherical upgrade at the two top corners (side + tip both active)
            in_corner = (d_lat < r) & (d_tip_arr < r) & in_body
            under_c   = r**2 - (r - d_lat)**2 - (r - d_tip_arr)**2
            dz_corner = np.where(
                in_corner,
                np.where(under_c >= 0.0,
                         r - np.sqrt(np.maximum(under_c, 0.0)),
                         r),
                0.0
            )
            # At each point take whichever fillet is deeper
            z_top = np.maximum(z_top - np.maximum(dz_tip, dz_corner), z_bot)
        else:
            # Other flat-tip shapes (ballerina): spherical corner only
            d_tip_arr  = np.maximum(L_total - grid_y, 0.0)
            in_corner  = (d_lat < r) & (d_tip_arr < r)
            under_c    = r**2 - (r - d_lat)**2 - (r - d_tip_arr)**2
            dz_corner  = np.where(
                in_corner,
                np.where(under_c >= 0.0,
                         r - np.sqrt(np.maximum(under_c, 0.0)),
                         r),
                0.0
            )
            z_top = np.maximum(z_top - dz_corner, z_bot)

        # Tip fillet — pointed shapes only (almond, stiletto):
        # domes the apex so it tapers smoothly rather than ending as a ridge.
        if shape not in ("round", "oval") and shape not in FLAT_TIP_SHAPES:
            d_tip  = np.maximum(L_total - grid_y, 0.0)  # (ny, nx)
            dz_tip = np.where(d_tip < r,
                              r - np.sqrt(np.maximum(r**2 - (r - d_tip)**2, 0.0)),
                              0.0)
            z_top  = np.maximum(z_top - dz_tip, z_bot)

    # ── 3-D point arrays ──────────────────────────────────────
    top3d = np.stack([grid_x, grid_y, z_top], axis=2)  # (ny, nx, 3)
    bot3d = np.stack([grid_x, grid_y, z_bot], axis=2)

    all_tris = []
    NX = nx - 1   # last column index

    # ── Top face  (normal +Z) ─────────────────────────────────
    # Winding: CCW from +Z view.
    # Interior quads (i, j) → (i, j+1) → (i+1, j) → (i+1, j+1):
    #   Tri1 [t[i,j], t[i,j+1], t[i+1,j]]
    #   Tri2 [t[i,j+1], t[i+1,j+1], t[i+1,j]]
    # Left boundary edge traversal: t[i+1,0]→t[i,0]  (opposite to left wall ✓)
    # Right boundary edge traversal: t[i,N]→t[i+1,N] (opposite to right wall ✓)
    for i in range(ny - 1):
        for j in range(nx - 1):
            all_tris.append([top3d[i, j],   top3d[i, j+1], top3d[i+1, j]])
            all_tris.append([top3d[i, j+1], top3d[i+1, j+1], top3d[i+1, j]])

    # ── Bottom face  (normal -Z) ──────────────────────────────
    # Winding: CW from +Z (= CCW from -Z).
    #   Tri1 [b[i,j+1], b[i,j],   b[i+1,j]]
    #   Tri2 [b[i+1,j+1], b[i,j+1], b[i+1,j]]
    # Left boundary edge: b[i,0]→b[i+1,0]  (opposite to left wall ✓)
    # Right boundary edge: b[i+1,N]→b[i,N] (opposite to right wall ✓)
    for i in range(ny - 1):
        for j in range(nx - 1):
            all_tris.append([bot3d[i, j+1], bot3d[i, j],   bot3d[i+1, j]])
            all_tris.append([bot3d[i+1, j+1], bot3d[i, j+1], bot3d[i+1, j]])

    # ── Left side wall  (j=0, normal -X) ─────────────────────
    # Top boundary edge: t[i,0]→t[i+1,0] (opposite to top face t[i+1,0]→t[i,0] ✓)
    # Bot boundary edge: b[i+1,0]→b[i,0] (opposite to bot face b[i,0]→b[i+1,0] ✓)
    for i in range(ny - 1):
        T0 = top3d[i,   0];  T1 = top3d[i+1, 0]
        B0 = bot3d[i,   0];  B1 = bot3d[i+1, 0]
        all_tris.append([T0, T1, B0])
        all_tris.append([T1, B1, B0])

    # ── Right side wall  (j=NX, normal +X) ───────────────────
    # Top boundary edge: t[i+1,N]→t[i,N] (opposite to top face t[i,N]→t[i+1,N] ✓)
    # Bot boundary edge: b[i,N]→b[i+1,N] (opposite to bot face b[i+1,N]→b[i,N] ✓)
    for i in range(ny - 1):
        T0 = top3d[i,   NX];  T1 = top3d[i+1, NX]
        B0 = bot3d[i,   NX];  B1 = bot3d[i+1, NX]
        all_tris.append([T0, B1, T1])
        all_tris.append([T0, B0, B1])

    # ── Tip cap  (flat-tip shapes + almond with tip truncation) ──
    # Flat-tip shapes (square, ballerina) and almond when ALMOND_TIP_R > 0
    # end with a finite-width cross-section that needs an explicit cap face.
    # Pure pointed shapes (stiletto, round, oval) converge to zero width and
    # are closed by the degenerate-triangle filter below.
    # Winding for +Y: viewed from +Y the vertices are CCW →
    #   Tri1 [t[M,j], t[M,j+1], b[M,j]]
    #   Tri2 [t[M,j+1], b[M,j+1], b[M,j]]
    if shape in FLAT_TIP_SHAPES:
        M = ny - 1
        for j in range(nx - 1):
            t0 = top3d[M, j];   t1 = top3d[M, j+1]
            b0 = bot3d[M, j];   b1 = bot3d[M, j+1]
            all_tris.append([t0, t1, b0])
            all_tris.append([t1, b1, b0])

    # Drop degenerate triangles (two or more identical vertices).
    # These appear at the tip and cuticle arch bottom where the grid
    # rows collapse to a single point.  The adjacent valid fan triangles
    # already close those ends; keeping the degenerate ones creates
    # duplicate edges that make the mesh non-manifold.
    EPS = 1e-4
    clean = []
    for tri in all_tris:
        v0 = np.asarray(tri[0]); v1 = np.asarray(tri[1]); v2 = np.asarray(tri[2])
        if (np.linalg.norm(v0 - v1) > EPS and
                np.linalg.norm(v1 - v2) > EPS and
                np.linalg.norm(v0 - v2) > EPS):
            clean.append(tri)
    all_tris = clean

    n_tris = write_binary_stl(output_path, all_tris)
    kb     = round(os.path.getsize(output_path) / 1024, 1)
    print(f"  [STL] {n_tris} tris  {kb} KB  -> {output_path}")

    return {
        "triangles": n_tris,
        "file_kb":   kb,
        "dimensions": {
            "shape":           shape,
            "width_mm":        round(W, 2),
            "length_mm":       round(L_total, 2),
            "c_curve_mm":      round(C, 2),
            "thickness_mm":    round(THICK, 2),
            "cuticle_depth_mm": round(CUT_DEPTH, 2),
        },
    }


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Exact-fit nail STL v15 from nail_measurements.json")
    p.add_argument("--input",          required=True,
                   help="nail_measurements.json from nail_measurer.py")
    p.add_argument("--finger",         default=None,
                   help="Only generate this finger (e.g. index)")
    p.add_argument("--output",         default="nail_stl",
                   help="Output directory (default: nail_stl)")
    p.add_argument("--tip-extension",  type=float, default=None,
                   help="Extra mm beyond nail tip (default: 7mm for stiletto/"
                        "coffin, 3mm for all other shapes)")
    p.add_argument("--cuticle-depth",  type=float, default=1.5,
                   help="Depth of cuticle arch below cuticle line in mm "
                        "(default 1.5 — increase for deeper arch)")
    p.add_argument("--cuticle-curve",  type=float, default=0.0,
                   help="Extra inner-dome height in mm at the cuticle end, "
                        "blending to the measured C-curve at the free edge "
                        "(default 0 — OFF, uses the plain measured C-curve; "
                        "set >0 to re-enable the cuticle-side rounding). "
                        "Ignored in --exact mode")
    p.add_argument("--thickness",      type=float, default=0.6,
                   help="Uniform shell thickness in mm (default 0.6)")
    p.add_argument("--exact",          action="store_true",
                   help="Exact replica of the measured nail: true width/length, "
                        "no comfort shrink, no tip extension (validation prints)")
    p.add_argument("--shape",          default="round", choices=SHAPES,
                   help="Tip shape: round | oval | almond | square | "
                        "stiletto | ballerina  (default: round)")
    p.add_argument("--edge-round",     type=float, default=None,
                   help="Fillet radius (mm) for the top perimeter edge — "
                        "rounds the sharp junction between top surface and "
                        "side walls (default: 1.0 for almond/ballerina/"
                        "square/stiletto, 0 for round/oval)")
    p.add_argument("--corner-round",   type=float, default=None,
                   help="Plan-view (XY) fillet radius (mm) for the square "
                        "shape's two tip corners — rounds off the sharp 90° "
                        "corners so they don't scratch skin (default: 1.5 "
                        "for square, ignored by other shapes; 0 = sharp)")
    p.add_argument("--taper",          type=float, default=None,
                   help="Total mm the square shape's plate narrows by "
                        "(split evenly across both sides) leading into the "
                        "tip corners — real square nails aren't perfectly "
                        "parallel-sided (default: 1.0 for square, ignored "
                        "by other shapes; 0 = parallel sides)")
    args = p.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    os.makedirs(args.output, exist_ok=True)
    nails = data.get("nails", [])
    if args.finger:
        nails = [n for n in nails if n["finger"] == args.finger]
    if not nails:
        sys.exit("ERROR: No matching nails in JSON")

    ext_default = 7.0 if args.shape in LONG_SHAPES else 3.0
    display_ext = args.tip_extension if args.tip_extension is not None else ext_default
    if args.exact:
        display_ext = 0.0

    _edge_round_default = 1.0 if args.shape in {"almond", "ballerina", "square", "stiletto"} else 0.0
    edge_round = args.edge_round if args.edge_round is not None else _edge_round_default

    _corner_round_default = 1.5 if args.shape == "square" else 0.0
    corner_round = args.corner_round if args.corner_round is not None else _corner_round_default

    _taper_default = 1.0 if args.shape == "square" else 0.0
    taper = args.taper if args.taper is not None else _taper_default

    print(f"\n{'='*55}")
    print(f"  Exact-Fit Nail STL  v15  |  shape: {args.shape}")
    _cc = 0.0 if args.exact else args.cuticle_curve
    print(f"  Tip +{display_ext}mm  CuticleArch {args.cuticle_depth}mm  "
          f"Thick {args.thickness}mm  EdgeRound {edge_round}mm  "
          f"CornerRound {corner_round}mm  Taper {taper}mm  CuticleCurve +{_cc}mm")
    print(f"{'='*55}")

    for nail in nails:
        finger = nail["finger"]
        print(f"\n  [{finger.upper()}]")
        print(f"  W={nail['width_mm']}mm  L={nail['length_mm']}mm  "
              f"C={nail['c_curve_mm']}mm  R={nail['arc_radius_mm']}mm")

        params = {
            "c_curve_mm":          nail["c_curve_mm"],
            "arc_radius_mm":       nail["arc_radius_mm"],
            "width_mm":            nail["width_mm"],
            "length_mm":           nail["length_mm"],
            "corrected_length_mm": nail.get("corrected_length_mm"),
            "tip_extension_mm":    args.tip_extension,
            "cuticle_depth_mm":    args.cuticle_depth,
            "cuticle_curve_mm":    args.cuticle_curve,
            "thickness_mm":        args.thickness,
            "shape":               args.shape,
            "edge_round_mm":       edge_round,
            "corner_round_mm":     corner_round,
            "taper_mm":            taper,
            "exact":               args.exact,
        }

        out   = os.path.join(args.output, f"nail_{finger}_{args.shape}.stl")
        stats = generate_stl(params, out)
        d     = stats["dimensions"]
        print(f"  shape={d['shape']}  W={d['width_mm']}mm  L={d['length_mm']}mm  "
              f"C={d['c_curve_mm']}mm  thick={d['thickness_mm']}mm  "
              f"{stats['triangles']} tris  {stats['file_kb']}KB")

    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    main()