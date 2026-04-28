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

SHAPES = ("oval", "almond", "square", "squoval", "stiletto", "coffin")

# Height of the tip taper region as a fraction of nail width W.
# Shapes whose tips taper to a point (oval, almond, stiletto) are closed
# by the degenerate-triangle filter.  Flat-ended shapes (square, squoval,
# coffin) need an explicit tip-cap face and a non-zero tip_h so the cap
# has a well-defined cross-section.
TIP_HEIGHT_FACTOR = {
    "oval":     0.50,   # semi-ellipse, smooth closed arc
    "almond":   0.70,   # cosine taper to a soft point
    "square":   0.00,   # no taper — flat perpendicular tip, full width
    "squoval":  0.20,   # square corners + quarter-circle fillet r = 0.2 W
    "stiletto": 1.50,   # linear taper to a sharp point (longer, more dramatic)
    "coffin":   1.00,   # linear taper to flat tip ~40 % of nail width
}

# Shapes whose tip is a flat face (need an explicit cap triangle strip).
FLAT_TIP_SHAPES = {"square", "squoval", "coffin"}


# ─────────────────────────────────────────────────────────────
# Nail shape: analytic cross-section width at each Y
# ─────────────────────────────────────────────────────────────

def x_extent(y_val, W, L_total, tip_h, cuticle_depth, shape="oval"):
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
            # No taper; flat perpendicular tip at full width.
            return 0.0, float(W)

        # Normalised position within tip: 0 at base, 1 at tip end.
        t = min((y_val - y_side_top) / tip_h, 1.0) if tip_h > 0 else 1.0

        if shape == "oval":
            # Ellipse: x²/a² + y²/b² = 1 → half_w = (W/2)·√(1−t²)
            half_w = W / 2 * float(np.sqrt(max(1.0 - t * t, 0.0)))

        elif shape == "almond":
            # Cosine taper — narrower than oval, closes to a soft point.
            half_w = W / 2 * float(np.cos(t * np.pi / 2))

        elif shape == "squoval":
            # Quarter-circle fillet of radius r = 0.2·W at each corner.
            # Arc centre: (r, L_total−r) and (W−r, L_total−r).
            # At dy above y_side_top:  xl = r − √(r²−dy²)
            #                          xr = (W−r) + √(r²−dy²)
            r   = W * 0.2
            dy  = y_val - y_side_top          # 0 → r
            arc = float(np.sqrt(max(r * r - dy * dy, 0.0)))
            return r - arc, (W - r) + arc

        elif shape == "stiletto":
            # Linear taper to a sharp point over a long region (1.5·W).
            # w(t) = W·(1−t)  →  0 at t=1
            half_w = W / 2 * (1.0 - t)

        elif shape == "coffin":
            # Linear taper from full width to ~40 % of W at the flat top.
            # w(t) = W·(1 − 0.6·t)  →  0.4·W at t=1
            half_w = W / 2 * (1.0 - 0.6 * t)

        else:
            # Fallback: oval
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
    THICK     = float(params.get("thickness_mm", 2.0))
    W         = float(params["width_mm"])
    # Prefer corrected_length_mm when the measurer flagged the raw length as
    # suspect (nail tip not fully visible in photo).
    L         = float(params.get("corrected_length_mm") or params["length_mm"])
    # Shape-specific default extensions: stiletto/coffin get 7 mm,
    # all other shapes default to 3 mm.  An explicit --tip-extension
    # value always wins (passed as a non-None entry in params).
    _shape_tmp = params.get("shape", "oval")
    _ext_default = 7.0 if _shape_tmp in ("stiletto", "coffin") else 3.0
    L_ext     = float(params.get("tip_extension_mm") or _ext_default)
    CUT_DEPTH = float(params.get("cuticle_depth_mm", 1.5))
    x_cen     = W / 2.0

    shape   = params.get("shape", "oval")
    L_total = L + L_ext
    # For stiletto and coffin the taper covers only the extension beyond the
    # natural nail — full width is maintained up to the free edge, then the
    # sides taper inward over exactly L_ext mm.
    if shape in ("stiletto", "coffin"):
        tip_h = L_ext
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
        xl, xr = x_extent(y, W, L_total, tip_h, CUT_DEPTH, shape)
        grid_x[i] = xl + np.linspace(0, 1, nx) * (xr - xl)
        grid_y[i] = y

    # ── Z values ──────────────────────────────────────────────
    # arc_z gives 0 at the nail centre, rising to C at the edges (bowl shape).
    # Inverting it gives a dome: C at centre, 0 at edges — this matches the
    # convex top surface of the real nail so the inner surface seats flush.
    # Adding THICK gives the outer surface, also dome-shaped (natural look).
    arc_off = arc_z(grid_x, x_cen, C, arc_r)     # (ny, nx)  0→C  bowl
    z_bot   = C - arc_off                          # (ny, nx)  C→0  dome (inner)
    z_top   = z_bot + THICK                        # (ny, nx)  uniform shell

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

    # ── Tip cap  (flat-tip shapes: square / squoval / coffin) ────
    # Pointed shapes (oval, almond, stiletto) converge to zero width and are
    # closed implicitly by the degenerate-triangle filter below.
    # Flat-tip shapes have a finite cross-section at i=ny-1 that must be
    # closed with an explicit cap face (normal +Y).
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
    p.add_argument("--thickness",      type=float, default=2.0,
                   help="Uniform shell thickness in mm (default 2.0)")
    p.add_argument("--shape",          default="oval", choices=SHAPES,
                   help="Tip shape: oval | almond | square | squoval | "
                        "stiletto | coffin  (default: oval)")
    args = p.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    os.makedirs(args.output, exist_ok=True)
    nails = data.get("nails", [])
    if args.finger:
        nails = [n for n in nails if n["finger"] == args.finger]
    if not nails:
        sys.exit("ERROR: No matching nails in JSON")

    ext_default = 7.0 if args.shape in ("stiletto", "coffin") else 3.0
    display_ext = args.tip_extension if args.tip_extension is not None else ext_default

    print(f"\n{'='*55}")
    print(f"  Exact-Fit Nail STL  v15  |  shape: {args.shape}")
    print(f"  Tip +{display_ext}mm  CuticleArch {args.cuticle_depth}mm  "
          f"Thick {args.thickness}mm")
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
            "thickness_mm":        args.thickness,
            "shape":               args.shape,
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
