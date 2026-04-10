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
# Nail shape: analytic cross-section width at each Y
# ─────────────────────────────────────────────────────────────

def x_extent(y_val, W, L_total, tip_h, cuticle_depth):
    """
    Return (x_left, x_right) for the nail footprint at height y_val.

    Three regions:
      y in [-cuticle_depth, 0]   : elliptical cuticle arch
                                   full width at y=0, closes at y=-cuticle_depth
      y in [0, L_total - tip_h]  : straight sides, full width W
      y in [L_total-tip_h, L_total]: semi-elliptical tip arc
                                   full width at y=L_total-tip_h, closes at tip
    """
    y_side_top = L_total - tip_h

    if y_val >= y_side_top:
        # Tip semi-ellipse
        t = min((y_val - y_side_top) / tip_h, 1.0)
        cos_v = float(np.sqrt(max(1.0 - t * t, 0.0)))
        return W / 2 - W / 2 * cos_v, W / 2 + W / 2 * cos_v

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
    L         = float(params["length_mm"])
    L_ext     = float(params.get("tip_extension_mm", 1.0))
    CUT_DEPTH = float(params.get("cuticle_depth_mm", 1.5))
    x_cen     = W / 2.0

    L_total = L + L_ext
    tip_h   = W * 0.45            # tip arc height (same formula as before)

    # ── Build structured grid ─────────────────────────────────
    nx = 50   # columns across width
    ny = 80   # rows along length

    ys = np.linspace(-CUT_DEPTH, L_total, ny)

    # grid_x[i, j], grid_y[i, j] = XY position of grid point (i, j)
    grid_x = np.zeros((ny, nx))
    grid_y = np.zeros((ny, nx))
    for i, y in enumerate(ys):
        xl, xr = x_extent(y, W, L_total, tip_h, CUT_DEPTH)
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
    p.add_argument("--tip-extension",  type=float, default=1.0,
                   help="Extra mm added beyond nail tip (default 1.0)")
    p.add_argument("--cuticle-depth",  type=float, default=1.5,
                   help="Depth of cuticle arch below cuticle line in mm "
                        "(default 1.5 — increase for deeper arch)")
    p.add_argument("--thickness",      type=float, default=2.0,
                   help="Uniform shell thickness in mm (default 2.0)")
    args = p.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    os.makedirs(args.output, exist_ok=True)
    nails = data.get("nails", [])
    if args.finger:
        nails = [n for n in nails if n["finger"] == args.finger]
    if not nails:
        sys.exit("ERROR: No matching nails in JSON")

    print(f"\n{'='*55}")
    print(f"  Exact-Fit Nail STL  v15")
    print(f"  Tip +{args.tip_extension}mm  CuticleArch {args.cuticle_depth}mm  "
          f"Thick {args.thickness}mm")
    print(f"{'='*55}")

    for nail in nails:
        finger = nail["finger"]
        print(f"\n  [{finger.upper()}]")
        print(f"  W={nail['width_mm']}mm  L={nail['length_mm']}mm  "
              f"C={nail['c_curve_mm']}mm  R={nail['arc_radius_mm']}mm")

        params = {
            "c_curve_mm":       nail["c_curve_mm"],
            "arc_radius_mm":    nail["arc_radius_mm"],
            "width_mm":         nail["width_mm"],
            "length_mm":        nail["length_mm"],
            "tip_extension_mm": args.tip_extension,
            "cuticle_depth_mm": args.cuticle_depth,
            "thickness_mm":     args.thickness,
        }

        out   = os.path.join(args.output, f"nail_{finger}_exact.stl")
        stats = generate_stl(params, out)
        d     = stats["dimensions"]
        print(f"  W={d['width_mm']}mm  L={d['length_mm']}mm  "
              f"C={d['c_curve_mm']}mm  thick={d['thickness_mm']}mm  "
              f"cuticle={d['cuticle_depth_mm']}mm  "
              f"{stats['triangles']} tris  {stats['file_kb']}KB")

    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    main()
