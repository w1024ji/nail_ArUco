"""
nail_shape_stl.py
-----------------
Generates an exact-fit nail STL from nail_measurements.json.
Uses the actual nail outline polygon (from nail_measurer_v6.py)
combined with the measured c-curve to create a 3D nail shape.

Usage:
    python nail_shape_stl.py --input results/nail_measurements.json
                             --finger index --output nail_stl/

How it works:
    1. Load nail_polygon_px from JSON (actual outline traced from photo)
    2. Convert polygon from pixels to mm using mpp
    3. For each polygon point, compute z height using c-curve arc formula
       - Centre of nail = lowest z (0)
       - Edges = highest z (c_curve_mm)
    4. Create top surface (outer shell following c-curve)
    5. Create flat bottom surface (inner shell, flat plate)
    6. Connect edges to make watertight STL
"""

import argparse
import json
import os
import struct
import sys

import numpy as np
from scipy.interpolate import splprep, splev


# ─────────────────────────────────────────────────────────────
# STL writer
# ─────────────────────────────────────────────────────────────

def write_binary_stl(filepath: str, triangles: np.ndarray):
    n = len(triangles)
    with open(filepath, "wb") as f:
        f.write(b"nail_shape_stl_v1" + b"\x00" * 63)
        f.write(struct.pack("<I", n))
        for tri in triangles:
            v0, v1, v2 = tri.astype(np.float32)
            nrm = np.cross(v1 - v0, v2 - v0).astype(np.float32)
            ln  = np.linalg.norm(nrm)
            if ln > 0:
                nrm /= ln
            f.write(struct.pack("<3f", *nrm))
            f.write(struct.pack("<3f", *v0))
            f.write(struct.pack("<3f", *v1))
            f.write(struct.pack("<3f", *v2))
            f.write(b"\x00\x00")


# ─────────────────────────────────────────────────────────────
# C-curve height calculator
# ─────────────────────────────────────────────────────────────

def ccurve_z(px: float, py: float,
             cx: float, cy: float,
             nail_half_w: float,
             c_curve_mm: float,
             arc_radius_mm: float) -> float:
    """
    Given a point (px, py) on the nail outline in mm,
    compute its z height based on c-curve arc.

    The c-curve goes across the nail width (x-axis).
    Centre (cx) = z=0 (lowest point of concave inner surface)
    Edges = z=c_curve_mm (highest)

    Arc formula: z = R - sqrt(R² - x_offset²)
    where x_offset = distance from nail centre in x
    """
    x_offset = px - cx
    R = arc_radius_mm
    under = R**2 - x_offset**2
    if under < 0:
        return float(c_curve_mm)
    z = R - np.sqrt(under)
    # Clamp to valid range
    return float(min(max(z, 0.0), c_curve_mm))


# ─────────────────────────────────────────────────────────────
# Main STL builder
# ─────────────────────────────────────────────────────────────

def build_nail_stl(nail_data: dict, mpp: float,
                   output_path: str,
                   wall_thickness: float = 1.2) -> dict:
    """
    Build a watertight nail STL from the polygon contour.

    Parameters:
        nail_data      : single nail dict from nail_measurements.json
        mpp            : mm per pixel (from ArUco measurement)
        output_path    : where to save the .stl file
        wall_thickness : thickness of nail shell in mm (default 1.2mm)
    """

    # ── Load polygon ──────────────────────────────────────────
    poly_px = np.array(nail_data["nail_polygon_px"], dtype=float)
    c_mm    = float(nail_data["c_curve_mm"])
    arc_r   = float(nail_data["arc_radius_mm"])
    w_mm    = float(nail_data["width_mm"])
    l_mm    = float(nail_data["length_mm"])

    print(f"  Polygon: {len(poly_px)} points")
    print(f"  Width: {w_mm}mm  Length: {l_mm}mm")
    print(f"  C-curve: {c_mm}mm  Arc R: {arc_r}mm")

    # ── Convert px → mm, origin at centroid ──────────────────
    # First, find bounding box
    x_px = poly_px[:, 0]
    y_px = poly_px[:, 1]
    x_min, x_max = x_px.min(), x_px.max()
    y_min, y_max = y_px.min(), y_px.max()

    # Convert to mm, origin at top-left of bounding box
    poly_mm = np.zeros_like(poly_px)
    poly_mm[:, 0] = (x_px - x_min) * mpp   # X = width direction
    poly_mm[:, 1] = (y_px - y_min) * mpp   # Y = length direction

    # Nail centre X for c-curve calculation
    cx_mm = poly_mm[:, 0].mean()
    cy_mm = poly_mm[:, 1].mean()

    print(f"  Polygon mm range: "
          f"x=[0, {poly_mm[:,0].max():.2f}]  "
          f"y=[0, {poly_mm[:,1].max():.2f}]")
    print(f"  Centre: ({cx_mm:.2f}, {cy_mm:.2f})")

    # ── Resample polygon to uniform spacing ───────────────────
    N = 200   # number of points around the outline
    pts = poly_mm
    try:
        tck, _ = splprep(
            [np.append(pts[:,0], pts[0,0]),
             np.append(pts[:,1], pts[0,1])],
            s=len(pts)*0.5, per=True, k=3)
        t_new = np.linspace(0, 1, N, endpoint=False)
        xs, ys = splev(t_new, tck)
        outline = np.column_stack([xs, ys])
    except Exception as e:
        print(f"  Spline failed ({e}), using raw polygon")
        # Resample manually
        idx = np.linspace(0, len(pts)-1, N, dtype=int)
        outline = pts[idx]

    print(f"  Resampled to {len(outline)} points")

    # ── Compute z heights for top surface ─────────────────────
    # Top surface = outer convex surface (c-curve)
    # z=0 at nail centre, z=c_mm at edges
    z_top = np.array([
        ccurve_z(p[0], p[1], cx_mm, cy_mm,
                 w_mm/2, c_mm, arc_r)
        for p in outline
    ])

    # Shift so minimum z = wall_thickness (nail has thickness)
    # Bottom surface is flat at z=0
    # Top surface ranges from wall_thickness (centre) to wall_thickness+c_mm (edges)
    z_top = z_top + wall_thickness

    # ── Build 3D vertices ─────────────────────────────────────
    # Top surface points (follow c-curve)
    top_verts = np.column_stack([outline[:, 0],
                                  outline[:, 1],
                                  z_top])

    # Bottom surface points (flat at z=0, same XY as top)
    bot_verts = np.column_stack([outline[:, 0],
                                  outline[:, 1],
                                  np.zeros(N)])

    # ── Triangulate ───────────────────────────────────────────
    all_tris = []

    # 1. TOP surface — fan triangulation from centroid
    top_centre = np.array([cx_mm, cy_mm,
                            wall_thickness])  # centre at min height
    for i in range(N):
        j = (i + 1) % N
        all_tris.append([top_centre,
                          top_verts[i],
                          top_verts[j]])

    # 2. BOTTOM surface — fan triangulation (reversed winding)
    bot_centre = np.array([cx_mm, cy_mm, 0.0])
    for i in range(N):
        j = (i + 1) % N
        all_tris.append([bot_centre,
                          bot_verts[j],
                          bot_verts[i]])

    # 3. SIDE wall — connect top and bottom outlines
    for i in range(N):
        j = (i + 1) % N
        t0, t1 = top_verts[i], top_verts[j]
        b0, b1 = bot_verts[i], bot_verts[j]
        all_tris.append([t0, b0, t1])
        all_tris.append([t1, b0, b1])

    tris_arr = np.array(all_tris, dtype=np.float32)
    write_binary_stl(output_path, tris_arr)

    file_kb = round(os.path.getsize(output_path) / 1024, 1)
    print(f"  Triangles: {len(tris_arr)}")
    print(f"  File: {file_kb} KB → {output_path}")

    return {
        "triangles":  len(tris_arr),
        "file_kb":    file_kb,
        "width_mm":   round(poly_mm[:,0].max(), 2),
        "length_mm":  round(poly_mm[:,1].max(), 2),
        "height_mm":  round(float(z_top.max()), 2),
        "c_curve_mm": c_mm,
    }


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Generate exact-fit nail STL from nail_measurements.json")
    p.add_argument("--input",     required=True,
                   help="Path to nail_measurements.json")
    p.add_argument("--finger",    default=None,
                   help="Which finger to generate (default: all)")
    p.add_argument("--output",    default="nail_stl",
                   help="Output folder (default: nail_stl/)")
    p.add_argument("--thickness", type=float, default=1.2,
                   help="Nail shell thickness in mm (default: 1.2)")
    p.add_argument("--aruco-size", type=float, default=20.0,
                   help="ArUco marker size in mm (default: 20)")
    args = p.parse_args()

    # Load JSON
    with open(args.input) as f:
        data = json.load(f)

    os.makedirs(args.output, exist_ok=True)

    nails = data.get("nails", [])
    if not nails:
        sys.exit("ERROR: No nails found in JSON")

    # Filter by finger if specified
    if args.finger:
        nails = [n for n in nails if n["finger"] == args.finger]
        if not nails:
            sys.exit(f"ERROR: Finger '{args.finger}' not found in JSON")

    print(f"\n{'='*55}")
    print(f"  Nail Shape STL Generator")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}/")
    print(f"{'='*55}")

    # Get mpp from mesh_params if available, else compute from aruco
    # We need mpp to convert polygon from px to mm
    # mpp is stored implicitly: width_mm / polygon_width_px
    results = []
    for nail in nails:
        finger = nail["finger"]
        print(f"\n  Processing: {finger.upper()}")

        poly_px = np.array(nail.get("nail_polygon_px", []))
        if len(poly_px) == 0:
            print(f"  ⚠ No polygon data for {finger}, skipping")
            continue

        # Compute mpp from polygon bounding box vs measured width
        poly_w_px = poly_px[:,0].max() - poly_px[:,0].min()
        poly_h_px = poly_px[:,1].max() - poly_px[:,1].min()
        mpp_from_w = nail["width_mm"]  / poly_w_px
        mpp_from_h = nail["length_mm"] / poly_h_px
        mpp = (mpp_from_w + mpp_from_h) / 2.0
        print(f"  mpp (from polygon): {mpp:.5f} mm/px "
              f"(w={mpp_from_w:.5f}, h={mpp_from_h:.5f})")

        out_file = os.path.join(args.output, f"nail_{finger}_exact.stl")
        stats = build_nail_stl(nail, mpp, out_file, args.thickness)
        results.append({"finger": finger, "file": out_file, **stats})

    # Summary
    print(f"\n{'='*55}")
    print(f"  ✅ Generated {len(results)} STL file(s):")
    for r in results:
        print(f"     {r['finger']:<8} → {os.path.basename(r['file'])}")
        print(f"              W={r['width_mm']}mm  "
              f"L={r['length_mm']}mm  "
              f"H={r['height_mm']}mm  "
              f"({r['triangles']} triangles, {r['file_kb']}KB)")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()