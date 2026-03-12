"""
nail_tip_generator.py  ·  v1
─────────────────────────────────────────────────────────────────────
Generates printable 3D nail-tip STL files from nail_measurements.json.

One STL per finger → ready to slice and print on any FDM or resin printer.

Usage
─────
  # Basic (uses all defaults):
  python nail_tip_generator.py --measurements nail_measurements.json

  # Custom tip length and C-curve:
  python nail_tip_generator.py --measurements nail_measurements.json \\
      --tip-length 6 --c-curve-override 3.5 --output stl_files/

  # Single finger only:
  python nail_tip_generator.py --measurements nail_measurements.json \\
      --finger middle --tip-length 8

Nail tip anatomy
────────────────
  Cross-section (looking down the nail):

       ╭──────────────╮  ← top (convex, uniform wall thickness)
      /                \\
     │   nail tip body  │   ← wall_thick at edges, center_thick at center
      \\                /
       ╰──────────────╯  ← bottom (concave well, matches natural nail C-curve)

  Side view:

       cuticle end           free tip
           │←── nail bed ──→│←─ extension ─→│
           │                                 │
           ╰─────────────────────────────────╯

  The cuticle end is OPEN (slides over the natural nail).
  The free tip is CLOSED (capped).

Printing recommendations
────────────────────────
  Material : Resin (best detail) or TPU (flexible, comfortable)
             PLA works but is brittle at 0.5 mm walls
  Layer height: 0.05–0.1 mm for resin / 0.1–0.15 mm for FDM
  Orientation : Print upside-down (top surface on build plate) for best
                surface finish on the visible top face
  Supports    : None needed when printed top-face-down
  Infill      : 100% (thin walls, solid is better)

Requirements
────────────
  pip install numpy   (no other dependencies — STL is written from scratch)
"""

import argparse
import json
import math
import os
import struct
import sys

import numpy as np


# ─────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────

def arc_z_profile(us: np.ndarray, width_mm: float, c_curve_mm: float) -> np.ndarray:
    """
    Circular-arc C-curve: z(u) for u∈[0,1] across nail width.
    z = 0 at center (u=0.5), z = c_curve at edges (u=0,1).
    Uses the sagitta formula: R = w²/(8c) + c/2.
    """
    if c_curve_mm < 0.05:
        return np.zeros_like(us)
    R   = (width_mm ** 2) / (8.0 * c_curve_mm) + c_curve_mm / 2.0
    x   = (us - 0.5) * width_mm          # centre at 0
    z   = R - np.sqrt(np.maximum(R**2 - x**2, 0.0))
    return z


def write_binary_stl(filepath: str, triangles: np.ndarray):
    """
    Write a binary STL file.
    triangles: float32 array of shape (N, 3, 3)  — N×[v0,v1,v2]×[x,y,z]
    """
    n = len(triangles)
    with open(filepath, "wb") as f:
        f.write(b"nail_tip_generator_v1" + b"\x00" * 59)   # 80-byte header
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
            f.write(b"\x00\x00")          # attribute byte count


def grid_to_tris(X, Y, Z, flip: bool = False) -> list:
    """Tessellate a (NV, NU) surface grid into triangles."""
    tris = []
    NV, NU = X.shape
    for j in range(NV - 1):
        for i in range(NU - 1):
            p00 = np.array([X[j,   i  ], Y[j,   i  ], Z[j,   i  ]], np.float32)
            p10 = np.array([X[j,   i+1], Y[j,   i+1], Z[j,   i+1]], np.float32)
            p01 = np.array([X[j+1, i  ], Y[j+1, i  ], Z[j+1, i  ]], np.float32)
            p11 = np.array([X[j+1, i+1], Y[j+1, i+1], Z[j+1, i+1]], np.float32)
            if flip:
                tris.append([p00, p10, p11])
                tris.append([p00, p11, p01])
            else:
                tris.append([p00, p11, p10])
                tris.append([p00, p01, p11])
    return tris


def edge_strip_tris(edge_A: np.ndarray, edge_B: np.ndarray,
                    reverse: bool = False) -> list:
    """
    Connect two parallel polylines (N×3 each) with a quad strip.
    Winding is set so normals point away from the body.
    """
    tris = []
    N = len(edge_A)
    for i in range(N - 1):
        a0, a1 = edge_A[i].astype(np.float32), edge_A[i+1].astype(np.float32)
        b0, b1 = edge_B[i].astype(np.float32), edge_B[i+1].astype(np.float32)
        if reverse:
            tris.append([a0, a1, b0])
            tris.append([a1, b1, b0])
        else:
            tris.append([a0, b0, a1])
            tris.append([a1, b0, b1])
    return tris


# ─────────────────────────────────────────────────────────────
# Main mesh builder
# ─────────────────────────────────────────────────────────────

def generate_nail_tip_stl(params: dict, output_path: str,
                           nu: int = 60, nv: int = 72) -> dict:
    """
    Build a watertight nail-tip mesh and write it as a binary STL.

    Parameters
    ──────────
    params dict keys:
      width_mm         Nail plate width (measured)
      length_mm        Natural nail bed length (measured)
      c_curve_mm       Arc sagitta across the width (cross-section curvature)
      tip_length_mm    Free-tip extension length beyond fingertip
      wall_thick_mm    Wall thickness at the side edges  (default 0.5)
      center_thick_mm  Wall thickness at the centre line (default 0.8)

    nu, nv: mesh resolution (U = width, V = length direction)

    Returns dict with mesh statistics.
    """
    W    = float(params["width_mm"])
    L_nb = float(params["length_mm"])
    L_tp = float(params["tip_length_mm"])
    L    = L_nb + L_tp
    C    = float(params["c_curve_mm"])
    TW   = float(params.get("wall_thick_mm",   0.5))
    TC   = float(params.get("center_thick_mm", 0.8))

    # ── Parameter grid ──────────────────────────────────────
    us = np.linspace(0.0, 1.0, nu)
    vs = np.linspace(0.0, 1.0, nv)
    UU, VV = np.meshgrid(us, vs)    # shape (nv, nu)

    # ── Bottom surface (concave well = matches natural nail) ─
    # z_bot = 0 at centre, rises to C at edges
    z_bot_1d = arc_z_profile(us, W, C)
    Z_bot    = np.tile(z_bot_1d, (nv, 1))

    # ── Wall thickness map ───────────────────────────────────
    # Thicker at centre, thinner at side edges (TC → TW)
    # Also very slightly thinner toward the free tip (cosmetic taper)
    u_center_dist     = np.abs(us - 0.5) * 2.0          # 0 at centre, 1 at edge
    thick_u           = TC - (TC - TW) * u_center_dist   # linear U-taper
    tip_taper_v       = 1.0 - 0.08 * vs                  # 8% thinner at free tip
    Thick             = np.outer(tip_taper_v, thick_u)

    # ── Top surface (bottom + thickness) ────────────────────
    Z_top = Z_bot + Thick

    # ── XY coordinates ───────────────────────────────────────
    X = UU * W     # 0 → W  (width direction)
    Y = VV * L     # 0 → L  (cuticle → free tip)

    # ── Assemble triangles ───────────────────────────────────
    all_tris = []

    # Top surface (normals up / outward)
    all_tris.extend(grid_to_tris(X, Y, Z_top, flip=False))

    # Bottom surface (normals down / inward toward nail)
    all_tris.extend(grid_to_tris(X, Y, Z_bot, flip=True))

    # Left side wall  (u=0, all v)
    l_top = np.column_stack([X[:, 0],  Y[:, 0],  Z_top[:, 0]])
    l_bot = np.column_stack([X[:, 0],  Y[:, 0],  Z_bot[:, 0]])
    all_tris.extend(edge_strip_tris(l_bot, l_top, reverse=False))

    # Right side wall (u=nu-1, all v)
    r_top = np.column_stack([X[:, -1], Y[:, -1], Z_top[:, -1]])
    r_bot = np.column_stack([X[:, -1], Y[:, -1], Z_bot[:, -1]])
    all_tris.extend(edge_strip_tris(r_top, r_bot, reverse=True))

    # Cuticle end (v=0, all u) — open slot (NOT capped) so it slides over the nail
    # We cap it to keep the mesh watertight; the slicer will show the opening
    c_top = np.column_stack([X[0, :], Y[0, :], Z_top[0, :]])
    c_bot = np.column_stack([X[0, :], Y[0, :], Z_bot[0, :]])
    all_tris.extend(edge_strip_tris(c_top, c_bot, reverse=True))

    # Free tip cap (v=nv-1, all u) — solid closed end
    t_top = np.column_stack([X[-1, :], Y[-1, :], Z_top[-1, :]])
    t_bot = np.column_stack([X[-1, :], Y[-1, :], Z_bot[-1, :]])
    all_tris.extend(edge_strip_tris(t_bot, t_top, reverse=False))

    # ── Write STL ────────────────────────────────────────────
    tris_arr = np.array(all_tris, dtype=np.float32)
    write_binary_stl(output_path, tris_arr)

    file_kb = os.path.getsize(output_path) / 1024
    stats = {
        "triangles":   len(tris_arr),
        "file_kb":     round(file_kb, 1),
        "dimensions":  {
            "width_mm":  round(W,  2),
            "length_mm": round(L,  2),
            "height_mm": round(float(Z_top.max()), 2),
        },
        "wall_thick":  {"center_mm": round(TC, 2), "edge_mm": round(TW, 2)},
    }
    return stats


# ─────────────────────────────────────────────────────────────
# Load measurements + run all fingers
# ─────────────────────────────────────────────────────────────

FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]

# Typical real cross-section C-curve values per finger (mm)
# (The photo measurement's c_curve_mm includes the length-direction curve too,
#  so for the 3D cross-section we use a more conservative estimate)
DEFAULT_CROSS_SECTION_C = {
    "thumb":  3.5,
    "index":  3.2,
    "middle": 3.0,
    "ring":   3.0,
    "pinky":  2.8,
}


def build_all(json_path: str, tip_length: float, c_override: float | None,
              output_dir: str, wall_thick: float, center_thick: float):

    with open(json_path) as f:
        data = json.load(f)

    nails = {n["finger"]: n for n in data["nails"]}
    os.makedirs(output_dir, exist_ok=True)
    summary = []

    print(f"\n{'─'*62}")
    print(f"{'Finger':<8} {'W':>6} {'L_nat':>7} {'L_tip':>7} {'C':>6} "
          f"{'Tris':>7}  File")
    print(f"{'─'*62}")

    for finger in FINGER_ORDER:
        if finger not in nails:
            print(f"{finger:<8}  (not in measurements, skipping)")
            continue

        m = nails[finger]
        C_cross = c_override if c_override else DEFAULT_CROSS_SECTION_C.get(finger, 3.0)

        params = {
            "width_mm":        m["width_mm"],
            "length_mm":       m["length_mm"],
            "c_curve_mm":      C_cross,
            "tip_length_mm":   tip_length,
            "wall_thick_mm":   wall_thick,
            "center_thick_mm": center_thick,
        }

        out_file = os.path.join(output_dir, f"nail_tip_{finger}.stl")
        stats    = generate_nail_tip_stl(params, out_file)
        d        = stats["dimensions"]
        summary.append({"finger": finger, "params": params, "stats": stats, "file": out_file})

        print(f"{finger:<8}  {m['width_mm']:>5.2f}  {m['length_mm']:>6.2f}  "
              f"{tip_length:>6.1f}  {C_cross:>5.2f}  "
              f"{stats['triangles']:>7}  {os.path.basename(out_file)}")

    print(f"{'─'*62}")
    print(f"\n✅  {len(summary)} STL file(s) saved to '{output_dir}/'")

    # Save a build manifest JSON
    manifest_path = os.path.join(output_dir, "print_manifest.json")
    manifest = {
        "source_measurements": json_path,
        "tip_length_mm":       tip_length,
        "wall_thick_mm":       wall_thick,
        "center_thick_mm":     center_thick,
        "nails": [
            {
                "finger":          s["finger"],
                "stl_file":        os.path.basename(s["file"]),
                "width_mm":        s["params"]["width_mm"],
                "length_natural_mm": s["params"]["length_mm"],
                "length_total_mm": round(s["params"]["length_mm"] + s["params"]["tip_length_mm"], 2),
                "c_curve_cross_section_mm": s["params"]["c_curve_mm"],
                "triangles":       s["stats"]["triangles"],
                "file_kb":         s["stats"]["file_kb"],
                "bounding_box_mm": s["stats"]["dimensions"],
                "printing_tips": {
                    "orientation":   "Print top-face-down on build plate",
                    "supports":      "None needed in recommended orientation",
                    "layer_height":  "0.05 mm resin / 0.1 mm FDM",
                    "material":      "Resin (best) or TPU (flexible)",
                },
            }
            for s in summary
        ],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"     print_manifest.json  (open this for all dimensions + print settings)\n")
    return summary


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate 3D-printable nail tip STL files from nail measurements")
    p.add_argument("--measurements",   required=True,
                   help="Path to nail_measurements.json from nail_measurer.py")
    p.add_argument("--output",         default="nail_stl",
                   help="Output folder for STL files (default: nail_stl/)")
    p.add_argument("--tip-length",     type=float, default=5.0,
                   help="Free-tip extension in mm beyond the natural nail (default: 5.0)")
    p.add_argument("--c-curve-override", type=float, default=None,
                   help="Override C-curve for all fingers (mm). "
                        "If omitted, uses per-finger defaults (2.8–3.5 mm)")
    p.add_argument("--wall-thick",     type=float, default=0.5,
                   help="Wall thickness at side edges in mm (default: 0.5)")
    p.add_argument("--center-thick",   type=float, default=0.8,
                   help="Wall thickness at centre line in mm (default: 0.8)")
    p.add_argument("--finger",         default=None,
                   help="Generate only this finger (thumb/index/middle/ring/pinky)")
    args = p.parse_args()

    if args.finger:
        # Temporarily restrict FINGER_ORDER to the chosen finger
        import builtins
        _orig = FINGER_ORDER.copy()
        FINGER_ORDER.clear()
        FINGER_ORDER.append(args.finger)

    build_all(
        json_path    = args.measurements,
        tip_length   = args.tip_length,
        c_override   = args.c_curve_override,
        output_dir   = args.output,
        wall_thick   = args.wall_thick,
        center_thick = args.center_thick,
    )