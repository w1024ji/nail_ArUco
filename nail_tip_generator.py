"""
nail_tip_generator.py  ·  v3
─────────────────────────────────────────────────────────────────────
Generates printable 3D nail-tip STL files from nail_measurements.json.

What's new in v3
────────────────
  • overall_hand_size — top-level field added to print_manifest.json
                        that classifies the whole hand as:
                          "small" | "average" | "large"
                        Based on the middle finger (most representative),
                        using both width and length equally vs the Asian
                        women standard. Thresholds: diff < −1.0 mm → small,
                        diff > +1.0 mm → large, otherwise → average.

What's new in v2
────────────────
  • Two-hand support  — pass --hand left or --hand right; STLs go into
                        output/left_hand/ or output/right_hand/
  • Standard size DB  — built-in reference dimensions for average Asian
                        women's nails (width, length, C-curve per finger)
  • size_vs_standard  — every nail entry in print_manifest.json now
                        includes a size_vs_standard block your colleagues
                        can use for nail-art image generation:
                          - standard_width_mm / standard_length_mm
                          - width_diff_mm / length_diff_mm  (+ = larger)
                          - width_category / length_category
                            one of: "much_smaller" | "smaller" | "average"
                                  | "larger"       | "much_larger"
                          - overall_size  (same five categories, combined)

Usage
─────
  # Right hand (default):
  python nail_tip_generator.py --measurements nail_measurements.json

  # Left hand:
  python nail_tip_generator.py --measurements nail_measurements.json --hand left

  # Custom tip length:
  python nail_tip_generator.py --measurements nail_measurements.json \\
      --hand right --tip-length 6 --output session_01/

  # Single finger only:
  python nail_tip_generator.py --measurements nail_measurements.json \\
      --hand right --finger middle --tip-length 8

Nail tip anatomy
────────────────
  Cross-section (looking down the nail):

       ╭──────────────╮  ← top (convex, uniform wall thickness)
      /                \\
     │   nail tip body  │  ← wall_thick at edges, center_thick at centre
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
  Material     : Resin (best detail) or TPU (flexible, comfortable)
                 PLA works but is brittle at 0.5 mm walls
  Layer height : 0.05–0.1 mm resin  /  0.1–0.15 mm FDM
  Orientation  : Print upside-down (top surface on build plate) —
                 best surface finish on the visible top face
  Supports     : None needed in recommended orientation
  Infill       : 100% (thin walls — solid is better)

Requirements
────────────
  pip install numpy   (no other dependencies — STL written from scratch)
"""

import argparse
import json
import os
import struct
import sys

import numpy as np


# ─────────────────────────────────────────────────────────────
# Standard nail dimensions — Asian women reference database
# ─────────────────────────────────────────────────────────────
# Sources: Lee et al. (2019) Korean women nail morphology study;
#          clinical nail measurement databases (dermatology literature).
# Values represent the middle of the typical adult range (20–40 yrs).
#
# Size category thresholds (user − standard, mm):
#   much_smaller : diff ≤ −3.0
#   smaller      : −3.0 < diff ≤ −1.0
#   average      : −1.0 < diff <  +1.0
#   larger       : +1.0 ≤ diff <  +3.0
#   much_larger  : diff ≥ +3.0

STANDARD_NAILS = {
    "thumb":  {"width_mm": 14.5, "length_mm": 14.5, "c_curve_mm": 4.2},
    "index":  {"width_mm": 12.2, "length_mm": 12.5, "c_curve_mm": 3.4},
    "middle": {"width_mm": 12.7, "length_mm": 13.5, "c_curve_mm": 3.4},
    "ring":   {"width_mm": 11.7, "length_mm": 12.5, "c_curve_mm": 3.2},
    "pinky":  {"width_mm":  9.7, "length_mm": 10.5, "c_curve_mm": 2.7},
}

SIZE_THRESHOLDS = [
    (-3.0,        "much_smaller"),
    (-1.0,        "smaller"),
    ( 1.0,        "average"),
    ( 3.0,        "larger"),
    (float("inf"),"much_larger"),
]


def size_category(diff_mm: float) -> str:
    """Map a mm difference (user − standard) to a size category string."""
    for threshold, label in SIZE_THRESHOLDS:
        if diff_mm < threshold:
            return label
    return "much_larger"


def overall_category(width_cat: str, length_cat: str) -> str:
    """
    Combine width and length categories into one overall size label.
    Ranks each on 0–4 scale, averages, maps back.
    """
    rank   = {"much_smaller": 0, "smaller": 1, "average": 2,
              "larger": 3, "much_larger": 4}
    labels = list(rank.keys())
    avg    = (rank[width_cat] + rank[length_cat]) / 2.0
    return labels[round(avg)]


def compare_to_standard(finger: str, width_mm: float, length_mm: float) -> dict:
    """Return a size_vs_standard dict for one finger."""
    std = STANDARD_NAILS.get(finger)
    if std is None:
        return {}

    w_diff = round(width_mm  - std["width_mm"],  2)
    l_diff = round(length_mm - std["length_mm"], 2)
    w_cat  = size_category(w_diff)
    l_cat  = size_category(l_diff)

    return {
        "reference":          "Average Asian women (Lee et al., 2019)",
        "standard_width_mm":  std["width_mm"],
        "standard_length_mm": std["length_mm"],
        "user_width_mm":      round(width_mm,  2),
        "user_length_mm":     round(length_mm, 2),
        "width_diff_mm":      w_diff,
        "length_diff_mm":     l_diff,
        "width_category":     w_cat,
        "length_category":    l_cat,
        "overall_size":       overall_category(w_cat, l_cat),
    }


def overall_hand_size(nails: dict) -> dict:
    """
    Classify the whole hand as "small", "average", or "large".

    Method: uses the middle finger only (most anatomically consistent),
    comparing both width and length equally against the Asian women standard.

    Average of the two diffs:
      avg_diff < −1.0 mm  →  "small"
      avg_diff > +1.0 mm  →  "large"
      otherwise           →  "average"

    Returns a dict with the classification and the supporting numbers.
    """
    middle = nails.get("middle")
    if middle is None:
        return {"hand_size": "unknown", "note": "middle finger not found in measurements"}

    std       = STANDARD_NAILS["middle"]
    w_diff    = round(middle["width_mm"]  - std["width_mm"],  2)
    l_diff    = round(middle["length_mm"] - std["length_mm"], 2)
    avg_diff  = round((w_diff + l_diff) / 2.0, 2)

    if avg_diff < -1.0:
        hand_size = "small"
    elif avg_diff > 1.0:
        hand_size = "large"
    else:
        hand_size = "average"

    return {
        "hand_size":                  hand_size,
        "basis":                      "middle finger vs Asian women standard",
        "middle_finger_width_mm":     round(middle["width_mm"],  2),
        "middle_finger_length_mm":    round(middle["length_mm"], 2),
        "standard_width_mm":          std["width_mm"],
        "standard_length_mm":         std["length_mm"],
        "width_diff_mm":              w_diff,
        "length_diff_mm":             l_diff,
        "avg_diff_mm":                avg_diff,
        "thresholds":                 "small: avg_diff < −1.0 mm  |  large: avg_diff > +1.0 mm",
    }


# ─────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────

def arc_z_profile(us: np.ndarray, width_mm: float, c_curve_mm: float) -> np.ndarray:
    """
    Circular-arc C-curve: z(u) for u in [0,1] across nail width.
    z = 0 at centre (u=0.5), z = c_curve at edges (u=0, 1).
    Sagitta formula: R = w²/(8c) + c/2.
    """
    if c_curve_mm < 0.05:
        return np.zeros_like(us)
    R = (width_mm ** 2) / (8.0 * c_curve_mm) + c_curve_mm / 2.0
    x = (us - 0.5) * width_mm
    return R - np.sqrt(np.maximum(R ** 2 - x ** 2, 0.0))


def write_binary_stl(filepath: str, triangles: np.ndarray):
    """
    Write a binary STL file.
    triangles: float32 array shape (N, 3, 3) — N × [v0,v1,v2] × [x,y,z]
    """
    n = len(triangles)
    with open(filepath, "wb") as f:
        f.write(b"nail_tip_generator_v2" + b"\x00" * 59)
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
    """Connect two parallel polylines (N×3 each) with a quad strip."""
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

    params keys:
      width_mm         Nail plate width (measured)
      length_mm        Natural nail bed length (measured)
      c_curve_mm       Arc sagitta across the width (cross-section curvature)
      tip_length_mm    Free-tip extension length beyond fingertip
      wall_thick_mm    Wall thickness at the side edges   (default 0.5)
      center_thick_mm  Wall thickness at the centre line  (default 0.8)

    Returns dict with mesh statistics.
    """
    W    = float(params["width_mm"])
    L_nb = float(params["length_mm"])
    L_tp = float(params["tip_length_mm"])
    L    = L_nb + L_tp
    C    = float(params["c_curve_mm"])
    TW   = float(params.get("wall_thick_mm",   0.5))
    TC   = float(params.get("center_thick_mm", 0.8))

    us = np.linspace(0.0, 1.0, nu)
    vs = np.linspace(0.0, 1.0, nv)
    UU, VV = np.meshgrid(us, vs)

    # Bottom surface — concave well matching natural nail
    z_bot_1d = arc_z_profile(us, W, C)
    Z_bot    = np.tile(z_bot_1d, (nv, 1))

    # Wall thickness: TC at centre → TW at edges; 8% taper toward free tip
    u_center_dist = np.abs(us - 0.5) * 2.0
    thick_u       = TC - (TC - TW) * u_center_dist
    tip_taper_v   = 1.0 - 0.08 * vs
    Thick         = np.outer(tip_taper_v, thick_u)

    Z_top = Z_bot + Thick
    X     = UU * W
    Y     = VV * L

    all_tris = []
    all_tris.extend(grid_to_tris(X, Y, Z_top, flip=False))         # top surface
    all_tris.extend(grid_to_tris(X, Y, Z_bot, flip=True))          # bottom surface

    l_top = np.column_stack([X[:, 0],  Y[:, 0],  Z_top[:, 0]])
    l_bot = np.column_stack([X[:, 0],  Y[:, 0],  Z_bot[:, 0]])
    all_tris.extend(edge_strip_tris(l_bot, l_top, reverse=False))  # left wall

    r_top = np.column_stack([X[:, -1], Y[:, -1], Z_top[:, -1]])
    r_bot = np.column_stack([X[:, -1], Y[:, -1], Z_bot[:, -1]])
    all_tris.extend(edge_strip_tris(r_top, r_bot, reverse=True))   # right wall

    c_top = np.column_stack([X[0, :], Y[0, :], Z_top[0, :]])
    c_bot = np.column_stack([X[0, :], Y[0, :], Z_bot[0, :]])
    all_tris.extend(edge_strip_tris(c_top, c_bot, reverse=True))   # cuticle end

    t_top = np.column_stack([X[-1, :], Y[-1, :], Z_top[-1, :]])
    t_bot = np.column_stack([X[-1, :], Y[-1, :], Z_bot[-1, :]])
    all_tris.extend(edge_strip_tris(t_bot, t_top, reverse=False))  # free tip cap

    tris_arr = np.array(all_tris, dtype=np.float32)
    write_binary_stl(output_path, tris_arr)

    return {
        "triangles": len(tris_arr),
        "file_kb":   round(os.path.getsize(output_path) / 1024, 1),
        "dimensions": {
            "width_mm":  round(W,  2),
            "length_mm": round(L,  2),
            "height_mm": round(float(Z_top.max()), 2),
        },
        "wall_thick": {"center_mm": round(TC, 2), "edge_mm": round(TW, 2)},
    }


# ─────────────────────────────────────────────────────────────
# Finger order + cross-section C-curve defaults
# ─────────────────────────────────────────────────────────────

FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]

# The photo-measured c_curve includes the length-direction curvature, so for
# the 3D cross-section we use per-finger anatomical defaults (more accurate).
DEFAULT_CROSS_SECTION_C = {
    "thumb":  3.5,
    "index":  3.2,
    "middle": 3.0,
    "ring":   3.0,
    "pinky":  2.8,
}


# ─────────────────────────────────────────────────────────────
# Build all STLs for one hand
# ─────────────────────────────────────────────────────────────

def build_all(json_path: str, hand: str, tip_length: float,
              c_override: float | None, output_dir: str,
              wall_thick: float, center_thick: float,
              fingers_filter: list | None = None) -> list:
    """
    Generate STL files for one hand and write print_manifest.json.

    Folder structure created:
        output_dir/
            left_hand/
                nail_tip_thumb.stl
                nail_tip_index.stl
                nail_tip_middle.stl
                nail_tip_ring.stl
                nail_tip_pinky.stl
                print_manifest.json
            right_hand/
                nail_tip_thumb.stl
                ...
                print_manifest.json
    """
    hand = hand.lower().strip()
    if hand not in ("left", "right"):
        sys.exit(f"ERROR: --hand must be 'left' or 'right', got '{hand}'")

    hand_dir = os.path.join(output_dir, f"{hand}_hand")
    os.makedirs(hand_dir, exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)

    nails          = {n["finger"]: n for n in data["nails"]}
    active_fingers = fingers_filter if fingers_filter else FINGER_ORDER
    summary        = []

    print(f"\n  Hand   : {hand.upper()}")
    print(f"  Folder : {hand_dir}/")
    print(f"\n  {'─'*72}")
    print(f"  {'Finger':<8} {'W(mm)':>7} {'L(mm)':>7} {'Tip(mm)':>8} "
          f"{'C(mm)':>7}  {'Overall vs standard':<22}  File")
    print(f"  {'─'*72}")

    for finger in active_fingers:
        if finger not in nails:
            print(f"  {finger:<8}  (not found in measurements, skipping)")
            continue

        m        = nails[finger]
        C_cross  = c_override if c_override else DEFAULT_CROSS_SECTION_C.get(finger, 3.0)
        size_cmp = compare_to_standard(finger, m["width_mm"], m["length_mm"])

        params = {
            "width_mm":        m["width_mm"],
            "length_mm":       m["length_mm"],
            "c_curve_mm":      C_cross,
            "tip_length_mm":   tip_length,
            "wall_thick_mm":   wall_thick,
            "center_thick_mm": center_thick,
        }

        out_file = os.path.join(hand_dir, f"nail_tip_{finger}.stl")
        stats    = generate_nail_tip_stl(params, out_file)

        summary.append({
            "finger":   finger,
            "params":   params,
            "stats":    stats,
            "file":     out_file,
            "size_cmp": size_cmp,
        })

        overall = size_cmp.get("overall_size", "—")
        print(f"  {finger:<8}  {m['width_mm']:>6.2f}  {m['length_mm']:>6.2f}  "
              f"{tip_length:>7.1f}  {C_cross:>6.2f}  {overall:<22}  "
              f"{os.path.basename(out_file)}")

    print(f"  {'─'*72}")
    print(f"\n  ✅  {len(summary)} STL file(s) saved to '{hand_dir}/'")

    # ── Overall hand size (middle finger vs standard) ────────
    hand_size_info = overall_hand_size(nails)
    print(f"  Overall hand size : {hand_size_info['hand_size'].upper()}"
          f"  (middle finger avg diff: {hand_size_info['avg_diff_mm']:+.2f} mm vs standard)\n")

    # ── Write print_manifest.json ────────────────────────────
    manifest = {
        "hand":                hand,
        "overall_hand_size":   hand_size_info,   # ← NEW in v3
        "source_measurements": json_path,
        "tip_length_mm":       tip_length,
        "wall_thick_mm":       wall_thick,
        "center_thick_mm":     center_thick,
        "standard_reference":  "Average Asian women nail dimensions (Lee et al., 2019)",
        "size_categories": {
            "much_smaller": "user nail ≤ −3.0 mm vs standard",
            "smaller":      "user nail −3.0 to −1.0 mm vs standard",
            "average":      "user nail within ±1.0 mm of standard",
            "larger":       "user nail +1.0 to +3.0 mm vs standard",
            "much_larger":  "user nail ≥ +3.0 mm vs standard",
        },
        "nails": [
            {
                "finger":                   s["finger"],
                "stl_file":                 os.path.basename(s["file"]),
                "width_mm":                 s["params"]["width_mm"],
                "length_natural_mm":        s["params"]["length_mm"],
                "length_total_mm":          round(
                    s["params"]["length_mm"] + s["params"]["tip_length_mm"], 2),
                "c_curve_cross_section_mm": s["params"]["c_curve_mm"],
                "triangles":                s["stats"]["triangles"],
                "file_kb":                  s["stats"]["file_kb"],
                "bounding_box_mm":          s["stats"]["dimensions"],
                "size_vs_standard":         s["size_cmp"],   # ← NEW in v2
                "printing_tips": {
                    "orientation":  "Print top-face-down on build plate",
                    "supports":     "None needed in recommended orientation",
                    "layer_height": "0.05 mm resin / 0.1 mm FDM",
                    "material":     "Resin (best detail) or TPU (flexible)",
                },
            }
            for s in summary
        ],
    }

    manifest_path = os.path.join(hand_dir, "print_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"     print_manifest.json written")

    # ── Console size comparison table ───────────────────────
    print(f"\n  Size comparison vs Asian women average:")
    print(f"  {'Finger':<8}  {'User W':>7}  {'Std W':>6}  {'W diff':>7}  "
          f"{'User L':>7}  {'Std L':>6}  {'L diff':>7}  Overall")
    print(f"  {'─'*72}")
    for s in summary:
        c = s["size_cmp"]
        if not c:
            continue
        print(f"  {s['finger']:<8}  "
              f"{c['user_width_mm']:>6.2f}mm  {c['standard_width_mm']:>5.1f}mm  "
              f"{c['width_diff_mm']:>+6.2f}mm  "
              f"{c['user_length_mm']:>6.2f}mm  {c['standard_length_mm']:>5.1f}mm  "
              f"{c['length_diff_mm']:>+6.2f}mm  "
              f"{c['overall_size']}")
    print()

    return summary


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate 3D-printable nail tip STL files from nail measurements")

    p.add_argument("--measurements",     required=True,
                   help="Path to nail_measurements.json from nail_measurer.py")
    p.add_argument("--hand",             default="right",
                   choices=["left", "right"],
                   help="Which hand this measurement is for (default: right). "
                        "STLs are saved into output/<hand>_hand/")
    p.add_argument("--output",           default="nail_stl",
                   help="Root output folder (default: nail_stl/). "
                        "Subfolders left_hand/ and right_hand/ are created inside.")
    p.add_argument("--tip-length",       type=float, default=5.0,
                   help="Free-tip extension in mm beyond the natural nail (default: 5.0)")
    p.add_argument("--c-curve-override", type=float, default=None,
                   help="Override cross-section C-curve for all fingers (mm). "
                        "Omit to use per-finger anatomical defaults (2.8–3.5 mm).")
    p.add_argument("--wall-thick",       type=float, default=0.5,
                   help="Wall thickness at side edges in mm (default: 0.5)")
    p.add_argument("--center-thick",     type=float, default=0.8,
                   help="Wall thickness at centre line in mm (default: 0.8)")
    p.add_argument("--finger",           default=None,
                   help="Generate only this finger "
                        "(thumb / index / middle / ring / pinky)")

    args = p.parse_args()

    build_all(
        json_path      = args.measurements,
        hand           = args.hand,
        tip_length     = args.tip_length,
        c_override     = args.c_curve_override,
        output_dir     = args.output,
        wall_thick     = args.wall_thick,
        center_thick   = args.center_thick,
        fingers_filter = [args.finger] if args.finger else None,
    )