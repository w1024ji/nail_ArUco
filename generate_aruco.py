"""
generate_aruco.py
-----------------
Generates a printable ArUco marker at an exact physical size.

Usage:
    python generate_aruco.py --size 20 --id 0 --output marker.png

Print at 100% scale (no "fit to page").
Measure the printed inner square with calipers and pass that exact value
to nail_measurer.py via --aruco-size for maximum accuracy.
"""

import argparse
import cv2
import numpy as np


DICT_MAP = {
    "4x4": cv2.aruco.DICT_4X4_50,
    "5x5": cv2.aruco.DICT_5X5_50,
    "6x6": cv2.aruco.DICT_6X6_50,
}


def generate(marker_id: int, size_mm: float, dpi: int,
             dict_name: str, output: str):
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_MAP[dict_name])

    px     = int(round(size_mm / 25.4 * dpi))
    border = px // 4
    total  = px + 2 * border

    marker_img = np.zeros((px, px), np.uint8)
    cv2.aruco.generateImageMarker(aruco_dict, marker_id, px, marker_img, 1)

    canvas = np.ones((total, total), np.uint8) * 255
    canvas[border:border+px, border:border+px] = marker_img
    cv2.rectangle(canvas, (1, 1), (total-2, total-2), 0, 1)

    # Label below marker
    label  = f"ArUco {dict_name.upper()}  id={marker_id}  {size_mm} mm"
    fscale = total / 700
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fscale, 1)
    labeled = np.ones((total + th + 10, total), np.uint8) * 255
    labeled[:total] = canvas
    cv2.putText(labeled, label,
                ((total - tw) // 2, total + th + 4),
                cv2.FONT_HERSHEY_SIMPLEX, fscale, 0, 1)

    cv2.imwrite(output, labeled)
    print(f"Saved: {output}")
    print(f"Print at 100% scale — inner square should measure exactly {size_mm} mm.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate printable ArUco marker")
    p.add_argument("--id",     type=int,   default=0)
    p.add_argument("--size",   type=float, default=20.0,
                   help="Desired printed size in mm (default 20)")
    p.add_argument("--dpi",    type=int,   default=300)
    p.add_argument("--dict",   default="4x4", choices=list(DICT_MAP.keys()))
    p.add_argument("--output", default="aruco_marker.png")
    args = p.parse_args()
    generate(args.id, args.size, args.dpi, args.dict, args.output)