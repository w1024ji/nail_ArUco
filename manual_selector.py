"""
manual_selector.py
------------------
Interactive fallback: manually outline each nail when auto-detection fails
(e.g. nail polish, uneven lighting, dark skin tones).

Controls:
  Left-click   → add a polygon point
  Right-click  → close current nail & save it
  'u'          → undo last point
  'n'          → skip this finger (no measurement saved)
  'q' / ESC    → done — compute measurements & write JSON

Usage:
    python manual_selector.py --image hand.jpg [--aruco-size 20] [--output results/]
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from nail_measurer import (
    detect_aruco, measure_nail, build_json_payload,
    print_summary, FINGER_NAMES
)

COLORS = [
    (255,  80,  80),   # thumb  – blue-ish
    ( 80, 200,  80),   # index  – green
    ( 80,  80, 255),   # middle – red
    (200,  80, 200),   # ring   – magenta
    ( 80, 200, 200),   # pinky  – cyan
]

HINTS = [
    "Left-click: add point  |  Right-click: save nail",
    "'u': undo point  |  'n': skip finger  |  'q': finish",
]


class ManualSelector:
    def __init__(self, image: np.ndarray, aruco_size_mm: float):
        self.orig    = image.copy()
        self.canvas  = image.copy()
        self.h, self.w = image.shape[:2]

        print("  Detecting ArUco marker …")
        self.mm_per_pixel, self.aruco_corners, self.marker_id = \
            detect_aruco(image, aruco_size_mm)
        print(f"  Scale = {self.mm_per_pixel:.5f} mm/pixel")

        self.current_pts: list  = []
        self.contours:    list  = []   # finished OpenCV contours
        self.finger_idx:  int   = 0    # which finger we're drawing

    # ── drawing ────────────────────────────────────────────────

    def _redraw(self):
        self.canvas = self.orig.copy()

        # ArUco corners
        if self.aruco_corners is not None:
            cv2.polylines(self.canvas,
                          [self.aruco_corners.astype(int)], True, (0, 255, 255), 2)

        # Finished nails
        for i, cnt in enumerate(self.contours):
            color = COLORS[i % len(COLORS)]
            cv2.polylines(self.canvas, [cnt], True, color, 2)
            M = cv2.moments(cnt)
            if M["m00"]:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(self.canvas, FINGER_NAMES[i],
                            (cx - 20, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # In-progress polygon
        if self.current_pts:
            color = COLORS[self.finger_idx % len(COLORS)]
            for p in self.current_pts:
                cv2.circle(self.canvas, p, 4, color, -1)
            if len(self.current_pts) > 1:
                cv2.polylines(self.canvas,
                              [np.array(self.current_pts, np.int32)],
                              False, color, 1)

        # HUD
        fname = FINGER_NAMES[self.finger_idx] if self.finger_idx < 5 else "done"
        cv2.putText(self.canvas,
                    f"Outlining: {fname.upper()}  "
                    f"({len(self.contours)}/5 saved)",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        for i, hint in enumerate(HINTS):
            cv2.putText(self.canvas, hint,
                        (10, self.h - 30 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Nail Selector", self.canvas)

    # ── mouse ──────────────────────────────────────────────────

    def _on_mouse(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_pts.append((x, y))
            self._redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._save_nail()

    def _save_nail(self):
        if len(self.current_pts) < 3:
            print("  Need at least 3 points.")
            return
        cnt = np.array(self.current_pts, np.int32).reshape(-1, 1, 2)
        self.contours.append(cnt)
        print(f"  ✓ Saved {FINGER_NAMES[self.finger_idx]}")
        self.current_pts = []
        self.finger_idx += 1
        self._redraw()

    # ── main loop ──────────────────────────────────────────────

    def run(self):
        cv2.namedWindow("Nail Selector", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Nail Selector", self._on_mouse)
        self._redraw()

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord('u'):
                if self.current_pts:
                    self.current_pts.pop()
                    self._redraw()
            elif key == ord('n'):
                print(f"  – Skipped {FINGER_NAMES[self.finger_idx]}")
                self.current_pts = []
                self.finger_idx += 1
                self._redraw()
            elif key in (ord('q'), 27):
                # auto-save if points pending
                if len(self.current_pts) >= 3:
                    self._save_nail()
                break

        cv2.destroyAllWindows()
        return self.contours, self.mm_per_pixel, self.aruco_corners, self.marker_id


# ─────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────

def run(image_path: str, aruco_size_mm: float, output_dir: str):
    image = cv2.imread(image_path)
    if image is None:
        sys.exit(f"ERROR: Cannot open '{image_path}'")

    selector = ManualSelector(image, aruco_size_mm)
    contours, mm_per_pixel, aruco_corners, marker_id = selector.run()

    if not contours:
        print("No nails saved — exiting.")
        return

    measurements = [
        measure_nail(cnt, mm_per_pixel, finger_id=i)
        for i, cnt in enumerate(contours)
    ]
    print_summary(measurements)

    os.makedirs(output_dir, exist_ok=True)
    payload   = build_json_payload(measurements, marker_id,
                                   aruco_size_mm, mm_per_pixel, image_path)
    json_path = os.path.join(output_dir, "nail_measurements.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  [JSON] Saved → {json_path}\n")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Manually outline nails for measurement")
    p.add_argument("--image",      required=True)
    p.add_argument("--aruco-size", type=float, default=20.0,
                   help="Physical ArUco marker side in mm (default 20)")
    p.add_argument("--output",     default="nail_results")
    args = p.parse_args()
    run(args.image, args.aruco_size, args.output)