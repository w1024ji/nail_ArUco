"""
merge_fingers.py

Loads 5 finger STL files (thumb -> pinky), applies a uniform 30-degree tilt
and 10mm spacing between each finger, and exports a single 3MF file with
5 separate objects (so Bambu Studio / Orca Slicer can still treat them as
individual parts for supports, etc.)

Usage:
    python merge_fingers.py

Requires:
    pip install trimesh numpy
"""

import trimesh
import numpy as np
import os

# ---------------------------------------------------------------------------
# CONFIG — adjust these to match your setup
# ---------------------------------------------------------------------------

STL_DIR = "stl"  # folder containing the 5 finger STLs
OUTPUT_PATH = "output/hand_merged.3mf"

# Order matters: this determines left-to-right spacing on the plate
FINGER_FILES = [
    "nail_thumb_round.stl",
    "nail_index_round.stl",
    "nail_middle_round.stl",
    "nail_ring_round.stl",
    "nail_pinky_round.stl",
]

SPACING_MM = 10.0      # gap between each finger's origin, along X axis
TILT_DEG = 30.0        # tilt angle applied to every finger
TILT_AXIS = [1, 0, 0]  # rotate around X axis (change to [0,1,0] if tilt should be front-to-back)

# ---------------------------------------------------------------------------


def load_finger(path):
    mesh = trimesh.load(path, force="mesh")
    if mesh.is_empty:
        raise ValueError(f"Loaded mesh is empty: {path}")
    return mesh


def build_scene():
    scene = trimesh.Scene()

    cursor_x = 0.0  # tracks the left edge where the next finger should start

    for i, filename in enumerate(FINGER_FILES):
        path = os.path.join(STL_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Could not find '{path}'. Check that FINGER_FILES matches "
                f"your actual filenames in {STL_DIR}/"
            )

        mesh = load_finger(path)

        # Center each mesh on its own origin first (so tilt happens around
        # its own center, not the world origin)
        mesh.apply_translation(-mesh.centroid)

        # Apply tilt
        tilt_matrix = trimesh.transformations.rotation_matrix(
            angle=np.radians(TILT_DEG),
            direction=TILT_AXIS,
            point=[0, 0, 0],
        )
        mesh.apply_transform(tilt_matrix)

        # Sit the mesh back on the build plate (Z=0 at lowest point)
        mesh.apply_translation([0, 0, -mesh.bounds[0][2]])

        # Re-check bounds after tilt (post-tilt bounding box, not original)
        min_x = mesh.bounds[0][0]
        max_x = mesh.bounds[1][0]
        width_x = max_x - min_x

        if i == 0:
            # place first finger's left edge at x=0
            x_offset = -min_x
        else:
            # place this finger's left edge SPACING_MM after the previous
            # finger's right edge
            x_offset = cursor_x - min_x

        mesh.apply_translation([x_offset, 0, 0])

        # update cursor to this finger's new right edge + gap
        cursor_x = x_offset + max_x + SPACING_MM

        finger_name = os.path.splitext(filename)[0]
        scene.add_geometry(mesh, node_name=finger_name, geom_name=finger_name)

    return scene


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    scene = build_scene()
    scene.export(OUTPUT_PATH)
    print(f"Exported merged 3MF to: {OUTPUT_PATH}")
    print(f"Objects in scene: {list(scene.geometry.keys())}")


if __name__ == "__main__":
    main()