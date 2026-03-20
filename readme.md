# Nail Measurement Tool
### ArUco-based · 5 fingers · JSON output for 3-D mesh pipeline

---
## commands
```bash
python nail_measurer.py --image fingers_left.jpg --aruco-size 20 --output my_results_left
python nail_tip_generator.py --measurements nail_measurements.json --hand right --curve flat
python nail_tip_generator.py --measurements nail_measurements.json --hand right --curve medium
python nail_tip_generator.py --measurements nail_measurements.json --hand right --curve steep
```

---

## Install

```bash
pip install opencv-python opencv-contrib-python numpy scipy
```
Python 3.10+ recommended.

---

## Quick-start workflow

### 1 · Print a marker

```bash
python generate_aruco.py --size 20 --id 0 --output marker.png
```
- Print `marker.png` at **100 % scale** (no "fit to page").
- Verify the inner black square with calipers. Note the exact mm value.

### 2 · Take the photo

| ✅ Do | ❌ Avoid |
|-------|---------|
| Hand flat, palm down | Tilted or angled shots |
| Marker in the same plane as nails | Marker propped up / on a book |
| Diffuse even lighting | Harsh shadows across nails |
| Plain light-coloured background | Patterned tablecloth |
| Shoot straight down (top-view) | Side angle |
| 12 MP+ camera | Low-res or blurry |

### 3 · Run auto-detection

```bash
python nail_measurer.py --image hand.jpg --aruco-size 20.3 --output results/
```
*(use your actual caliper measurement for `--aruco-size`)*

Output: `results/nail_measurements.json`

### 4 · Manual fallback (painted nails / poor lighting)

```bash
python manual_selector.py --image hand.jpg --aruco-size 20.3 --output results/
```
- **Left-click** to place outline points around a nail
- **Right-click** to save and move to the next finger
- **`u`** to undo last point
- **`n`** to skip a finger
- **`q`** or **ESC** to finish

---

## JSON output structure

```jsonc
{
  "meta": {
    "source_image": "hand.jpg",
    "aruco_marker_id": 0,
    "aruco_physical_size_mm": 20.3,
    "mm_per_pixel": 0.07314,
    "nails_detected": 5,
    "measurement_notes": { ... }   // explains each field
  },

  // Full detail per nail
  "nails": [
    {
      "finger":            "thumb",
      "finger_id":         0,
      "length_mm":         16.4,
      "width_mm":          14.2,
      "c_curve_mm":        2.3,     // arc depth (sagitta)
      "arc_radius_mm":     23.1,    // radius of curvature
      "thickness_mm":      0.45,    // estimated (see note)
      "thickness_source":  "geometric_estimate",
      "aspect_ratio":      0.866,
      "pixel_scale_mm_per_px": 0.07314,
      "nail_polygon_px":   [[x,y], ...]
    }
    // ... index, middle, ring, pinky
  ],

  // Quick lookup by finger name
  "by_finger": {
    "thumb":  { "length_mm": 16.4, "width_mm": 14.2, ... },
    ...
  },

  // Ready-to-use 3-D mesh parameters
  "mesh_params": {
    "thumb": {
      "bounding_box_mm": { "x": 14.2, "y": 0.45, "z": 16.4 },
      "curvature": {
        "c_curve_sagitta_mm": 2.3,
        "arc_radius_mm":      23.1
      },
      "shape_hint": "barrel_arc  AR=0.866"
    },
    ...
  }
}
```

---

## Measurements explained

| Field | Definition | How it's computed |
|-------|-----------|-------------------|
| `length_mm` | Base-to-tip along the nail's long axis | Rotated bounding box long side |
| `width_mm` | Widest point across the nail | Rotated bounding box short side |
| `c_curve_mm` | Arc depth (sagitta) — how much the nail curves across its width | Max perpendicular deviation from the width chord |
| `arc_radius_mm` | Radius of the circular arc | Sagitta formula: R = w²/(8h) + h/2 |
| `thickness_mm` | Nail plate thickness base-to-surface | Estimated: c_curve × 1.5, clamped to 0.25–0.85 mm |

### Thickness note
Thickness **cannot** be measured from a top-down photo alone.
The value in the JSON is a geometric estimate based on nail anatomy research
(average human nail: 0.35–0.65 mm).
For precision 3-D printing, measure thickness from a **side-view** photo:
```bash
# future: side_view_measurer.py --image thumb_side.jpg --aruco-size 20.3
```
Then manually update `thickness_mm` in the JSON.

---

## Using the JSON in your 3-D pipeline

The `mesh_params` block gives you everything needed to construct a parametric nail mesh:

```python
import json

with open("nail_measurements.json") as f:
    data = json.load(f)

for finger, params in data["mesh_params"].items():
    bb  = params["bounding_box_mm"]   # x=width, y=thickness, z=length
    R   = params["curvature"]["arc_radius_mm"]
    h   = params["curvature"]["c_curve_sagitta_mm"]
    # → feed bb + R into your parametric nail mesh generator
```