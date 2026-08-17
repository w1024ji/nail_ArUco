# Nail ArUco — Handoff (2026-08-10)

Custom nail-tip generator: photo of a finger → real-world measurements → 3D-printable STL. Written for picking this project back up on a new Claude account. Repo: `/mnt/c/nail_ArUco`, branch `nail-shape-detection`.

## Pipeline

1. **Photo/Capture** — one finger per shot, finger pointing up, ArUco marker (6x6_50 dict) beside it on the same flat surface, dark background.
   - `scan/nail_capture.py` — auto-capture still-photo flow (marker stability + skin-fraction gate before countdown).
   - `scan/nail_live.py` — **preferred now.** Runs measurement continuously on the live camera feed; operator presses ENTER to accept the measurement they're looking at (not a raw photo to be judged blind afterwards). Has a `--rotate {0,90,180,270}` flag because the rig's camera is mounted upside down. Shows a live fold-coverage meter and a peak-width tracker (see below).
2. **Measure** — `scan/nail_measurer.py` (the core, heavily-tuned module). ArUco gives mm/px scale; then detects finger silhouette, nail width (lateral folds), length, cuticle position, and c-curve. Outputs `nail_measurements.json` + `profile.json` + annotated image.
3. **C-curve (separate, end-on shot)** — `scan/measure_ccurve.py`, needs a photo looking straight at the fingertip. Not yet integrated into the box rig (see Open Problems).
4. **Generate STL** — `scan/nail_exact_stl.py` (primary; v15-ish internally) turns the JSON into a parametric watertight STL. Shapes: round, almond, square, stiletto, ballerina, oval.
5. **Arrange for printing** — `printer/merge_fingers.py` (new, untracked as of this handoff) loads the 5 finger STLs, tilts + spaces them, exports one `.3mf` for the slicer.
6. **Upload** — `scan/s3_upload.py`. **Upload-only, on purpose** — never add delete/sync logic here; a local mistake could wipe the S3 bucket. This was an explicit user decision, don't revisit it.

## Where things stand (measurement accuracy)

Validated against ruler ground truth (`scan/sh_images/`, see details below): width and length both land within **±0.5–0.9mm** of a ruler on well-lit reference photos. The box rig (in-progress hardware) is not yet at that accuracy — see below.

### The central finding: light direction decides what's detectable
A groove only casts a shadow when light rakes **across** it, not along it.
- **Lateral nail folds** (run vertically along the finger) need light from the **side** (left/right).
- **Cuticle** (runs horizontally across the finger) needs light along the **finger axis** (from the tip or base end), and is actually found by *pigment color* now, not shadow (see below), so it's lighting-independent as long as it's not blown out.
- These two needs conflict — one side light can't reveal both. Plan: two lights, switchable, ideally two frames per finger using whichever light reveals that feature.
- The code's own header comment ("even lighting, no harsh shadows") is **wrong** for this pipeline and should eventually be corrected/removed.

### Width — from lateral nail folds
`detect_lateral_edges()` in `nail_measurer.py` finds the fold groove via a color/texture ridge (`S/12 + a*/4 - L/12`), validated by requiring skin on both sides of a candidate peak (distinguishes a real groove from the finger's outer silhouette, which only has skin on one side).
- Needs a raking side light. Target: skin brightness gradient (L\*) **~40+** between the lit and shadow side of the finger, and lit-side L\* around ~195 with near-0% clipped (overexposed) pixels — overexposing erases the very shadow the detector needs.
- If fold coverage is too low, width silently falls back to the plain finger-silhouette mask, which is measurably worse (this was traced as the main driver of instability in earlier box-rig tests).
- `nail_live.py`'s on-screen meter shows live fold-coverage % and lit-side clipping % so the operator can aim the light while watching the numbers — use this when tuning the new box's LEDs.
- Width is also foreshortened by finger roll (rotation about its long axis) and this is invisible in the overlay — `nail_live.py` tracks a rolling peak-W window and shows SQUARE-ON vs ROLLED so the operator rolls the finger until W stops increasing.
- **Known noise issue**: width is *bistable* under small pixel noise / JPEG re-compression — it can flip between two discrete values ~0.6mm apart on a sizeable minority of frames (root cause not fully isolated, lives somewhere in the row-scan width path). Fix in place: `nail_live.py` takes a **median across several accepted frames** rather than trusting one. This is the strongest reason the live-camera flow exists at all — a single static photo can't average.

### Cuticle — from color, not shadow
`detect_cuticle_by_color()` finds the cuticle by pigment: a\* (redness) dips to a local minimum right at the cuticle, and b\* (yellowness) climbs just below it (skin is more yellow than nail plate). Score = `(b_rise) + (a_dip)`; this beat either signal alone on every test photo, and it works regardless of light direction. Search window is built from the *measured* width, not a fixed anthropometric ratio (see below).

### The W/L "Jung plate ratio" 0.91 prior is wrong — don't reintroduce it
An old assumption that cuticle-to-tip length ≈ width / 0.91 is baked into `WL_STANDARD` / `corrected_length_mm` / `apply_wl_correction()`. Ruler ground truth shows the *true* ratio is more like 0.64–0.86 depending on finger, and using 0.91 as ground truth actively corrupts results (it shrank a true 14mm-long thumb print down to 12.65mm on one test). Current code measures cuticle/length directly and only uses the 0.91-based path as a last-resort fallback. If you see width/length numbers that look "corrected" back toward a suspicious ratio, that's this legacy prior leaking back in — don't retune it, trust the direct measurement instead.

### End-on c-curve (dome curvature)
`measure_ccurve.py` needs a photo looking straight at the fingertip (not top-down), fits a circle to the free-edge band, gets sagitta `h` and radius `R`. Validated once (sh_middle: h=2.38mm, R=6.06mm, matched ruler within ~0.1mm). **The current box rig cannot produce this view** — the finger enters from below toward a wall-mounted marker, so there's no way to point the fingertip at the camera inside it. Runs are currently top-view only (no `--ccurve`), and STL dome curvature from box-rig runs should not be trusted yet — that's real open work for the new box design if you want accurate c-curve without a second manual shot.

## Print-fit corrections (from real test prints, `nail_exact_stl.py`)
These are empirical corrections layered on top of the *measured* geometry (only in normal mode — `--exact` mode skips all of them and prints a true replica for validation):
- **Width**: printed nail was smaller than real even at +0, because bending flat measured width into the 3D C-curve shrinks the wrapping chord. Fix: add `WIDTH_FIT_MARGIN_MM = 3.0` before building geometry.
- **Length**: was printing ~4mm short — root cause was the bad 0.91 prior described above overriding the real measured length. Fixed by preferring measured length; do not re-add a flat "+4mm" band-aid, that was the wrong fix and has been removed.
- **Cuticle-curve blend**: an attempt to round the inner base near the cuticle. Currently **off by default** (`--cuticle-curve 0.0`) — user reported it "not working". Code is still there, dormant, if you want to revisit.
- **Thickness default**: 0.6mm (down from 1.2mm), per request.

## Open problems / where to focus next (given the new box build)
1. **Get the side light right first** — this is the single biggest lever on width accuracy right now. Aim for the ~40+ L\* gradient target above and watch `nail_live.py`'s coverage meter live while positioning LEDs. Don't make the box interior uniformly diffuse/matte-white — that recreates flat lighting and kills the fold shadow the whole method depends on.
2. **Cuticle-axis light** — currently unaddressed in the box; a second, switchable light along the finger's long axis was the plan but wasn't built as of the last session. Not strictly required now that cuticle detection is color-based rather than shadow-based, but worth checking that the color signal isn't getting washed out by whatever light is on.
3. **End-on c-curve capture** — the box's finger-from-below geometry blocks this shot entirely. If you want in-box c-curve, the new box needs either a second camera position/angle or a way to present the fingertip to the main camera.
4. **Width bistability** — root cause not fully pinned down (something in the row-scan / width path is sensitive to single-pixel noise). Currently papered over by median-of-N-live-frames; worth root-causing properly if you have time, since it would let single-photo capture become reliable again.
5. **`printer/merge_fingers.py`** is new and not yet committed to git — review/commit it if it's in a good state, or continue iterating on plate layout (tilt/spacing constants at the top of the file).

## Practical gotchas
- No `cv2` in WSL python — everything must run under Windows Python: `/mnt/c/Users/USER/AppData/Local/Programs/Python/Python311/python.exe`.
- Environment variables do **not** propagate from WSL to Windows Python — gate any debug flags on `sys.argv`, not `os.environ`.
- Windows Python cannot write to WSL `/tmp` paths — write debug/scratch files inside the repo instead (e.g. `scan/test/`).
- Live capture must measure at the exact resolution that will be accepted (never downscale for speed) — measurement is resolution-sensitive; the same photo gave wildly different widths at 640/960/1280/1920px.
- Accept/save frames as **PNG**, never JPEG — JPEG re-encoding alone can flip the bistable width or shift c-curve sagitta.
- ArUco marker is **DICT_6X6_50**, printed once via `generate_aruco.py`. Always pass the true printed `--aruco-size` — a wrong value scales every measurement linearly and the overlay will still look plausible.

## Reference: ruler ground truth (sh_images)
| nail | width (from above) | width incl. c-curve | length |
|---|---|---|---|
| sh_thumb | 12mm | 13mm | 14mm |
| sh_middle | 9mm | 11mm | 14mm |

Pipeline came within ±0.5mm on width for both (thumb 11.51 vs 12, middle 9.50 vs 9) and ±0.9mm on length across several test photos.
