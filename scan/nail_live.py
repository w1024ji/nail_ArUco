"""
nail_live.py
------------
Live-camera nail measurement with human confirmation.

The difference from nail_capture.py:
    nail_capture.py  → you confirm the PHOTO, then measurement runs blind.
    nail_live.py     → measurement runs continuously on the live feed and the
                       annotated result is what you see on screen. You press
                       OK to accept the MEASUREMENT you are looking at.

That is the whole point: a bad measurement is obvious in the annotated
overlay, so if the human only ever accepts good overlays, bad measurements
can never reach the STL stage.

Usage:
    python nail_live.py --finger index --output out_live
    python nail_live.py --finger index --output out_live --ccurve
    python nail_live.py --finger index --output out_live --shape round almond

Controls:
    ENTER / SPACE : accept the measurement currently on screen
    R             : clear stability + peak-width history and keep scanning
    Q / ESC       : quit without saving

Finding the true width:
    Rolling the finger about its long axis foreshortens the nail, so W can only
    read smaller than the truth, never larger - and the overlay looks perfectly
    correct either way. The status bar therefore tracks the peak W and tells you
    when the current reading is below it: roll the finger slowly and accept at
    SQUARE-ON, where W stops rising.

Display:
    Main image    : the latest completed measurement overlay (~1 fps).
                    This is what gets accepted — what you see is what you get.
    Top-right PiP : the raw live feed (30 fps) for positioning your finger.
"""

import argparse
import json
import math
import os
import contextlib
import subprocess
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

import nail_measurer as nm

BASE   = os.path.dirname(os.path.abspath(__file__))
SHAPES = ("round", "oval", "almond", "square", "stiletto", "ballerina")

# Measurement is resolution-sensitive (width shifted ~2.5mm between 1920 and
# 640 on the same photo), so the live loop must measure at exactly the
# resolution that gets accepted. No downscaling for speed.
CAM_WIDTH  = 1920
CAM_HEIGHT = 1080

# Stability gate: advisory only, it never blocks the OK key. The human is the
# judge; this just flags a measurement that is jittering between frames.
STABLE_N       = 3      # consecutive results compared
STABLE_TOL_MM  = 0.3    # max spread in width and length to count as stable

# Peak-width tracking. Rolling the finger about its long axis foreshortens the
# nail, so the projected width can only ever be smaller than the true width,
# never larger (observed: 93px vs 82px on the same nail across two runs, with
# nail height and marker size unchanged — a 0.75mm swing that looks perfectly
# correct in the overlay). The true width is therefore the peak of the
# projected width: roll the finger and accept where W stops rising.
PEAK_N         = 20     # results kept for the peak (~20s at 1 Hz)
PEAK_TOL_MM    = 0.15   # how close to the peak counts as square-on

DISPLAY_H = 820         # window height; the full-res frame is what gets saved


# ─────────────────────────────────────────────────────────────
# Per-thread stdout muting
# ─────────────────────────────────────────────────────────────

class _ThreadQuietStdout:
    """sys.stdout replacement that drops writes from threads marked quiet.

    nail_measurer prints a block of diagnostics per measurement, and the
    measurement runs on a background thread ~1x/second. contextlib.
    redirect_stdout cannot be used to silence it: that swaps the *global*
    sys.stdout, so when the worker and the main thread enter/leave their
    contexts in an interleaved order, the wrong stream gets restored and the
    console goes silent for the rest of the run. Muting per thread has no
    such race.
    """

    def __init__(self, real):
        self._real  = real
        self._local = threading.local()

    def set_quiet(self, value):
        prev = getattr(self._local, "quiet", False)
        self._local.quiet = value
        return prev

    def write(self, s):
        if getattr(self._local, "quiet", False):
            return len(s)
        return self._real.write(s)

    def flush(self):
        self._real.flush()

    def __getattr__(self, name):
        return getattr(self._real, name)


_filter      = None
_filter_lock = threading.Lock()


def _ensure_filter():
    global _filter
    with _filter_lock:
        if _filter is None:
            _filter = _ThreadQuietStdout(sys.stdout)
            sys.stdout = _filter
        return _filter


@contextlib.contextmanager
def quiet():
    """Silence prints made by the calling thread only."""
    f    = _ensure_filter()
    prev = f.set_quiet(True)
    try:
        yield
    finally:
        f.set_quiet(prev)


# ─────────────────────────────────────────────────────────────
# Measurement worker
# ─────────────────────────────────────────────────────────────

def clipped_fraction(frame, data):
    """Fraction of the brighter skin band beside the nail that is blown out.

    Sampled from the two bands 1.2-2.4 nail-half-widths either side of the nail
    axis, over the nail's own rows, and reported for whichever side is brighter
    — that is the lit side, and the one whose groove the measurement needs.
    """
    try:
        cx   = int(data["_tip_x"])
        half = float(data["_nail_half"])
        y0   = int(data["_tip_y"])
        y1   = int(data["_cuticle_y"])
    except (KeyError, TypeError, ValueError):
        return None
    if y1 <= y0 or half <= 0:
        return None

    lo, hi = int(half * 1.2), int(half * 2.4)
    H, W   = frame.shape[:2]
    y0, y1 = max(0, y0), min(H, y1)
    worst  = None
    for x0, x1 in ((cx - hi, cx - lo), (cx + lo, cx + hi)):
        x0, x1 = max(0, x0), min(W, x1)
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        band = frame[y0:y1, x0:x1]
        if band.size == 0:
            continue
        # brightest side wins: compare means, keep that side's clipped fraction
        clipped = float(np.mean(band.max(axis=2) >= 250)) * 100.0
        if worst is None or band.mean() > worst[1]:
            worst = (clipped, band.mean())
    return None if worst is None else worst[0]


def unusable_values(data):
    """Names of measurements that came out None or non-finite.

    A failed c-curve still renders as a perfectly plausible overlay - it just
    prints "C: nanmm" among four good-looking numbers - so an operator watching
    W and L can accept it and push NaN into the JSON and on into the STL stage.
    These frames must be unacceptable, not merely ugly.
    """
    bad = []
    for key in ("width_mm", "length_mm", "corrected_length_mm",
                "c_curve_mm", "arc_radius_mm", "thickness_mm"):
        v = data.get(key)
        if v is None:
            bad.append(f"{key}=None")
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            bad.append(f"{key}={v!r}")
            continue
        if not math.isfinite(f) or f <= 0:
            bad.append(f"{key}={v}")
    return bad


def draw_width_marker(overlay, data, mpp):
    """Mark the span that width_mm actually measures.

    The green polygon comes from the lateral-edge fit and can be much narrower
    than the reported width (50px vs 69px on one real frame), so an operator
    checking only the outline is not checking W at all.

    The span is derived from width_mm itself rather than from _nail_half: those
    two agree on some frames but not all (sh_middle.jpg reports width_mm 9.52mm
    while 2*_nail_half is 7.74mm), and a bracket that disagrees with its own
    label is worse than no bracket.
    """
    try:
        cx   = int(data["_tip_x"])
        half = int(round(float(data["width_mm"]) / mpp / 2.0))
        y    = int((float(data["_tip_y"]) + float(data["_cuticle_y"])) / 2)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return overlay

    x1, x2 = cx - half, cx + half
    cv2.line(overlay, (x1, y), (x2, y), (255, 0, 255), 3)
    for x in (x1, x2):
        cv2.line(overlay, (x, y - 18), (x, y + 18), (255, 0, 255), 3)
    put(overlay, f"W {data['width_mm']}mm", (x1, y - 28), (255, 0, 255), 1.0, 2)
    return overlay


def measure_frame(frame, finger, aruco_size_mm):
    """Run the full measurement on one frame.

    Returns a result dict. Never raises: a failed frame (no marker, no finger)
    is a normal event when the user is still positioning their hand.
    """
    try:
        with quiet():
            mpp, corners, marker_id = nm.detect_aruco(frame, aruco_size_mm)
            finger_mask, _, bbox    = nm.segment_finger(frame, corners)
            data = nm.measure_top(frame, mpp, finger_mask, bbox,
                                  aruco_corners=corners)

            data.update(nm.apply_wl_correction(finger,
                                               data["width_mm"],
                                               data["length_mm"]))
            data["aspect_ratio"] = round(
                data["width_mm"] / data["length_mm"], 3
            ) if data["length_mm"] else 0.0
            data["_ccurve_method"] = "brightness fallback"

            # Exposure of the lit side. A clipped highlight has no texture left
            # in it, so the groove shadow the width measurement depends on is
            # simply gone: at 77% clipped the fold coverage was 21%, while the
            # sh reference photos clip 0% and reach 58-84%. Coverage alone tells
            # the operator something is wrong; this tells them what to change.
            data["_clip_pct"] = clipped_fraction(frame, data)

            bad = unusable_values(data)
            if bad:
                return {"ok": False, "err": "UNUSABLE " + ", ".join(bad),
                        "t": time.time()}

            # Stashed for the accept path, which redraws the overlay after the
            # c-curve stage. Leading underscore keeps it out of the JSON.
            data["_mpp"] = mpp

            overlay = draw_width_marker(
                nm.draw_annotated(frame, data, corners, finger), data, mpp)

        return {"ok": True, "frame": frame, "data": data,
                "corners": corners, "overlay": overlay, "t": time.time()}

    except Exception as e:
        return {"ok": False, "err": str(e).split("\n")[0], "t": time.time()}


class MeasureWorker(threading.Thread):
    """Measures the most recent frame, over and over, in the background."""

    def __init__(self, finger, aruco_size_mm):
        super().__init__(daemon=True)
        self.finger     = finger
        self.aruco_size = aruco_size_mm
        self._lock      = threading.Lock()
        self._pending   = None      # newest frame waiting to be measured
        self._result    = None      # newest completed result
        self._stop      = threading.Event()

    def submit(self, frame):
        with self._lock:
            self._pending = frame

    def latest(self):
        with self._lock:
            return self._result

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            with self._lock:
                frame, self._pending = self._pending, None
            if frame is None:
                time.sleep(0.01)
                continue
            result = measure_frame(frame, self.finger, self.aruco_size)
            with self._lock:
                self._result = result


# ─────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────

def stability(history):
    """(is_stable, width_spread, length_spread) over the recent results."""
    if len(history) < STABLE_N:
        return False, None, None
    ws = [h[0] for h in history]
    ls = [h[1] for h in history]
    dw, dl = max(ws) - min(ws), max(ls) - min(ls)
    return (dw <= STABLE_TOL_MM and dl <= STABLE_TOL_MM), dw, dl


def put(img, text, xy, color, scale=0.62, thick=2):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thick, cv2.LINE_AA)


def hint_bar(view, text):
    """Append a hint strip below the image instead of drawing over it."""
    bar = np.zeros((34, view.shape[1], 3), np.uint8)
    put(bar, text, (14, 24), (0, 220, 255), 0.6)
    return np.vstack([view, bar])


def compose(result, live_frame, history, finger, n_measured, peak_w=None):
    """Build the window image: measured overlay + live PiP + status bar."""
    have = result is not None and result.get("ok")
    base = result["overlay"] if have else live_frame

    scale = DISPLAY_H / base.shape[0]
    view  = cv2.resize(base, (int(base.shape[1] * scale), DISPLAY_H))
    h, w  = view.shape[:2]

    # Live picture-in-picture so the user can position the finger while the
    # main image is up to a second behind.
    pip_w = w // 4
    pip   = cv2.resize(live_frame, (pip_w, int(live_frame.shape[0] * pip_w /
                                               live_frame.shape[1])))
    ph, pw = pip.shape[:2]
    view[10:10 + ph, w - pw - 10:w - 10] = pip
    cv2.rectangle(view, (w - pw - 10, 10), (w - 10, 10 + ph), (200, 200, 200), 2)
    put(view, "LIVE", (w - pw - 4, 30), (200, 200, 200), 0.5, 1)

    if not have:
        msg = "Measuring..." if result is None else f"NO READING: {result['err']}"
        cv2.rectangle(view, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
        put(view, msg, (14, 40), (0, 80, 255), 0.8)
        put(view, "Place finger + ArUco marker in view   |   Q: quit",
            (14, h - 20), (200, 200, 0), 0.6)
        return view

    d = result["data"]
    is_stable, dw, dl = stability(history)
    age = time.time() - result["t"]

    border = (0, 220, 0) if is_stable else (0, 200, 255)
    cv2.rectangle(view, (0, 0), (w - 1, h - 1), border, 8)

    # W/L is shown as a bare number, not as wl_ratio_check's ok/length_suspect
    # verdict: that verdict compares against a 0.91 prior, while ruler-validated
    # nails here measure 0.64-0.86, so it flags correct measurements as suspect
    # on essentially every real nail. An alarm that is always on would just
    # teach the operator to ignore this bar.
    wl = d.get("wl_ratio_check", {}).get("measured_wl", "?")
    rows = [
        (f"{finger.upper()}   W {d['width_mm']}mm   L {d['length_mm']}mm"
         f"   (corrected L {d['corrected_length_mm']}mm)", (255, 255, 255)),
        (f"C-curve {d['c_curve_mm']}mm   R {d['arc_radius_mm']}mm"
         f"   skin {d['skin_tone_hex']}   W/L {wl}", (255, 255, 255)),
    ]

    # Fold coverage: what fraction of the nail's rows the lateral-groove scan
    # actually found an edge on. width_mm is a fit over exactly those rows, so
    # this is a direct trust indicator — and it doubles as a live meter for
    # aiming the side light, which is what the groove signature needs.
    found, span = d.get("_lateral_rows"), d.get("_lateral_span")
    if found is not None and span:
        cov = 100.0 * found / span
        if cov >= 50:
            col, verdict = (0, 255, 0), "good"
        elif cov >= 25:
            col, verdict = (0, 200, 255), "thin"
        else:
            col, verdict = (0, 80, 255), "WIDTH UNRELIABLE"
        clip = d.get("_clip_pct")
        extra = ""
        if clip is not None:
            extra = f"   |   lit side blown out {clip:.0f}%"
            if clip >= 20:
                extra += " - DIM THE LIGHT (reference clips 0%)"
                col = (0, 80, 255)
        rows.append((f"fold coverage {cov:.0f}%  ({found} of {span} rows) - "
                     f"{verdict}{extra}", col))

    if peak_w is not None:
        short = peak_w - d["width_mm"]
        if short <= PEAK_TOL_MM:
            rows.append((f"SQUARE-ON  (W is at its peak {peak_w}mm)",
                         (0, 255, 0)))
        else:
            rows.append((f"ROLLED  (W is {short:.2f}mm under peak {peak_w}mm) "
                         f"- roll the finger until W stops rising",
                         (0, 200, 255)))
    if is_stable:
        rows.append((f"STABLE  (dW {dw:.2f}  dL {dl:.2f} over last {STABLE_N})",
                     (0, 255, 0)))
    else:
        detail = f"dW {dw:.2f}  dL {dl:.2f}" if dw is not None else "warming up"
        rows.append((f"UNSTABLE  ({detail}) - hold still, check the outline",
                     (0, 200, 255)))

    for i, (txt, col) in enumerate(rows):
        put(view, txt, (14, 34 + i * 30), col)

    put(view, f"frame age {age:.1f}s   accepted so far: {n_measured}",
        (14, h - 48), (180, 180, 180), 0.55, 1)
    put(view, "ENTER/SPACE: accept THIS measurement   |   R: reset   |   Q: quit",
        (14, h - 18), (0, 220, 255), 0.62)
    return view


# ─────────────────────────────────────────────────────────────
# Live stages
# ─────────────────────────────────────────────────────────────

def _log_row(result, finger, elapsed):
    """One CSV line for a completed measurement, good or rejected."""
    if not result["ok"]:
        reason = result.get("err", "?").replace(",", ";")
        return f"{elapsed:.2f},{finger},,,,,,,{reason}\n"
    d = result["data"]
    poly = d.get("nail_polygon_px") or []
    poly_px = (max(p[0] for p in poly) - min(p[0] for p in poly)) if poly else ""
    scan_px = round(2 * float(d["_nail_half"]), 1) if "_nail_half" in d else ""
    return (f"{elapsed:.2f},{finger},{d['width_mm']},{d['length_mm']},"
            f"{d['c_curve_mm']},{d['arc_radius_mm']},{scan_px},{poly_px},ok\n")


def live_top_stage(cap, finger, aruco_size_mm, log_path=None):
    """Live measure loop. Returns the accepted result dict, or None if quit.

    log_path : optional CSV to append every completed measurement to. The
        per-frame scatter is the thing worth measuring right now (pixel-level
        noise alone moves W by up to 0.8mm), and a log removes the need to read
        it off the screen and remember it.
    """
    worker = MeasureWorker(finger, aruco_size_mm)
    worker.start()

    window  = f"nail_live - {finger}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    history = deque(maxlen=STABLE_N)
    peaks   = deque(maxlen=PEAK_N)
    last_t  = 0.0
    accepted = None

    print(f"\n[Live] Measuring '{finger}' - accept with ENTER when the "
          f"outline looks right.")

    log = None
    if log_path:
        # One file per run. Appending to a shared file silently interleaved six
        # sessions of different fingers into one CSV, which made the scatter
        # statistics meaningless until they were split apart again.
        stamp     = time.strftime("%Y%m%d_%H%M%S")
        root, ext = os.path.splitext(log_path)
        log_path  = f"{root}_{stamp}{ext}"
        log = open(log_path, "w", buffering=1)
        log.write("time_s,finger,width_mm,length_mm,c_curve_mm,"
                  "arc_radius_mm,scan_px,poly_px,status\n")
        print(f"  [Log] writing every measurement to {log_path}")
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[Error] Cannot read from camera.")
            break

        worker.submit(frame)
        result = worker.latest()

        # Record each newly completed result for the stability check.
        if result is not None and result["t"] != last_t:
            last_t = result["t"]
            if result["ok"]:
                history.append((result["data"]["width_mm"],
                                result["data"]["length_mm"]))
                peaks.append(result["data"]["width_mm"])
            else:
                history.clear()

            if log is not None:
                log.write(_log_row(result, finger, time.time() - t0))

        cv2.imshow(window, compose(result, frame, history, finger, 0,
                                   peak_w=max(peaks) if peaks else None))
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        if key == ord('r'):
            history.clear()
            peaks.clear()
            print("  [Reset] stability and peak-width history cleared.")
        if key in (13, 10, ord(' ')):
            if result is not None and result["ok"]:
                accepted = result
                print(f"  [Accepted] W={result['data']['width_mm']}mm  "
                      f"L={result['data']['length_mm']}mm")
                break
            print("  [Ignored] no valid measurement on screen yet.")

    worker.stop()
    if log is not None:
        log.close()
    cv2.destroyWindow(window)
    return accepted


def live_ccurve_stage(cap, finger, width_mm, output_dir):
    """End-on C-curve stage. Returns the accepted ccurve dict, or None."""
    try:
        from measure_ccurve import measure_ccurve
    except ImportError:
        print("  [C-curve] measure_ccurve.py unavailable, skipping.")
        return None

    window = f"nail_live c-curve - {finger}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    # PNG, not JPEG: measure_ccurve reads from disk, and a lossy re-encode of
    # the frame shifts the sagitta by ~0.03mm for no reason.
    tmp_img   = os.path.join(output_dir, f"_{finger}_ccurve_live.png")
    debug_img = os.path.join(output_dir, f"{finger}_ccurve_debug.jpg")

    print(f"\n[Live] C-curve stage - point the fingertip AT the camera.")
    print("  ENTER: accept   |   S: skip (brightness fallback)   |   Q: quit")

    last_try  = 0.0
    cc, shown = None, None
    accepted  = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ~1 measurement per second; measure_ccurve reads from disk.
        if time.time() - last_try > 1.0:
            last_try = time.time()
            cv2.imwrite(tmp_img, frame)
            try:
                with quiet():
                    cc = measure_ccurve(tmp_img, width_mm=width_mm,
                                        debug_out=debug_img)
                dbg = cv2.imread(debug_img)
                shown = (cc, dbg, frame.copy()) if dbg is not None else None
            except Exception as e:
                cc, shown = None, None
                print(f"  [C-curve] {str(e).split(chr(10))[0]}")

        if shown is not None:
            cc_d, dbg, src = shown
            scale = DISPLAY_H / dbg.shape[0]
            view  = cv2.resize(dbg, (int(dbg.shape[1] * scale), DISPLAY_H))
            cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1),
                          (0, 220, 0), 8)
            put(view, f"C-curve {cc_d['c_curve_mm']}mm   "
                      f"R {cc_d['arc_radius_mm']}mm", (14, 38), (255, 255, 255), 0.75)
            view = hint_bar(view, "ENTER: accept   |   S: skip   |   Q: quit")
        else:
            scale = DISPLAY_H / frame.shape[0]
            view  = cv2.resize(frame, (int(frame.shape[1] * scale), DISPLAY_H))
            cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1),
                          (0, 0, 255), 8)
            put(view, "No C-curve reading - show the fingertip end-on",
                (14, 38), (0, 80, 255), 0.7)
            view = hint_bar(view, "S: skip   |   Q: quit")

        cv2.imshow(window, view)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27, ord('s')):
            break
        if key in (13, 10, ord(' ')) and shown is not None:
            accepted = shown[0]
            print(f"  [Accepted] C-curve h={accepted['c_curve_mm']}mm  "
                  f"R={accepted['arc_radius_mm']}mm")
            break

    cv2.destroyWindow(window)
    if os.path.isfile(tmp_img):
        os.remove(tmp_img)
    return accepted


# ─────────────────────────────────────────────────────────────
# Save + STL
# ─────────────────────────────────────────────────────────────

def save_results(results, aruco_size_mm, output_dir):
    payload   = nm.build_payload(results, aruco_size_mm)
    json_path = os.path.join(output_dir, "nail_measurements.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    profiles     = nm.build_profile(results)
    profile_path = os.path.join(output_dir, "profile.json")
    with open(profile_path, "w") as f:
        json.dump(profiles[0] if len(profiles) == 1 else profiles, f, indent=2)

    print(f"\n[Saved] {json_path}")
    print(f"[Saved] {profile_path}")
    return json_path


def generate_stl(json_path, finger, shapes, output_dir):
    stl_dir = os.path.join(output_dir, "stl")
    os.makedirs(stl_dir, exist_ok=True)
    made = []
    for shape in shapes:
        print(f"\n[STL] {shape} ...")
        r = subprocess.run([sys.executable,
                            os.path.join(BASE, "nail_exact_stl.py"),
                            "--input",  json_path,
                            "--shape",  shape,
                            "--finger", finger,
                            "--output", stl_dir], cwd=BASE)
        if r.returncode == 0:
            made.append(os.path.join(stl_dir, f"nail_{finger}_{shape}.stl"))
        else:
            print(f"  [Warning] STL failed for '{shape}', skipping.")
    return made


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

class Camera:
    """VideoCapture wrapper that applies a fixed rotation to every frame.

    The measurement pipeline assumes the finger points UP: the free edge is
    found above the cuticle and length is cuticle_y - tip_y. Feed it an
    upside-down frame and it does not fail, it silently returns nonsense
    (measured 13.94mm x 28.62mm on a nail that is really 10.07 x 13.85).

    Rotating at the single point of entry means the preview, the measurement,
    the accepted frame and the saved JPEG all see the same upright image, so
    nothing downstream needs to know how the camera is mounted.
    """

    _MODES = {90:  cv2.ROTATE_90_CLOCKWISE,
              180: cv2.ROTATE_180,
              270: cv2.ROTATE_90_COUNTERCLOCKWISE}

    def __init__(self, cap, rotate=0):
        self._cap = cap
        self._rot = self._MODES.get(rotate)

    def read(self):
        ok, frame = self._cap.read()
        if ok and self._rot is not None:
            frame = cv2.rotate(frame, self._rot)
        return ok, frame

    def release(self):
        self._cap.release()


def open_camera(index, rotate=0):
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open camera index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Camera] index {index}  {w}x{h}"
          f"{f'  rotate {rotate} deg' if rotate else ''}")
    if (w, h) != (CAM_WIDTH, CAM_HEIGHT):
        print(f"  [Warning] camera gave {w}x{h}, not {CAM_WIDTH}x{CAM_HEIGHT}. "
              f"Measurements are resolution-sensitive - results will differ "
              f"from photos taken at {CAM_WIDTH}x{CAM_HEIGHT}.")
    return Camera(cap, rotate)


def main():
    p = argparse.ArgumentParser(
        description="Live camera nail measurement with human confirmation")
    p.add_argument("--finger", default="index", choices=nm.FINGER_NAMES,
                   help="single finger to measure")
    p.add_argument("--fingers", nargs="+", default=None,
                   help="measure several fingers in one run, in this order")
    p.add_argument("--camera", type=int, default=0, help="camera index")
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                   help="rotate every frame by this many degrees clockwise. "
                        "Use 180 for an upside-down camera: the pipeline "
                        "assumes the finger points UP and returns silent "
                        "nonsense otherwise")
    p.add_argument("--aruco-size", type=float, default=20.0,
                   help="physical ArUco side length in mm")
    p.add_argument("--ccurve", action="store_true",
                   help="add the end-on C-curve stage after each finger")
    p.add_argument("--shape", nargs="*", default=[], choices=SHAPES,
                   help="also generate STL(s) for these shapes")
    p.add_argument("--output", default="out_live", help="output directory")
    p.add_argument("--log-frames", action="store_true",
                   help="append every completed measurement to "
                        "{output}/{finger}_frames.csv, for measuring how much "
                        "the reading scatters frame to frame")
    p.add_argument("--debug", action="store_true",
                   help="verbose detector diagnostics (read by nail_measurer)")
    p.add_argument("--no-lateral", action="store_true",
                   help="disable side-lit lateral fold-edge detection")
    args = p.parse_args()

    _ensure_filter()   # install before any worker thread starts

    fingers = args.fingers if args.fingers else [args.finger]
    bad = [f for f in fingers if f not in nm.FINGER_NAMES]
    if bad:
        sys.exit(f"ERROR: unknown finger(s): {bad}")

    output_dir = args.output if os.path.isabs(args.output) \
        else os.path.join(BASE, args.output)
    os.makedirs(output_dir, exist_ok=True)

    cap = open_camera(args.camera, args.rotate)
    results = []

    try:
        for i, finger in enumerate(fingers):
            print(f"\n{'='*55}\n  [{i+1}/{len(fingers)}] {finger.upper()}\n{'='*55}")

            log_path = os.path.join(output_dir, f"{finger}_frames.csv") \
                if args.log_frames else None
            accepted = live_top_stage(cap, finger, args.aruco_size, log_path)
            if accepted is None:
                print(f"  [Skipped] {finger} - no measurement accepted.")
                continue

            data = accepted["data"]

            if args.ccurve:
                cc = live_ccurve_stage(cap, finger,
                                       data["width_mm"], output_dir)
                if cc:
                    data["c_curve_mm"]     = cc["c_curve_mm"]
                    data["arc_radius_mm"]  = cc["arc_radius_mm"]
                    data["_ccurve_method"] = "end-on photo (live)"

            # Save the exact frame that was accepted, and its overlay, so the
            # saved annotated JPG matches what the human approved.
            # PNG, not JPEG: the measurement is chaotically sensitive to
            # pixel-level noise. Re-encoding ys_index.jpg moved W by -0.06mm at
            # q95 and -0.79mm at q100, while a PNG round-trip reproduced 10.07mm
            # exactly. A JPEG archive would not re-measure to the value that was
            # actually approved.
            raw_path = os.path.join(output_dir, f"{finger}.png")
            cv2.imwrite(raw_path, accepted["frame"])

            # Width marker included, so reviewing the JPEG later shows exactly
            # what was approved on screen — the green outline alone is ~28%
            # narrower than the span width_mm reports.
            vis   = draw_width_marker(
                nm.draw_annotated(accepted["frame"], data,
                                  accepted["corners"], finger),
                data, data.get("_mpp"))
            scale = 900 / vis.shape[0]
            ann_path = os.path.join(output_dir, f"{finger}_annotated.jpg")
            cv2.imwrite(ann_path,
                        cv2.resize(vis, (int(vis.shape[1] * scale), 900)))
            print(f"  [Saved] {raw_path}")
            print(f"  [Saved] {ann_path}")

            results.append({"finger": finger, **data})
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not results:
        sys.exit("\nNothing accepted - no output written.")

    json_path = save_results(results, args.aruco_size, output_dir)

    if args.shape:
        for r in results:
            generate_stl(json_path, r["finger"], args.shape, output_dir)

    print(f"\n{'='*55}")
    print(f"  Done. Accepted: {[r['finger'] for r in results]}")
    print(f"  Results: {output_dir}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
