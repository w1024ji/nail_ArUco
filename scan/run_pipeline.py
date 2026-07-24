"""
run_pipeline.py
---------------
End-to-end nail pipeline: crop -> measure -> generate STL -> upload to S3

Usage (finger mode, default):
    python run_pipeline.py --userid u001 --session 1 --hand right --shape round

Usage (hand mode — single full-hand photo):
    python run_pipeline.py --userid u001 --session 1 --hand right --shape round --mode hand

Finger mode — photos must be placed in:
    photos/{userid}/{session}/{hand}/thumb.jpg
    photos/{userid}/{session}/{hand}/index.jpg
    photos/{userid}/{session}/{hand}/middle.jpg
    photos/{userid}/{session}/{hand}/ring.jpg
    photos/{userid}/{session}/{hand}/pinky.jpg

Hand mode — one photo:
    photos/{userid}/{session}/{hand}/hand.jpg

Output structure (finger mode):
    results/{userid}/{session}/{hand}/thumb/
        nail_measurements.json
        profile.json
        thumb_annotated.jpg
    results/{userid}/{session}/{hand}/index/  ...
    results/{userid}/{session}/{hand}/stl/
        nail_thumb_{shape}.stl
        nail_index_{shape}.stl
        nail_middle_{shape}.stl
        nail_ring_{shape}.stl
        nail_pinky_{shape}.stl

Output structure (hand mode):
    results/{userid}/{session}/{hand}/
        nail_measurements.json
        profile.json
        hand_annotated.jpg
        index/index_annotated.jpg
        middle/middle_annotated.jpg
        ring/ring_annotated.jpg
        pinky/pinky_annotated.jpg
        thumb/thumb_annotated.jpg   (if thumb detected)
        stl/
            nail_index_{shape}.stl  ...
"""

import argparse
import json
import os
import sys
import subprocess

import cv2

from s3_upload import upload_session

FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
SHAPES       = ("round", "oval", "almond", "square", "stiletto", "ballerina")
BASE         = os.path.dirname(os.path.abspath(__file__))


def crop_image(src_path: str, crop_fraction: float) -> str:
    img = cv2.imread(src_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open: {src_path}")
    h, w = img.shape[:2]
    cut  = int(h * (1.0 - crop_fraction))
    cropped = img[:cut, :]
    out_path = src_path.replace(".jpg", "_cropped.jpg")
    cv2.imwrite(out_path, cropped)
    print(f"  Cropped: {h}x{w} -> {cut}x{w}  ->  {out_path}")
    return out_path


def run(cmd: list):
    result = subprocess.run(cmd, cwd=BASE)
    if result.returncode != 0:
        sys.exit(f"ERROR: command failed: {' '.join(cmd)}")


def main():
    p = argparse.ArgumentParser(description="Nail pipeline: crop -> measure -> STL -> S3")
    p.add_argument("--userid",        required=True,
                   help="User ID (provided by Seha's backend)")
    p.add_argument("--session",       required=True,
                   help="Session number (provided by Seha's backend)")
    p.add_argument("--hand",          default="right", choices=["right", "left"],
                   help="Which hand: right or left (default: right)")
    p.add_argument("--shape",         default="round", choices=SHAPES,
                   help="Nail tip shape (default: round)")
    p.add_argument("--aruco-size",    type=float, default=20.0,
                   help="Physical ArUco marker side length in mm (default: 20)")
    p.add_argument("--crop-fraction", type=float, default=0.30,
                   help="Fraction of image height to remove from bottom (default: 0.30)")
    p.add_argument("--mode", default="finger", choices=["finger", "hand"],
                   help="'finger': 5 separate photos (default); 'hand': single full-hand photo")
    args = p.parse_args()

    # ── Resolve folder paths ──────────────────────────────────────
    photos_root  = os.path.join(BASE, "photos",  args.userid, args.session, args.hand)
    results_root = os.path.join(BASE, "results", args.userid, args.session, args.hand)

    if args.mode == "finger":
        # ── Verify all five photos exist before starting ──────────
        photo_paths = {}
        missing = []
        for finger in FINGER_ORDER:
            path = os.path.join(photos_root, f"{finger}.jpg")
            if not os.path.isfile(path):
                missing.append(path)
            else:
                photo_paths[finger] = path
        if missing:
            sys.exit(
                "ERROR: Missing photo(s):\n" +
                "\n".join(f"  {p}" for p in missing) +
                f"\n\nExpected folder: photos/{args.userid}/{args.session}/{args.hand}/"
            )

        # STL folder is shared across all five fingers (one shape per session)
        stl_dir = os.path.join(results_root, "stl")
        os.makedirs(stl_dir, exist_ok=True)

        for i, finger in enumerate(FINGER_ORDER):
            photo = photo_paths[finger]
            print(f"\n{'='*60}")
            print(f"  [{i+1}/5] {finger.upper()}")
            print(f"{'='*60}")

            # ── Step 1: Crop ──────────────────────────────────────
            print("\n[Step 1] Crop")
            cropped_path = crop_image(photo, args.crop_fraction)

            # ── Step 2: Measure ───────────────────────────────────
            print("\n[Step 2] Measure")
            finger_out = os.path.join(results_root, finger)
            os.makedirs(finger_out, exist_ok=True)

            measure_cmd = [
                sys.executable,
                os.path.join(BASE, "nail_measurer.py"),
                "--top",        cropped_path,
                "--finger",     finger,
                "--aruco-size", str(args.aruco_size),
                "--output",     finger_out,
            ]

            # Auto-detect end-on C-curve photo: {finger}_ccurve.jpg
            # placed in the same folder as the top photo.
            ccurve_photo = os.path.join(photos_root, f"{finger}_ccurve.jpg")
            if os.path.isfile(ccurve_photo):
                print(f"  [C-curve] end-on photo found: {ccurve_photo}")
                measure_cmd += ["--ccurve-top", ccurve_photo]
            else:
                print(f"  [C-curve] no end-on photo "
                      f"({finger}_ccurve.jpg) → brightness fallback")

            run(measure_cmd)

            # ── Step 3: Generate STL ──────────────────────────────
            print(f"\n[Step 3] Generate STL ({args.shape})")
            json_path = os.path.join(finger_out, "nail_measurements.json")
            run([
                sys.executable,
                os.path.join(BASE, "nail_exact_stl.py"),
                "--input",  json_path,
                "--shape",  args.shape,
                "--finger", finger,
                "--output", stl_dir,
            ])

        print(f"\n{'='*60}")
        print("All done!")
        print(f"Results: results/{args.userid}/{args.session}/{args.hand}/")
        print(f"{'='*60}")

        # ── Upload this session to S3 ─────────────────────────────
        upload_session(args.userid, args.session, args.hand, BASE)

    elif args.mode == "hand":
        # ── Verify single hand photo exists ──────────────────────
        hand_photo = os.path.join(photos_root, "hand.jpg")
        if not os.path.isfile(hand_photo):
            sys.exit(
                f"ERROR: Missing hand photo:\n  {hand_photo}\n\n"
                f"Expected: photos/{args.userid}/{args.session}/{args.hand}/hand.jpg"
            )

        # ── Step 1: Crop bottom 30% ───────────────────────────────
        print(f"\n{'='*60}")
        print("  [Hand Mode] Step 1: Crop")
        print(f"{'='*60}")
        cropped_path = crop_image(hand_photo, args.crop_fraction)

        # ── Step 2: Measure all fingers via hand_measurer.py ─────
        print(f"\n{'='*60}")
        print("  [Hand Mode] Step 2: Measure (hand_measurer.py)")
        print(f"{'='*60}")
        os.makedirs(results_root, exist_ok=True)
        run([
            sys.executable,
            os.path.join(BASE, "hand_measurer.py"),
            "--image",      cropped_path,
            "--aruco-size", str(args.aruco_size),
            "--output",     results_root,
        ])

        # ── Step 3: Read output JSON to find measured fingers ─────
        json_path = os.path.join(results_root, "nail_measurements.json")
        if not os.path.isfile(json_path):
            sys.exit(f"ERROR: hand_measurer did not produce {json_path}")

        with open(json_path) as f:
            payload = json.load(f)

        measured_fingers = [nail["finger"] for nail in payload.get("nails", [])]
        if not measured_fingers:
            sys.exit("ERROR: nail_measurements.json contains no measured fingers.")

        print(f"\n  Measured fingers: {measured_fingers}")

        # ── Step 4: Generate STL for each measured finger ─────────
        print(f"\n{'='*60}")
        print(f"  [Hand Mode] Step 3: Generate STLs ({args.shape})")
        print(f"{'='*60}")
        stl_dir = os.path.join(results_root, "stl")
        os.makedirs(stl_dir, exist_ok=True)

        for finger in measured_fingers:
            print(f"\n  --- STL for {finger.upper()} ---")
            run([
                sys.executable,
                os.path.join(BASE, "nail_exact_stl.py"),
                "--input",  json_path,
                "--shape",  args.shape,
                "--finger", finger,
                "--output", stl_dir,
            ])

        print(f"\n{'='*60}")
        print("All done!")
        print(f"Results: results/{args.userid}/{args.session}/{args.hand}/")
        print(f"{'='*60}")

        # ── Upload this session to S3 ─────────────────────────────
        upload_session(args.userid, args.session, args.hand, BASE)


if __name__ == "__main__":
    main()