"""Manual verification for the optional debugging helper
`camerachatbot.debugging.box_review`.

Not a pytest test: this repo has no test runner, so this is a plain
runnable script using bare ``assert`` statements, printing a PASS/FAIL
summary and exiting 0/1. Run it directly:

    python tests_manual/test_box_review.py

What it verifies (only `save_annotated_and_crops` — `review_interactive`
needs a real display and cannot be exercised headlessly, same documented
limitation as `geometry/calibrate_camera.py`):

1. Folder-per-frame output structure: each real frame_id gets its own
   `<output_dir>/<frame_id>/overview.jpg`.
2. Crop count matches the number of entries for a frame, in JSON order,
   named `box_000_...`, `box_001_...`, etc.
3. The `"neighborhood"` pseudo-frame key is skipped entirely (no output
   folder, no manifest entry) — same convention as
   `hand_detector.py`/`face_detector.py`/`secondary_detector.py`.
4. A frame_id with no locatable image is handled gracefully: no exception,
   omitted from the manifest, one `[WARN]` printed.
5. A frame_id mapped to an empty entry list still gets an `overview.jpg`
   and stays in the manifest with `"boxes": []` — distinct from case 4.
6. Crop dimensions match the *clamped* bbox (not the raw, possibly
   out-of-bounds one).

Uses only synthetic in-memory images written to a temp dir — no real
models, no real keyframes.
"""

import contextlib
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from camerachatbot.debugging.box_review import save_annotated_and_crops  # noqa: E402

results = []  # list[tuple[str, bool, str]]


def check(name, fn):
    try:
        fn()
        results.append((name, True, ""))
    except AssertionError as e:
        results.append((name, False, f"AssertionError: {e}"))
    except Exception as e:  # noqa: BLE001 - report any unexpected exception as a failure
        results.append((name, False, f"{type(e).__name__}: {e}"))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _write_dummy_image(frames_dir, frame_id, size=(64, 64)):
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    path = os.path.join(frames_dir, f"{frame_id}.jpg")
    cv2.imwrite(path, img)
    return path


def _entry(kind="person", class_name="person", bbox=(1, 1, 10, 10), confidence=0.9):
    return {
        "kind": kind,
        "class_name": class_name,
        "bbox": list(bbox),
        "confidence": confidence,
        "attributes": {},
    }


def _write_json(tmpdir, data):
    path = os.path.join(tmpdir, "input.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


# ---------------------------------------------------------------------------
# 1. Folder-per-frame structure
# ---------------------------------------------------------------------------

def test_folder_per_frame_structure():
    with tempfile.TemporaryDirectory(prefix="box_review_structure_") as tmp:
        frames_dir = os.path.join(tmp, "frames")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(frames_dir)

        for fid in ("100", "200"):
            _write_dummy_image(frames_dir, fid)
        data = {
            "100": [_entry()],
            "200": [_entry(kind="object", class_name="backpack")],
        }
        json_path = _write_json(tmp, data)

        manifest = save_annotated_and_crops(json_path, frames_dir, out_dir)

        assert set(manifest.keys()) == {"100", "200"}, manifest.keys()
        for fid in ("100", "200"):
            overview_path = os.path.join(out_dir, fid, "overview.jpg")
            assert os.path.exists(overview_path), overview_path
            img = cv2.imread(overview_path)
            assert img is not None, f"overview.jpg for {fid} is not a valid image"
            assert manifest[fid]["overview"] == overview_path


# ---------------------------------------------------------------------------
# 2. Crop count matches box count, in JSON order
# ---------------------------------------------------------------------------

def test_crop_count_matches_box_count():
    with tempfile.TemporaryDirectory(prefix="box_review_crops_") as tmp:
        frames_dir = os.path.join(tmp, "frames")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(frames_dir)

        fid = "300"
        _write_dummy_image(frames_dir, fid)
        entries = [
            _entry(kind="person", class_name="person", bbox=(0, 0, 20, 20)),
            _entry(kind="object", class_name="backpack", bbox=(20, 20, 40, 40)),
            _entry(kind="object", class_name="knife", bbox=(40, 40, 60, 60)),
        ]
        json_path = _write_json(tmp, {fid: entries})

        manifest = save_annotated_and_crops(json_path, frames_dir, out_dir)

        boxes = manifest[fid]["boxes"]
        assert len(boxes) == 3, boxes
        expected_names = ["box_000_person.jpg", "box_001_backpack.jpg", "box_002_knife.jpg"]
        for expected_name, box in zip(expected_names, boxes):
            assert os.path.basename(box["path"]) == expected_name, box["path"]
            assert os.path.exists(box["path"])
            assert cv2.imread(box["path"]) is not None


# ---------------------------------------------------------------------------
# 3. "neighborhood" pseudo-key is skipped
# ---------------------------------------------------------------------------

def test_neighborhood_key_is_skipped():
    with tempfile.TemporaryDirectory(prefix="box_review_neighborhood_") as tmp:
        frames_dir = os.path.join(tmp, "frames")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(frames_dir)

        fid = "400"
        _write_dummy_image(frames_dir, fid)
        data = {
            fid: [_entry()],
            "neighborhood": {"some": "unrelated structure, not a list of entries"},
        }
        json_path = _write_json(tmp, data)

        manifest = save_annotated_and_crops(json_path, frames_dir, out_dir)

        assert "neighborhood" not in manifest, manifest.keys()
        assert not os.path.isdir(os.path.join(out_dir, "neighborhood"))
        assert fid in manifest


# ---------------------------------------------------------------------------
# 4. Missing frame image -> graceful skip, one [WARN]
# ---------------------------------------------------------------------------

def test_missing_frame_image_handled_gracefully():
    with tempfile.TemporaryDirectory(prefix="box_review_missing_") as tmp:
        frames_dir = os.path.join(tmp, "frames")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(frames_dir)
        # No image written for "500" at all.
        json_path = _write_json(tmp, {"500": [_entry()]})

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            manifest = save_annotated_and_crops(json_path, frames_dir, out_dir)

        assert "500" not in manifest, manifest.keys()
        assert not os.path.isdir(os.path.join(out_dir, "500"))
        printed = buf.getvalue()
        assert "[WARN]" in printed, f"expected a [WARN] print, got: {printed!r}"


# ---------------------------------------------------------------------------
# 5. Zero-box frame still gets an overview, is NOT dropped
# ---------------------------------------------------------------------------

def test_zero_box_frame_still_gets_overview():
    with tempfile.TemporaryDirectory(prefix="box_review_zero_box_") as tmp:
        frames_dir = os.path.join(tmp, "frames")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(frames_dir)

        fid = "600"
        _write_dummy_image(frames_dir, fid)
        json_path = _write_json(tmp, {fid: []})

        manifest = save_annotated_and_crops(json_path, frames_dir, out_dir)

        assert fid in manifest, "a zero-entry frame must not be dropped from the manifest"
        assert manifest[fid]["boxes"] == []
        overview_path = os.path.join(out_dir, fid, "overview.jpg")
        assert os.path.exists(overview_path)
        assert cv2.imread(overview_path) is not None


# ---------------------------------------------------------------------------
# 6. Crop dimensions match the clamped bbox, not the raw one
# ---------------------------------------------------------------------------

def test_crop_bbox_matches_clamped_bbox_dimensions():
    with tempfile.TemporaryDirectory(prefix="box_review_clamp_") as tmp:
        frames_dir = os.path.join(tmp, "frames")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(frames_dir)

        fid = "700"
        _write_dummy_image(frames_dir, fid, size=(64, 64))
        # bbox extends well past the 64x64 image bounds on both axes.
        entries = [_entry(bbox=(50, 50, 200, 200))]
        json_path = _write_json(tmp, {fid: entries})

        manifest = save_annotated_and_crops(json_path, frames_dir, out_dir)

        box = manifest[fid]["boxes"][0]
        crop = cv2.imread(box["path"])
        assert crop is not None
        # Clamped to image bounds: x2<=64, y2<=64 -> width/height == 14 (64-50).
        h, w = crop.shape[:2]
        assert (w, h) == (14, 14), f"expected crop clamped to (14, 14), got {(w, h)}"


def main():
    check("save_annotated_and_crops(): folder-per-frame structure", test_folder_per_frame_structure)
    check("save_annotated_and_crops(): crop count matches box count, JSON order",
          test_crop_count_matches_box_count)
    check("save_annotated_and_crops(): \"neighborhood\" pseudo-key is skipped",
          test_neighborhood_key_is_skipped)
    check("save_annotated_and_crops(): missing frame image handled gracefully",
          test_missing_frame_image_handled_gracefully)
    check("save_annotated_and_crops(): zero-box frame still gets an overview",
          test_zero_box_frame_still_gets_overview)
    check("save_annotated_and_crops(): crop dimensions match the clamped bbox",
          test_crop_bbox_matches_clamped_bbox_dimensions)

    print("\n=== box_review.py verification ===")
    n_pass = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        line = f"[{status}] {name}"
        if not ok:
            line += f" — {detail}"
        print(line)

    print(f"\n{n_pass}/{len(results)} checks passed.")
    if n_pass != len(results):
        print("SUMMARY: FAIL")
        return 1
    print("SUMMARY: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
