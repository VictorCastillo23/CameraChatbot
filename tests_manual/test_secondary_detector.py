"""Manual verification for Fase 6 (PR10)'s `detectors.secondary_detector`.

Not a pytest test: this repo has no test runner, so this is a plain
runnable script using bare ``assert`` statements, printing a PASS/FAIL
summary and exiting 0/1. Run it directly:

    python tests_manual/test_secondary_detector.py

What it verifies:

1. Unconfigured (`model_path=None`, the shipped default via
   `security_config.WEAPONS_DETECTOR`): `AllowlistedDetector` is a genuine
   no-op -- exactly one warning printed at construction, `.enabled is
   False`, `run_on_json()` returns its input path unchanged, no file
   written to disk.
2. Configured: wraps `YOLOOnnxDetector` (monkeypatched -- no real ONNX
   weights needed, same pattern as `test_enroll_person.py`) and appends
   `kind="object"` entries only for classes in the configured allowlist.
3. The documented empty-`class_names` edge case on a *configured* detector:
   zero entries ever pass -- this is intentional, not a bug, and gets its
   own test so it is never mistaken for a regression later.
4. `pipeline_service.build_detail_detectors()` wiring: excludes the
   detector when `WEAPONS_DETECTOR["model_path"] is None` (today's
   default) and includes exactly one instance when configured.

Uses only synthetic in-memory fakes and tiny throwaway images written to a
temp dir -- no real ONNX weights, no live models.
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

import camerachatbot.detectors.secondary_detector as secondary_detector_mod  # noqa: E402
from camerachatbot.detectors.secondary_detector import AllowlistedDetector  # noqa: E402

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
# Shared fakes: fake YOLOOnnxDetector result shape (no real ONNX weights)
# ---------------------------------------------------------------------------

class _FakeBox:
    def __init__(self, cls_id, conf, xyxy):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [xyxy]


class _FakeDetResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class _FakeYOLOOnnxDetector:
    """Returns one allowlisted ("knife") and one non-allowlisted ("chair")
    box per call, enough to exercise the allowlist filter without real
    ONNX weights."""

    def __init__(self, model_path, providers=None, conf=0.35):
        self.model_path = model_path
        self.conf = conf

    def __call__(self, img, verbose=False):
        h, w = img.shape[:2]
        knife_box = _FakeBox(0, 0.9, [0.0, 0.0, float(w) / 2, float(h) / 2])
        chair_box = _FakeBox(1, 0.8, [float(w) / 2, float(h) / 2, float(w), float(h)])
        names = {0: "knife", 1: "chair"}
        return [_FakeDetResult([knife_box, chair_box], names)]


def _write_dummy_image(frames_dir, frame_id):
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    path = os.path.join(frames_dir, f"{frame_id}.jpg")
    cv2.imwrite(path, img)
    return path


def _write_input_json(res_dir, frame_id):
    data = {frame_id: [{"kind": "person", "bbox": [1, 1, 10, 10], "attributes": {}}]}
    path = os.path.join(res_dir, "tracking_input.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path, data


# ---------------------------------------------------------------------------
# 1. Unconfigured -> genuine no-op
# ---------------------------------------------------------------------------

def test_unconfigured_detector_is_noop_with_one_warning():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        det = AllowlistedDetector()
    printed_lines = [line for line in buf.getvalue().splitlines() if line.strip()]
    assert len(printed_lines) == 1, f"expected exactly one warning, got: {printed_lines!r}"

    assert det.enabled is False

    with tempfile.TemporaryDirectory(prefix="secondary_detector_noop_") as tmpdir:
        input_path = os.path.join(tmpdir, "some_path.json")
        with open(input_path, "w", encoding="utf-8") as f:
            f.write("{}")

        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            out_path = det.run_on_json(input_path)
        assert out_path == input_path, f"expected unchanged input path, got: {out_path}"
        assert not buf2.getvalue().strip(), "run_on_json() must print nothing when unconfigured"

        # No new file was created alongside the input.
        created = set(os.listdir(tmpdir)) - {"some_path.json"}
        assert not created, f"run_on_json() must not write any file when unconfigured: {created}"


# ---------------------------------------------------------------------------
# 2. Configured -> wraps YOLOOnnxDetector, filters by allowlist
# ---------------------------------------------------------------------------

def _with_fake_yolo_onnx_detector(fn):
    original = secondary_detector_mod.YOLOOnnxDetector
    secondary_detector_mod.YOLOOnnxDetector = _FakeYOLOOnnxDetector
    try:
        return fn()
    finally:
        secondary_detector_mod.YOLOOnnxDetector = original


def test_configured_detector_wraps_yoloonnxdetector_and_filters_by_allowlist():
    def _call():
        with tempfile.TemporaryDirectory(prefix="secondary_detector_configured_") as tmpdir:
            frames_dir = os.path.join(tmpdir, "frames")
            res_dir = os.path.join(tmpdir, "res")
            os.makedirs(frames_dir)
            os.makedirs(res_dir)

            frame_id = "frame_001"
            _write_dummy_image(frames_dir, frame_id)
            input_path, original_data = _write_input_json(res_dir, frame_id)

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                det = AllowlistedDetector(model_path="fake.onnx", class_names={"knife"})
            det.frames_folder = frames_dir

            assert det.enabled is True

            out_path = det.run_on_json(input_path)
            assert out_path == os.path.join(res_dir, "tracking_weapons_6.json"), out_path

            with open(out_path, "r", encoding="utf-8") as f:
                out_data = json.load(f)

            entries = out_data[frame_id]
            object_entries = [e for e in entries if e.get("kind") == "object"]
            assert len(object_entries) == 1, f"expected 1 allowlisted object entry, got {object_entries!r}"
            assert object_entries[0]["class_name"] == "knife"

            # Original entries (the "person" entry) must be untouched.
            person_entries = [e for e in entries if e.get("kind") == "person"]
            assert person_entries == original_data[frame_id]

    _with_fake_yolo_onnx_detector(_call)


# ---------------------------------------------------------------------------
# 3. Configured but empty class_names -> zero entries appended (documented
#    intentional edge case, not a bug)
# ---------------------------------------------------------------------------

def test_empty_class_names_configured_detector_appends_nothing():
    def _call():
        with tempfile.TemporaryDirectory(prefix="secondary_detector_empty_allowlist_") as tmpdir:
            frames_dir = os.path.join(tmpdir, "frames")
            res_dir = os.path.join(tmpdir, "res")
            os.makedirs(frames_dir)
            os.makedirs(res_dir)

            frame_id = "frame_001"
            _write_dummy_image(frames_dir, frame_id)
            input_path, _ = _write_input_json(res_dir, frame_id)

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                det = AllowlistedDetector(model_path="fake.onnx", class_names=())
            det.frames_folder = frames_dir

            assert det.enabled is True

            out_path = det.run_on_json(input_path)
            with open(out_path, "r", encoding="utf-8") as f:
                out_data = json.load(f)

            object_entries = [e for e in out_data[frame_id] if e.get("kind") == "object"]
            assert object_entries == [], (
                f"empty class_names on a configured detector must append zero entries, "
                f"got: {object_entries!r}"
            )

    _with_fake_yolo_onnx_detector(_call)


# ---------------------------------------------------------------------------
# 4. `pipeline_service.build_detail_detectors()` wiring
# ---------------------------------------------------------------------------

class _FakePoseCls:
    """Minimal stand-in for `PoseClsOnnxClassifier` -- `PoseActionClassifier`'s
    constructor only reads `.names` off it, no real ONNX weights needed."""
    names = {0: "sentado", 1: "de pie"}


def test_build_detail_detectors_excludes_allowlisted_detector_when_unconfigured():
    import camerachatbot.pipeline.pipeline_service as pipeline_service_mod

    runtime = {"yolo_posecls": _FakePoseCls()}
    detectors = pipeline_service_mod.build_detail_detectors(runtime, "unused_frames_folder")
    assert not any(isinstance(d, AllowlistedDetector) for d in detectors), (
        "AllowlistedDetector must not be built when WEAPONS_DETECTOR['model_path'] is None"
    )


def test_build_detail_detectors_includes_allowlisted_detector_when_configured():
    import camerachatbot.pipeline.pipeline_service as pipeline_service_mod

    original_weapons_detector = pipeline_service_mod.WEAPONS_DETECTOR
    pipeline_service_mod.WEAPONS_DETECTOR = {
        "model_path": "fake.onnx", "class_names": ("knife",), "conf": 0.35,
    }
    original_onnx = secondary_detector_mod.YOLOOnnxDetector
    secondary_detector_mod.YOLOOnnxDetector = _FakeYOLOOnnxDetector
    try:
        runtime = {"yolo_posecls": _FakePoseCls()}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            detectors = pipeline_service_mod.build_detail_detectors(runtime, "unused_frames_folder")
        matches = [d for d in detectors if isinstance(d, AllowlistedDetector)]
        assert len(matches) == 1, f"expected exactly one AllowlistedDetector, got {matches!r}"
    finally:
        pipeline_service_mod.WEAPONS_DETECTOR = original_weapons_detector
        secondary_detector_mod.YOLOOnnxDetector = original_onnx


def main():
    check("AllowlistedDetector(): unconfigured is a genuine no-op (one warning)",
          test_unconfigured_detector_is_noop_with_one_warning)
    check("AllowlistedDetector(): configured wraps YOLOOnnxDetector, filters by allowlist",
          test_configured_detector_wraps_yoloonnxdetector_and_filters_by_allowlist)
    check("AllowlistedDetector(): empty class_names on configured detector appends nothing",
          test_empty_class_names_configured_detector_appends_nothing)
    check("build_detail_detectors(): excludes AllowlistedDetector when unconfigured",
          test_build_detail_detectors_excludes_allowlisted_detector_when_unconfigured)
    check("build_detail_detectors(): includes AllowlistedDetector when configured",
          test_build_detail_detectors_includes_allowlisted_detector_when_configured)

    print("\n=== Fase 6 (PR10) verification: secondary_detector.py ===")
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
