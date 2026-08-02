"""Manual verification for the Fase 1 attribute-gating / allowlist contract.

Not a pytest test: this repo has no test runner, so this is a plain
runnable script using bare ``assert`` statements, printing a PASS/FAIL
summary and exiting 0/1. Run it directly:

    python tests_manual/test_fase1_attribute_gating.py

What it verifies (PR2 / Fase 1 tasks 1.1-1.5):

1.1 `pipeline_service.build_detail_detectors()` only builds detectors whose
    `security_config.DETECTOR_FLAGS` entry is True. With this repo's current
    flags (pose=True, everything else False), only `PoseActionClassifier`
    should be built — `FaceDetector`/`HandDetector` (which import mediapipe,
    not installed in this sandbox) must NOT even be instantiated.
1.2 `bootstrap.load_face_attr_sessions()` skips loading the emotion/age ONNX
    sessions when both `DETECTOR_FLAGS["emotion"]` and `["age"]` are False —
    verified for real (not reimplemented): this sandbox has no
    `models/emotion-ferplus-8.onnx` / `age_googlenet.onnx`, so if the gate
    were missing this call would raise `FileNotFoundError`.
1.3 `FaceAttributesDetector` logs a warning at construction when
    emotion/age is enabled but `face_attention` is not (the silent
    label=None/confidence=0.0 risk), and stays silent otherwise.
1.4 `YOLOPersonReID.detect_and_embed()` discards (does not just hide)
    detections whose class name is outside `COCO_ALLOWLIST` — verified via
    a fake YOLO model over real images from `keyFrames/` (no network, no
    real model weights needed).
1.5 `formatter.reformat_to_video_schema_uniform()` no longer emits an
    `object.metadata` `depth` node, even when the source JSON carries a
    `depth` dict on an entry.

Uses fakes/synthetic fixtures throughout — never touches the repo's real
`models/`, `gallery.index`/`id_map.json`/`proto_store.npy`, or a live
Postgres connection (see the "Identity persistence gotcha" note in
CLAUDE.md).
"""

import io
import json
import os
import sys
import tempfile
import contextlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camerachatbot import paths  # noqa: E402
from camerachatbot.security_config import DETECTOR_FLAGS, COCO_ALLOWLIST  # noqa: E402
from camerachatbot.detectors.person_reid import YOLOPersonReID  # noqa: E402
from camerachatbot.detectors.face_attributes_detector import FaceAttributesDetector  # noqa: E402
from camerachatbot.pipeline.pipeline_service import build_detail_detectors  # noqa: E402
from camerachatbot.video_schema.formatter import reformat_to_video_schema_uniform  # noqa: E402

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
# 1.1 build_detail_detectors() honors DETECTOR_FLAGS
# ---------------------------------------------------------------------------

class _FakeYoloPoseCls:
    """Stand-in for the ultralytics YOLO pose-cls model PoseActionClassifier
    expects: only needs `.to()`, `.fuse()`, `.names` to satisfy its ctor."""
    names = {0: "sit", 1: "stand"}

    def to(self, device):
        return self

    def fuse(self):
        pass


def test_build_detail_detectors_honors_flags():
    assert DETECTOR_FLAGS["pose"] is True
    assert DETECTOR_FLAGS["face_attention"] is False
    assert DETECTOR_FLAGS["hands"] is False
    assert DETECTOR_FLAGS["emotion"] is False
    assert DETECTOR_FLAGS["age"] is False

    fake_runtime = {
        "yolo_posecls": _FakeYoloPoseCls(),
        "emotion": {"sess": None, "input": None, "output": None},
        "age": {"sess": None, "input": None, "output": None},
    }
    detectors = build_detail_detectors(fake_runtime, "unused_frames_folder")

    assert len(detectors) == 1, f"expected exactly 1 detector (pose only), got {len(detectors)}"
    assert type(detectors[0]).__name__ == "PoseActionClassifier", \
        f"expected PoseActionClassifier, got {type(detectors[0]).__name__}"
    # FaceDetector/HandDetector (mediapipe) and FaceAttributesDetector must be
    # absent, not just disabled — nothing else was appended to the list.


# ---------------------------------------------------------------------------
# 1.2 bootstrap.load_face_attr_sessions() skips loading when flags are off
# ---------------------------------------------------------------------------

def test_load_face_attr_sessions_skips_when_both_flags_false():
    assert not (DETECTOR_FLAGS["emotion"] or DETECTOR_FLAGS["age"]), \
        "this check assumes today's default flags (emotion=False, age=False)"

    # Imported lazily: bootstrap.py pulls in flask/onnxruntime/torchreid/
    # supabase, which most tests_manual scripts in this repo don't need.
    from camerachatbot.runtime import bootstrap

    (emotion_sess, emo_in, emo_out), (age_sess, age_in, age_out) = bootstrap.load_face_attr_sessions()

    # If the gate were missing, this call would try to open
    # models/emotion-ferplus-8.onnx / age_googlenet.onnx, which do not exist
    # in this sandbox, and raise FileNotFoundError instead of returning None.
    assert emotion_sess is None and emo_in is None and emo_out is None
    assert age_sess is None and age_in is None and age_out is None


# ---------------------------------------------------------------------------
# 1.3 FaceAttributesDetector warns on emotion/age=True, face_attention=False
# ---------------------------------------------------------------------------

def test_face_attributes_detector_warns_on_inconsistent_flags():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        FaceAttributesDetector(
            emotion_input="in", emotion_output="out",
            age_input="in", age_output="out",
            emotion_sess=object(),  # truthy stand-in for a real ORT session
            age_sess=None,
            face_attention_enabled=False,
        )
    out = buf.getvalue()
    assert "WARN" in out and "face_attention" in out, \
        f"expected a face_attention WARN, got: {out!r}"


def test_face_attributes_detector_silent_when_face_attention_enabled():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        FaceAttributesDetector(
            emotion_input="in", emotion_output="out",
            age_input="in", age_output="out",
            emotion_sess=object(),
            age_sess=object(),
            face_attention_enabled=True,
        )
    out = buf.getvalue()
    assert "WARN" not in out, f"expected no WARN when face_attention=True, got: {out!r}"


def test_face_attributes_detector_silent_when_emotion_and_age_disabled():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        FaceAttributesDetector(
            emotion_input="in", emotion_output="out",
            age_input="in", age_output="out",
            emotion_sess=None,
            age_sess=None,
            face_attention_enabled=False,
        )
    out = buf.getvalue()
    assert "WARN" not in out, \
        f"expected no WARN when emotion/age are both disabled (nothing depends on face bbox), got: {out!r}"


# ---------------------------------------------------------------------------
# 1.4 detect_and_embed() discards non-allowlisted COCO classes
# ---------------------------------------------------------------------------

class _FakeBox:
    def __init__(self, cls_id, bbox, conf=0.9):
        self.cls = [cls_id]
        self.xyxy = [bbox]
        self.conf = [conf]


class _FakeYoloResult:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes


class _FakeYoloDetModel:
    """Stand-in for the ultralytics YOLO detector: batch-callable, returns a
    fixed set of detections per image (allowlisted person + knife, plus a
    non-allowlisted bicycle) so the allowlist filter can be exercised without
    real model weights."""

    NAMES = {0: "person", 1: "bicycle", 43: "knife"}

    def __call__(self, batch_imgs, verbose=False, conf=0.25, iou=0.45):
        boxes = [
            _FakeBox(0, [10, 10, 60, 60]),     # person -> allowlisted
            _FakeBox(1, [70, 70, 120, 120]),   # bicycle -> NOT allowlisted
            _FakeBox(43, [130, 130, 180, 180]),  # knife -> allowlisted
        ]
        return [_FakeYoloResult(self.NAMES, boxes) for _ in batch_imgs]


def test_detect_and_embed_discards_non_allowlisted_classes():
    frames_folder = str(paths.KEYFRAMES_SAMPLE_DIR)
    assert os.path.isdir(frames_folder), f"expected real fixture images at {frames_folder}"

    with tempfile.TemporaryDirectory(prefix="fase1_allowlist_test_") as tmpdir:
        tracker = YOLOPersonReID(
            model=_FakeYoloDetModel(),
            frames_folder=frames_folder,
            output_folder=tmpdir,
            transform=None, reid_model=None, device=None,
            depth_model=None, depth_transform=None, save_deph=False,
        )
        tmp_json_path = tracker.detect_and_embed(
            batch_size=16, save_outputs=False, allowlist=COCO_ALLOWLIST
        )

        with open(tmp_json_path, "r", encoding="utf-8") as f:
            results_json = json.load(f)

        assert len(results_json) > 0, "expected at least one processed frame from keyFrames/"

        seen_classes = set()
        for frame_id, entries in results_json.items():
            for e in entries:
                seen_classes.add(e["class_name"])
                assert e["class_name"] in COCO_ALLOWLIST, \
                    f"non-allowlisted class leaked into frame_list: {e['class_name']!r}"
            # Exactly person + knife per frame, bicycle discarded entirely.
            assert len(entries) == 2, f"expected 2 kept entries (person+knife), got {len(entries)}"

        assert seen_classes == {"person", "knife"}, \
            f"expected only person/knife to survive filtering, got {seen_classes}"
        assert "bicycle" not in seen_classes


# ---------------------------------------------------------------------------
# 1.5 formatter no longer emits a depth metadata node
# ---------------------------------------------------------------------------

def test_formatter_omits_depth_node():
    with tempfile.TemporaryDirectory(prefix="fase1_depth_test_") as tmpdir:
        src_path = os.path.join(tmpdir, "src.json")
        dst_path = os.path.join(tmpdir, "dst.json")

        src = {
            "0": [
                {
                    "kind": "person",
                    "class_name": "person",
                    "bbox": [1, 2, 3, 4],
                    "confidence": 0.9,
                    "attributes": {"pose": "sentado"},
                    "depth": {"mean": 1.23, "median": 1.1, "min": 0.5},
                    "user_id": 7,
                },
                {
                    "kind": "object",
                    "class_name": "knife",
                    "bbox": [5, 6, 7, 8],
                    "confidence": 0.8,
                    "depth": {"mean": 2.0, "median": 2.0, "min": 1.0},
                },
            ],
            "neighborhood": {},
        }
        with open(src_path, "w", encoding="utf-8") as f:
            json.dump(src, f)

        reformat_to_video_schema_uniform(
            src_path, dst_path,
            video_key="test_video",
            start_at="2024-01-01T00:00:00Z",
            size_xy=(640, 480),
            fps=30,
            per_frame_inference=0.0,
            per_frame_preprocess=0.0,
            per_frame_postprocess=0.0,
        )

        with open(dst_path, "r", encoding="utf-8") as f:
            out = json.load(f)

        raw = json.dumps(out)
        assert '"depth"' not in raw, "found a lingering 'depth' key in formatter output"

        objects = out["video"]["key_frames"][0]["objects"]
        assert len(objects) == 2
        for obj in objects:
            metadata_keys = {m["metadata_key"] for m in obj["metadata"]}
            assert "depth" not in metadata_keys, f"depth node present in metadata: {metadata_keys}"


def main():
    check("1.1 build_detail_detectors() only builds pose (current DETECTOR_FLAGS)",
          test_build_detail_detectors_honors_flags)
    check("1.2 load_face_attr_sessions() skips ONNX load when emotion/age both False",
          test_load_face_attr_sessions_skips_when_both_flags_false)
    check("1.3 FaceAttributesDetector warns: emotion/age=True, face_attention=False",
          test_face_attributes_detector_warns_on_inconsistent_flags)
    check("1.3 FaceAttributesDetector silent when face_attention=True",
          test_face_attributes_detector_silent_when_face_attention_enabled)
    check("1.3 FaceAttributesDetector silent when emotion/age both disabled",
          test_face_attributes_detector_silent_when_emotion_and_age_disabled)
    check("1.4 detect_and_embed() discards non-allowlisted COCO classes",
          test_detect_and_embed_discards_non_allowlisted_classes)
    check("1.5 formatter output has no depth metadata node",
          test_formatter_omits_depth_node)

    print("\n=== Fase 1 attribute-gating contract verification ===")
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
