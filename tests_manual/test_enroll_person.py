"""Manual verification for the Fase 5 (PR9) gate-review fix batch's
`identity.enroll_person` coverage.

Not a pytest test: this repo has no test runner, so this is a plain
runnable script using bare ``assert`` statements, printing a PASS/FAIL
summary and exiting 0/1. Run it directly:

    python tests_manual/test_enroll_person.py

What it verifies:

1. (Fix A, CRITICAL reliability gap) `enroll()`'s documented gray-zone-raise
   behavior: when `gallery.assign_or_create()` returns `(None, sim, False)`
   (the gray-zone shape -- see `global_identity_service.py::
   assign_or_create()`), `enroll()` raises `RuntimeError` instead of
   silently picking a person id, and never reaches
   `_upsert_authorized_identity()`. No live gallery, ONNX weights, or
   Postgres needed -- `load_detector()`/`load_reid()` are monkeypatched
   with fakes (same workaround pattern as `test_identity_thresholds.py`'s
   `_with_fake_auth_registry`), and a fake `gallery` object is injected via
   `enroll()`'s own `gallery=` parameter.
2. (Fix C, split-state visibility) when `gallery.assign_or_create()`
   succeeds (a real, non-gray-zone match/creation) but the subsequent
   `authorized_identity` upsert fails, `enroll()` raises a `RuntimeError`
   whose message explicitly conveys the split state (gallery write
   succeeded, authorization record failed, person is UNAUTHORIZED until
   retried) -- not a bare/opaque DB error.
3. (Fix B, resilience) `main()`'s top-level exception handling: expected
   failure modes (`FileNotFoundError` from `--images` matching nothing,
   `RuntimeError` from a failed/gray-zone enroll, `psycopg2.Error` from a
   DB failure) are caught, print a clean `[ERROR] ...` message (not a raw
   traceback), and exit non-zero via `SystemExit`, rather than propagating
   uncaught.

Uses only synthetic in-memory fakes and a tiny throwaway image written to a
temp dir -- no live Postgres connection, no real ONNX weights, and never
touches the repo's real `gallery.index`/`id_map.json`/`proto_store.npy`.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import psycopg2  # noqa: E402

import camerachatbot.identity.enroll_person as enroll_person_mod  # noqa: E402
from camerachatbot.identity.enroll_person import enroll  # noqa: E402

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
# Shared fakes: detector/embedder (avoid needing real ONNX weights), gallery
# ---------------------------------------------------------------------------

class _FakeBox:
    def __init__(self, cls_id, xyxy):
        self.cls = [cls_id]
        self.xyxy = [xyxy]


class _FakeDetResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class _FakeDetector:
    """Detects one full-frame "person" box per image -- enough to drive
    `_largest_person_bbox()`/`_embed_images()` without a real YOLO model."""

    def __call__(self, img, verbose=False):
        h, w = img.shape[:2]
        box = _FakeBox(0, [0.0, 0.0, float(w), float(h)])
        return [_FakeDetResult([box], {0: "person"})]


class _FakeEmbedder:
    """Deterministic unit-norm embedding -- enough to drive `_centroid()`
    and `assign_or_create()` without real ONNX weights."""

    def embed_one(self, crop):
        v = np.ones(512, dtype="float32")
        return v / np.linalg.norm(v)


class _FakeGrayZoneGallery:
    """`assign_or_create()` always returns the gray-zone shape (`pid=None`),
    matching `GlobalIdentityService.assign_or_create()`'s real return
    contract for an ambiguous match (see that module's `# zona gris`
    branch)."""

    def assign_or_create(self, emb, modality="body", *, t_accept, t_reject,
                          k=10, proto_meta=None, return_hits=False):
        return (None, 0.6, False)


class _FakeAcceptedGallery:
    """`assign_or_create()` always succeeds (confident match/creation)."""

    def assign_or_create(self, emb, modality="body", *, t_accept, t_reject,
                          k=10, proto_meta=None, return_hits=False):
        return (7, 0.95, True)


def _write_dummy_image(tmpdir, name="ref.jpg"):
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    path = os.path.join(tmpdir, name)
    cv2.imwrite(path, img)
    return path


def _with_fake_detector_reid(fn):
    """Monkeypatches `enroll_person.load_detector`/`load_reid` for the
    duration of `fn()`."""
    original_load_detector = enroll_person_mod.load_detector
    original_load_reid = enroll_person_mod.load_reid
    enroll_person_mod.load_detector = lambda: _FakeDetector()
    enroll_person_mod.load_reid = lambda: _FakeEmbedder()
    try:
        return fn()
    finally:
        enroll_person_mod.load_detector = original_load_detector
        enroll_person_mod.load_reid = original_load_reid


# ---------------------------------------------------------------------------
# Fix A: gray-zone match raises RuntimeError, never reaches the DB upsert
# ---------------------------------------------------------------------------

def test_enroll_raises_runtime_error_on_gray_zone_match():
    with tempfile.TemporaryDirectory(prefix="enroll_gray_zone_") as tmpdir:
        _write_dummy_image(tmpdir)

        upsert_calls = []
        original_upsert = enroll_person_mod._upsert_authorized_identity
        enroll_person_mod._upsert_authorized_identity = lambda *a, **kw: upsert_calls.append((a, kw))
        try:
            def _call():
                enroll("Gray Zone Person", tmpdir, gallery=_FakeGrayZoneGallery())

            try:
                _with_fake_detector_reid(_call)
                raise AssertionError("expected RuntimeError, enroll() succeeded instead")
            except RuntimeError as e:
                assert "ambigua" in str(e).lower() or "zona gris" in str(e).lower(), (
                    f"RuntimeError message does not mention the gray-zone ambiguity: {e}"
                )
        finally:
            enroll_person_mod._upsert_authorized_identity = original_upsert

        assert upsert_calls == [], (
            "enroll() must raise BEFORE calling _upsert_authorized_identity() on a "
            "gray-zone match -- it must never silently create an authorized_identity "
            "row for an ambiguous pid"
        )


# ---------------------------------------------------------------------------
# Fix C: gallery write succeeds, authorized_identity upsert fails -> split
# state must be surfaced explicitly, not a bare DB error
# ---------------------------------------------------------------------------

def test_enroll_surfaces_split_state_when_upsert_fails_after_gallery_write():
    with tempfile.TemporaryDirectory(prefix="enroll_split_state_") as tmpdir:
        _write_dummy_image(tmpdir)

        original_upsert = enroll_person_mod._upsert_authorized_identity

        def _failing_upsert(*a, **kw):
            raise psycopg2.OperationalError("simulated connection failure")

        enroll_person_mod._upsert_authorized_identity = _failing_upsert
        try:
            def _call():
                enroll("Split State Person", tmpdir, gallery=_FakeAcceptedGallery())

            try:
                _with_fake_detector_reid(_call)
                raise AssertionError("expected RuntimeError, enroll() succeeded instead")
            except RuntimeError as e:
                msg = str(e)
                assert "person_global_id=7" in msg, f"message must name the resolved pid: {msg}"
                assert "galería" in msg.lower(), f"message must mention the gallery write succeeded: {msg}"
                assert "no autorizada" in msg.lower(), f"message must state the UNAUTHORIZED interim state: {msg}"
        finally:
            enroll_person_mod._upsert_authorized_identity = original_upsert


# ---------------------------------------------------------------------------
# Fix B: main()'s top-level exception handling -- clean exit, no traceback
# ---------------------------------------------------------------------------

def _run_main_expect_clean_exit(argv):
    """Runs `main(argv)`, asserting it exits via `SystemExit` (not an
    uncaught exception) with a non-zero code."""
    try:
        enroll_person_mod.main(argv)
        raise AssertionError("expected SystemExit, main() returned normally instead")
    except SystemExit as e:
        assert e.code not in (None, 0), f"expected non-zero exit code, got {e.code!r}"


def test_main_reports_clean_error_on_missing_images_dir():
    with tempfile.TemporaryDirectory(prefix="enroll_no_images_") as tmpdir:
        empty_dir = os.path.join(tmpdir, "empty")
        os.makedirs(empty_dir)
        _run_main_expect_clean_exit(["--name", "Nobody", "--images", empty_dir])


def test_main_reports_clean_error_on_runtime_error_from_enroll():
    original_enroll = enroll_person_mod.enroll

    def _fake_enroll(name, images_dir, role=None, gallery=None):
        raise RuntimeError("simulated gray-zone failure")

    enroll_person_mod.enroll = _fake_enroll
    try:
        _run_main_expect_clean_exit(["--name", "X", "--images", "unused"])
    finally:
        enroll_person_mod.enroll = original_enroll


def test_main_reports_clean_error_on_psycopg2_error_from_deactivate():
    original_deactivate = enroll_person_mod.deactivate

    def _fake_deactivate(person_global_id):
        raise psycopg2.OperationalError("simulated connection failure")

    enroll_person_mod.deactivate = _fake_deactivate
    try:
        _run_main_expect_clean_exit(["--deactivate", "42"])
    finally:
        enroll_person_mod.deactivate = original_deactivate


def main():
    check("enroll(): raises RuntimeError on gray-zone match, never upserts",
          test_enroll_raises_runtime_error_on_gray_zone_match)
    check("enroll(): surfaces split state when upsert fails after gallery write",
          test_enroll_surfaces_split_state_when_upsert_fails_after_gallery_write)
    check("main(): clean exit on FileNotFoundError (--images has no matching files)",
          test_main_reports_clean_error_on_missing_images_dir)
    check("main(): clean exit on RuntimeError from enroll()",
          test_main_reports_clean_error_on_runtime_error_from_enroll)
    check("main(): clean exit on psycopg2.Error from deactivate()",
          test_main_reports_clean_error_on_psycopg2_error_from_deactivate)

    print("\n=== Fase 5 (PR9) gate-review fix: enroll_person.py verification ===")
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
