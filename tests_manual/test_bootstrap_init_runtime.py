"""Manual verification for `camerachatbot.runtime.bootstrap.init_runtime()`.

Not a pytest test: this repo has no test runner. Plain runnable script with
bare ``assert`` statements, printing a PASS/FAIL summary and exiting 0/1.

    python tests_manual/test_bootstrap_init_runtime.py

Why this file exists (PR6 gate-review finding): `tests_manual/
test_fase3b_switch_flip.py`, deleted in Fase 3c (PR6) once its premise (two
possible `load_reid()` shapes, torch vs. ONNX) no longer existed, was ALSO
the only test anywhere in this repo that ran the real
`bootstrap.init_runtime()` end-to-end. Deleting it silently dropped that
coverage — no other `tests_manual/` file calls `init_runtime()`;
`test_fase1_attribute_gating.py` only exercises `load_face_attr_sessions()`
in isolation, and the Fase 3a files (`test_fase3a_onnx_detector_reid.py`,
`test_fase3a_pose_clustering_parity.py`) test the ONNX detector/reid/pose/
clustering modules directly, never through `bootstrap`. This file replaces
that lost coverage against the CURRENT (post-Fase-3c) shape of
`init_runtime()`: a single ONNX-only path, no tuple-shape absorption, no
`reid_transform`/`device` keys in `RUNTIME`.

Loader/network seams (`load_detector`/`load_posecls`/`load_reid`/
`load_face_attr_sessions`/`build_supabase`) are monkeypatched with
lightweight fakes so `init_runtime()` runs to completion in-process without
touching `.env`, real model files/weights, or a network connection — every
patched attribute is restored in a `finally` block regardless of outcome
(same "seams restored" pattern the deleted switch-flip test used).

Includes one mutation-style sanity check (`test_init_runtime_does_not_
absorb_tuple_shape`) that feeds `load_reid()` a `(model, transform)` tuple —
the shape the old torch loader used to return. Post-Fase-3c `init_runtime()`
has no `isinstance(reid_loaded, tuple)` absorption branch left, so the tuple
must land in `RUNTIME["reid_model"]` completely unpacked/unmodified. If the
absorption branch were ever reintroduced, this check would fail — proving
this file is a genuine tripwire on the post-cleanup shape, not a tautology
that would pass no matter what `init_runtime()` does with its input.
"""

import io
import os
import sys
import contextlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imported lazily inside a helper below (not at module scope): bootstrap.py
# pulls in flask/onnxruntime/supabase/python-dotenv, which most
# tests_manual scripts in this repo don't need.

results = []  # list[tuple[str, bool, str]]


def check(name, fn):
    try:
        fn()
        results.append((name, True, ""))
    except AssertionError as e:
        results.append((name, False, f"AssertionError: {e}"))
    except Exception as e:  # noqa: BLE001 - surfaced in the summary, not swallowed
        results.append((name, False, f"{type(e).__name__}: {e}"))


# ---------------------------------------------------------------------------
# Seam-patching helper
# ---------------------------------------------------------------------------

_SEAM_NAMES = (
    "build_supabase",
    "load_detector",
    "load_posecls",
    "load_reid",
    "load_face_attr_sessions",
)


def _run_init_runtime(
    detector_return="fake_detector",
    posecls_return="fake_posecls",
    reid_return="fake_reid_model",
    face_attr_return=((None, None, None), (None, None, None)),
    supabase_return=(None, None),
):
    """Runs the REAL `bootstrap.init_runtime()` with its loader/network
    seams monkeypatched to hermetic, argument-configurable stubs, restoring
    every patched attribute in a `finally` block regardless of outcome. This
    exercises `init_runtime()`'s actual wiring logic — not a
    reimplementation of it.
    """
    from camerachatbot.runtime import bootstrap

    originals = {name: getattr(bootstrap, name) for name in _SEAM_NAMES}
    try:
        bootstrap.build_supabase = lambda: supabase_return
        bootstrap.load_detector = lambda: detector_return
        bootstrap.load_posecls = lambda: posecls_return
        bootstrap.load_reid = lambda: reid_return
        bootstrap.load_face_attr_sessions = lambda: face_attr_return
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            runtime = bootstrap.init_runtime()
        return runtime, buf.getvalue(), bootstrap
    finally:
        for name, fn in originals.items():
            setattr(bootstrap, name, fn)


# ---------------------------------------------------------------------------
# 1. init_runtime() wires the expected RUNTIME keys, no crash
# ---------------------------------------------------------------------------

def test_init_runtime_wires_expected_keys_no_crash():
    runtime, stdout, _bootstrap = _run_init_runtime()

    expected_keys = {
        "app", "supabase", "BUCKET_NAME",
        "yolo_det", "yolo_posecls", "reid_model",
        "emotion", "age",
    }
    assert set(runtime.keys()) == expected_keys, (
        f"unexpected RUNTIME key set: {set(runtime.keys())}"
    )
    assert "[INIT] Modelos y sesiones cargados correctamente." in stdout, (
        f"expected init_runtime() to print its success line, got: {stdout!r}"
    )


# ---------------------------------------------------------------------------
# 2. Fake return values land verbatim in RUNTIME (real wiring, not reimplemented)
# ---------------------------------------------------------------------------

def test_init_runtime_values_are_wired_from_loaders_verbatim():
    sentinel_det = object()
    sentinel_posecls = object()
    sentinel_reid = object()
    runtime, _stdout, _bootstrap = _run_init_runtime(
        detector_return=sentinel_det,
        posecls_return=sentinel_posecls,
        reid_return=sentinel_reid,
        supabase_return=("fake_client", "fake_bucket"),
    )
    assert runtime["yolo_det"] is sentinel_det
    assert runtime["yolo_posecls"] is sentinel_posecls
    assert runtime["reid_model"] is sentinel_reid, (
        "reid_model must be load_reid()'s return value verbatim — "
        "there is no shape absorption left post-Fase-3c"
    )
    assert runtime["supabase"] == "fake_client"
    assert runtime["BUCKET_NAME"] == "fake_bucket"


def test_init_runtime_emotion_age_structure():
    fake_emotion = ("emo_sess", "emo_in", "emo_out")
    fake_age = ("age_sess", "age_in", "age_out")
    runtime, _stdout, _bootstrap = _run_init_runtime(
        face_attr_return=(fake_emotion, fake_age),
    )
    assert runtime["emotion"] == {"sess": "emo_sess", "input": "emo_in", "output": "emo_out"}
    assert runtime["age"] == {"sess": "age_sess", "input": "age_in", "output": "age_out"}


# ---------------------------------------------------------------------------
# 3. Post-Fase-3c: reid_transform/device are genuinely gone, not just unset
# ---------------------------------------------------------------------------

def test_init_runtime_has_no_reid_transform_or_device_keys():
    runtime, _stdout, _bootstrap = _run_init_runtime()
    assert "reid_transform" not in runtime, (
        "RUNTIME['reid_transform'] was removed in Fase 3c — the dual-shape "
        "torch/ONNX absorption that produced it no longer exists"
    )
    assert "device" not in runtime, (
        "RUNTIME['device'] was already removed in Fase 3b — onnx_providers() "
        "is the sole device-selection path"
    )


# ---------------------------------------------------------------------------
# 4. Mutation-style tripwire: no isinstance(reid_loaded, tuple) absorption left
# ---------------------------------------------------------------------------

def test_init_runtime_does_not_absorb_tuple_shape():
    # This is exactly the shape the OLD torch loader used to return
    # (`(reid_model, transform)`). Post-Fase-3c, init_runtime() has no
    # isinstance(reid_loaded, tuple) branch to unpack it — if this ever
    # regressed (the absorption branch reintroduced), RUNTIME["reid_model"]
    # would silently become just the first tuple element instead of the
    # whole tuple, and this assertion would catch it.
    fake_transform = object()
    old_shaped_tuple = ("fake_torch_reid_model", fake_transform)
    runtime, _stdout, _bootstrap = _run_init_runtime(reid_return=old_shaped_tuple)

    assert runtime["reid_model"] == old_shaped_tuple, (
        "expected the tuple to land in RUNTIME['reid_model'] completely "
        "unpacked/unmodified (no absorption branch left); got "
        f"{runtime['reid_model']!r} instead — a reid-shape absorption "
        "branch appears to have been reintroduced"
    )
    assert "reid_transform" not in runtime


# ---------------------------------------------------------------------------
# 5. Monkeypatched seams are restored after use
# ---------------------------------------------------------------------------

def test_init_runtime_seams_restored_after_each_run():
    _runtime, _stdout, bootstrap = _run_init_runtime()
    # Sanity check on the test harness itself: confirm the monkeypatches
    # above don't leak into module state for any other consumer of
    # bootstrap.py (e.g. run_local.py / run_webhook.py importing it later).
    assert bootstrap.load_reid.__module__ == "camerachatbot.runtime.loaders_onnx", (
        "bootstrap.load_reid must be restored to the real loaders_onnx.load_reid "
        "after this suite's monkeypatching, not left pointing at a test stub"
    )
    assert bootstrap.load_detector.__module__ == "camerachatbot.runtime.loaders_onnx"
    assert bootstrap.load_posecls.__module__ == "camerachatbot.runtime.loaders_onnx"
    assert bootstrap.build_supabase.__module__ == "camerachatbot.runtime.bootstrap"
    assert bootstrap.load_face_attr_sessions.__module__ == "camerachatbot.runtime.bootstrap"


def main():
    check("init_runtime() wires the expected RUNTIME key set, no crash",
          test_init_runtime_wires_expected_keys_no_crash)
    check("init_runtime() wires loader return values into RUNTIME verbatim",
          test_init_runtime_values_are_wired_from_loaders_verbatim)
    check("init_runtime() wires emotion/age sess/input/output structure",
          test_init_runtime_emotion_age_structure)
    check("init_runtime() RUNTIME has no reid_transform/device keys (Fase 3c/3b)",
          test_init_runtime_has_no_reid_transform_or_device_keys)
    check("init_runtime() does NOT absorb the old (model, transform) tuple shape",
          test_init_runtime_does_not_absorb_tuple_shape)
    check("bootstrap.py monkeypatched seams restored after use",
          test_init_runtime_seams_restored_after_each_run)

    print("\n=== bootstrap.init_runtime() verification (post-Fase-3c ONNX-only shape) ===")
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
