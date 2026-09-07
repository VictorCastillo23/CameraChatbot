"""Manual verification for the Fase 0 identity-threshold contract.

Not a pytest test: this repo has no test runner, so this is a plain
runnable script using bare ``assert`` statements, printing a PASS/FAIL
summary and exiting 0/1. Run it directly:

    python tests_manual/test_identity_thresholds.py

What it verifies (PR1 / Fase 0 threshold-centralization contract):

1. `person_reid.YOLOPersonReID.enroll_and_assign_global_ids`,
   `person_reid.YOLOPersonReID._enforce_unique_global_per_frame_post`,
   `global_identity_service.GlobalIdentityService.assign_or_create`, and its
   `maybe_add_support_prototype` helper are all keyword-only on their
   threshold params with NO default — omitting one raises `TypeError`
   immediately instead of silently falling back to a stale value.
2. Calling each of those four with every required kwarg explicitly supplied
   (sourced from `security_config.IDENTITY_THRESHOLDS`) does NOT raise
   `TypeError` — the happy path still works.
3. `orchestrator.multi_models`'s sentinel-resolution: leaving `t_accept`/
   `t_reject`/`min_sim_add`/`max_protos_per_person` as their `None` default
   resolves them from `IDENTITY_THRESHOLDS` instead of raising, and the
   resolved values match `IDENTITY_THRESHOLDS` exactly.

Uses minimal/mocked inputs throughout (a temp-dir-backed
`GlobalIdentityService`, a fake tracker object for `multi_models`) rather
than a real gallery on the repo's production `gallery.index`/`id_map.json`/
`proto_store.npy` or a real detection pipeline — see the "Identity
persistence gotcha" note in CLAUDE.md for why those production files must
never be touched by a verification script.
"""

import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from camerachatbot.security_config import IDENTITY_THRESHOLDS  # noqa: E402
from camerachatbot.detectors.person_reid import YOLOPersonReID  # noqa: E402
from camerachatbot.identity.global_identity_service import GlobalIdentityService  # noqa: E402
from camerachatbot.pipeline.orchestrator import multi_models  # noqa: E402

RUNTIME_T_ACCEPT = IDENTITY_THRESHOLDS["runtime"]["t_accept"]
RUNTIME_T_REJECT = IDENTITY_THRESHOLDS["runtime"]["t_reject"]
SUPPORT_MIN_SIM_ADD = IDENTITY_THRESHOLDS["support"]["min_sim_add"]
SUPPORT_MAX_PROTOS = IDENTITY_THRESHOLDS["support"]["max_protos_per_person"]

results = []  # list[tuple[str, bool, str]]


def check(name, fn):
    try:
        fn()
        results.append((name, True, ""))
    except AssertionError as e:
        results.append((name, False, f"AssertionError: {e}"))
    except Exception as e:  # noqa: BLE001 - report any unexpected exception as a failure
        results.append((name, False, f"{type(e).__name__}: {e}"))


def _fresh_gallery(tmpdir, suffix):
    return GlobalIdentityService(
        dim=8,
        index_path=os.path.join(tmpdir, f"gallery_{suffix}.index"),
        map_path=os.path.join(tmpdir, f"id_map_{suffix}.json"),
        store_path=os.path.join(tmpdir, f"proto_store_{suffix}.npy"),
    )


# ---------------------------------------------------------------------------
# Part 1: TypeError when a threshold kwarg is omitted (no-default contract)
# ---------------------------------------------------------------------------

def test_enroll_and_assign_global_ids_typeerror_on_missing_kwarg():
    fake_self = types.SimpleNamespace()
    try:
        YOLOPersonReID.enroll_and_assign_global_ids(
            fake_self, gallery=None, labels=[],
            t_accept=RUNTIME_T_ACCEPT,
            # t_reject deliberately omitted
            min_sim_add=SUPPORT_MIN_SIM_ADD,
            max_protos_per_person=SUPPORT_MAX_PROTOS,
        )
        raise AssertionError("expected TypeError, call succeeded instead")
    except TypeError:
        pass


def test_enforce_unique_global_per_frame_post_typeerror_on_missing_kwarg():
    fake_self = types.SimpleNamespace(results_json={})
    try:
        YOLOPersonReID._enforce_unique_global_per_frame_post(
            fake_self, E_norm=None, gallery=None,
            t_accept=RUNTIME_T_ACCEPT,
            # t_reject deliberately omitted
            min_sim_add=SUPPORT_MIN_SIM_ADD,
            max_protos_per_person=SUPPORT_MAX_PROTOS,
        )
        raise AssertionError("expected TypeError, call succeeded instead")
    except TypeError:
        pass


def test_assign_or_create_typeerror_on_missing_kwarg():
    fake_self = types.SimpleNamespace()
    emb = np.zeros(8, dtype="float32")
    try:
        GlobalIdentityService.assign_or_create(
            fake_self, emb,
            t_accept=RUNTIME_T_ACCEPT,
            # t_reject deliberately omitted
        )
        raise AssertionError("expected TypeError, call succeeded instead")
    except TypeError:
        pass


def test_maybe_add_support_prototype_typeerror_on_missing_kwarg():
    fake_self = types.SimpleNamespace()
    emb = np.zeros(8, dtype="float32")
    try:
        GlobalIdentityService.maybe_add_support_prototype(
            fake_self, pid=1, emb=emb,
            min_sim_add=SUPPORT_MIN_SIM_ADD,
            # max_protos_per_person deliberately omitted
        )
        raise AssertionError("expected TypeError, call succeeded instead")
    except TypeError:
        pass


# ---------------------------------------------------------------------------
# Part 2: happy path — all required kwargs supplied, no TypeError
# ---------------------------------------------------------------------------

def test_enroll_and_assign_global_ids_happy_path():
    # labels=[] hits the function's own early-return before touching
    # self.embeddings/gallery at all — the minimal input that exercises the
    # keyword-only signature without needing a real detection run.
    fake_self = types.SimpleNamespace()
    result = YOLOPersonReID.enroll_and_assign_global_ids(
        fake_self, gallery=None, labels=[],
        t_accept=RUNTIME_T_ACCEPT, t_reject=RUNTIME_T_REJECT,
        min_sim_add=SUPPORT_MIN_SIM_ADD, max_protos_per_person=SUPPORT_MAX_PROTOS,
    )
    assert result is None, f"expected early-return None for empty labels, got {result!r}"


def test_enforce_unique_global_per_frame_post_happy_path():
    # results_json={} makes the whole per-frame loop a no-op.
    fake_self = types.SimpleNamespace(results_json={})
    YOLOPersonReID._enforce_unique_global_per_frame_post(
        fake_self, E_norm=None, gallery=None,
        t_accept=RUNTIME_T_ACCEPT, t_reject=RUNTIME_T_REJECT,
        min_sim_add=SUPPORT_MIN_SIM_ADD, max_protos_per_person=SUPPORT_MAX_PROTOS,
    )


def test_assign_or_create_happy_path(tmpdir):
    gallery = _fresh_gallery(tmpdir, "assign")
    emb = np.random.RandomState(0).rand(8).astype("float32")
    pid, sim, created = gallery.assign_or_create(
        emb, t_accept=RUNTIME_T_ACCEPT, t_reject=RUNTIME_T_REJECT,
    )
    # Empty gallery -> no hits -> a brand-new person is created.
    assert created is True
    assert isinstance(pid, int)


def test_maybe_add_support_prototype_happy_path(tmpdir):
    gallery = _fresh_gallery(tmpdir, "support")
    emb = np.random.RandomState(1).rand(8).astype("float32")
    added = gallery.maybe_add_support_prototype(
        pid=1, emb=emb,
        min_sim_add=SUPPORT_MIN_SIM_ADD, max_protos_per_person=SUPPORT_MAX_PROTOS,
    )
    # Empty gallery -> search() returns no hits -> declines to add, but this
    # must be a normal `False` return, never a TypeError.
    assert added is False


# ---------------------------------------------------------------------------
# Part 3: orchestrator.multi_models sentinel resolution
# ---------------------------------------------------------------------------

class _FakeTracker:
    """Minimal stand-in for YOLOPersonReID, just enough to drive
    multi_models() through its threshold-resolution logic without a real
    detection pipeline. Records the kwargs multi_models forwards downstream.
    """

    def __init__(self):
        self.captured_kwargs = None

    def detect_and_embed(self, **kwargs):
        # PR2/Fase1 added a required `allowlist` kwarg to the real
        # `detect_and_embed()`; accept and ignore it here so this fake keeps
        # driving multi_models()'s threshold-resolution logic without
        # depending on the (unrelated) allowlist-filtering contract.
        return {}

    def cluster_locally(self, eps=None, min_samples=None):
        return []

    def enforce_unique_label_per_frame(self, labels):
        return labels

    def enroll_and_assign_global_ids(self, gallery, labels, *, t_accept, t_reject,
                                      min_sim_add, max_protos_per_person):
        self.captured_kwargs = dict(
            t_accept=t_accept, t_reject=t_reject,
            min_sim_add=min_sim_add, max_protos_per_person=max_protos_per_person,
        )
        return {}


def test_multi_models_resolves_none_sentinels_from_identity_thresholds():
    tracker = _FakeTracker()
    # t_accept/t_reject/min_sim_add/max_protos_per_person all left at their
    # None default deliberately.
    multi_models(tracker, detail_detectors=[], keyframes_path="unused", gallery=None)

    assert tracker.captured_kwargs is not None, "enroll_and_assign_global_ids was never called"
    assert tracker.captured_kwargs["t_accept"] == RUNTIME_T_ACCEPT
    assert tracker.captured_kwargs["t_reject"] == RUNTIME_T_REJECT
    assert tracker.captured_kwargs["min_sim_add"] == SUPPORT_MIN_SIM_ADD
    assert tracker.captured_kwargs["max_protos_per_person"] == SUPPORT_MAX_PROTOS


def test_multi_models_explicit_override_wins_over_sentinel():
    tracker = _FakeTracker()
    multi_models(
        tracker, detail_detectors=[], keyframes_path="unused", gallery=None,
        t_accept=0.99, t_reject=0.01, min_sim_add=0.5, max_protos_per_person=1,
    )
    assert tracker.captured_kwargs["t_accept"] == 0.99
    assert tracker.captured_kwargs["t_reject"] == 0.01
    assert tracker.captured_kwargs["min_sim_add"] == 0.5
    assert tracker.captured_kwargs["max_protos_per_person"] == 1


def main():
    with tempfile.TemporaryDirectory(prefix="identity_thresholds_test_") as tmpdir:
        check("enroll_and_assign_global_ids: TypeError on missing t_reject",
              test_enroll_and_assign_global_ids_typeerror_on_missing_kwarg)
        check("_enforce_unique_global_per_frame_post: TypeError on missing t_reject",
              test_enforce_unique_global_per_frame_post_typeerror_on_missing_kwarg)
        check("assign_or_create: TypeError on missing t_reject",
              test_assign_or_create_typeerror_on_missing_kwarg)
        check("maybe_add_support_prototype: TypeError on missing max_protos_per_person",
              test_maybe_add_support_prototype_typeerror_on_missing_kwarg)

        check("enroll_and_assign_global_ids: happy path, no TypeError",
              test_enroll_and_assign_global_ids_happy_path)
        check("_enforce_unique_global_per_frame_post: happy path, no TypeError",
              test_enforce_unique_global_per_frame_post_happy_path)
        check("assign_or_create: happy path, no TypeError",
              lambda: test_assign_or_create_happy_path(tmpdir))
        check("maybe_add_support_prototype: happy path, no TypeError",
              lambda: test_maybe_add_support_prototype_happy_path(tmpdir))

        check("multi_models: None sentinels resolve to IDENTITY_THRESHOLDS",
              test_multi_models_resolves_none_sentinels_from_identity_thresholds)
        check("multi_models: explicit kwargs override the resolved sentinel",
              test_multi_models_explicit_override_wins_over_sentinel)

    print("\n=== Identity threshold contract verification ===")
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
