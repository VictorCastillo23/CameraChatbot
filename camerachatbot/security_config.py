"""Central configuration for the security-oriented pipeline overhaul.

Single source of truth for detector toggles, identity-matching thresholds,
the COCO class allowlist, and security-rule constants. Plain module-level
dicts, no class, no env parsing beyond what already exists elsewhere in the
codebase.

Convention: this module is imported only by top-level callers
(``orchestrator``, ``pipeline_service``, ``bootstrap``, CLI scripts) — never
by ``identity/`` or ``detectors/`` internals. Those modules receive resolved
values as explicit parameters instead of importing this module directly.

All exported collections below are frozen (``frozenset``/``tuple``) or
wrapped in ``types.MappingProxyType`` (including nested dicts). They are
module-level singletons shared by every importer — mutating a live reference
(e.g. ``COCO_ALLOWLIST.add(...)``) would silently leak state across every
other consumer of this module. Freezing converts that mistake into an
immediate ``TypeError`` at the point of misuse instead of corrupting shared
state. Read access (``in``, ``[]``, ``.get()``, iteration) is unaffected.
"""

from types import MappingProxyType

# Class NAMES (not COCO ids) kept after detection filtering.
COCO_ALLOWLIST = frozenset({
    "person",
    "backpack",
    "handbag",
    "suitcase",
    "knife",
    "cell phone",
    "laptop",
    "bottle",
})

# Which detail detectors run in `pipeline_service.build_detail_detectors()`.
DETECTOR_FLAGS = MappingProxyType({
    "pose": True,
    "face_attention": False,
    "hands": False,
    "emotion": False,
    "age": False,
})

# Identity-matching thresholds, centralized so every call site resolves
# from here instead of carrying its own (possibly stale) local default.
IDENTITY_THRESHOLDS = MappingProxyType({
    "runtime": MappingProxyType({"t_accept": 0.80, "t_reject": 0.45}),  # live values today
    "enroll": MappingProxyType({"t_accept": 0.85, "t_reject": 0.45}),   # stricter: manual enrollment
    "cluster": MappingProxyType({"eps": 0.5, "min_samples": 4}),
    "support": MappingProxyType({"min_sim_add": 0.68, "max_protos_per_person": 5}),
})

# Zone/event/tracking rule constants consumed starting in Fase 4.
SECURITY_RULES = MappingProxyType({
    "label_source": "cluster",  # "cluster" | "tracker" (Fase 4 flips to "tracker")
    "default_loiter_seconds": 30,
    "intrusion_gap_frames": 2,
    "unenrolled_debounce_frames": 5,
    "tracker": MappingProxyType({
        "high_thresh": 0.5,
        "low_thresh": 0.1,
        "match_thresh": 0.8,
        "max_age": 30,
        "min_hits": 3,
    }),
})

# Placeholder for the Fase 6 secondary (weapons) detector. Unset model_path
# means `detectors.secondary_detector.AllowlistedDetector` stays a no-op.
WEAPONS_DETECTOR = MappingProxyType({"model_path": None, "class_names": (), "conf": 0.35})
