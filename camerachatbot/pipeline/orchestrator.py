import json, os, time
from datetime import datetime, timezone

from camerachatbot.security_config import IDENTITY_THRESHOLDS, COCO_ALLOWLIST, SECURITY_RULES
from camerachatbot.security.tracker import assign_track_ids
from camerachatbot.security.zones import load_zones, bbox_world_and_zone
from camerachatbot.security.events import (
    build_tracks_timeline,
    evaluate_intrusion,
    evaluate_loitering,
    evaluate_unenrolled,
    event_to_dict,
)
from camerachatbot.geometry.homography import CameraCalibration
from camerachatbot.identity.authorization import AuthorizationRegistry
from camerachatbot.video_schema.timing import parse_start_at


def _write_zone_and_world_xy(tracker, calib, zones):
    """Writes `entry["zone_id"]`/`entry["world_xy"]` onto every person entry
    in `tracker.results_json` (Fase 4b / PR8b), via the single shared
    `security.zones.bbox_world_and_zone()` helper -- the SAME helper
    `build_tracks_timeline()` uses per observation, so a person entry's
    `zone_id`/`world_xy` can never drift out of sync with its
    `TrackObservation`'s `zone_id`/`world_xy` (one implementation instead of
    two independently computing `bbox -> world -> zone`).

    Degrades to `None`/`None` when there's no calibration for this camera
    (design.md section 4: "world_xy/zone_id stay None") -- this must never
    raise just because a camera hasn't been calibrated yet.
    """
    for frame, entries in tracker.results_json.items():
        for p in entries:
            if p.get("kind") != "person":
                continue
            bbox = p.get("bbox")
            if calib is None or bbox is None:
                p["zone_id"] = None
                p["world_xy"] = None
                continue
            world_xy, zone = bbox_world_and_zone(bbox, calib, zones)
            p["world_xy"] = [world_xy[0], world_xy[1]]
            p["zone_id"] = zone.id if zone is not None else None


def _resync_json_output(tracker, json_output_path):
    """After `_write_zone_and_world_xy()` mutates `tracker.results_json` in
    place, re-dump it to `json_output_path` (Fase 4b / PR8b).

    Every stage in this pipeline (`enroll_and_assign_global_ids()`, then
    each detail detector's `run_on_json()`) reads/writes its JSON checkpoint
    from DISK, not from `tracker.results_json` directly -- so an in-memory-
    only mutation of `results_json` would be silently lost the moment the
    next stage (the first detail detector) re-reads `json_output_path` from
    disk. This re-dump keeps the checkpoint-per-stage convention intact:
    same file, now carrying `zone_id`/`world_xy`, same `neighborhood`
    ride-along `enroll_and_assign_global_ids()` already wrote.
    """
    neighborhood = None
    try:
        with open(json_output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if isinstance(existing, dict):
            neighborhood = existing.get("neighborhood")
    except Exception as e:
        print(f"[WARN] No se pudo releer {json_output_path} para resync de zona/world_xy: {e}")

    data = dict(tracker.results_json)
    if neighborhood is not None:
        data["neighborhood"] = neighborhood

    # This write overwrites `json_output_path` -- the actual working
    # checkpoint the next pipeline stage reads via `det.run_on_json(...)`.
    # Exception-safe (log and continue): a write failure here must not
    # abort a pipeline run that otherwise succeeded through zones+events;
    # worst case, the on-disk checkpoint simply misses this run's
    # zone_id/world_xy ride-along (the in-memory `tracker.results_json`
    # already has it).
    try:
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[SECURITY-DEGRADED] No se pudo escribir el checkpoint resincronizado "
              f"{json_output_path} (zone_id/world_xy pueden faltar en el JSON en disco): {e}")


def _dump_security_events(output_folder, events):
    """Dumps `events` (list[SecurityEvent]) to `security_events.json` next
    to this run's other stage checkpoints (`tracking_pose_2.json`, etc.),
    consistent with every other stage's checkpoint convention.

    The write is exception-safe (log and continue, returns `None` on
    failure) -- a checkpoint-write failure (e.g. disk full, permissions)
    must not abort a pipeline run that otherwise completed successfully."""
    path = os.path.join(output_folder, "security_events.json")
    try:
        serializable = [event_to_dict(e) for e in events]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[SECURITY-DEGRADED] No se pudo escribir el checkpoint de eventos en {path}: {e}")
        return None
    return path


def _run_zones_events_stage(tracker, camera_id, start_at, fps):
    """Fase 4b (PR8b) + Fase 5 (PR9): classifies each person detection's
    world position/zone and evaluates intrusion/loitering/unenrolled-person
    rules over the resulting tracks timeline. Placed between
    `enroll_and_assign_global_ids()` and the `detail_detectors` loop per
    design.md section 4's diagram.

    Must never raise -- for ANY reason -- out into `multi_models()`: a
    failure here (bad zone data reaching `zone_is_armed()`'s time parsing,
    `fps=0` reaching `frame_timestamp()`'s division, a bug in one of the
    evaluators, a disk write failure) must degrade this stage to
    `events = []` rather than crashing pose classification and Postgres
    persistence for the whole batch. The calib/zones/registry DB lookups
    below keep their OWN finer-grained try/excepts (for better per-lookup
    logging, and because a lookup failure alone is still recoverable -- run
    the stage with `calib=None`/`zones=[]`/`registry=None`); this outer
    try/except is an ADDITIONAL safety net around everything downstream of
    those lookups, which the finer-grained ones do not cover.
    """
    calib = None
    zones = []

    if camera_id is not None:
        try:
            calib = CameraCalibration.load(camera_id)
        except Exception as e:
            # [SECURITY-DEGRADED] (vs. the calm, print-free "camera not yet
            # calibrated" case, where CameraCalibration.load()/load_zones()
            # simply return None/[] without raising): a real lookup failure
            # (e.g. Postgres outage) must be visibly distinguishable in logs
            # from an unconfigured camera, so an operator/alerting rule can
            # tell "not set up" apart from "a real failure just happened
            # during a possible intrusion".
            print(f"[SECURITY-DEGRADED] No se pudo cargar CameraCalibration para camera_id={camera_id}: {e}")
            calib = None
        try:
            zones = load_zones(camera_id)
        except Exception as e:
            print(f"[SECURITY-DEGRADED] No se pudieron cargar zonas para camera_id={camera_id}: {e}")
            zones = []

    # Fase 5 (PR9): AuthorizationRegistry is camera-independent
    # (`authorized_identity` has no `camera_id` column) -- loaded
    # unconditionally, regardless of whether `camera_id` is set/calibrated,
    # so `evaluate_unenrolled` is usable before any camera is calibrated
    # (design.md section 4). A lookup failure degrades to `registry=None`,
    # which SKIPS `evaluate_unenrolled` entirely below rather than falling
    # back to an empty (fail-closed-for-everyone) registry -- an empty
    # registry would flag every currently-tracked person as unenrolled,
    # flooding false positives during what is really a DB outage, not an
    # authorization gap.
    registry = None
    try:
        registry = AuthorizationRegistry.load()
    except Exception as e:
        print(f"[SECURITY-DEGRADED] No se pudo cargar AuthorizationRegistry: {e}")
        registry = None

    try:
        _write_zone_and_world_xy(tracker, calib, zones)

        if start_at:
            try:
                t0 = parse_start_at(start_at)
            except Exception as e:
                print(f"[WARN] No se pudo parsear start_at={start_at!r}, usando now(UTC): {e}")
                t0 = datetime.now(timezone.utc)
        else:
            t0 = datetime.now(timezone.utc)

        timeline = build_tracks_timeline(tracker, calib, zones, t0, fps)

        events = []
        events.extend(evaluate_intrusion(timeline, zones, SECURITY_RULES))
        events.extend(evaluate_loitering(timeline, zones, SECURITY_RULES))
        if registry is not None:
            events.extend(evaluate_unenrolled(timeline, registry, SECURITY_RULES))
    except Exception as e:
        print(f"[SECURITY-DEGRADED] Fase 4b (zonas+eventos) fallo inesperado para "
              f"camera_id={camera_id}, degradando a events=[]: {e}")
        timeline = {}
        events = []

    events_path = _dump_security_events(tracker.output_folder, events)
    n_intrusion = sum(1 for e in events if e.event_type == "intrusion")
    n_loitering = sum(1 for e in events if e.event_type == "loitering")
    n_unenrolled = sum(1 for e in events if e.event_type == "unenrolled_person")
    print(f"[STAGE 4b] camera_id={camera_id}, calib={'yes' if calib else 'no'}, zonas={len(zones)}, "
          f"auth={'yes' if registry is not None else 'no'}, tracks={len(timeline)}, "
          f"eventos={len(events)} (intrusion={n_intrusion}, loitering={n_loitering}, "
          f"unenrolled={n_unenrolled}) -> {events_path}")

    return events


def multi_models(
    yoloPersonReID, detail_detectors, keyframes_path, gallery, eps=0.5, min_samples=4,
    *, t_accept=None, t_reject=None, min_sim_add=None, max_protos_per_person=None,
    camera_id=None, start_at=None, fps=30
):
    # orchestrator.multi_models is the top-level caller for identity thresholds:
    # it resolves any unset value from security_config.IDENTITY_THRESHOLDS and
    # passes explicit values down — the lower-level functions never default.
    if t_accept is None:
        t_accept = IDENTITY_THRESHOLDS["runtime"]["t_accept"]
    if t_reject is None:
        t_reject = IDENTITY_THRESHOLDS["runtime"]["t_reject"]
    if min_sim_add is None:
        min_sim_add = IDENTITY_THRESHOLDS["support"]["min_sim_add"]
    if max_protos_per_person is None:
        max_protos_per_person = IDENTITY_THRESHOLDS["support"]["max_protos_per_person"]

    print("\n[INFO] === multi_models ===")
    print(f"[INFO] keyframes_path: {keyframes_path}")
    print(f"[INFO] params -> eps={eps}, min_samples={min_samples}, t_accept={t_accept}, t_reject={t_reject}")

    start_pre_process = time.time()

    tracker = yoloPersonReID

    tmp_json = tracker.detect_and_embed(allowlist=COCO_ALLOWLIST)

    final_pre_process = f"{time.time() - start_pre_process:.3f}"
    print(f"[STAGE PRE] Total pre-process: {final_pre_process}s")

    start_post_process = time.time()

    # Fase 4a: label_source is the rollback switch between the pre-Fase-4
    # cosine-clustering label source and the new continuous tracker. Default
    # stays "cluster" — this PR adds the CAPABILITY, it does not flip the
    # default. When "tracker" is active, cluster_locally()'s O(N^2)
    # similarity pass is skipped entirely.
    label_source = SECURITY_RULES["label_source"]
    print(f"[INFO] label_source={label_source}")

    if label_source == "tracker":
        labels = assign_track_ids(tracker, SECURITY_RULES["tracker"])
    else:
        labels = tracker.cluster_locally(eps=eps, min_samples=min_samples)

    try:
        n_items = len(labels) if labels is not None else 0
        n_labels = len({l for l in labels if l != -1}) if labels is not None else 0
        print(f"[STAGE POST] label_source={label_source}, items etiquetados: {n_items}, labels distintos (sin ruido): {n_labels}")
    except Exception as _:
        pass

    labels = tracker.enforce_unique_label_per_frame(labels)

    json_output = tracker.enroll_and_assign_global_ids(
        gallery, labels, t_accept=t_accept, t_reject=t_reject,
        min_sim_add=min_sim_add, max_protos_per_person=max_protos_per_person
    )

    # Fase 4b (PR8b): zones + events stage, between enroll_and_assign_global_ids()
    # and the detail_detectors loop per design.md section 4's diagram.
    events = _run_zones_events_stage(tracker, camera_id, start_at, fps)
    if isinstance(json_output, str) and json_output:
        _resync_json_output(tracker, json_output)

    for i, det in enumerate(detail_detectors, 1):
        det.frames_folder = keyframes_path
        json_output = det.run_on_json(json_output)

    final_post_process = f"{time.time() - start_post_process:.3f}"
    print(f"[STAGE POST] Total post-process: {final_post_process}s")

    try:
        if isinstance(json_output, dict):
            print(f"[RESULT] json_output keys: {list(json_output.keys())}")
        else:
            print(f"[RESULT] json_output type: {type(json_output).__name__}")
    except Exception:
        pass

    return json_output, events, final_pre_process, final_post_process
