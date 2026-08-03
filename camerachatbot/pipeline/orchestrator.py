import json, os, time
from datetime import datetime, timezone

from camerachatbot.security_config import IDENTITY_THRESHOLDS, COCO_ALLOWLIST, SECURITY_RULES
from camerachatbot.security.tracker import assign_track_ids
from camerachatbot.security.zones import load_zones, bbox_world_and_zone
from camerachatbot.security.events import (
    build_tracks_timeline,
    evaluate_intrusion,
    evaluate_loitering,
    event_to_dict,
)
from camerachatbot.geometry.homography import CameraCalibration
from camerachatbot.video_schema.timing import parse_start_at


def _write_zone_and_world_xy(reid, calib, zones):
    """Writes `entry["zone_id"]`/`entry["world_xy"]` onto every person entry
    in `reid.results_json` (Fase 4b / PR8b), via the single shared
    `security.zones.bbox_world_and_zone()` helper -- the SAME helper
    `build_tracks_timeline()` uses per observation, so a person entry's
    `zone_id`/`world_xy` can never drift out of sync with its
    `TrackObservation`'s `zone_id`/`world_xy` (one implementation instead of
    two independently computing `bbox -> world -> zone`).

    Degrades to `None`/`None` when there's no calibration for this camera
    (design.md section 4: "world_xy/zone_id stay None") -- this must never
    raise just because a camera hasn't been calibrated yet.
    """
    for frame, entries in reid.results_json.items():
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

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def _dump_security_events(output_folder, events):
    """Dumps `events` (list[SecurityEvent]) to `security_events.json` next
    to this run's other stage checkpoints (`tracking_pose_2.json`, etc.),
    consistent with every other stage's checkpoint convention."""
    path = os.path.join(output_folder, "security_events.json")
    serializable = [event_to_dict(e) for e in events]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    return path


def _run_zones_events_stage(tracker, camera_id, start_at, fps):
    """Fase 4b (PR8b): classifies each person detection's world position/
    zone and evaluates intrusion/loitering rules over the resulting tracks
    timeline. Placed between `enroll_and_assign_global_ids()` and the
    `detail_detectors` loop per design.md section 4's diagram.

    Degrades gracefully -- must never raise just because a camera hasn't
    been calibrated/zoned yet, or `camera_id` wasn't supplied at all (no
    caller currently threads a real one through `run_pipeline_and_persist()`,
    see apply-progress): tracking and `person_global_id` assignment (4a) are
    calibration-independent, `build_tracks_timeline()` already handles
    `calib=None` internally (world_xy/zone_id stay `None`), and
    `evaluate_intrusion`/`evaluate_loitering` over an empty `zones` list
    simply yield no events.
    """
    calib = None
    zones = []

    if camera_id is not None:
        try:
            calib = CameraCalibration.load(camera_id)
        except Exception as e:
            print(f"[WARN] No se pudo cargar CameraCalibration para camera_id={camera_id}: {e}")
            calib = None
        try:
            zones = load_zones(camera_id)
        except Exception as e:
            print(f"[WARN] No se pudieron cargar zonas para camera_id={camera_id}: {e}")
            zones = []

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

    events_path = _dump_security_events(tracker.output_folder, events)
    print(f"[STAGE 4b] zonas+eventos: calib={'yes' if calib else 'no'}, zonas={len(zones)}, "
          f"tracks={len(timeline)}, eventos={len(events)} -> {events_path}")

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
