# Security subsystem reference

The modules built specifically for the `security-pipeline-overhaul` program: configuration, continuous tracking, camera calibration, and zone/event logic. This is the part of the codebase evolving fastest, and the part with the most uneven wiring status — each section below states plainly whether the module is actually called by `orchestrator.py`/`pipeline_service.py` today, because that's easy to get wrong by reading the code alone.

## `security_config.py`

The single configuration surface for everything in this file. Convention, stated in its own docstring: imported only by top-level callers (`orchestrator`, `pipeline_service`, `bootstrap`, CLI scripts) — never by `identity/` or `detectors/` internals, which receive resolved values as explicit parameters instead. Every export is frozen (`frozenset`/`tuple`/`MappingProxyType`, including nested dicts) so an accidental in-place mutation (`COCO_ALLOWLIST.add(...)`) raises immediately instead of silently leaking state to every other importer.

| Export | Purpose |
|---|---|
| `COCO_ALLOWLIST` | The 8 class names kept after detection filtering: `person`, `backpack`, `handbag`, `suitcase`, `knife`, `cell phone`, `laptop`, `bottle` |
| `DETECTOR_FLAGS` | Which detail detectors run: `pose=True`, everything else (`face_attention`, `hands`, `emotion`, `age`) `False` |
| `IDENTITY_THRESHOLDS` | Identity-matching thresholds by context: `runtime` (live values, `t_accept=0.80`/`t_reject=0.45`), `enroll` (stricter, for manual enrollment), `cluster` (`eps`/`min_samples` for the default label source), `support` (thresholds for adding extra reference embeddings to a person) |
| `SECURITY_RULES` | `label_source` (`"cluster"` default \| `"tracker"`), `default_loiter_seconds=30`, `intrusion_gap_frames=2`, `unenrolled_debounce_frames=5`, and a nested `tracker` dict of ByteTrack parameters |
| `WEAPONS_DETECTOR` | Reserved placeholder for a future secondary detector (`model_path=None` means it's a no-op) |

## `security/tracker.py` — continuous tracking

**Wired in**: yes. `orchestrator.py` imports `assign_track_ids` directly and calls it whenever `SECURITY_RULES["label_source"] == "tracker"` — confirmed by reading `orchestrator.py`'s imports, not assumed. Off by default (`label_source` defaults to `"cluster"`); reachable by changing one config value.

A from-scratch SORT-style Kalman filter (`KalmanBoxTracker`) plus a ByteTrack-style two-stage association tracker (`ByteTracker`), and the `assign_track_ids(reid, cfg)` driver that runs it as a sequential pass over one video's frames — independent of the detector's inference batch size, so a batch boundary never truncates a track.

**Why the two-stage design matters here**: this pipeline's crops come from a security camera feed, where partial occlusion (someone walking behind furniture or another person) is routine. A tracker that discards every low-confidence detection loses the track the instant confidence dips, then re-identifies the same person as a "new" track a few frames later — exactly the kind of churn that would generate spurious enter/leave events in a security context. `ByteTracker.update()` matches high-confidence detections against every track first, then matches remaining low-confidence detections against whatever's still unmatched — recovering an occluded track from a faint detection instead of discarding it or spawning a new id.

`KalmanBoxTracker` tracks a `confirmed` flag: `True` once a track has ever satisfied the reporting threshold (`hits >= min_hits`, or the tracker-startup exception), and it stays `True` for the track's whole life even though the underlying `hits` counter keeps resetting on missed frames. This is what lets a track recovered after an occlusion gap report again immediately with its original id, instead of re-earning `min_hits` from scratch — real ByteTrack/DeepSORT-family "Lost track re-activation" semantics, not the plainer SORT behavior that would otherwise reintroduce the same churn problem the two-stage matching exists to avoid.

## `security/zones.py` — restricted/monitored/safe zone classification

**Wired in**: no. Only its own tests import it — `orchestrator.py` and `pipeline_service.py` never do, confirmed by grep. Fully built and tested, sitting unused until something calls it.

- `Zone` — a dataclass: `id`, `camera_id`, `name`, `zone_type` (`"restricted"` \| `"monitored"` \| `"safe"`), `polygon` (world coordinates — the same units `CameraCalibration` maps into), `schedule` (optional dict: days/time window/loiter threshold), `is_active`.
- `point_in_polygon(pt, polygon)` — textbook ray-casting, pure Python/NumPy, O(V). Chosen deliberately over OpenCV's `cv2.pointPolygonTest`: world coordinates are floats in metres, and `cv2` wants integer pixel contours — using it here would quantize a metric polygon down to whole units. A point exactly on a polygon edge counts as inside, on purpose: a zone boundary shouldn't silently exclude someone standing on the line.
- `load_zones(camera_id)` — loads active zones for a camera, ordered by `id`. Same graceful-degrade convention as `CameraCalibration.load()` below: returns `[]` if none are configured, rather than raising.
- `zone_is_armed(zone, at)` — `True` if the zone has no schedule (always armed) or `at` falls within the schedule's day-of-week + time window. Correctly handles a schedule that crosses midnight (e.g. `"from": "22:00", "to": "06:00"`): the day-of-week filter is applied to each side of the wrap separately, so a Friday-night-through-Saturday-morning window still reports "armed" at 1am Saturday — a naive same-day check would incorrectly reject that as unarmed, since Saturday itself isn't in a `days: [4]` (Friday) list. This exact bug was found and fixed during development; the fixed logic and a test covering it both exist in the current code.
- `classify_bbox_zone(bbox, calib, zones)` — maps a detection's ground point into world coordinates and returns the first active zone (in list order) whose polygon contains it, or `None`. Deliberately does **not** check `zone_is_armed()` — that's a separate, rule-specific temporal question, left to whichever evaluator needs it (see `evaluate_intrusion` below, the only one that calls `zone_is_armed()`).

## `security/events.py` — intrusion & loitering rule evaluation

**Wired in**: no, same as `zones.py` — no production caller. Pure logic module, explicitly: no database writes, no orchestrator wiring. Both are future work, per this module's own docstring.

- `build_tracks_timeline(reid, calib, zones, t0, fps)` — reads a tracked run's detections and groups them per `track_id`, frame-ordered, into a `TracksTimeline`. **Skips any detection without a `track_id`** — which means this function only produces a non-empty timeline when `label_source == "tracker"` was active for the run that produced `reid`'s results (the default `"cluster"` path never sets `track_id`). This dependency isn't stated anywhere else in the code; if zones/events logic is ever wired into the live pipeline without also flipping `label_source`, it will silently produce empty timelines with no error. Degrades gracefully without a camera calibration (`calib=None`): `world_xy`/`zone_id` are `None` on every observation instead of raising, so identity tracking stays usable even before any camera is calibrated.
- `evaluate_intrusion(timeline, zones, cfg)` — one event per continuous run of observations where the zone is `"restricted"` **and** `zone_is_armed()` says so at that observation's timestamp. Tolerates gaps of up to `cfg["intrusion_gap_frames"]` frames within one run (via a shared `_runs()` helper), so one dropped detection doesn't split a single intrusion into several separate events.
- `evaluate_loitering(timeline, zones, cfg)` — one event per zone whose effective `loiter_seconds` (the zone's own schedule, falling back to `cfg["default_loiter_seconds"]`) is set, per continuous run spent in that zone, emitted only once the run's dwell time meets or exceeds the threshold.
- `evaluate_unenrolled` (unauthorized-person detection) and `evaluate_weapons` are not implemented — deliberately reserved slots for future work, per this module's own docstring.

## `geometry/homography.py` — `CameraCalibration`

**Wired in**: partially. Only the write side (`geometry/calibrate_camera.py`, the manual CLI below) touches the `camera_calibration` table in production code today — the read side, `CameraCalibration.load()`, has no caller outside `zones.py`/`events.py`, which are themselves unwired. Confirmed by grep: nothing in `orchestrator.py`/`pipeline_service.py` imports `CameraCalibration`.

Maps image pixel coordinates to real-world ground-plane coordinates via a single-camera planar homography — a 3×3 transform valid only on the ground plane. `bbox_ground_point()` returns a bbox's bottom-center, not its center: for a standing person the feet are the only bbox point reliably resting on the ground plane, so using the center would place them meters away from their real position.

- `CameraCalibration.load(camera_id)` — loads the active calibration row for a camera, or `None` if none exists. Graceful degrade by design: callers must treat `None` as "skip world-coordinate work for this camera," not an error.
- `image_to_world(pts_xy)` — maps image points to world coordinates.
- `bbox_to_world(bbox)` — the ground-point shortcut most callers actually want.

## `geometry/calibrate_camera.py` — manual calibration CLI

The only code path that writes to the `camera_calibration` table today. Run once per physical camera:

```
python -m camerachatbot.geometry.calibrate_camera --camera-id 1 --image path/to/frame.jpg --units m
```

Opens an OpenCV window over the given image; click ≥4 points on the ground plane, typing each one's real-world coordinate at the terminal prompt. Computes the homography via `cv2.findHomography(..., cv2.RANSAC)` plus a mean reprojection error, then upserts the result (deactivating any prior calibration for the same camera). `reference_points` are stored verbatim so a calibration is auditable and re-runnable without re-clicking.

This is a manual dev tool needing a display and a real ground-plane image — neither available in an automated/CI/sandbox environment. Per its own docstring: **it has not been exercised interactively; verify it on a real workstation before relying on it.**

## `SECURITY_RULES["tracker"]` parameters

Consumed by `assign_track_ids()`/`ByteTracker`, defined in `security_config.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `high_thresh` | 0.5 | Confidence floor for stage-1 matching |
| `low_thresh` | 0.1 | Confidence floor for stage-2 (occlusion-recovery) matching |
| `match_thresh` | 0.8 | Minimum IoU for a detection-to-track match to be accepted |
| `max_age` | 30 | Frames a track survives with no matching detection before being dropped |
| `min_hits` | 3 | Consecutive confirmations a brand-new track needs before being reported |
