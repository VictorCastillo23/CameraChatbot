import os,time

from camerachatbot.security_config import IDENTITY_THRESHOLDS, COCO_ALLOWLIST, SECURITY_RULES
from camerachatbot.security.tracker import assign_track_ids

def multi_models(
    yoloPersonReID, detail_detectors, keyframes_path, gallery, eps=0.5, min_samples=4,
    *, t_accept=None, t_reject=None, min_sim_add=None, max_protos_per_person=None
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

    return json_output, final_pre_process, final_post_process
