import os,time

def multi_models(
    yoloPersonReID, detail_detectors, keyframes_path, gallery, eps=0.5, min_samples=4, t_accept=0.80, t_reject=0.45
):
    print("\n[INFO] === multi_models ===")
    print(f"[INFO] keyframes_path: {keyframes_path}")
    print(f"[INFO] params -> eps={eps}, min_samples={min_samples}, t_accept={t_accept}, t_reject={t_reject}")

    start_pre_process = time.time()

    tracker = yoloPersonReID

    tmp_json = tracker.detect_and_embed()

    final_pre_process = f"{time.time() - start_pre_process:.3f}"
    print(f"[STAGE PRE] Total pre-process: {final_pre_process}s")

    start_post_process = time.time()

    labels = tracker.cluster_locally(eps=eps, min_samples=min_samples)

    try:
        n_items = len(labels) if labels is not None else 0
        n_clusters = len({l for l in labels if l != -1}) if labels is not None else 0
        print(f"[STAGE POST] Items clusterizados: {n_items}, clusters (sin ruido): {n_clusters}")
    except Exception as _:
        pass

    labels = tracker.enforce_unique_label_per_frame(labels)

    json_output = tracker.enroll_and_assign_global_ids(gallery, labels, t_accept=t_accept, t_reject=t_reject)

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
