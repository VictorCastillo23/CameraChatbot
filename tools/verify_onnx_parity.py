"""GATE.1 — real-hardware ONNX parity verification. USER-RUN, not automated CI.

**Fase 3c note**: `camerachatbot/runtime/loaders_torch.py` (this script's torch
comparison side) was deleted once GATE.1 passed and Fase 3b was exercised on
real data — see `sdd/security-pipeline-overhaul/gate1-signoff`. This script
is kept for historical/audit reference (the exact parity methodology GATE.1
was signed off against) but can no longer run: `main()`'s `loaders_torch`
import at the bottom of this file will raise `ModuleNotFoundError`. Reviving
it would require reverting to a pre-Fase-3c commit.

Imports `camerachatbot.runtime.loaders_torch` and
`camerachatbot.runtime.loaders_onnx` side by side and, for every image in
`keyFrames/`, compares the torch/ultralytics stack against the ONNX
Runtime stack across all three models this program's Fase 3 touches:

1. **Detector** (`YOLO` vs `YOLOOnnxDetector`). Boxes are greedily matched
   by descending IoU (class-agnostic matching — a class mismatch surfaces
   as a MATCHED pair with differing `cls_id`, not two silently-unmatched
   boxes). Per matched pair: `IoU >= 0.90`, identical `cls_id`,
   `|conf_torch - conf_onnx| <= 0.02`. Additionally: no UNMATCHED box on
   either side may have `conf >= 0.25` — a pairs-only check would miss a
   wholesale missing/spurious detection.
2. **ReID** (`torchreid` OSNet vs `OSNetOnnxEmbedder`). Crops come from the
   TORCH detector's boxes ONLY, so both embedders see byte-identical
   pixels — detector disagreement must never contaminate this comparison.
   Assert `cosine(e_torch, e_onnx) >= 0.99` per crop.
3. **Pose classifier** (`YOLO`-CLS vs `PoseClsOnnxClassifier`). Same
   torch-derived person crops. Assert identical `argmax` class and
   `|p_torch - p_onnx| <= 0.02`.

Prints a per-image failure table (the failure SHAPE matters more than the
pass/fail bit — near-zero IoU everywhere means the wrong detector
output-layout branch; a uniformly ~0.85 cosine means wrong OSNet
normalization; systematically shifted probabilities mean a wrong YOLO-CLS
transform) plus aggregate min/mean per metric. `sys.exit(0)` ONLY when
every check passes on every image; any failure is a non-zero exit — the
explicit signal that Fase 3b (flipping `bootstrap.py`'s loader switch line)
must NOT proceed.

Requires the real `models/*.pt` weights AND their ONNX exports
(`python tools/export_to_onnx.py --all` first) AND `keyFrames/` populated
with representative real frames. Usage:

    python tools/export_to_onnx.py --all
    python tools/verify_onnx_parity.py
    python tools/verify_onnx_parity.py --limit 5      # first 5 images only
    python tools/verify_onnx_parity.py --conf 0.25 --iou 0.45
"""

import argparse
import sys

import cv2
import numpy as np

from camerachatbot import paths
from camerachatbot.geometry.bbox_utils import iou as bbox_iou

DET_IOU_MIN = 0.90
DET_CONF_TOL = 0.02
DET_UNMATCHED_CONF_FLOOR = 0.25
REID_COSINE_MIN = 0.99
POSECLS_PROB_TOL = 0.02


# ---------------------------------------------------------------------------
# Pure comparison-logic helpers — no model loading, independently testable
# with synthetic data (see tests_manual/... for exercised coverage of these).
# ---------------------------------------------------------------------------

def greedy_match_boxes(dets_a, dets_b, min_iou: float = 0.0):
    """Greedily pair `dets_a[i]` with `dets_b[j]` by descending IoU.

    `dets_a`/`dets_b` are lists of `{"bbox": [x1,y1,x2,y2], ...}` dicts.
    Matching is bbox-only / class-agnostic on purpose: a class mismatch
    between two spatially-overlapping boxes should surface as a MATCHED
    pair with differing `cls_id` (caught by the caller's cls_id assertion),
    not silently become two separate "unmatched" boxes that a downstream
    `conf >= 0.25` check might not even flag.

    Returns `(matches, unmatched_a, unmatched_b)` where `matches` is a list
    of `(i, j, iou)` tuples, and `unmatched_a`/`unmatched_b` are index lists.
    """
    candidates = []
    for i, da in enumerate(dets_a):
        for j, db in enumerate(dets_b):
            iou_val = bbox_iou(da["bbox"], db["bbox"])
            if iou_val > min_iou:
                candidates.append((iou_val, i, j))
    candidates.sort(key=lambda t: t[0], reverse=True)

    matched_a, matched_b = set(), set()
    matches = []
    for iou_val, i, j in candidates:
        if i in matched_a or j in matched_b:
            continue
        matched_a.add(i)
        matched_b.add(j)
        matches.append((i, j, iou_val))

    unmatched_a = [i for i in range(len(dets_a)) if i not in matched_a]
    unmatched_b = [j for j in range(len(dets_b)) if j not in matched_b]
    return matches, unmatched_a, unmatched_b


def extract_detections(det_result):
    """Duck-typed `Results`/`DetResult` (torch OR onnx, same iteration shape
    per `camerachatbot/detectors/onnx_yolo.py`'s docstring) -> list of plain
    dicts, so the rest of this tool never branches on which backend produced
    a detection.
    """
    boxes = det_result.boxes
    names = det_result.names
    out = []
    for b in boxes:
        xyxy = np.asarray(b.xyxy).reshape(-1)[:4].astype(float).tolist()
        conf = float(np.asarray(b.conf).reshape(-1)[0])
        cls_id = int(np.asarray(b.cls).reshape(-1)[0])
        cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        out.append({"bbox": xyxy, "conf": conf, "cls_id": cls_id, "cls_name": cls_name})
    return out


def compare_detections(dets_t, dets_o):
    """Compare two already-extracted detection lists. Returns a report dict."""
    matches, unmatched_t, unmatched_o = greedy_match_boxes(dets_t, dets_o)

    iou_values, conf_diffs, cls_mismatches = [], [], []
    for i, j, iou_val in matches:
        iou_values.append(iou_val)
        conf_diffs.append(abs(dets_t[i]["conf"] - dets_o[j]["conf"]))
        if dets_t[i]["cls_id"] != dets_o[j]["cls_id"]:
            cls_mismatches.append((i, j, dets_t[i]["cls_id"], dets_o[j]["cls_id"]))

    unmatched_high_conf = []
    for i in unmatched_t:
        if dets_t[i]["conf"] >= DET_UNMATCHED_CONF_FLOOR:
            unmatched_high_conf.append(("torch", i, dets_t[i]["conf"]))
    for j in unmatched_o:
        if dets_o[j]["conf"] >= DET_UNMATCHED_CONF_FLOOR:
            unmatched_high_conf.append(("onnx", j, dets_o[j]["conf"]))

    low_iou = [v for v in iou_values if v < DET_IOU_MIN]
    high_conf_diff = [d for d in conf_diffs if d > DET_CONF_TOL]

    passed = not low_iou and not high_conf_diff and not cls_mismatches and not unmatched_high_conf

    return {
        "n_torch": len(dets_t),
        "n_onnx": len(dets_o),
        "n_matched": len(matches),
        "iou_values": iou_values,
        "conf_diffs": conf_diffs,
        "cls_mismatches": cls_mismatches,
        "unmatched_high_conf": unmatched_high_conf,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Model-backed comparisons — require real weights, real images.
# ---------------------------------------------------------------------------

def run_torch_detector(model, img_bgr, conf, iou_thr):
    results = model(img_bgr, verbose=False, conf=conf, iou=iou_thr)
    return results[0] if isinstance(results, (list, tuple)) else results


def torch_reid_embed(reid_model, transform, device, crop_bgr):
    import torch
    from PIL import Image

    if crop_bgr is None or crop_bgr.size == 0:
        return None
    img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    ten = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = reid_model(ten).detach().cpu().numpy()[0]
    feat = feat / (np.linalg.norm(feat) + 1e-12)
    return feat.astype(np.float32)


def torch_posecls_predict(model, crops, imgsz=224, batch=16):
    """Mirrors `pose_action_classifier.py::run_on_json()`'s own
    `results[i].probs` parsing, kept in lockstep with that logic since this
    tool exists specifically to validate the ONNX replacement against it.
    """
    import torch

    if not crops:
        return []

    use_half = torch.cuda.is_available()
    with torch.inference_mode():
        results = model(crops, imgsz=imgsz, batch=batch, half=use_half, verbose=False)
    if not isinstance(results, (list, tuple)):
        results = list(results)

    preds = []
    for r in results:
        probs_obj = getattr(r, "probs", None)
        if probs_obj is None:
            preds.append((0, 0.0))
            continue
        scores = probs_obj.data.detach().cpu().numpy()
        if scores.ndim > 1:
            scores = scores[0]
        pred_idx = int(np.argmax(scores))
        conf = float(scores[pred_idx])
        preds.append((pred_idx, conf))
    return preds


def cosine(a, b):
    if a is None or b is None:
        return None
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return None
    return float(np.dot(a, b) / (na * nb))


def compare_reid_and_posecls(img_bgr, dets_t, torch_reid, torch_transform, torch_device,
                              onnx_reid, torch_posecls, onnx_posecls, person_cls_name="person"):
    H, W = img_bgr.shape[:2]
    person_dets = [d for d in dets_t if d["cls_name"] == person_cls_name]

    crops = []
    for d in person_dets:
        x1, y1, x2, y2 = [int(round(v)) for v in d["bbox"]]
        x1, x2 = max(0, min(x1, W)), max(0, min(x2, W))
        y1, y2 = max(0, min(y1, H)), max(0, min(y2, H))
        crop = img_bgr[y1:y2, x1:x2]
        crops.append(crop if crop.size > 0 else None)

    reid_cosines = []
    for crop in crops:
        e_t = torch_reid_embed(torch_reid, torch_transform, torch_device, crop)
        e_o = onnx_reid.embed_one(crop) if crop is not None else None
        c = cosine(e_t, e_o)
        if c is not None:
            reid_cosines.append(c)

    valid_crops = [c for c in crops if c is not None]
    torch_preds = torch_posecls_predict(torch_posecls, valid_crops)
    onnx_preds = onnx_posecls.predict(valid_crops)

    posecls_prob_diffs = []
    posecls_cls_mismatches = 0
    for (idx_t, p_t), (idx_o, p_o) in zip(torch_preds, onnx_preds):
        posecls_prob_diffs.append(abs(p_t - p_o))
        if idx_t != idx_o:
            posecls_cls_mismatches += 1

    reid_passed = all(c >= REID_COSINE_MIN for c in reid_cosines) if reid_cosines else True
    posecls_passed = (
        all(d <= POSECLS_PROB_TOL for d in posecls_prob_diffs) and posecls_cls_mismatches == 0
    )

    return {
        "n_person_crops": len(person_dets),
        "reid_cosines": reid_cosines,
        "reid_passed": reid_passed,
        "posecls_prob_diffs": posecls_prob_diffs,
        "posecls_cls_mismatches": posecls_cls_mismatches,
        "posecls_passed": posecls_passed,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _fmt_stats(values):
    if not values:
        return "n/a"
    return f"min={min(values):.4f} mean={sum(values) / len(values):.4f}"


def _fmt_stats_max(values):
    """Same as `_fmt_stats()` but also reports `max` — aggregate min/mean alone
    can hide a single large outlier (e.g. one 0.068 conf_diff buried in a
    mean of 0.008 across dozens of matched pairs). Kept as a separate helper
    rather than changing `_fmt_stats()`'s output shape everywhere, since not
    every caller wants max (e.g. IoU's max is always uninteresting — 1.0-ish).
    """
    if not values:
        return "n/a"
    return f"min={min(values):.4f} mean={sum(values) / len(values):.4f} max={max(values):.4f}"


def print_report(per_image_rows):
    print("\n=== tools/verify_onnx_parity.py — per-image report ===")
    for name, det_row, extra_row in per_image_rows:
        status = "PASS" if (det_row["passed"] and extra_row["reid_passed"] and extra_row["posecls_passed"]) else "FAIL"
        print(f"\n[{status}] {name}")
        print(
            f"  detector: torch={det_row['n_torch']} onnx={det_row['n_onnx']} "
            f"matched={det_row['n_matched']} "
            f"IoU[{_fmt_stats(det_row['iou_values'])}] "
            f"conf_diff[{_fmt_stats_max(det_row['conf_diffs'])}] "
            f"cls_mismatches={len(det_row['cls_mismatches'])} "
            f"unmatched_high_conf={len(det_row['unmatched_high_conf'])}"
        )
        # SUGGESTION from PR4b's gate review: a bare unmatched-box COUNT let a
        # prior batch's own narrative undercount this — print the actual
        # side/index/confidence of every unmatched high-confidence box so a
        # future run's summary can't repeat that mistake.
        for side, idx, conf in det_row["unmatched_high_conf"]:
            print(f"    unmatched high-conf box: side={side} idx={idx} conf={conf:.4f}")
        print(
            f"  reid: n_crops={extra_row['n_person_crops']} "
            f"cosine[{_fmt_stats(extra_row['reid_cosines'])}]"
        )
        print(
            f"  posecls: prob_diff[{_fmt_stats_max(extra_row['posecls_prob_diffs'])}] "
            f"cls_mismatches={extra_row['posecls_cls_mismatches']}"
        )

    all_iou = [v for _, d, _ in per_image_rows for v in d["iou_values"]]
    all_conf_diff = [v for _, d, _ in per_image_rows for v in d["conf_diffs"]]
    all_cosine = [v for _, _, e in per_image_rows for v in e["reid_cosines"]]
    all_prob_diff = [v for _, _, e in per_image_rows for v in e["posecls_prob_diffs"]]
    all_unmatched_high_conf = [
        (name, side, idx, conf)
        for name, d, _ in per_image_rows
        for side, idx, conf in d["unmatched_high_conf"]
    ]

    print("\n=== aggregate ===")
    print(f"detector IoU:        {_fmt_stats(all_iou)}")
    print(f"detector conf diff:  {_fmt_stats_max(all_conf_diff)}")
    print(f"reid cosine:         {_fmt_stats(all_cosine)}")
    print(f"posecls prob diff:   {_fmt_stats_max(all_prob_diff)}")
    print(
        f"unmatched high-conf boxes: {len(all_unmatched_high_conf)} total "
        f"across {len({name for name, *_ in all_unmatched_high_conf})} image(s)"
    )
    if all_unmatched_high_conf:
        confs = [c for *_, c in all_unmatched_high_conf]
        print(f"  confidence range: min={min(confs):.4f} max={max(confs):.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--limit", type=int, default=None, help="only check the first N images")
    args = parser.parse_args()

    keyframes_dir = paths.KEYFRAMES_SAMPLE_DIR
    exts = (".jpg", ".jpeg", ".png")
    image_paths = sorted(
        p for p in keyframes_dir.glob("*") if p.suffix.lower() in exts
    ) if keyframes_dir.exists() else []

    if not image_paths:
        print(f"[verify_onnx_parity] No images found in {keyframes_dir} — nothing to verify.")
        return 1

    if args.limit:
        image_paths = image_paths[: args.limit]

    # Import here, not at module level: importing either loader module
    # requires the real torch/ultralytics/torchreid/onnxruntime stack AND
    # (for loaders_torch) the real .pt weights AND (for loaders_onnx) the
    # real ONNX exports — none of which should be a hard import-time
    # requirement just to run `--help` or exercise the pure comparison
    # helpers above from a test script.
    from camerachatbot.runtime import loaders_torch, loaders_onnx

    print("[verify_onnx_parity] Loading torch models...")
    torch_det = loaders_torch.load_detector()
    torch_posecls = loaders_torch.load_posecls()
    torch_reid, torch_transform = loaders_torch.load_reid()
    torch_device = loaders_torch.get_device()

    print("[verify_onnx_parity] Loading ONNX models...")
    onnx_det = loaders_onnx.load_detector()
    onnx_posecls = loaders_onnx.load_posecls()
    onnx_reid = loaders_onnx.load_reid()

    per_image_rows = []
    for path in image_paths:
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            print(f"[verify_onnx_parity] WARN: could not read {path}, skipping.")
            continue

        torch_result = run_torch_detector(torch_det, img_bgr, args.conf, args.iou)
        onnx_result = onnx_det(img_bgr, verbose=False, conf=args.conf, iou=args.iou)[0]

        dets_t = extract_detections(torch_result)
        dets_o = extract_detections(onnx_result)
        det_row = compare_detections(dets_t, dets_o)

        extra_row = compare_reid_and_posecls(
            img_bgr, dets_t, torch_reid, torch_transform, torch_device,
            onnx_reid, torch_posecls, onnx_posecls,
        )

        per_image_rows.append((path.name, det_row, extra_row))

    print_report(per_image_rows)

    all_passed = all(
        d["passed"] and e["reid_passed"] and e["posecls_passed"] for _, d, e in per_image_rows
    )
    if not all_passed:
        print(
            "\n[verify_onnx_parity] FAIL — at least one check failed on at least one "
            "image. Fase 3b (flipping bootstrap.py's loader switch line) must NOT proceed."
        )
        return 1

    print(f"\n[verify_onnx_parity] PASS — all checks passed on all {len(per_image_rows)} images.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
