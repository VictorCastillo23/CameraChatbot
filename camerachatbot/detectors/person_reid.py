import os, glob, cv2, json,numpy as np,torch,re,time
from ultralytics import YOLO
from collections import defaultdict
from sklearn.cluster import DBSCAN
from PIL import Image
from time import perf_counter as now
class YOLOPersonReID:
    def __init__(self,model: YOLO,frames_folder=None,output_folder=None,transform=None,
                 reid_model=None,device=None,depth_model=None,depth_transform=None,save_deph=False):
        self.model = model
        self.frames_folder=frames_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

        self.reid_transform = transform
        self.reid_model = reid_model.eval().to(device) if reid_model is not None else None

        self.depth_model = depth_model.eval().to(device) if depth_model is not None else None
        self.depth_transform = depth_transform

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.person_class_id = 0
        self.embeddings = []
        self.metadata = []
        self.save_deph = save_deph
        self.results_json = defaultdict(list)

    def extract_embedding(self, crop):
        if self.reid_model is None or self.reid_transform is None:
            return None
        if crop is None or crop.size == 0:
            return None
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ten = self.reid_transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.reid_model(ten).detach().cpu().numpy()[0]
        feat = feat / (np.linalg.norm(feat) + 1e-12)
        return feat.astype("float32")

    def _compute_depth_map(self, img_rgb):
        if self.depth_model is None or self.depth_transform is None:
            return None, None

        input_batch = self.depth_transform(img_rgb).to(self.device)
        with torch.no_grad():
            pred = self.depth_model(input_batch)

        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map = pred.detach().cpu().numpy().astype(np.float32)
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return depth_map, depth_norm

    @staticmethod
    def _depth_stats(roi):
        if roi is None or roi.size == 0:
            return None
        roi_flat = roi[np.isfinite(roi)]
        if roi_flat.size == 0:
            return None
        return {
            "mean": float(np.mean(roi_flat)),
            "median": float(np.median(roi_flat)),
            "min": min(0.0,float(np.min(roi_flat)))
        }

    def detect_and_embed(self,
                         conf=0.25,
                         iou=0.45,
                         batch_size=16,
                         exts=(".jpg", ".jpeg", ".png",),
                         save_outputs=True,
                         *,
                         allowlist
                         ) -> str:

        t = defaultdict(float)
        ctr = defaultdict(int)

        def _acc(name, dt):
            t[name] += float(dt)

        def _natural_key(p):
            b = os.path.basename(p)
            return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r'(\d+)', b)]

        t_total0 = now()
        t_scan0 = now()

        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(self.frames_folder, f"*{ext}")))
        files = sorted(set(files), key=_natural_key)

        _acc("scan_files", now() - t_scan0)

        print(f"[YOLO+ReID+Depth] Procesando {len(files)} imágenes...")
        if not files:
            # Sin I/O: devolvemos la ruta "esperada" pero avisamos que no se guardó nada
            empty_path = os.path.join(self.output_folder, "results_tmp.json")
            print("[INFO] No hay archivos; no se guardó JSON porque save_outputs=False.")
            return empty_path

        idx_global = 0
        t_loop0 = now()

        for start in range(0, len(files), batch_size):
            batch_files = files[start:start + batch_size]

            # 2) Lectura de imágenes
            t_read0 = now()
            batch_imgs, batch_frames = [], []
            for fp in batch_files:
                img_bgr = cv2.imread(fp)
                if img_bgr is None:
                    print(f"[WARN] No se pudo leer: {fp}")
                    continue
                frame = os.path.splitext(os.path.basename(fp))[0]
                batch_imgs.append(img_bgr)
                batch_frames.append((frame, fp))
                if frame not in self.results_json:
                    self.results_json[frame] = []
            _acc("read_images", now() - t_read0)
            ctr["images"] += len(batch_imgs)
            if not batch_imgs:
                continue

            # 3) Inference YOLO (batch, con fallback individual si falla)
            try:
                t_yolo0 = now()
                yres_list = self.model(batch_imgs, verbose=False, conf=conf, iou=iou)
                _acc("yolo_batch", now() - t_yolo0)
            except Exception as e:
                print(f"[ERROR] YOLO batch failed: {e}")
                yres_list = []
                for img in batch_imgs:
                    try:
                        t_fallback0 = now()
                        yres_list.append(self.model(img, verbose=False, conf=conf, iou=iou)[0])
                        _acc("yolo_fallback", now() - t_fallback0)
                    except Exception as ee:
                        print(f"[ERROR] YOLO single failed: {ee}")
                        yres_list.append(None)

            # 4) Post-proceso por imagen: depth_map, depth_stats ROI, ReID y JSON en memoria
            for (frame, fp), img_bgr, yres in zip(batch_frames, batch_imgs, yres_list):
                if yres is None:
                    continue

                # 4.1 depth_map por imagen
                depth_map = None
                if self.save_deph:
                    try:
                        t_dm0 = now()
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        depth_map, _ = self._compute_depth_map(img_rgb)
                        _acc("depth_map", now() - t_dm0)
                    except Exception as e:
                        print(f"[Depth] WARN ({frame}): {e}")

                names = getattr(yres, "names", None) or getattr(self.model, "names", {}) or {}
                boxes = getattr(yres, "boxes", None)
                if boxes is None:
                    continue

                H, W = img_bgr.shape[:2]
                frame_list = self.results_json[frame]

                persons_frame = 0
                objects_frame = 0

                for b in boxes:
                    try:
                        cls_id = int(b.cls[0])
                        x1, y1, x2, y2 = map(lambda v: int(round(float(v))), b.xyxy[0])

                        # clamp a la imagen
                        x1 = max(0, min(W - 1, x1));
                        x2 = max(0, min(W - 1, x2))
                        y1 = max(0, min(H - 1, y1));
                        y2 = max(0, min(H - 1, y2))
                        if x2 <= x1 or y2 <= y1:
                            continue

                        confidence = float(b.conf[0]) if hasattr(b, "conf") and b.conf is not None else None
                        class_name = names.get(cls_id, str(cls_id))

                        # Discard (not just hide) detections outside the allowlist —
                        # never written to frame_list, so they never reach downstream
                        # stages (video schema, Postgres).
                        if class_name not in allowlist:
                            continue

                        # 4.2 depth_stats por ROI (si hay depth_map)
                        depth_info = None
                        if depth_map is not None:
                            roi = depth_map[y1:y2, x1:x2]
                            if roi.size > 0:
                                try:
                                    t_ds0 = now()
                                    depth_info = self._depth_stats(roi)
                                    _acc("depth_stats_roi", now() - t_ds0)
                                except Exception as e:
                                    print(f"[Depth] WARN stats ({frame}): {e}")

                        if cls_id == self.person_class_id:
                            persons_frame += 1
                            rid_assigned = None
                            crop = img_bgr[y1:y2, x1:x2]
                            if crop.size > 0:
                                try:
                                    # 4.3 ReID (embedding)
                                    t_emb0 = now()
                                    emb = self.extract_embedding(crop)
                                    _acc("embedding", now() - t_emb0)
                                    if emb is not None:
                                        self.embeddings.append(emb)
                                        self.metadata.append((frame, [x1, y1, x2, y2]))
                                        rid_assigned = idx_global
                                        idx_global += 1
                                except Exception as e:
                                    print(f"[ReID] WARN ({frame}): {e}")

                            entry = {
                                "kind": "person",
                                "frame": frame,
                                "class_name": "person",
                                "bbox": [x1, y1, x2, y2],
                                "confidence": confidence,
                                "attributes": {"pose": None, "hands": None, "face": None, "eyes": None},
                                "track_id": None,
                                "person_global_id": None
                            }
                            if depth_info is not None:
                                entry["depth"] = depth_info
                            if rid_assigned is not None:
                                entry["_rid"] = rid_assigned
                                entry["user_id"] = rid_assigned
                            frame_list.append(entry)
                        else:
                            objects_frame += 1
                            entry = {
                                "kind": "object",
                                "frame": frame,
                                "class_name": class_name,
                                "bbox": [x1, y1, x2, y2],
                                "confidence": confidence
                            }
                            if depth_info is not None:
                                entry["depth"] = depth_info
                            frame_list.append(entry)

                    except Exception as e:
                        print(f"[Post] WARN ({frame}): {e}")
                        continue

                ctr["persons"] += persons_frame
                ctr["objects"] += objects_frame
                print(f"[Frame {frame}] personas: {persons_frame}, objetos: {objects_frame}")

        if len(self.embeddings) > 0 and save_outputs:
            t_sv0 = now()
            np.save(os.path.join(self.output_folder, "embeddings.npy"),
                    np.array(self.embeddings, dtype="float32"))
            _acc("save_embeddings", now() - t_sv0)

        tmp_json = os.path.join(self.output_folder, "results_tmp.json")
        t_json0 = now()
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(self.results_json, f, indent=4, ensure_ascii=False)
        _acc("save_json", now() - t_json0)

        total_time = now() - t_total0
        _acc("total", total_time)

        # Reporte bonito
        imgs = max(1, ctr["images"])
        print("\n=== PERF REPORT (detect_and_embed) ===")
        ordered_keys = [
            "scan_files",
            "read_images",
            "yolo_batch", "yolo_fallback",
            "depth_map", "depth_stats_roi",
            "embedding",
            # "json_serialize_only",  # si lo activas
            "save_embeddings", "save_json",
            "total"
        ]
        for key in ordered_keys:
            if key in t:
                print(f"{key:>20s}: {t[key]:8.3f} s   ({t[key] / imgs:.4f} s/img)")

        print(
            f"imgs procesadas: {ctr['images']}, personas totales: {ctr['persons']}, objetos totales: {ctr['objects']}")
        print(f"save_outputs={save_outputs} -> {'SE guardaron archivos' if save_outputs else 'NO se guardó nada'}")
        print("=== END PERF REPORT ===\n")

        # Conservamos la misma interfaz de retorno
        return tmp_json

    def cluster_locally(self, eps=0.5, min_samples=4):
        if not self.embeddings:
            return None
        E = np.array(self.embeddings, dtype="float32")
        # ✅ normaliza (igual que haces en enroll_and_assign_global_ids)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(E)

        rid2label = {rid: int(lab) for rid, lab in enumerate(labels)}
        for frame, entries in self.results_json.items():
            for p in entries:
                if p.get("kind") == "person" and "_rid" in p:
                    p["track_id"] = rid2label.get(int(p["_rid"]), -1)
        return labels

    def enforce_unique_label_per_frame(self, labels: np.ndarray) -> np.ndarray:
        if labels is None:
            return None
        from collections import defaultdict
        labels = labels.copy()
        frame_lab_rids = defaultdict(lambda: defaultdict(list))

        for frame, entries in self.results_json.items():
            for p in entries:
                if p.get("kind") != "person" or "_rid" not in p:
                    continue
                rid = int(p["_rid"])
                lab = int(labels[rid])
                frame_lab_rids[frame][lab].append(rid)

        for frame, lab_dict in frame_lab_rids.items():
            for lab, rids in lab_dict.items():
                if lab == -1 or len(rids) <= 1:
                    continue
                for rid in rids[1:]:
                    labels[rid] = -1
        return labels

    def _score_for_person(self, p):
        ev = p.get("person_global_evidence") or {}
        if "best_sim" in ev and ev["best_sim"] is not None:
            return float(ev["best_sim"])
        return float(p.get("confidence", 0.0))

    def _enforce_unique_global_per_frame_post(self, E_norm, gallery, *, t_accept, t_reject,
                                               min_sim_add, max_protos_per_person):

        for frame, entries in self.results_json.items():
            # agrupa por pid global
            pid_groups = {}
            for idx, p in enumerate(entries):
                if p.get("kind") != "person":
                    continue
                pid = p.get("person_global_id")
                if pid is None:
                    continue
                pid = int(pid)
                pid_groups.setdefault(pid, []).append((idx, p))

            # para cada pid con más de una detección en el frame…
            for pid, items in pid_groups.items():
                if len(items) <= 1:
                    continue

                # ordena por "score" descendente y conserva el primero
                items_sorted = sorted(items, key=lambda it: self._score_for_person(it[1]), reverse=True)
                keep_idx, keep_p = items_sorted[0]
                used_pids = {int(pid)}  # ya usado en frame

                # procesar duplicados (perdedores)
                for dup_idx, dup_p in items_sorted[1:]:
                    # intenta reasignar a un pid diferente ya existente entre sus top_hits
                    ev = dup_p.get("person_global_evidence") or {}
                    hits = ev.get("top_hits", []) or []

                    reassigned = False
                    # 1) usa top_hits si hay candidatos válidos y no usados en el frame
                    for h in hits:
                        cand_pid = int(h.get("person_global_id", -1))
                        sim = float(h.get("sim", 0.0))
                        if cand_pid < 0 or cand_pid in used_pids:
                            continue
                        if sim >= t_accept:
                            dup_p["person_global_id"] = cand_pid
                            dup_p["user_id"] = cand_pid
                            dup_p["person_global_evidence"] = {
                                "best_sim": sim,
                                "created": False,
                                "top_hits": hits[:3]
                            }
                            used_pids.add(cand_pid)
                            reassigned = True
                            break

                    if reassigned:
                        continue

                    # 2) si top_hits no sirvió: busca en galería con el embedding del rid
                    rid = dup_p.get("_rid")
                    e = None
                    if rid is not None:
                        rid = int(rid)
                        if 0 <= rid < len(E_norm):
                            e = E_norm[rid]

                    if e is not None:
                        search_hits = gallery.search(e, k=10) or []
                        # elige el primer pid no usado y con sim aceptable
                        picked = None
                        for h in search_hits:
                            cand_pid = int(h.get("person_global_id", -1))
                            sim = float(h.get("sim", 0.0))
                            if cand_pid >= 0 and cand_pid not in used_pids and sim >= t_accept:
                                picked = (cand_pid, sim, search_hits[:3])
                                break

                        if picked:
                            cand_pid, sim, top3 = picked
                            dup_p["person_global_id"] = cand_pid
                            dup_p["user_id"] = cand_pid
                            dup_p["person_global_evidence"] = {
                                "best_sim": sim,
                                "created": False,
                                "top_hits": top3
                            }
                            # (opcional) aprendizaje incremental
                            gallery.maybe_add_support_prototype(
                                cand_pid, e, meta={"type": "support_from_reassign"},
                                min_sim_add=min_sim_add, max_protos_per_person=max_protos_per_person
                            )
                            used_pids.add(cand_pid)
                            continue

                        # 3) zona gris / sin buenos matches → crear identidad nueva
                        top_sim = float(search_hits[0]["sim"]) if (search_hits and "sim" in search_hits[0]) else 0.0
                        if (not search_hits) or top_sim <= t_reject:
                            new_pid = int(gallery.create_person([{
                                "embedding": e,
                                "proto_id": f"dup_{rid}",
                                "modality": "body",
                                "meta": {"type": "dup_created"}
                            }]))
                            dup_p["person_global_id"] = new_pid
                            dup_p["user_id"] = new_pid
                            dup_p["person_global_evidence"] = {
                                "best_sim": top_sim,
                                "created": True,
                                "top_hits": search_hits[:3] if search_hits else []
                            }
                            used_pids.add(new_pid)
                        else:
                            # zona gris → crea para no dejar colisión
                            new_pid = int(gallery.create_person([{
                                "embedding": e,
                                "proto_id": f"dup_gray_{rid}",
                                "modality": "body",
                                "meta": {"type": "dup_created_gray"}
                            }]))
                            dup_p["person_global_id"] = new_pid
                            dup_p["user_id"] = new_pid
                            dup_p["person_global_evidence"] = {
                                "best_sim": top_sim,
                                "created": True,
                                "top_hits": search_hits[:3]
                            }
                            used_pids.add(new_pid)
                    else:
                        # sin embedding (no _rid) → no podemos reasignar inteligentemente
                        # para no duplicar pid, dejamos sin id global
                        dup_p["person_global_id"] = None
                        dup_p["user_id"] = None
                        ev = dup_p.get("person_global_evidence") or {}
                        ev["created"] = ev.get("created", False)
                        dup_p["person_global_evidence"] = ev

    def enroll_and_assign_global_ids(self, gallery, labels, *, t_accept, t_reject,
                                      min_sim_add, max_protos_per_person):
        cluster_debug = {}

        if labels is None or len(labels) == 0:
            return

        E = np.asarray(self.embeddings, dtype="float32")
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

        label_to_rids = defaultdict(list)
        outliers = []
        for rid, lab in enumerate(labels):
            lab = int(lab)
            if lab >= 0:
                label_to_rids[lab].append(int(rid))
            else:
                outliers.append(int(rid))

        def _assign_pid_and_evidence(rids, pid, evidence_dict):
            for frame, persons in self.results_json.items():
                for p in persons:
                    if p.get("kind") != "person":
                        continue
                    rid_p = int(p.get("_rid", -1))
                    if rid_p in rids:
                        if pid is not None:
                            p["person_global_id"] = int(pid)
                            p["user_id"] = int(pid)
                        p["person_global_evidence"] = evidence_dict

        for lab, rids in label_to_rids.items():
            centroid = E[rids].mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

            (pid, sim, created), hits = gallery.assign_or_create(
                centroid,
                modality="body",
                t_accept=t_accept,
                t_reject=t_reject,
                k=30,
                proto_meta={"type": "body_centroid", "size": len(rids)},
                return_hits=True
            )

            evidence = {"best_sim": float(sim), "created": bool(created), "top_hits": hits[:3]}
            cluster_debug[int(lab)] = evidence

            _assign_pid_and_evidence(rids, pid, evidence)

            if not created:
                gallery.maybe_add_support_prototype(
                    pid, centroid,
                    meta={"type": "support_from_centroid", "size": len(rids)},
                    min_sim_add=min_sim_add, max_protos_per_person=max_protos_per_person
                )

            if gallery.count_active_prototypes(pid) >= 6:
                gallery.consolidate_person(pid, keep_k=4, add_centroid=True, deactivate_old=True)

        for rid in outliers:
            e = E[rid]
            hits = gallery.search(e, k=10)

            top_sim = float(hits[0]["sim"]) if (hits and "sim" in hits[0]) else 0.0
            evidence = {}
            evidence = {"best_sim": top_sim, "created": False, "top_hits": hits[:3] if hits else []}

            if hits and top_sim >= t_accept:
                pid = int(hits[0]["person_global_id"])
                evidence.update({"created": False})
                gallery.maybe_add_support_prototype(
                    pid, e, meta={"type": "support_from_outlier"},
                    min_sim_add=min_sim_add, max_protos_per_person=max_protos_per_person
                )

            elif (not hits) or top_sim <= t_reject:
                pid = int(gallery.create_person([{
                    "embedding": e,
                    "proto_id": f"out_{rid}",
                    "modality": "body",
                    "meta": {"type": "outlier_single"}
                }]))
                evidence.update({"created": True})

            else:
                pid = int(gallery.create_person([{
                    "embedding": e,
                    "proto_id": f"out_{rid}",
                    "modality": "body",
                    "meta": {"type": "outlier_single_gray"}
                }]))
                evidence.update({"created": True})

            _assign_pid_and_evidence([rid], pid, evidence)

            if gallery.count_active_prototypes(pid) >= 6:
                gallery.consolidate_person(pid, keep_k=4, add_centroid=True, deactivate_old=True)

        # ... al final de enroll_and_assign_global_ids, justo antes de volcar a disco:

        # E ya está normalizado más arriba:
        self._enforce_unique_global_per_frame_post(
            E, gallery, t_accept=t_accept, t_reject=t_reject,
            min_sim_add=min_sim_add, max_protos_per_person=max_protos_per_person
        )

        out_json = os.path.join(self.output_folder, "tracking_results_1.json")

        contextual_path = os.path.join(self.output_folder, "contextual_neighborhood.json")
        neigh_data = self.build_neighborhood(out_path=contextual_path)

        base_data = dict(self.results_json)
        base_data["neighborhood"] = (neigh_data or {}).get("relations", [])

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(base_data, f, indent=4, ensure_ascii=False)

        print(f"[OK] JSON global con person_global_id + neighborhood → {out_json}")
        return out_json

    def build_neighborhood(self, min_iou=0.01, near_thresh=0.15, out_path=None, video_id="session_001"):
        relations = []
        rid = 1

        def center(b):
            x1, y1, x2, y2 = b
            return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

        def iou(b1, b2):
            xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
            xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
            inter = max(0, xB - xA) * max(0, yB - yA)
            if inter <= 0: return 0.0
            a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            return inter / (a1 + a2 - inter + 1e-12)

        def spatial(b_ref, b_other):
            (cx1, cy1), (cx2, cy2) = center(b_ref), center(b_other)
            w1 = max(1e-6, b_ref[2] - b_ref[0])
            h1 = max(1e-6, b_ref[3] - b_ref[1])
            nx, ny = (cx2 - cx1) / w1, (cy2 - cy1) / h1

            rels = []
            if abs(nx) < near_thresh and abs(ny) < near_thresh:
                rels.append("cerca")
            if nx > 0.2:
                rels.append("a la derecha")
            elif nx < -0.2:
                rels.append("a la izquierda")
            if ny > 0.3:
                rels.append("abajo")
            elif ny < -0.3:
                rels.append("arriba")
            return rels, nx, ny

        for f_idx, (frame, entries) in enumerate(self.results_json.items(), start=1):
            persons = [p for p in entries if p.get("kind") == "person"]
            objects = [o for o in entries if o.get("kind") == "object"]

            for j_obj, obj in enumerate(objects):
                b_obj = obj["bbox"]
                class_name = obj.get("class_name", f"objeto {j_obj}")
                for p in persons:
                    pid = p.get("person_global_id") or p.get("track_id") or 0
                    b_p = p["bbox"]

                    rels, nx, ny = spatial(b_p, b_obj)
                    ov = iou(b_p, b_obj)

                    if not rels and ov < min_iou:
                        continue
                    if not rels and ov >= min_iou:
                        rels = ["superpuesto a"]

                    rel_txt = " y ".join(rels)
                    relations.append({
                        "id": rid,
                        "video_id": video_id,
                        "key_frame_id": f_idx,
                        "parent_object_id": int(pid),
                        "related_object_id": int(j_obj),
                        "relation": ",".join(rels),
                        "intersection": round(ov, 4),
                        "x_alignment": round(nx, 4),
                        "y_alignment": round(ny, 4),
                        "markdown": f"Persona {pid} está {rel_txt} de {class_name}."
                    })
                    rid += 1

            n = len(persons)
            for i in range(n):
                p1 = persons[i]
                b1 = p1["bbox"]
                pid1 = p1.get("person_global_id") or p1.get("track_id") or 0
                for j in range(i + 1, n):
                    p2 = persons[j]
                    b2 = p2["bbox"]
                    pid2 = p2.get("person_global_id") or p2.get("track_id") or 0
                    rels, nx, ny = spatial(b1, b2)
                    if not rels:
                        continue
                    rel_txt = " y ".join(rels)
                    relations.append({
                        "id": rid,
                        "video_id": video_id,
                        "key_frame_id": f_idx,
                        "parent_object_id": int(pid1),
                        "related_object_id": int(pid2),
                        "relation": ",".join(rels),
                        "intersection": None,
                        "x_alignment": round(nx, 4),
                        "y_alignment": round(ny, 4),
                        "markdown": f"Persona {pid1} está {rel_txt} de Persona {pid2}."
                    })
                    rid += 1

        out = {"relations": relations}

        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            print(f"[OK] Vecindario contextual global guardado en {out_path}")

        return out