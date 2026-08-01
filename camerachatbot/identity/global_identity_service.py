import os, json,numpy as np, time,faiss
def l2_normalize(x: np.ndarray, eps=1e-12):
    x = x.astype("float32")
    n = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / n

class GlobalIdentityService:
    def __init__(self, dim=512, index_path="gallery.index", map_path="id_map.json", store_path="proto_store.npy"):
        self.dim = int(dim)
        self.index_path = index_path
        self.map_path = map_path
        self.store_path = store_path

        self.id_map = []
        print('[creating gallery]')

        # --- cargar/crear índice (HNSW, L2) ---
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.IndexHNSWFlat(self.dim, 32)
            self.index.hnsw.efSearch = 64
            self.index.hnsw.efConstruction = 200

        # --- cargar id_map ---
        if os.path.exists(map_path):
            with open(map_path, "r", encoding="utf-8") as f:
                self.id_map = json.load(f)
        else:
            self.id_map = []

        # --- cargar store ---
        if os.path.exists(store_path):
            store = np.load(store_path, allow_pickle=False)
            if store.ndim == 1:
                # si está vacío, mantén (0, dim); si no es múltiplo, reset
                if store.size == 0:
                    store = np.zeros((0, self.dim), dtype="float32")
                elif store.size % self.dim == 0:
                    store = store.reshape(-1, self.dim).astype("float32")
                else:
                    print("[WARN] proto_store.npy con tamaño inválido. Reiniciando store.")
                    store = np.zeros((0, self.dim), dtype="float32")
            elif store.ndim == 2 and store.shape[1] != self.dim:
                print("[WARN] proto_store.npy con dim incompatible. Reiniciando store.")
                store = np.zeros((0, self.dim), dtype="float32")
            self.store = store.astype("float32")
        else:
            self.store = np.zeros((0, self.dim), dtype="float32")

        # --- coherencia básica y reconstrucción si hace falta ---
        n_map = len(self.id_map)
        n_store = self.store.shape[0]
        n_index = self.index.ntotal

        # recorta al mínimo y reconstruye índice desde store
        n = min(n_map, n_store)
        if n_map != n_store or n_index != n:
            if n_map != n_store:
                print("[WARN] Desalineación id_map/store. Recortando al mínimo común.")
                self.id_map = self.id_map[:n]
                self.store = self.store[:n]
            # reconstruir índice desde store normalizado
            self.index = faiss.IndexHNSWFlat(self.dim, 32)
            self.index.hnsw.efSearch = 64
            self.index.hnsw.efConstruction = 200
            if n > 0:
                self.index.add(self._l2norm(self.store))
            # no guardamos en disco aún; se guardará en la próxima mutación

        # contador de IDs globales
        self._next_person_id = 1 + max([m.get("person_global_id", 0) for m in self.id_map] or [0])

    # ----------------- utils internos -----------------
    @staticmethod
    def _l2norm(x: np.ndarray, axis=1, eps=1e-12) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.map_path, "w", encoding="utf-8") as f:
            json.dump(self.id_map, f, indent=2, ensure_ascii=False)
        np.save(self.store_path, self.store.astype("float32"))

    def _append_store(self, Xn: np.ndarray):
        if Xn.ndim == 1:
            Xn = Xn[None, :]
        self.store = np.vstack([self.store, Xn.astype("float32")])

    def _new_person_id(self) -> int:
        pid = self._next_person_id
        self._next_person_id += 1
        return int(pid)

    # ----------------- API pública -----------------
    def add_prototypes(self, person_global_id: int, prototypes: list):
        if not prototypes:
            return
        X = np.stack([np.asarray(p["embedding"], dtype="float32") for p in prototypes])
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"Embeddings deben tener shape [N,{self.dim}], got {X.shape}")

        Xn = self._l2norm(X)
        self.index.add(Xn)
        self._append_store(Xn)

        now = time.time()
        for p in prototypes:
            meta = dict(p.get("meta", {}))
            meta.setdefault("is_active", True)
            meta.setdefault("created_at", now)
            self.id_map.append({
                "person_global_id": int(person_global_id),
                "proto_id": p.get("proto_id"),
                "modality": p.get("modality", "body"),
                "meta": meta,
            })
        self._save()

    def create_person(self, prototypes: list) -> int:
        pid = self._new_person_id()
        self.add_prototypes(pid, prototypes)
        return pid

    def search(self, emb: np.ndarray, k=10):
        if self.index.ntotal == 0:
            return []
        x = self._l2norm(np.asarray(emb, dtype="float32").reshape(1, -1))
        D, I = self.index.search(x, int(max(1, k)))

        metric_type = getattr(self.index, "metric_type", None)
        if metric_type is not None and metric_type == faiss.METRIC_INNER_PRODUCT:
            sims = D[0]
            order = np.arange(len(sims))
        else:
            sims = 1.0 - 0.5 * D[0]  # L2 -> cos para vectores L2-normalizados
            order = np.argsort(-sims)

        idxs = I[0][order]
        sims = sims[order]

        hits = []
        for sim, idx in zip(sims, idxs):
            if idx < 0 or idx >= len(self.id_map):
                continue
            meta = self.id_map[idx]
            if meta.get("meta", {}).get("is_active", True) is False:
                continue
            hits.append({
                "sim": float(sim),
                "person_global_id": int(meta["person_global_id"]),
                "proto_id": meta.get("proto_id"),
                "modality": meta.get("modality", "body"),
                "meta": meta.get("meta", {}),
                "index_row": int(idx),
            })
        return hits

    def assign_or_create(self, emb, modality="body",
                         t_accept=0.60, t_reject=0.45, k=10, proto_meta=None,
                         return_hits=False):
        emb = np.asarray(emb, dtype="float32")
        if emb.ndim != 1 or emb.shape[0] != self.dim:
            raise ValueError(f"Embedding debe tener shape [{self.dim}], got {emb.shape}")

        hits = self.search(emb, k=k)
        best = hits[0] if hits else None

        if best and best["sim"] >= t_accept:
            out = (best["person_global_id"], best["sim"], False)
            return (out, hits) if return_hits else out

        if (not best) or best["sim"] <= t_reject:
            pid = self.create_person([{
                "embedding": emb,
                "proto_id": f"{int(time.time())}_{np.random.randint(1e9)}",
                "modality": modality,
                "meta": (proto_meta or {})
            }])
            out = (pid, float(best["sim"]) if best else 0.0, True)
            return (out, hits) if return_hits else out

        out = (None, float(best["sim"]), False)  # zona gris
        return (out, hits) if return_hits else out

    def count_active_prototypes(self, pid: int) -> int:
        return sum(1 for m in self.id_map
                   if m["person_global_id"] == pid and m.get("meta", {}).get("is_active", True))

    def maybe_add_support_prototype(self, pid: int, emb: np.ndarray, meta: dict | None = None,
                                    min_sim_add: float = 0.68, max_protos_per_person: int = 5, k: int = 10):
        emb = np.asarray(emb, dtype="float32")
        hits = self.search(emb, k=k)
        if not hits or hits[0]["person_global_id"] != pid:
            return False
        if hits[0]["sim"] < float(min_sim_add):
            return False
        if self.count_active_prototypes(pid) >= int(max_protos_per_person):
            return False

        self.add_prototypes(pid, [{
            "embedding": emb,
            "proto_id": f"{pid}_supp_{int(time.time())}",
            "modality": "body",
            "meta": (meta or {}) | {"type": "support"}
        }])
        return True

    def consolidate_person(self, pid: int, keep_k: int = 4, add_centroid: bool = True, deactivate_old: bool = True):
        rows = [i for i, m in enumerate(self.id_map)
                if m["person_global_id"] == pid and m.get("meta", {}).get("is_active", True)]
        if len(rows) == 0:
            return {"ok": False, "reason": "no_active_prototypes"}

        vecs = self.store[rows]
        if vecs.size == 0:
            return {"ok": False, "reason": "empty_vectors"}

        centroid = vecs.mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-12)

        added_centroid = False
        if add_centroid:
            self.add_prototypes(pid, [{
                "embedding": centroid.astype("float32"),
                "proto_id": f"{pid}_centroid_{int(time.time())}",
                "modality": "body",
                "meta": {"type": "centroid", "n": len(rows)}
            }])
            added_centroid = True

        if deactivate_old:
            dists = 1.0 - np.dot(vecs, centroid)  # 1 - cos
            order = np.argsort(dists)  # más similares primero
            keep = set(rows[i] for i in order[:max(int(keep_k), 1)])
            for i in rows:
                if i not in keep:
                    self.id_map[i]["meta"]["is_active"] = False
            self._save()

        return {"ok": True, "added_centroid": added_centroid, "kept": int(keep_k),
                "deactivated": max(0, len(rows) - int(keep_k))}
