"""Pure-numpy drop-in replacement for
`sklearn.cluster.DBSCAN(eps, min_samples, metric="cosine").fit_predict(E)`.

Used today at `camerachatbot/detectors/person_reid.py::cluster_locally()`.
This module is **inert** in this PR: `cluster_locally()` still imports and
calls `sklearn.cluster.DBSCAN` directly — swapping it for `cluster_cosine`
is Fase 3b's job (`SECURITY_RULES["label_source"]` flip, gated on the
user's `tools/verify_onnx_parity.py` sign-off), since one of Fase 3's
stated goals is eventually dropping the `scikit-learn` runtime dependency
entirely (Fase 3c).

Why reimplement DBSCAN instead of keeping scikit-learn: the design's Fase 3c
requirements-slimming goal explicitly targets removing scikit-learn from the
runtime `requirements.txt` once the ONNX switch is verified — this module is
the piece of groundwork that makes that removal possible without changing
`cluster_locally()`'s observable contract (same `np.ndarray[int]` return
shape, same `-1` sentinel for noise).
"""

import warnings
from collections import deque

import numpy as np

_LARGE_N_WARNING_THRESHOLD = 5000


def cluster_cosine(E: np.ndarray, eps: float = 0.5, min_samples: int = 4) -> np.ndarray:
    """Cluster rows of `E` by cosine distance, DBSCAN semantics.

    Drop-in for `DBSCAN(eps, min_samples, metric="cosine").fit_predict(E)`:
    returns an `(N,)` int array, one label per row, with `-1` marking noise
    points. Label *numbering* and the specific cluster a border point that
    is reachable from two mutually-unconnected clusters ends up in are not
    guaranteed to match sklearn's — sklearn documents both as
    nondeterministic/order-dependent itself (border-point cluster
    assignment depends on scan order), so this is an accepted, not a bug.

    Algorithm (mirrors sklearn's own `dbscan_inner` flood-fill exactly,
    same core-point definition, same border-point-joins-whichever-cluster-
    finds-it-first semantics — only the container (queue vs. sklearn's
    internal stack) differs, which does not change which points end up
    core/noise, only tie-break order for ambiguous border points):

    1. Rows are L2-normalized (re-normalized defensively — callers such as
       `person_reid.py::cluster_locally()` already normalize, but this
       function must not silently produce wrong distances if a future
       caller forgets to).
    2. `S = E_norm @ E_norm.T` (cosine similarity matrix), `D = 1.0 - S`
       (cosine distance). The diagonal is forced to exactly `0.0` — a
       point's distance to itself must never fail the `eps` threshold due
       to float32 round-off in step 1's normalization.
    3. `neigh = D <= eps`, a boolean `(N, N)` adjacency matrix (includes
       self, since `D[i, i] == 0 <= eps` for any `eps >= 0`).
    4. **Core points**: `neigh.sum(axis=1) >= min_samples`. sklearn counts
       the point itself in this sum (a point with exactly `min_samples - 1`
       *other* neighbors within `eps`, plus itself, IS core) — subtracting
       1 here would be an off-by-one that silently shifts every cluster
       boundary. This function does not subtract 1, matching sklearn.
    5. BFS flood-fill from each unvisited core point (scan order 0..N-1):
       seed a new cluster label, enqueue the seed's neighbors: FOR EACH
       neighbor `j` popped, if not yet labeled, label it with the current
       cluster; if `j` is ALSO core, enqueue `j`'s neighbors too (this is
       what absorbs an entire density-connected chain of core points into
       one cluster and picks up each core point's border points along the
       way). Non-core (border) points get labeled but never expand the
       frontier further, matching DBSCAN's definition precisely.

    Memory is `O(N^2)` float32 (the full pairwise similarity/distance
    matrix) — negligible for this pipeline's actual scale (person crops per
    batch: tens to low hundreds). Above `_LARGE_N_WARNING_THRESHOLD` this
    function emits a `UserWarning` rather than silently allocating an
    unexpectedly large matrix; it does not implement chunked comparison
    (out of scope here — batch sizes at that scale are not expected for
    this pipeline's per-run person-crop clustering step).
    """
    E = np.asarray(E, dtype=np.float32)
    n = E.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return labels

    if n > _LARGE_N_WARNING_THRESHOLD:
        warnings.warn(
            f"cluster_cosine: N={n} exceeds {_LARGE_N_WARNING_THRESHOLD} — "
            f"the O(N^2) similarity matrix will use ~{(n * n * 4) / 1e9:.2f} GB "
            f"of memory. This pipeline's expected batch scale is tens to low "
            f"hundreds of person crops; chunked comparison is not implemented.",
            stacklevel=2,
        )

    norm = np.linalg.norm(E, axis=1, keepdims=True)
    norm = np.where(norm < 1e-12, 1.0, norm)  # guard against a zero-norm row
    E_norm = (E / norm).astype(np.float32)

    S = E_norm @ E_norm.T
    np.clip(S, -1.0, 1.0, out=S)
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)

    neigh = D <= eps
    n_neighbors = neigh.sum(axis=1)
    is_core = n_neighbors >= min_samples

    label_num = 0
    for seed in range(n):
        if labels[seed] != -1 or not is_core[seed]:
            continue

        labels[seed] = label_num
        queue = deque(np.nonzero(neigh[seed])[0].tolist())
        while queue:
            j = queue.popleft()
            if labels[j] == -1:
                labels[j] = label_num
                if is_core[j]:
                    queue.extend(np.nonzero(neigh[j])[0].tolist())

        label_num += 1

    return labels
