# Glossary

Terms used throughout these docs that the codebase itself doesn't define inline.

**Fase / PR numbering** — The `security-pipeline-overhaul` program is broken into numbered phases ("Fase 0" through "Fase 6"), each split into one or more chained pull requests ("PR1", "PR4a"/"PR4b", "PR8a"/"PR8b", etc.). You'll see these labels in code comments, commit messages, and this documentation. It's an internal planning convention, not a versioning scheme.

**Re-ID (Re-identification)** — Generating a numeric fingerprint (embedding) for a detected person's appearance, so the same individual can be recognized again in a different frame, video, or camera session, without knowing their identity in advance.

**Embedding** — A fixed-length vector of numbers representing an image crop (in this codebase, 512 dimensions for a person). Two embeddings of the same person end up numerically close together; different people end up far apart.

**Cosine similarity / cosine distance** — A way of comparing two embeddings by the angle between them rather than their raw magnitude. Cosine distance = 1 − cosine similarity. This codebase L2-normalizes embeddings first, which makes cosine similarity equivalent to a plain dot product.

**DBSCAN** — A clustering algorithm that groups points by density (repeatedly expanding outward from "core" points that have enough nearby neighbors) rather than requiring you to specify the number of clusters up front. Originally from `scikit-learn`; this codebase reimplements it from scratch (`identity/cosine_clustering.py`) to drop the dependency.

**ByteTrack** — A multi-object tracking approach whose key idea is a two-stage matching pass: match high-confidence detections to existing tracks first, then match *remaining* low-confidence detections against whatever tracks are still unmatched, instead of discarding low-confidence detections outright. This recovers tracks through brief occlusion instead of losing and re-creating them.

**Kalman filter** — A mathematical technique for estimating a moving object's current state (here: bounding box position/size and their rates of change) from noisy observations, and predicting where it'll be next frame even without a new observation. Used by `KalmanBoxTracker` to keep a track "alive" through occlusion.

**SORT** — The tracking approach ByteTrack builds on: bounding boxes + a Kalman filter + a "match by predicted overlap" association step. "Confirmed"/"reporting" semantics (how many consecutive successful matches a track needs before being trusted) trace back to SORT's original design.

**Homography** — A single 3×3 mathematical transform that maps points from one flat plane to another — here, from image pixels to real-world ground positions, for one fixed, calibrated camera. Only valid for points that actually lie on the ground plane (see `bbox_ground_point()` in [`04`](04-security-subsystem-reference.md) for why that matters).

**FAISS / HNSW** — FAISS is a library for fast nearest-neighbor search over large sets of embeddings. HNSW ("Hierarchical Navigable Small World") is the specific index structure this codebase uses inside FAISS for the persistent identity gallery.

**COCO allowlist** — The COCO dataset defines 80 general object classes a detector can recognize. This codebase filters detections down to a short allowlist of security-relevant classes (`security_config.COCO_ALLOWLIST`) rather than keeping all 80.

**`label_source`** — The `security_config.SECURITY_RULES` setting that picks how detections get grouped into "the same person": `"cluster"` (default, similarity-based, within one batch) or `"tracker"` (frame-to-frame continuous tracking across a whole video). See [`02-architecture.md`](02-architecture.md).

**Zone types** — `restricted` (intrusion detection applies), `monitored` (watched but not restricted), `safe` (no special rules) — the three values `Zone.zone_type` can take.

**Armed (zone)** — Whether a zone's restriction is currently active, based on its optional schedule (day-of-week + time window). A zone with no schedule is always armed. See `zone_is_armed()` in [`04`](04-security-subsystem-reference.md).
