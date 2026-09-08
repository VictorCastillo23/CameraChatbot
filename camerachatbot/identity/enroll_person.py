"""Manual person-enrollment CLI (Fase 5).

Run once per person to authorize, from a small folder of reference images::

    python -m camerachatbot.identity.enroll_person \\
        --name "Ana" --images ./enroll/ana [--role staff]

    python -m camerachatbot.identity.enroll_person --deactivate 42

`--name`/`--images`: detects the largest `person` box per image (largest
crop is assumed to be the enrollment subject, same convention as a passport
photo -- the intended subject fills most of the frame), embeds every crop
with the same `OSNetOnnxEmbedder` the live pipeline uses
(`runtime.loaders_onnx.load_reid()`), L2-normalizes the mean (centroid) of
those per-image embeddings, and resolves it against the gallery via
`GlobalIdentityService.assign_or_create()` -- using thresholds taken
**explicitly** from `security_config.IDENTITY_THRESHOLDS["enroll"]`
(stricter than the live `["runtime"]` values: manual enrollment can afford
to demand a tighter match before accepting/creating an identity). There is
no default value for `t_accept`/`t_reject` at this call site by design --
see design.md section 2, "the exact caller the exploration warned would
otherwise silently inherit the stale 0.60 default".

The resolved `person_global_id` is then upserted into `authorized_identity`
(insert-or-update via `ON CONFLICT`). `--deactivate <person_global_id>` only
flips that row's `is_active` to `FALSE` -- it never touches the gallery/FAISS
index (`gallery.index`/`id_map.json`/`proto_store.npy`): the person stays
recognizable and starts generating `unenrolled_person` events instead (see
`security.events.evaluate_unenrolled`), which is the intended Fase 5
behavior, not a gap.

This is a manual dev/ops tool, same category as `geometry.calibrate_camera`:
it needs real ONNX model weights (`models/yolov10m.onnx`,
`models/osnet_x1_0.onnx`) and real reference images, neither of which is
available in this sandbox/CI environment. It has not been exercised against
real weights here -- verify it on a real workstation with the exported
models before relying on it.
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import psycopg

from camerachatbot import paths
from camerachatbot.db.postgres_writer import get_conn, SCHEMA, ensure_schema_and_tables
from camerachatbot.detectors.onnx_reid import OSNetOnnxEmbedder
from camerachatbot.identity.global_identity_service import GlobalIdentityService
from camerachatbot.runtime.loaders_onnx import load_detector, load_reid
from camerachatbot.security_config import IDENTITY_THRESHOLDS

_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _iter_image_paths(images_dir: str):
    files = []
    for ext in _IMAGE_EXTS:
        files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
        files.extend(glob.glob(os.path.join(images_dir, f"*{ext.upper()}")))
    return sorted(set(files))


def _largest_person_bbox(det_result):
    """The largest-area `person`-class box in one `DetResult`, or `None`.

    "Largest" is the enrollment convention: the intended subject is assumed
    to dominate the frame, same as a passport-photo crop.
    """
    if det_result is None:
        return None
    best_bbox = None
    best_area = 0.0
    for b in det_result.boxes:
        cls_id = int(b.cls[0])
        class_name = det_result.names.get(cls_id, str(cls_id))
        if class_name != "person":
            continue
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area > best_area:
            best_area = area
            best_bbox = (int(x1), int(y1), int(x2), int(y2))
    return best_bbox


def _embed_images(detector, embedder, image_paths):
    """Detect the largest person crop per image and embed it.

    Returns `(embeddings, skipped)` -- `embeddings` is a list of `(512,)`
    L2-normalized float32 arrays (one per usable image); `skipped` is a
    list of `(path, reason)` for images that yielded no usable embedding
    (unreadable file, no person detected, invalid/empty crop). Skips are
    reported to the caller rather than raised per-image, so one bad photo
    in the folder does not abort the whole enrollment.
    """
    embeddings = []
    skipped = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            skipped.append((path, "no se pudo leer la imagen"))
            continue

        det_results = detector(img, verbose=False)
        det_result = det_results[0] if det_results else None
        bbox = _largest_person_bbox(det_result)
        if bbox is None:
            skipped.append((path, "no se detectó ninguna persona"))
            continue

        x1, y1, x2, y2 = bbox
        crop = img[y1:y2, x1:x2]
        emb = embedder.embed_one(crop)
        if emb is None or not OSNetOnnxEmbedder.is_valid_embedding(emb):
            skipped.append((path, "el embedding falló sobre el recorte detectado"))
            continue

        embeddings.append(emb)

    return embeddings, skipped


def _centroid(embeddings):
    """Mean of per-image embeddings, re-normalized to unit L2 norm.

    Each individual embedding is already unit-normalized by
    `OSNetOnnxEmbedder.embed_one()`, but their mean is not -- re-normalizing
    the centroid is required before it can be compared via cosine similarity
    in `GlobalIdentityService.assign_or_create()`.
    """
    E = np.stack(embeddings).astype("float32")
    c = E.mean(axis=0)
    norm = float(np.linalg.norm(c))
    if norm < 1e-9:
        raise RuntimeError(
            "El centroide de enrolamiento tiene norma ~0 -- los embeddings "
            "recolectados son degenerados (todos cero o se cancelan entre sí). "
            "Vuelva a enrolar con imágenes de referencia distintas."
        )
    return (c / norm).astype("float32")


def _upsert_authorized_identity(person_global_id: int, *, display_name: str, role=None, notes=None):
    """Insert-or-update the `authorized_identity` row for `person_global_id`.

    `ON CONFLICT (person_global_id) DO UPDATE` handles both cases (brand-new
    enrollment vs. re-enrolling/updating an already-authorized person) in one
    round trip, re-activating the row (`is_active = TRUE`) even if it had
    been previously deactivated via `--deactivate`.
    """
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                ensure_schema_and_tables(cur)
                cur.execute(f"SET search_path TO {SCHEMA}")
                cur.execute(
                    """
                    INSERT INTO authorized_identity
                        (person_global_id, display_name, role, is_active, enrolled_at, notes)
                    VALUES (%s, %s, %s, TRUE, NOW(), %s)
                    ON CONFLICT (person_global_id) DO UPDATE SET
                        display_name = EXCLUDED.display_name,
                        role = EXCLUDED.role,
                        is_active = TRUE,
                        enrolled_at = NOW(),
                        notes = EXCLUDED.notes
                    """,
                    (person_global_id, display_name, role, notes),
                )
    finally:
        conn.close()


def _deactivate_authorized_identity(person_global_id: int) -> int:
    """Sets `is_active = FALSE` for `person_global_id`'s `authorized_identity`
    row. Returns the number of rows updated (0 if no such row exists).

    Never touches the gallery/FAISS index -- deauthorization is an
    `authorized_identity`-only operation per design.md's Fase 5 section.
    """
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                ensure_schema_and_tables(cur)
                cur.execute(f"SET search_path TO {SCHEMA}")
                cur.execute(
                    "UPDATE authorized_identity SET is_active = FALSE WHERE person_global_id = %s",
                    (person_global_id,),
                )
                updated = cur.rowcount
    finally:
        conn.close()
    return updated


def _load_gallery() -> GlobalIdentityService:
    """The same production gallery every real entry point (`run_local.py`,
    `run_webhook.py`) opens, from the same `paths.py`-resolved files."""
    return GlobalIdentityService(
        dim=512,
        index_path=str(paths.GALLERY_INDEX_PATH),
        map_path=str(paths.ID_MAP_PATH),
        store_path=str(paths.PROTO_STORE_PATH),
    )


def enroll(name: str, images_dir: str, role=None, gallery: GlobalIdentityService = None) -> int:
    """Enroll `name` from every usable person crop under `images_dir`.

    Returns the resolved `person_global_id`. Raises `RuntimeError` if no
    image yields a usable embedding, or if the resolved match lands in the
    gallery's "gray zone" (ambiguous similarity -- neither a confident
    accept nor a confident reject) -- a manual enrollment should never
    silently create an ambiguous identity; the operator must re-enroll with
    clearer images or resolve the ambiguity by hand.
    """
    image_paths = _iter_image_paths(images_dir)
    if not image_paths:
        raise FileNotFoundError(
            f"No se encontraron imágenes ({', '.join(_IMAGE_EXTS)}) en: {images_dir}"
        )

    detector = load_detector()
    embedder = load_reid()

    embeddings, skipped = _embed_images(detector, embedder, image_paths)
    for path, reason in skipped:
        print(f"[WARN] Omitiendo {path}: {reason}")

    if not embeddings:
        raise RuntimeError(
            f"No se obtuvo ningún recorte/embedding de persona utilizable entre "
            f"{len(image_paths)} imagen(es) en {images_dir} -- no se puede enrolar '{name}'."
        )

    centroid = _centroid(embeddings)

    if gallery is None:
        gallery = _load_gallery()

    # Fase 5 requires the STRICTER enrollment thresholds, explicitly -- see
    # this module's docstring and design.md section 2. Never fall back to
    # IDENTITY_THRESHOLDS["runtime"] here.
    t_accept = IDENTITY_THRESHOLDS["enroll"]["t_accept"]
    t_reject = IDENTITY_THRESHOLDS["enroll"]["t_reject"]

    pid, sim, created = gallery.assign_or_create(
        centroid,
        modality="body",
        t_accept=t_accept,
        t_reject=t_reject,
        proto_meta={"type": "enroll_centroid", "name": name, "n_images": len(embeddings)},
    )

    if pid is None:
        raise RuntimeError(
            f"La coincidencia de galería para '{name}' es ambigua (zona gris, sim={sim:.3f}, "
            f"t_accept={t_accept}, t_reject={t_reject}) -- se rechaza adivinar. "
            f"Vuelva a enrolar con imágenes más claras/distintivas."
        )

    # `gallery.assign_or_create()` above already wrote the new/matched
    # prototype to gallery.index/id_map.json/proto_store.npy (via
    # GlobalIdentityService._save(), synchronous, no rollback primitive) --
    # if the authorized_identity upsert below fails now, we're left in a
    # split state: person_global_id=pid exists in the gallery but has no
    # authorization row, so AuthorizationRegistry.is_authorized(pid) will be
    # fail-closed False until this is retried. Surface that explicitly
    # rather than letting a generic DB error obscure it -- this is
    # deliberately NOT rolled back (no clean primitive for it, out of scope
    # for this fix), the goal is operator visibility.
    try:
        _upsert_authorized_identity(pid, display_name=name, role=role)
    except Exception as e:
        raise RuntimeError(
            f"person_global_id={pid} se agregó correctamente a la galería de identidades, "
            f"pero falló la escritura del registro de autorización: {e}. Esta persona será "
            f"tratada como NO AUTORIZADA hasta que se reintente (volver a enrolar con las "
            f"mismas imágenes debería volver a coincidir con el mismo person_global_id={pid} "
            f"vía la galería, y solo faltaría que la escritura de authorized_identity tenga éxito)."
        ) from e

    print(
        f"[OK] '{name}' enrolado como person_global_id={pid} "
        f"({'identidad nueva' if created else 'coincidió con identidad existente'}, sim={sim:.3f}, "
        f"imágenes_usadas={len(embeddings)}/{len(image_paths)}, role={role!r})."
    )
    return pid


def deactivate(person_global_id: int) -> int:
    updated = _deactivate_authorized_identity(person_global_id)
    if updated == 0:
        print(
            f"[WARN] No se encontró fila authorized_identity para person_global_id={person_global_id} "
            f"-- no se desactivó nada."
        )
    else:
        print(
            f"[OK] person_global_id={person_global_id} desactivado (is_active=FALSE). "
            f"La entrada de galería/FAISS quedó intacta -- esta persona sigue siendo reconocible "
            f"y ahora generará eventos unenrolled_person."
        )
    return updated


def main(argv=None):
    parser = argparse.ArgumentParser(description="Manual person-enrollment CLI (Fase 5).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--name", type=str, help="Display name to enroll (requires --images).")
    group.add_argument(
        "--deactivate", type=int, metavar="PERSON_GLOBAL_ID",
        help="Deactivate an existing authorized_identity row. Does not touch the gallery/FAISS index.",
    )
    parser.add_argument("--images", type=str, help="Folder of reference images for --name.")
    parser.add_argument("--role", type=str, default=None, help="Optional role label (e.g. 'staff').")
    args = parser.parse_args(argv)

    if args.deactivate is None and not args.images:
        parser.error("--images is required when using --name.")

    # Expected, real failure modes get a clean CLI message + non-zero exit
    # instead of a raw traceback: FileNotFoundError (--images has no
    # matching files), RuntimeError (no image yielded a usable embedding,
    # gray-zone match, or the fix-C split-state case above), and any
    # psycopg connection/operational error from the authorized_identity
    # upsert/deactivate path. Per-image failures (unreadable file, no
    # person detected) are already handled gracefully via the `skipped`
    # list inside `enroll()` -- this does not touch that path.
    try:
        if args.deactivate is not None:
            deactivate(args.deactivate)
        else:
            enroll(args.name, args.images, role=args.role)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except psycopg.Error as e:
        print(f"[ERROR] Error de base de datos: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
