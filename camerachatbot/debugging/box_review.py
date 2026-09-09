"""Helper de debugging OPCIONAL: guarda, por cada keyframe, una imagen
completa con todas las boxes dibujadas y etiquetadas (aunque se solapen,
sin ninguna logica de supresion/deduplicacion), mas un recorte individual
por cada deteccion, organizados en una carpeta por keyframe. Incluye ademas
un visor interactivo por teclado para recorrer el resultado sin salir de
la terminal.

No forma parte del pipeline por defecto -- se invoca manualmente desde
`run_local.py` detras de un flag (`SAVE_BOX_REVIEW`).

`save_annotated_and_crops()` es headless y totalmente verificable sin
display (ver `tests_manual/test_box_review.py`). `review_interactive()`
en cambio necesita una ventana real de OpenCV -- no se puede ejercitar ni
verificar en un entorno automatizado/sandbox, la misma limitacion ya
documentada en `geometry/calibrate_camera.py`.
"""

import os
import json

import cv2

from camerachatbot.debugging.annotate import (
    ensure_int_bbox,
    put_multiline,
    fmt_float,
    find_image_for_frame,
)


def _label_for(entry: dict) -> str:
    """'<class_name> <conf:.3f>' -- mismo header que usa la imagen completa."""
    cls_name = entry.get("class_name", "object")
    conf = entry.get("confidence")
    return cls_name + (f" {fmt_float(conf, 3)}" if conf is not None else "")


def _clamp_bbox(bbox, w, h):
    x1, y1, x2, y2 = ensure_int_bbox(bbox)
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    return x1, y1, x2, y2


def save_annotated_and_crops(json_path: str, images_dir: str, output_dir: str) -> dict:
    """Para cada frame_id real (se salta el pseudo-frame "neighborhood",
    mismo guard que `hand_detector.py`/`face_detector.py`/
    `secondary_detector.py`):

    - busca la imagen del frame (`find_image_for_frame`); si no la
      encuentra, imprime un [WARN] y omite ese frame_id del manifest
      (no lanza excepcion).
    - dibuja TODAS las entradas (bbox + etiqueta) sobre una copia de la
      imagen, sin ninguna supresion por solapamiento, y la guarda en
      `<output_dir>/<frame_id>/overview.jpg`. Un frame sin entradas
      igual obtiene su overview.jpg (sin boxes) y queda en el manifest
      con `"boxes": []` -- caso distinto al de imagen faltante.
    - por cada entrada, recorta la imagen ORIGINAL (sin anotar) a su
      bbox clampeado, le quema encima la misma etiqueta, y la guarda
      como `<output_dir>/<frame_id>/box_<NNN>_<class_name>.jpg` (NNN
      es el indice en el orden del JSON, con ceros a la izquierda).

    Devuelve un manifest:
        {frame_id: {"overview": "<path>",
                    "boxes": [{"path": "<path>", "class_name": ...,
                               "label": ...}, ...]}}
    en el mismo orden que el JSON de entrada, listo para que
    `review_interactive()` lo consuma sin volver a parsear nada.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    manifest = {}

    for frame_id, entries in data.items():
        if "neighborhood" in frame_id:
            continue  # pseudo-frame, no es una lista de detecciones

        img_path = find_image_for_frame(images_dir, frame_id)
        if not img_path:
            print(f"[WARN] No se encontro imagen para el frame '{frame_id}' en {images_dir}")
            continue

        original = cv2.imread(img_path)
        if original is None:
            print(f"[WARN] No se pudo leer la imagen '{img_path}' para el frame '{frame_id}'")
            continue

        h, w = original.shape[:2]
        frame_dir = os.path.join(output_dir, frame_id)
        os.makedirs(frame_dir, exist_ok=True)

        overview = original.copy()
        n = len(entries)
        width = max(3, len(str(max(n - 1, 0))))
        boxes = []

        for i, entry in enumerate(entries):
            bbox = entry.get("bbox") or entry.get("shape")
            if bbox is None:
                continue
            x1, y1, x2, y2 = _clamp_bbox(bbox, w, h)
            if x2 <= x1 or y2 <= y1:
                continue

            label = _label_for(entry)

            cv2.rectangle(overview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            put_multiline(overview, (x1, max(18, y1 - 8)), [label])

            crop = original[y1:y2, x1:x2].copy()
            put_multiline(crop, (2, 18), [label])

            class_name = entry.get("class_name", "object")
            crop_name = f"box_{i:0{width}d}_{class_name}.jpg"
            crop_path = os.path.join(frame_dir, crop_name)
            cv2.imwrite(crop_path, crop)
            boxes.append({"path": crop_path, "class_name": class_name, "label": label})

        overview_path = os.path.join(frame_dir, "overview.jpg")
        cv2.imwrite(overview_path, overview)

        manifest[frame_id] = {"overview": overview_path, "boxes": boxes}

    return manifest


def review_interactive(manifest: dict,
                        advance_key: str = " ",
                        skip_key: str = "n",
                        quit_keys=(27, ord("q"), ord("Q"))) -> None:
    """Visor por teclado sobre el manifest de `save_annotated_and_crops()`.

    Requiere una ventana real de OpenCV -- no se puede ejercitar ni
    verificar en un entorno automatizado/sandbox (misma limitacion que
    `geometry/calibrate_camera.py`, verificar en una maquina real).

    Teclas:
      - SPACE (`advance_key`): avanza a la siguiente box del keyframe
        actual; al llegar a la ultima, pasa automaticamente al overview
        del siguiente keyframe.
      - N (`skip_key`): salta directo al siguiente keyframe, sin ver el
        resto de las boxes del actual.
      - ESC / Q (`quit_keys`): cierra el visor inmediatamente.
    """
    frame_ids = list(manifest.keys())
    n_frames = len(frame_ids)
    window = "box_review"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    advance_code = ord(advance_key)
    skip_code = ord(skip_key.lower())
    skip_code_upper = ord(skip_key.upper())

    try:
        for fi, frame_id in enumerate(frame_ids, start=1):
            entry = manifest[frame_id]

            overview = cv2.imread(entry["overview"])
            cv2.imshow(window, overview)
            cv2.setWindowTitle(window, f"Keyframe {fi}/{n_frames} — overview")
            key = cv2.waitKey(0) & 0xFF

            if key in quit_keys:
                return
            if key in (skip_code, skip_code_upper):
                continue

            boxes = entry["boxes"]
            n_boxes = len(boxes)
            for bi, box in enumerate(boxes, start=1):
                crop = cv2.imread(box["path"])
                cv2.imshow(window, crop)
                cv2.setWindowTitle(
                    window,
                    f"Keyframe {fi}/{n_frames} — box {bi}/{n_boxes} ({box['class_name']})",
                )
                key = cv2.waitKey(0) & 0xFF

                if key in quit_keys:
                    return
                if key in (skip_code, skip_code_upper):
                    break
                # advance_key o cualquier otra tecla: sigue a la proxima box
    finally:
        cv2.destroyAllWindows()
