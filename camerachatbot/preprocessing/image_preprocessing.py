import os
import cv2
import numpy as np
from typing import Tuple, Union, Optional, Dict

def _resize_keep_ar_multiple32(
    img_bgr: np.ndarray,
    max_side: int = 1280,
    multiple: int = 32
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    h, w = img_bgr.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    new_w = int(np.round(w * scale))
    new_h = int(np.round(h * scale))

    new_w = max(multiple, int(np.round(new_w / multiple) * multiple))
    new_h = max(multiple, int(np.round(new_h / multiple) * multiple))

    new_w = min(new_w, w)
    new_h = min(new_h, h)

    if new_w == w and new_h == h:
        return img_bgr, (h, w), (new_h, new_w)

    img_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_resized, (h, w), (new_h, new_w)

def preprocess_image(
    input_img: Union[str, np.ndarray],
    *,
    max_side: int = 1280,
    multiple: int = 32,
    save_path: Optional[str] = None
) -> Dict[str, Union[str, Tuple[int, int], np.ndarray]]:
    # 1) cargar
    in_path = None
    if isinstance(input_img, str):
        in_path = input_img
        img_bgr = cv2.imread(input_img)
        if img_bgr is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {input_img}")
    elif isinstance(input_img, np.ndarray):
        if input_img.ndim != 3 or input_img.shape[2] != 3:
            raise ValueError("input_img debe ser BGR (H,W,3).")
        img_bgr = input_img
    else:
        raise TypeError("input_img debe ser ruta (str) o np.ndarray BGR.")

    # 2) resize seguro (AR + múltiplos de 32)
    img_pp, (h0, w0), (h1, w1) = _resize_keep_ar_multiple32(
        img_bgr, max_side=max_side, multiple=multiple
    )

    # 3) guardar si se solicita
    out_path = None
    if save_path:
        print('/\/\guardando imagen')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img_pp)
        out_path = save_path

    return {
        "in_path": in_path,
        "out_path": out_path,
        "orig_size": (h0, w0),
        "new_size": (h1, w1),
        "image": img_pp,
    }
