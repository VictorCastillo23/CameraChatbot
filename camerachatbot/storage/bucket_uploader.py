import os , numpy as np, time, io, cv2
from datetime import datetime

from camerachatbot.runtime.bootstrap import build_supabase

supabase, BUCKET_NAME = build_supabase()
if supabase is None or BUCKET_NAME is None:
    raise RuntimeError("[FATAL] Supabase no configurado; revisa SUPABASE_URL/SUPABASE_KEY/SUPABASE_BUCKET")

def formatted_timestamp(ts=None):
    ts = ts or time.time()
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def upload_cv2_images_to_folder(images,timestamp=None,speed_inference=None):
    folder = f"{str(timestamp).split('.')[0]}/"
    for name, img in images:

        if not isinstance(img, np.ndarray):
            raise TypeError(f"[FATAL] {name} no es un frame válido, llegó {type(img)}")

        # Convertir imagen a JPEG en memoria
        _, buffer = cv2.imencode(".jpg", img)
        file_bytes = io.BytesIO(buffer)

        path = f"{folder}{name}.jpg"
        res = supabase.storage.from_(BUCKET_NAME).upload(
            path,
            file_bytes.getvalue(),
            {"content-type": "image/jpeg"}
        )

        if res is None or (isinstance(res, dict) and "error" in res):
            print(f"❌ Error al subir {path}: {res}")
        else:
            print(f"✅ Subida: {path}")

    #print(f"folder: {folder}")
    res = supabase.table("uploaded_folders").insert(
        {
            "folder_path": folder,
            "timestamp": formatted_timestamp(timestamp),
            "speed_inference" : speed_inference,
        }
    ).execute()

    if res is None or (isinstance(res, dict) and "error" in res):
        print(f"❌ Error al subir los datos: {res}")
    else:
        print(f"✅ Subida: de los datos")

