import os, numpy as np, cv2

from camerachatbot.preprocessing.image_preprocessing import preprocess_image


def download_folder(folder_path: str, supabase, bucket_name: str, local_dir: str = "mis_frames"):
    # Carpeta de destino local
    local_base = os.path.join(local_dir, folder_path)
    os.makedirs(local_base, exist_ok=True)

    files = supabase.storage.from_(bucket_name).list(folder_path)
    if not files:
        print(f"⚠️ No se encontraron archivos en {folder_path}")
        return 0, 0, 0

    n_valid = 0
    w0 = h0 = 0

    for file in files:
        name = file.get("name")
        if not name:
            continue

        remote_path = f"{folder_path}/{name}"
        local_path  = os.path.join(local_base, name)

        print(f"⬇️ Descargando {remote_path} → {local_path}")
        res = supabase.storage.from_(bucket_name).download(remote_path)
        if res is None:
            print(f"❌ Error al descargar {remote_path}")
            continue

        # Decodifica bytes → BGR
        nparr = np.frombuffer(res, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"❌ Imagen corrupta o no soportada: {remote_path}")
            continue

        try:
            result = preprocess_image(input_img=img_bgr, save_path=local_path)
            # Guarda dims del primer válido
            if n_valid == 0:
                h0, w0 = result['orig_size']
        except Exception as e:
            print(f"❌ Error en preprocess para {remote_path}: {e}")
            continue

        print(f"✅ Guardado en {local_path}")
        n_valid += 1

    return n_valid, w0, h0
