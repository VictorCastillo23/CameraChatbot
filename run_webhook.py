from flask import request, abort

from camerachatbot import paths
from camerachatbot.runtime.bootstrap import init_runtime
from camerachatbot.identity.global_identity_service import GlobalIdentityService
from camerachatbot.pipeline.pipeline_service import run_pipeline_and_persist, _safe_float
from camerachatbot.storage.bucket_downloader import download_folder

R = init_runtime()

app = R["app"]

gallery = GlobalIdentityService(
    dim=512,
    index_path=str(paths.GALLERY_INDEX_PATH),
    map_path=str(paths.ID_MAP_PATH),
    store_path=str(paths.PROTO_STORE_PATH),
)


@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json(silent=True)
    if not data or 'record' not in data:
        abort(400, description="JSON inválido: falta 'record'")

    record = data['record']
    folder_path     = record.get('folder_path')
    timestamp       = record.get('timestamp')
    final_inference = _safe_float(record.get('speed_inference', 0.0))

    if not folder_path:
        abort(400, description="Falta 'folder_path' en record")

    print(f'Extrayendo : {folder_path}')
    n_keyframes, w, h = download_folder(
        folder_path,
        supabase=R["supabase"],
        bucket_name=R["BUCKET_NAME"],
        local_dir=str(paths.MIS_FRAMES_DIR),
    )

    # Si no hubo frames válidos, corta temprano
    if not n_keyframes or n_keyframes <= 0:
        print("⚠️ No se descargaron frames válidos.")
        return 'No frames', 200

    frames_folder = str(paths.MIS_FRAMES_DIR / folder_path)

    run_pipeline_and_persist(
        runtime=R,
        gallery=gallery,
        frames_folder=frames_folder,
        n_keyframes=n_keyframes,
        size_xy=(w, h),
        start_at=timestamp,
        video_key=R["BUCKET_NAME"] + folder_path,
        final_inference=final_inference,
        draw_debug=True,
        debug_images_dir=frames_folder,
    )
    return 'Success', 200


if __name__ == '__main__':
    app.run()
