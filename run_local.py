import time
from datetime import datetime

from camerachatbot import paths
from camerachatbot.runtime.bootstrap import init_runtime
from camerachatbot.identity.global_identity_service import GlobalIdentityService
from camerachatbot.pipeline.pipeline_service import run_pipeline_and_persist
from camerachatbot.debugging.box_review import save_annotated_and_crops, review_interactive

# Helper de debugging opcional: guarda todas las boxes dibujadas + recortes
# individuales por keyframe y abre un visor interactivo al final de la
# corrida (SPACE = siguiente box, N = saltar al siguiente keyframe,
# ESC/Q = salir). Ver camerachatbot/debugging/box_review.py. Apagado por
# defecto -- no afecta corridas normales.
SAVE_BOX_REVIEW = True


def main():
    start_total = time.time()

    runtime = init_runtime()

    gallery = GlobalIdentityService(
        dim=512,
        index_path=str(paths.GALLERY_INDEX_PATH),
        map_path=str(paths.ID_MAP_PATH),
        store_path=str(paths.PROTO_STORE_PATH),
    )

    print(f"[PRE] Inicialización de modelos: {time.time() - start_total:.3f}s")

    n_keyframes = 5
    final_inference = 5
    folder_path = 'keyFrames'
    frames_folder = str(paths.KEYFRAMES_SAMPLE_DIR)

    BUCKET_NAME = 'prueba'
    video_key = BUCKET_NAME + folder_path

    final_json = run_pipeline_and_persist(
        runtime=runtime,
        gallery=gallery,
        frames_folder=frames_folder,
        n_keyframes=n_keyframes,
        size_xy=(848, 478),
        start_at=datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        video_key=video_key,
        final_inference=final_inference,
        draw_debug=False,
    )

    if SAVE_BOX_REVIEW:
        output_dir = str(paths.debug_boxes_dir_for(frames_folder))
        manifest = save_annotated_and_crops(final_json, frames_folder, output_dir)
        review_interactive(manifest)


if __name__ == "__main__":
    main()
