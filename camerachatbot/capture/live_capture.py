import cv2, asyncio, time
from legacy.keyframe_extractor_async import extract_keyframes
from camerachatbot.storage.bucket_uploader import upload_cv2_images_to_folder

async def video_producer(queue: asyncio.Queue, chunk_seconds=5, fps=30, camera_index=1):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")

    index = 0
    frames = []
    chunk_size = int(chunk_seconds * fps)
    start_timestamp = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if len(frames) == 0:
            start_timestamp = time.time()

        frames.append(frame)

        if len(frames) >= chunk_size:
            await queue.put((frames, start_timestamp, index))

            print(f"[Producer] Clip {index} ({start_timestamp}) con {len(frames)} frames enviado a la cola")
            frames = []
            index += 1

        await asyncio.sleep(1 / fps)

    cap.release()

async def video_consumer(queue: asyncio.Queue, consumer_id=1):
    while True:
        frames, timestamp, index = await queue.get()
        print(f"\n{consumer_id}")

        start_keyFrame_extractor = time.time()

        keyframes = await asyncio.to_thread(
            extract_keyframes, normalFrames=frames, consumer_id=index
        )
        speed_inference = f'{time.time() - start_keyFrame_extractor:.3f}'

        await asyncio.to_thread(
            upload_cv2_images_to_folder,
            [(str(i), frames[i]) for i in keyframes],
            timestamp = timestamp,
            speed_inference = speed_inference
        )
        cv2.waitKey(5000)
        #print(f"⏱ Latencia Global: {time.time() - timestamp:.3f} s")
        queue.task_done()


async def main():
    queue = asyncio.Queue(maxsize=10)

    producer = asyncio.create_task(video_producer(queue, chunk_seconds=5))
    consumers = [asyncio.create_task(video_consumer(queue, i)) for i in range(10)]

    await producer
    await queue.join()

    for c in consumers:
        c.cancel()
