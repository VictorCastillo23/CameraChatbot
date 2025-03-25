import cv2
import os
from video import Video

def procesar_video(video_path, start_time, end_time, speed):
    video = Video(video_path)

    return f"\nVídeo procesado y guardado en: Video_1_cut.mp4\n"

"""
    if not video.check_video():
        return "The video could not be opened. Please check the file name."

    fps = video.get_fps()
    total_frames = video.get_frame_count()
    frame_width = video.get_width()
    frame_height = video.get_height()
    total_duration = video.get_duration()

    if end_time > total_duration:
        return "The final time is longer than the total duration of the video."
    if start_time < 0:
        return "The start time cannot be less than 0."
    if end_time < start_time:
        return "The start time cannot be greater than the end time."

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video_path = f"{os.path.splitext(video_path)[0]}_procesado.mp4"
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    video.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for current_frame in range(start_frame, end_frame + 1):
        ret, frame = video.video.read()

        if not ret:
            break
        if current_frame % speed == 0:
            out.write(frame)

    out.release()

    return f"Video processed and saved in: {output_video_path}"
"""