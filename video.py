import cv2

class Video:
    def __init__(self, path):
        self.video = cv2.VideoCapture(path)

    def check_video(self):
        return self.video.isOpened()

    def get_fps(self):
        return self.video.get(cv2.CAP_PROP_FPS)

    def get_frame_count(self):
        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_duration(self):
        return self.get_frame_count() / self.get_fps()

    def get_width(self):
        return int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_height(self):
        return int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))