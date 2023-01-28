import cv2
import numpy as np
from time import time
from video_tools.video_info import VideoInfo


class VideoSink:
    """
    A context manager that uses OpenCV to save video frames to a file.

    :param output_path: str : The path to the output file where the video will be saved.
    :param video_info: VideoInfo : An instance of VideoInfo containing information about the video resolution, fps, and total frame count.
    """

    def __init__(self, output_path: str, reduction_factor: int, video_info: VideoInfo):
        """
        Initializes the VideoSink with the specified output path and video information.
        """
        self.output_path = output_path
        self.video_info = video_info
        self.reduction_factor = reduction_factor
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = None
        self.start, self.end = 0, 0
        self.frame_index = 0

    def __enter__(self):
        """
        Opens the output file and returns the VideoSink instance.
        """
        self.writer = cv2.VideoWriter(
            self.output_path,
            self.fourcc,
            int(self.video_info.fps / self.reduction_factor),
            self.video_info.resolution,
        )
        self.start = time()
        return self

    @property
    def elapsed(self) -> float:
        return self.end - self.start

    @property
    def avg_fps(self) -> float:
        return self.video_info.total_frames / self.elapsed

    def write_frame(self, frame: np.ndarray):

        """
        Writes a frame to the output video file.

        :param frame: np.ndarray : The frame to be written.
        """
        if self.frame_index % self.reduction_factor == 0:
            self.writer.write(frame)
        self.frame_index += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Closes the output file.
        """
        self.writer.release()
        self.end = time()
        print(
            f"Elapsed time: {self.elapsed:.2f} seconds for {self.video_info.total_frames} frames"
        )
        print(f"Average FPS: {self.avg_fps:.2f} frames per second")
