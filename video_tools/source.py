from typing import Generator
import cv2
import numpy as np


def shift_frame(frame: np.ndarray, shift: int) -> np.ndarray:
    """
    Shifts the frame to the left or right by the specified number of pixels.

    :param frame: np.ndarray : The frame to shift.
    :param shift: int : The number of pixels to shift the frame.
    :return: np.ndarray : The shifted frame.
    """
    height, width = frame.shape[:2]
    black_img = np.zeros((height, width, 3), np.uint8)

    if shift > 0:
        img_left = frame[:, shift:]
        black_img[:, : width - shift] = img_left
    elif shift < 0:
        img_right = frame[:, : width + shift]
        black_img[:, -width - shift :] = img_right

    return black_img


# create generator that yields the shifted frame from original frame
def generate_shifted_frames(frame: np.ndarray, shift: int, stride: int) -> np.ndarray:
    for i in range(1, shift, stride):
        yield shift_frame(frame, i)  # shift frame to the right


def get_video_frames_generator(
    video_path: str, stride: int = 10
) -> Generator[int, None, None]:
    """
    Returns a generator that yields the frames of the video.

    :param video_path: str : The path of the video file.
    :return: Generator[int, None, None] : Generator that yields the frames of the video.
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise Exception(f"Could not open video at {video_path}")
    success, frame = video.read()
    while success:
        if (
            video.get(cv2.CAP_PROP_POS_FRAMES)
            == video.get(cv2.CAP_PROP_FRAME_COUNT) - 1
        ):
            print("This is the last frame")
            hor_size = frame.shape[1]
            yield from generate_shifted_frames(frame, hor_size, stride)
        yield frame
        success, frame = video.read()
    video.release()
