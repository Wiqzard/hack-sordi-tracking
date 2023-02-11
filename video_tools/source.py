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
    video_path: str, stride: int = 10, reduction_factor: int = 1
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
    hor_size = frame.shape[1]
    idx = 0

    while success:
        if (
            video.get(cv2.CAP_PROP_POS_FRAMES)
            == video.get(cv2.CAP_PROP_FRAME_COUNT) - 1
        ):
            print("This is the last frame")
            yield from generate_shifted_frames(frame, int(0.85 * hor_size), stride)

        success, frame = video.read()
        if idx % reduction_factor == 0:
            yield frame
        else:
            yield None

        idx += 1

    video.release()


def get_video_frames_batch_generator(
    video_path: str, batch_size: int = 1, stride: int = 10, reduction_factor: int = 1
) -> Generator:
    """
    Returns a generator that yields the frames of the video in batches.

    :param video_path: str : The path of the video file.
    :param batch_size: int : The size of the batch.
    :return: Generator : Generator that yields the frames of the video in batches.
    """
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise Exception(f"Could not open video at {video_path}")

    success, frame = video.read()
    hor_size = frame.shape[1]
    idx = 0

    batch = []
    while success:
        if (
            video.get(cv2.CAP_PROP_POS_FRAMES)
            == video.get(cv2.CAP_PROP_FRAME_COUNT) - 1
        ):
            print("This is the last frame")
            yield from generate_shifted_frames(frame, int(0.85 * hor_size), stride)

        success, frame = video.read()

        # add frame to batch
        if len(batch) == batch_size and idx % reduction_factor == 0:
            yield batch
            batch = []
        else:
            yield None

        idx += 1

    video.release()


# from decord import VideoReader
# from decord import cpu, gpu
def get_video_frames_batch_generator_v2(
    video_path: str, batch_size: int = 1, stride: int = 10, reduction_factor: int = 1
) -> Generator:
    """
    Returns a generator that yields the frames of the video in batches.

    :param video_path: str : The path of the video file.
    :param batch_size: int : The size of the batch.
    :return: Generator : Generator that yields the frames of the video in batches.
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    frames_list = list(range(0, total_frames, reduction_factor))
    saved_count = 0

    for idx in range(0, frames_list, batch_size):
        if idx + batch_size > len(frames_list):
            continue
            # yield vr.get_batch(frames_list[idx:]).asnumpy()

        # Last frame
        if idx + batch_size >= len(frames_list):
            print("This is the last frame.")
            frame = vr[idx].asnumpy()
            hor_size = frame.shape[1]
            yield from generate_shifted_frames(frame, int(0.85 * hor_size), stride)

        yield vr.get_batch(frames_list[idx : idx + batch_size]).asnumpy()
