from detection import Detection
from typing import List
import numpy as np
from dataclasses import dataclass
from typing import Generator
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


@dataclass(frozen=True)
class VideoConfig:
    fps: float
    width: int
    height: int



# converts List[Detection] into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: List[Detection], with_confidence: bool = True) -> np.ndarray:
    return np.array([
        [
            detection.rect.top_left.x, 
            detection.rect.top_left.y,
            detection.rect.bottom_right.x,
            detection.rect.bottom_right.y,
            detection.confidence
        ] if with_confidence else [
            detection.rect.top_left.x, 
            detection.rect.top_left.y,
            detection.rect.bottom_right.x,
            detection.rect.bottom_right.y
        ]
        for detection
        in detections
    ], dtype=float)

    
    
def generate_frames(video_file: str) -> Generator[np.ndarray, None, None]:
    video = cv2.VideoCapture(video_file)
    frame_count = 0

    while video.isOpened():
        success, frame = video.read()

        if not success:
            break
        cv2.imwrite('/content/temp/frame_{:04d}.jpg'.format(frame_count), frame)
        frame_path = "/content/temp/frame_{:04d}.jpg".format(frame_count)
        frame_count += 1
        yield frame_path

    video.release()


def plot_image(image: np.ndarray, size: int = 12) -> None:
    plt.figure(figsize=(size, size))
    plt.imshow(image[...,::-1])
    plt.show() 
    
    
   # create cv2.VideoWriter object that we can use to save output video
def get_video_writer(target_video_path: str, video_config: VideoConfig) -> cv2.VideoWriter:
    video_target_dir = os.path.dirname(os.path.abspath(target_video_path))
    os.makedirs(video_target_dir, exist_ok=True)
    return cv2.VideoWriter(
        target_video_path, 
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"), 
        fps=video_config.fps, 
        frameSize=(video_config.width, video_config.height), 
        isColor=True
    ) 
