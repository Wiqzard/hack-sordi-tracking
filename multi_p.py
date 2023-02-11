import pickle
from tqdm import tqdm
from time import time

from video_tools.video_info import VideoInfo
from geometry.geometry import *
from video_tools.sink import *
from detection.detection_tools import Detections, BoxAnnotator
from draw.color import ColorPalette, Color
from tracking.tracking_counter import create_submission_dict
from constants.bboxes import *
from tracking.rack_counter import RackScanner, ScannerCounterAnnotator
from video_tools.source import get_video_frames_generator
from draw.color import ColorPalette

# from tracking.rack_counter import RackScanner, ScannerCounterAnnotator
from tracking.rack_counter_new import RackScanner, ScannerCounterAnnotator
from video_tools.video_info import VideoInfo

import concurrent.futures
from time import perf_counter
import itertools


@dataclass(frozen=True)
class YoloArgs:
    SOURCE_VIDEO_PATH: str = "dataset/eval_video_1.mp4"
    TARGET_VIDEO_PATH: str = "dataset/result/eval_video_1.mp4"
    MODEL_PATH: str = "dataset/best.pt"


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


video_info = VideoInfo.from_video_path(YoloArgs.SOURCE_VIDEO_PATH)
generator = get_video_frames_generator(
    YoloArgs.SOURCE_VIDEO_PATH, stride=10, reduction_factor=2
)

with open("dataset/detections.pickle", "rb") as handle:
    detections_list = pickle.load(handle)

box_annotator = BoxAnnotator(
    color=ColorPalette(),
    thickness=2,
    text_thickness=1,
    text_scale=0.3,
    text_padding=2,
)
scanner = RackScanner(Point(x=300, y=50), 620)
scanner_annotator = ScannerCounterAnnotator(
    thickness=2,
    color=Color.white(),
    text_thickness=2,
    text_color=Color.red(),
    text_scale=0.6,
    text_offset=1.5,
    text_padding=10,
)
## without write 750, with write 120, with anno 40-60
from time import time


def process_detections(detections):
    labels = [
        f"#{tracker_id} {CONSTANTS.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id in detections
    ]
    return labels


def process_frame(frame, detections):
    labels = [
        f"#{tracker_id} {CONSTANTS.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id in detections
    ]
    frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
    sink.write_frame(frame)


#    return frame


if __name__ == "__main__":
    with VideoSink(YoloArgs.TARGET_VIDEO_PATH, 1, video_info) as sink:
        (
            detect_time,
            track_up_time,
            track_matcher,
            box_anno,
            scanner_up,
            scanner_anno,
            writer,
        ) = (0, 0, 0, 0, 0, 0, 0)

        frame_batch = []
        detections_batch = []
        for i, frame in tqdm(enumerate(generator), total=video_info.total_frames):
            if frame is None:
                continue
            if i >= len(detections_list):
                continue
            detections_batch.append(detections_list[i])
            frame_batch.append(frame)
            if len(frame_batch) > 4:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    executor.map(process_frame, frame_batch, detections_batch)
                #                    frames = [
                #                        executor.submit(process_frame, frame, detections)
                #                        for frame, detections in zip(frame_batch, detections_batch)
                #                    ]
                #                    for i in range(len(frame_batch)):
                #                        executor.submit(sink.write_frame, frames[i].result())
                #
                detections_batch = []
                frame_batch = []
