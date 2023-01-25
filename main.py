import numpy as np

from video_tools.video_info import VideoInfo
from geometry.geometry import *  
from video_tools.video_info import VideoInfo
from video_tools.sink import *


from detection.detection_tools import Detections, BoxAnnotator
from draw.color import ColorPalette
from constants.bboxes import *
from dataclasses import dataclass
from ultralytics import YOLO

from tracking.rack_counter import RackScanner, ScannerCounterAnnotator

from tqdm import tqdm
from video_tools.source import get_video_frames_generator
from video_tools.sink import VideoSink
from detection.detection_tools import BoxAnnotator, Detections
from draw.color import ColorPalette
from geometry.geometry import Point
from tracking.rack_counter import RackScanner, ScannerCounterAnnotator

from bytetrack.yolox.tracker.byte_tracker import BYTETracker
from video_tools.video_info import VideoInfo
from tracking.tracking_utils import detections2boxes, match_detections_with_tracks


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


def main():

    model = YOLO(YoloArgs.MODEL_PATH)  
    model.fuse()

    CLASS_NAMES_DICT = model.model.names

    print(VideoInfo.from_video_path(YoloArgs.SOURCE_VIDEO_PATH))

    # create BYTETracker instance
    byte_tracker = BYTETracker(BYTETrackerArgs())
    # create VideoInfo instance
    video_info = VideoInfo.from_video_path(YoloArgs.SOURCE_VIDEO_PATH)
    # create frame generator
    generator = get_video_frames_generator(YoloArgs.SOURCE_VIDEO_PATH)
    # create LineCounter instance
    # create instance of BoxAnnotator and LineCounterAnnotator

    box_annotator = BoxAnnotator(color=ColorPalette(), thickness=2, text_thickness=2, text_scale=0.5)
    scanner = RackScanner(Point(x=300, y=50), 620)
    scanner_annotator = ScannerCounterAnnotator() 

    # open target video file
    with VideoSink(YoloArgs.TARGET_VIDEO_PATH, video_info) as sink:

        # loop over video frames
        for frame in tqdm(generator, total=video_info.total_frames):

            # model prediction on single frame and conversion to supervision Detections
            results = model(frame)
            detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int)
            )
            # tracking detections
            tracks = byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape
            )
            tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
            detections.tracker_id = np.array(tracker_id)

            # filtering out detections without trackers
            mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)

            # format custom labels
            labels = [
                f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id
                in detections
            ]

            # updating line counter
            scanner.update(detections=detections)

            # annotate and display frame
            frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
            scanner_annotator.annotate(frame=frame, rack_scanner=scanner)
            sink.write_frame(frame)

if __name__ == "__main__":
    main()