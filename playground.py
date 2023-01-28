import pickle
from tqdm import tqdm


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
from tracking.rack_counter import RackScanner, ScannerCounterAnnotator
from video_tools.video_info import VideoInfo


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
generator = get_video_frames_generator(YoloArgs.SOURCE_VIDEO_PATH, stride=10)

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

with VideoSink(YoloArgs.TARGET_VIDEO_PATH, 3, video_info) as sink:

    for i, frame in enumerate(tqdm(generator, total=video_info.total_frames)):
        if i < video_info.total_frames:
            detections = detections_list[i]

            labels = [
                f"#{tracker_id} {CONSTANTS.CLASS_ID_LABELS[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id in detections
            ]

            # annotate and display frame
            frame = box_annotator.annotate(
                frame=frame, detections=detections, labels=labels
            )

            scanner.update(detections=detections)
            scanner_annotator.annotate(frame=frame, rack_scanner=scanner)

        sink.write_frame(frame)
    #
    # print(create_submission_dict(scanner.rack_tracks, 0.9, 20))


## without write 750, with write 120, with anno 40-60
