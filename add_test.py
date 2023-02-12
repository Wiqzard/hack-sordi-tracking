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
rack_scanner = RackScanner(start=Point(x=300, y=50), height=620)
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

from detection.detection_tools import remove_placeholders_iou


def annotate_placeholders(frame: np.ndarray, detections: Detections) -> np.ndarray:
    rack_mask = np.isin(detections.class_id, CONSTANTS.RACK_IDS) & (
        np.less(detections.xyxy[:, 0], rack_scanner.scanner.x)
        & np.greater(detections.confidence, 0.9)
    )
    if np.any(rack_mask):

        rack_detections_after_scanner = detections.filter(rack_mask)
        if placeholders := Detections.get_placeholders_for_racks(
            rack_detections_after_scanner
        ):
            if placeholders := remove_placeholders_iou(
                detections.filter(~rack_mask), placeholders
            ):
                placeholder_labels = ["placeholder"] * len(placeholders)

                frame = box_annotator.annotate(
                    frame=frame, detections=placeholders, labels=placeholder_labels
                )
    return frame


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

    for i, frame in tqdm(enumerate(generator), total=video_info.total_frames):
        if frame is None:
            continue
        if i >= len(detections_list):
            continue

        detections = detections_list[i]

        # mask for detections: True if rack in scannner
        rack_mask = np.isin(detections.class_id, CONSTANTS.RACK_IDS) & (
            np.less(detections.xyxy[:, 0], rack_scanner.scanner.x)
            & np.greater(detections.confidence, 0.9)
        )
        if np.any(rack_mask):

            rack_detections_after_scanner = detections.filter(rack_mask)
            if placeholders := Detections.get_placeholders_for_racks(
                rack_detections_after_scanner
            ):
                if placeholders := remove_placeholders_iou(
                    detections.filter(~rack_mask), placeholders
                ):
                    placeholder_labels = ["placeholder"] * len(placeholders)

                    frame = box_annotator.annotate(
                        frame=frame, detections=placeholders, labels=placeholder_labels
                    )

        # format custom labels
        labels = [
            f"#{tracker_id} {CONSTANTS.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id in detections
        ]

        # annotatoe detection boxes
        start = time()
        frame = box_annotator.annotate(
            frame=frame, detections=detections, labels=labels
        )
        end = time()
        box_anno += end - start

        # update the scanner
        start = time()
        rack_scanner.update(detections=detections)
        end = time()
        scanner_up += end - start
        # draw the scanner
        start = time()
        scanner_annotator.annotate(frame=frame, rack_scanner=rack_scanner)
        end = time()
        scanner_anno += end - start
        # add the annotated frame to video
        start = time()
        sink.write_frame(frame)
        end = time()
        writer += end - start
print(round(box_anno, 3) * 1000)
print(round(scanner_up, 3) * 1000)
print(round(scanner_anno, 3) * 1000)
print(round(writer, 3) * 1000)
