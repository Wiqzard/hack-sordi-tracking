from typing import Generator, Tuple, Any, Union, List
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import multiprocessing as mp
import paddle

paddle.utils.run_check()

from PaddleDetection.deploy.python.infer import Detector
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

from video_tools.video_info import VideoInfo
from video_tools.source import get_video_frames_batch_generator_v2
from video_tools.sink import VideoSink
from tracking.rack_counter import RackScanner, ScannerCounterAnnotator
from tracking.tracking_utils import detections2boxes, match_detections_with_tracks
from tracking.tracking_counter import create_submission_dict, write_submission
from detection.detection_tools import BoxAnnotator, Detections, process_placeholders
from draw.color import ColorPalette, Color
from geometry.geometry import Point

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger()
handler = logging.FileHandler("logs.log")
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

Frame = np.ndarray
Path = Union[str, Path]


class VideoProcessor:
    """
    A class to detect and track boxes in a rack.
    ...

    Attributes
    ----------
    args : Args
        contains all the necessary information
    source_video_path : str | Path
        path of the video source
    target_video_path : str | Path
        path where the result is written

    Methods
    -------
    process_video(additional="" ):
        Processes the video and writes the result to the target video path.
    create_submission(additional=""):
        Creates a submission file from the processed video.
    """

    def __init__(
        self,
        args: Any,
        source_video_path: Union[str, Path],
        target_dir: Union[str, Path] = "/temp/",
    ) -> None:
        self.args = args

        if not os.path.exists(source_video_path) or not source_video_path.endswith(
            ".mp4"
        ):
            raise ValueError("Invalid source video path")
        self.source_video_path = source_video_path

        self.target_dir = os.path.dirname(target_dir)
        # os.makedirs(target_dir, exist_ok=True)

        self.target_video_path = os.path.join(target_dir, "processed_eval_video.mp4")

        logger.info("<---------- BUILD VIDEOPROCESSOR ---------->")
        self.video_info: VideoInfo = VideoInfo.from_video_path(self.source_video_path)
        self._frame_shape: Tuple[int, int] = self.video_info.shape
        self.video_sink: VideoSink = VideoSink(
            self.target_video_path, self.args.REDUCTION_FACTOR, self.video_info
        )

        # self.detector: Detector = self._build_detector()
        self.tracker: BYTETracker = self._build_tracker()
        self.scanner: RackScanner = self._build_scanner()
        self.box_annotator: BoxAnnotator = self._build_box_annotator()
        self.scanner_annotator: ScannerCounterAnnotator = (
            self._build_scanner_annotator()
        )
        logger.info("<--------- INITILIAZATION COMPLETE ---------> \n")

    def _build_detector(self) -> Detector:
        logger.info("*** BUILD DETECTOR ***")
        return Detector(
            model_dir=self.args.MODEL_DIR,
            device="GPU",
            run_mode=self.args.RUN_MODE,
            batch_size=self.args.BATCH_SIZE,
            trt_min_shape=1,
            trt_max_shape=1280,
            trt_opt_shape=640,
            trt_calib_mode=False,
            cpu_threads=self.args.CPU_THREADS,
            enable_mkldnn=False,
            enable_mkldnn_bfloat16=False,
            output_dir="output_paddle",
            threshold=0.5,
            delete_shuffle_pass=False,
        )

    def _build_tracker(self) -> BYTETracker:
        logger.info("*** BUILD TRACKER ***")
        return BYTETracker(self.args.BYTE_TRACKER_ARGS())

    def _build_scanner(self) -> RackScanner:
        logger.info("*** BUILD SCANNER ***")
        return RackScanner(Point(x=self.args.SCANNER_X, y=self.args.SCANNER_Y), 620)

    def _build_box_annotator(self) -> BoxAnnotator:
        logger.info("*** BUILD BOX ANNOTATOR ***")
        return BoxAnnotator(
            color=ColorPalette(),
            thickness=2,
            text_thickness=1,
            text_scale=0.3,
            text_padding=2,
        )

    def _build_scanner_annotator(self) -> ScannerCounterAnnotator:
        logger.info("*** BUILD SCNANER ANNOTATOR ***")
        return ScannerCounterAnnotator(
            thickness=2,
            color=Color.white(),
            text_thickness=2,
            text_color=Color.red(),
            text_scale=0.6,
            text_offset=1.5,
            text_padding=10,
        )

    def _build_generator(self) -> Generator:
        return get_video_frames_batch_generator_v2(
            self.source_video_path,
            batch_size=self.args.BATCH_SIZE,
            stride=self.args.STRIDE,
            reduction_factor=self.args.REDUCTION_FACTOR,
        )

    def _infer_batch(self, detector, batch: List[Frame]) -> List[Detections]:
        """
        > It takes a list of frames, preprocesses them, runs the predictor, and then postprocesses the
        results

        :param detector: The detector object that we created earlier
        :param batch: List[Frame]
        :type batch: List[Frame]
        :return: A list of detections.
        """
        inputs = detector.preprocess(batch)
        detector.predictor.run()
        results = self._postprocess(detector)
        return results

    def _postprocess(self, detector):
        """
        :param detector: the detector object
        :return: The result is a dictionary with the keys "boxes" and "boxes_num". The value of "boxes"
        is a numpy array of shape (n, 6) where n is the number of detected objects. The value of
        "boxes_num" is a numpy array of shape (1,) containing the number of detected objects.
        """
        np_boxes_num, np_boxes, np_masks = np.array([0]), None, None
        output_names = detector.predictor.get_output_names()
        boxes_tensor = detector.predictor.get_output_handle(output_names[0])
        np_boxes = boxes_tensor.copy_to_cpu()
        boxes_num = detector.predictor.get_output_handle(output_names[1])
        np_boxes_num = boxes_num.copy_to_cpu()

        result = dict(boxes=np_boxes, masks=np_masks, boxes_num=np_boxes_num)
        np_boxes_num = result["boxes_num"]
        if not isinstance(np_boxes_num, np.ndarray):
            raise ValueError("np_boxes_num` should be a `numpy.ndarray`")

        if np_boxes_num.sum() <= 0:
            logger.warning("[WARNNING] No object detected.")
            result = {"boxes": np.zeros([0, 6]), "boxes_num": np_boxes_num}
        result = {k: v for k, v in result.items() if v is not None}
        return result

    def _initial_results_to_detections(
        self, results, idx: int
    ) -> Tuple[int, Detections]:
        boxes = results["boxes"][
            idx * self.args.MAX_DETECTIONS : (idx + 1) * self.args.MAX_DETECTIONS, :
        ]
        boxes = boxes[boxes[:, 1] > self.args.BOX_THRESHOLD]
        detections = Detections(
            xyxy=boxes[:, 2:], confidence=boxes[:, 1], class_id=boxes[:, 0].astype(int)
        )
        # filter where center of box below threshold
        position_mask = (
            detections.xyxy[:, 1] + (detections.xyxy[:, 3] - detections.xyxy[:, 1]) / 2
        ) < 600
        area_mask = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * (
            detections.xyxy[:, 3] - detections.xyxy[:, 1]
        ) < 2500
        mask_conf_klt = np.isin(detections.class_id, [0, 1])
        mask_klt = np.logical_and(area_mask, mask_conf_klt)
        mask_conf_rack = np.logical_and(
            detections.confidence > self.args.RACK_THRESHOLD,
            np.isin(detections.class_id, [2, 3, 4, 5]),
        )
        mask = np.logical_and(
            np.logical_or(mask_conf_klt, mask_conf_rack), position_mask
        )

        detections.filter(mask=mask, inplace=True)
        return (idx, detections)

    def _get_tracks(self, detections: Detections) -> Detections:
        tracks = self.tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=self.video_info.shape,
            img_size=self.video_info.shape,
        )
        if len(detections) == 0 or len(tracks) == 0:
            return detections

        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)

        # filtering out detections without trackers
        mask = np.array(
            [tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool
        )
        detections.filter(mask=mask, inplace=True)
        return detections

    def _annotate_placeholders(
        self, frame_detections: Tuple[int, Frame, Detections]
    ) -> Tuple[int, Frame]:
        idx, frame, detections = frame_detections
        placeholders, placeholder_labels = process_placeholders(
            detections, self.scanner.scanner.x
        )
        if placeholders and placeholder_labels:
            frame = self.box_annotator.annotate(
                frame=frame, detections=placeholders, labels=placeholder_labels
            )
        return idx, frame

    def _annotate_detections(
        self, frame_detections: Tuple[int, Frame, Detections]
    ) -> Tuple[int, Frame]:
        idx, frame, detections = frame_detections
        labels = [
            f"#{tracker_id} {self.args.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id in detections
        ]
        # annotatoe detection boxes
        frame = self.box_annotator.annotate(
            frame=frame, detections=detections, labels=labels
        )
        return idx, frame

    def _update_scanner(self, detections: Detections) -> None:
        self.scanner.update(detections)

    def _annotate_scanner(self, idx: int, frame: Frame) -> Tuple[int, Frame]:
        return idx, self.scanner_annotator.annotate(
            frame=frame, rack_scanner=self.scanner
        )

    def process_video(
        self,
        detector,
        with_scanner: bool = True,
        with_placeholders: bool = True,
        with_annotate_scanner: bool = True,
    ) -> None:
        """
        Reads in the video frames, runs the detector and tracker, and writes the annotated video.
        :param detector: The detector object
        :param with_scanner: If True, the scanner will be used, defaults to True
        :type with_scanner: bool (optional)
        :param with_placeholders: If True, the placeholders will be drawn on the frame, defaults to True
        :type with_placeholders: bool (optional)
        :param with_annotate_scanner: If True, the scanner will be drawn on the frame, defaults to True
        :type with_annotate_scanner: bool (optional)
        """
        if not with_scanner and with_annotate_scanner:
            raise ValueError(
                "Cannot annotate scanner if scanner is not used. Set `with_annotate_scanner` to False."
            )

        self.with_scanner, self.with_placeholders = with_scanner, with_placeholders

        generator = self._build_generator()
        with self.video_sink as sink:
            for _, batch in tqdm(
                enumerate(generator),
                total=int(
                    self.video_info.total_frames
                    / self.args.BATCH_SIZE
                    / self.args.REDUCTION_FACTOR
                ),
            ):
                if batch is None:
                    continue
                # Run detector in batches
                results = self._infer_batch(detector, batch)

                # Process each frame in batch
                # with ProcessPoolExecutor(max_workers=self.args.BATCH_SIZE) as executor:
                with ThreadPoolExecutor() as executor:
                    frames_gen = ((i, frame) for i, frame in enumerate(batch))
                    # with ProcessPoolExecutor(max_workers=self.args.BATCH_SIZE) as executor:
                    # results_gen = executor.map(
                    #    partial(self._initial_results_to_detections, results),
                    #    range(len(batch)),
                    # )
                    results_gen = (
                        (self._initial_results_to_detections(results, i))
                        for i in range(len(batch))
                    )
                    detections_dict: dict[int, Detections] = dict(results_gen)

                    # if not detections, simply write the frames
                    if not detections_dict:
                        frames = dict(frames_gen)
                        for i in len(batch):
                            sink.write_frame(frames[i])
                        continue

                    detections_dict = {
                        i: self._get_tracks(detections_dict[i])
                        for i in range(len(batch))
                    }

                    # if not detections, simply write the frames
                    if not detections_dict:
                        frames = dict(frames_gen)
                        for i in len(batch):
                            sink.write_frame(frames[i])
                        continue

                    if with_scanner:
                        temp = [
                            self._update_scanner(detections_dict[i])
                            for i in range(len(batch))
                        ]
                    # if tracks
                    # if not all(val is None for val in detections_dict.values()):
                    #    if with_scanner and with_annotate_scanner:
                    #        frames_gen: Generator[int, Frame] = executor.map(self._annotate_scanner, batch, range(len(batch)))

                    frames_detections_gen = (
                        (i, frame, detections_dict[i]) for i, frame in frames_gen
                    )
                    if with_placeholders:
                        frames_gen = executor.map(
                            self._annotate_placeholders, frames_detections_gen
                        )

                    frames_detections_gen = (
                        (i, frame, detections_dict[i]) for i, frame in frames_gen
                    )

                    # annotate detections
                    frames_gen = executor.map(
                        self._annotate_detections, frames_detections_gen
                    )
                    # annotate scanner
                    if with_annotate_scanner:
                        frames = dict(frames_gen)
                        # frames_gen = (
                        #    self._annotate_scanner(frames[i], i)
                        #    for i in range(len(batch))  # frame, i in frames_gen
                        # )
                        frames_gen = (
                            self._annotate_scanner(frames[i], i)
                            for i in range(len(batch))  # frame, i in frames_gen
                        )
                    print(next(iter(frames_gen)))
                    # sort the frames depending on intital batch index
                    frames_ordered = sorted(list(frames_gen), key=lambda x: x[1])
                    frames_ordered = [x[0] for x in frames_ordered]
                    for frame in frames_ordered:
                        sink.write_frame(frame)

    def create_submission(self, mAP: float, fps: float, save: bool = False) -> dict:
        """
        It takes the output of the `scanner` and creates a submission dictionary

        :param mAP: mean average precision, in percent points
        :type mAP: float
        :param fps: frames per second
        :type fps: float
        :param save: bool = False, defaults to False
        :type save: bool (optional)
        :return: A dictionary with the following keys:
            - "scanned_racks": A list of dictionaries with the following keys:
                - "rack_id": The id of the rack
                - "track": A list of dictionaries with the following keys:
                    - "frame_id": The id of the frame
                    - "bbox": A list of 4
        """
        if fps < 25 or mAP < 10:
            raise ValueError("fps or mAP in wrong format")
        submission_dict = create_submission_dict(
            scanned_racks=self.scanner.rack_tracks, mAP=mAP, fps=fps
        )
        if save:
            submission_path = os.path.join(self.target_dir, "AcademicWeapons.json")
            write_submission(
                submission_dict=submission_dict, submission_path=submission_path
            )
        return submission_dict
