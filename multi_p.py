from typing import Generator, Tuple
import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
import concurrent.futures
import paddle
paddle.utils.run_check()

from PaddleDetection.deploy.python.infer import Detector
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

from video_tools.video_info import VideoInfo
from video_tools.sink import VideoSink
from tracking.rack_counter_new import RackScanner, ScannerCounterAnnotator
from detection.detection_tools import BoxAnnotator, Detections, process_placeholders
from draw.color import ColorPalette, Color
from geometry.geometry import Point

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

logger = logging.getLogger()
logger.setLevel(logging.INFO)
#handler = logging.FileHandler('logs.log')
#handler.setLevel(logging.DEBUG)
# create a formatter
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#handler.setFormatter(formatter)
#logger.addHandler(handler)

Frame = np.ndarray
Path = Union[str, Path]

class VideoProcessor:
    """
    A class to detect and track objects in a video.
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
    info(additional=""):
        Prints the person's name and age.
    """
    
    def __init__(
        self,
        args: Args,
        source_video_path: Union[str, Path],
        target_video_path: Union[str, Path],
    ) -> None:
        self.args = args
        
        if not os.path.exists(source_video_path) or not source_video_path.endswith(".mp4"):
            raise ValueError("Invalid source video path")
        self.source_video_path = source_video_path
        
        target_dir = os.path.dirname(target_video_path)
        os.makedirs(target_dir, exist_ok=True)
        self.target_video_path = target_video_path
        
        logger.info("<---------- BUILD VIDEOPROCESSOR ---------->")
        self.video_info: VideoInfo = VideoInfo.from_video_path(args.SOURCE_VIDEO_PATH)
        self._frame_shape: Tuple[int, int] = self.video_info.shape
        
        self.detector: Detector = self._build_detector()
        self.tracker: Tracker = self._build_tracker()
        self.scanner: Scanner = self._build_scanner()
        self.box_annotator: BoxAnnotator = self._build_box_annotator()
        self.scanner_annotator: ScannerAnnotator = self._build_scanner_annotator()
        logger.info("<--------- INITILIAZATION COMPLETE ---------> \n")
    


    def _build_detector(self) -> Detector:
        logger.info("*** BUILD DETECTOR ***")
        return Detector(model_dir=self.args.MODEL_DIR,
                 device='GPU',
                 run_mode=self.args.RUN_MODE,
                 batch_size=self.args.BATCH_SIZE,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=self.args.CPU_THREADS,
                 enable_mkldnn=False,
                 enable_mkldnn_bfloat16=False,
                 output_dir='output_paddle',
                 threshold=0.5,
                 delete_shuffle_pass=False)
    
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
            self.source_video_path, batch_size=self.args.BATCH_SIZE, stride=self.args.STRIDE, reduction_factor=self.args.REDUCTION_FACTOR
        )
    
    def postprocess(self):
        np_boxes_num, np_boxes, np_masks = np.array([0]), None, None
        output_names = self.detector.predictor.get_output_names()
        boxes_tensor = self.detector.predictor.get_output_handle(output_names[0])
        np_boxes = boxes_tensor.copy_to_cpu()
        boxes_num = self.detector.predictor.get_output_handle(output_names[1])
        np_boxes_num = boxes_num.copy_to_cpu()

        result = dict(boxes=np_boxes, masks=np_masks, boxes_num=np_boxes_num)
        np_boxes_num = result['boxes_num']
        if not isinstance(np_boxes_num, np.ndarray):
            raise ValueError("np_boxes_num` should be a `numpy.ndarray`")

        if np_boxes_num.sum() <= 0:
            logger.warning('[WARNNING] No object detected.')
            result = {'boxes': np.zeros([0, 6]), 'boxes_num': np_boxes_num}
        result = {k: v for k, v in result.items() if v is not None}
        return result

    def process_video(self, with_scanner: bool=True, with_placeholders: bool=True, with_annotate_scanner: bool=True) -> None:
        self.with_scanner, self.with_placeholders = with_scanner, with_placeholders
              
        generator = self._build_generator()
        with VideoSink(self.target_video_path, 1, self.video_info) as sink:
            for idx, batch in tqdm(
                enumerate(generator),
                total=int(self.video_info.total_frames / self.args.BATCH_SIZE),
            ):
                if batch is None:
                    continue
                # Run detector in batches
                inputs = self.detector.preprocess(batch)
                self.detector.predictor.run()
                results = self.postprocess()

                # Process each frame in batch
                with ProcessPoolExecuter(max_workers=2) as executor:
                    frames_gen = (i, frame for i, frame in enumerate(batch))
                    results_gen = executor.map(self._initial_results_to_detections, results, range(len(batch)))
                    detections_dict: dict[int, Detections] = dict(results_gen)
                
                for i in range(len(batch)):
                    detections_dict[i] = self._get_tracks(detections_dict[i])
                    if with_scanner:
                        self._update_scanner(detections_dict[i])

                if with_scanner and with_annotate_scanner: 
                    frames_gen: Generator[int, Frame] = executor.map(self._annotate_scanner, batch, range(len(batch)))

                frames_detections_gen = (i, frame, detections_dict[i] for i, frame in list(frames_gen))
                if with_placeholders:
                    frames_detections_gen = executor.map(self._annotate_placeholders, frames_detections_gen)                    

                frames_gen = executor.map(self._annotate_detections, frames_detections_gen)
                frames_ordered = list(frames_gen).sort(key=lambda x: x[0])
                frames_ordered = [x[1] for x in frames_ordered] 

                for frame in frames_ordered:
                    sink.write(frame)

    def sort_indexed_tuple(self, tup: Tuple[Tuple[int, Any]]):
        tup.sort(key=lambda x: x[0]) 
        return tup 

    def _initial_results_to_detections(self, results, idx: int) -> Tuple[int, Detections]:
        boxes = results['boxes'][idx*self.args.MAX_DETECTIONS : (idx+1) * self.args.MAX_DETECTIONS, :]
        boxes = boxes[boxes[:,1] > 0.3]
        detections = Detections(
                xyxy=boxes[:,2:],
                confidence=boxes[:,1],
                class_id=boxes[:,0].astype(int)
            )
        boxes = results["boxes"][:self.args.MAX_DETECTIONS, :]
        masks_conf_klt = np.logical_and(detections.confidence > 0.3, np.isin(detections.class_id, [0, 1]))
        masks_conf_rack = np.logical_and(detections.confidence > 0.7, np.isin(detections.class_id, [2, 3, 4, 5]))
        masks = np.logical_or(masks_conf_klt, masks_conf_rack)
        detections.filter(mask=masks, inplace=True)
        print("init: ",detections.xyxy.shape)
        return idx, detections
    
    def _get_tracks(self, detections: Detections) -> Detections:
        tracks = self.tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=self.video_info.shape,
                img_size=self.video_info.shape
            )

        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)
        # filtering out detections without trackers
        mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)
        print("tracks: ",detections.xyxy.shape)

        return detections
    
    def _annotate_placeholders(self, frame_detections) -> np.ndarray:
        idx, frame, detections = frame_detections
        placeholders, placeholder_labels= process_placeholders(detections, self.scanner.scanner.x)
        if placeholders and placeholder_labels:
            frame = self.box_annotator.annotate(
                    frame=frame, detections=placeholders, labels=placeholder_labels
                )   
            return idx, frame, detections
              
    def _annotate_detections(self, frame_detections: Tuple[int, Frame, Detections]) -> Tuple[int, Frame]:
        idx, frame, detections = frame_detections
        labels = [
            f"#{tracker_id} {self.args.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections]
        # annotatoe detection boxes
        frame = self.box_annotator.annotate(
            frame=frame, detections=detections, labels=labels
        )
        return idx, frame
    
    def _update_scanner(self, detections: Detections) -> None:
        self.scanner.update(detections)
              
    def _annotate_scanner(self, frame: Frame, idx: int) -> Tuple[int, Frame]:
        return idx, self.scanner_annotator.annotate(frame=frame, rack_scanner=self.scanner)
    
    