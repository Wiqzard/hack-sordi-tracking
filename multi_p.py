from typing import Generator
import numpy as np
from tqdm import tqdm
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
class VideoProcessor:
    def __init__(
        self,
        args: Args,
        source_video_path: str,
        target_video_path: str,
    ) -> None:
        self.args = args
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        

        print("<----- BUILD VIDEOPROCESSOR ----->")
        self.video_info = VideoInfo.from_video_path(args.SOURCE_VIDEO_PATH)
        self._frame_shape = self.video_info.shape
 
        self.detector = self._build_detector()
        self.tracker = self._build_tracker()
        self.scanner = self._build_scanner()
        self.box_annotator = self._build_box_annotator()
        self.scanner_annotator = self._build_scanner_annotator()
        print("<----- INITILIAZATION COMPLETE ----->")
    
    def _build_detector(self):
        print("*** BUILD DETECTOR ***")
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
    
    def _build_tracker(self):
        print("*** BUILD TRACKER ***")
        return BYTETracker(self.args.BYTE_TRACKER_ARGS())
    
    def _build_scanner(self):
        print("*** BUILD SCANNER ***")
        return RackScanner(Point(x=self.args.SCANNER_X, y=self.args.SCANNER_Y), 620)

    
    def _build_box_annotator(self):
        print("*** BUILD BOX ANNOTATOR ***")
        return BoxAnnotator(
                color=ColorPalette(),
                thickness=2,
                text_thickness=1,
                text_scale=0.3,
                text_padding=2,
            )
    
    def _build_scanner_annotator(self):
        print("*** BUILD SCNANER ANNOTATOR ***")
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
            self.source_video_path, stride=10, reduction_factor=self.reduction_factor
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
        assert isinstance(np_boxes_num, np.ndarray), \
        '`np_boxes_num` should be a `numpy.ndarray`'

        if np_boxes_num.sum() <= 0:
            print('[WARNNING] No object detected.')
            result = {'boxes': np.zeros([0, 6]), 'boxes_num': np_boxes_num}
        result = {k: v for k, v in result.items() if v is not None}
        return result

    def process_video(self, with_scanner=True, with_placeholders=True):
        self.with_scanner, self.with_placeholders = with_scanner, with_placeholders
              
        generator = self._build_generator()
        with VideoSink(self.target_video_path, 1, self.video_info) as sink:
            for idx, batch in tqdm(
                enumerate(generator),
                total=int(self.video_info.total_frames / self.BATCH_SIZE),
            ):
                if batch is None:
                    continue
                
                # Run detector in batches
                inputs = self.detector.preprocess(batch)
                self.detector.predictor.run()
                results = self.postprocess()
                
                last_tracks = None
                frame_tracks = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
                    future_to_frame = {
                        executor.submit(
                            self.process_frame_in_batch, frame, detections, last_tracks
                        ): idx
                        for idx, (frame, detections) in enumerate(
                            zip(batch, results["boxes_num"])
                        )
                    }
                    for future in concurrent.futures.as_completed(future_to_frame):
                        idx = future_to_frame[future]
                        last_tracks, processed_frame = future.result()
                        frame_tracks.append((idx, processed_frame, last_tracks))
                frame_tracks.sort(key=lambda x: x[0])
                for _, frame, last_tracks in frame_tracks:
                    sink.write_frame(frame)

    def process_frame_in_batch(self, results, idx, num, frame, last_tracks):
        detections = self._initial_results_to_detections(results, idx, num)
        if detections.xyxy.shape[0] == 0:
            return None
                
        detections = self._get_tracks(detections, last_tracks)
        if self.with_placeholders:
            self._annotate_placeholders_to_frame(frame, detections)
        frame = self._annotate_detections(frame, detections)
        if self.with_scanner:
              self._upadte_scanner(detections)
              frame = self._annotate_scanner(frame, detections)
              
        return last_tracks, frame
    
    def _initial_results_to_detections(self, results, idx, num) -> Detections:
        boxes = results['boxes'][idx*num : (idx+1) * num, :]
        boxes = boxes[boxes[:,1] > 0.3]
        detections = Detections(
                xyxy=boxes[:,2:],
                confidence=boxes[:,1],
                class_id=boxes[:,0].astype(int)
            )
        boxes = results["boxes"][:num, :]
        masks_conf_klt = np.logical_and(detections.confidence > 0.3, np.isin(detections.class_id, [0, 1]))
        masks_conf_rack = np.logical_and(detections.confidence > 0.7, np.isin(detections.class_id, [2, 3, 4, 5]))
        masks = np.logical_or(masks_conf_klt, masks_conf_rack)
        detections.filter(mask=masks, inplace=True)
        return detections
    
    def _get_tracks(self, detections):
        tracks = self.tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=self.frame_shape,
                img_size=self.frame_shape
            )

        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)
        # filtering out detections without trackers
        mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)
        return detections
    
    def _annotate_placeholders(self, frame, detections):
        placeholders, placeholder_labels= process_placeholders(detections, self.scanner.scanner.x)
        if placeholders and placeholder_labels:
            frame = self.box_annotator.annotate(
                    frame=frame, detections=placeholders, labels=placeholder_labels
                )   
            return frame
              
    def _annotate_detections(self, frame, detections):
        labels = [
            f"#{tracker_id} {self.args.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]
        # annotatoe detection boxes
        frame = self.box_annotator.annotate(
            frame=frame, detections=detections, labels=labels
        )
        return frame
    
    def _update_scanner(self, detections):
        self.scanner.update(detections)
              
    def _annotate_scanner(self, frame):
        return self.scanner_annotator.annotate(frame=frame, rack_scanner=self.scanner)
    
    