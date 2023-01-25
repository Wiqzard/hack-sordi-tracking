import sys

from PIL import Image
import cv2
import numpy as np

from video_tools.video_info import VideoInfo
from geometry.geometry import *  
from video_tools.video_info import VideoInfo
from video_tools.sink import *

#from ultralytics import YOLO

import pickle

from detection.detection_tools import Detections, BoxAnnotator
from draw.color import ColorPalette
from constants.bboxes import *


from tracking.rack_counter import RackScanner, ScannerCounterAnnotator

def main():

    box_annotator = BoxAnnotator(color=ColorPalette(), thickness=2, text_thickness=1, text_scale=1)

    PATH_TO_FRAME = "dataset/1.jpg"
    PATH_TO_MODEL = "dataset/best.pt"

    image = Image.open(PATH_TO_FRAME)
    frame = np.asarray(image)
    frame = cv2.imread(PATH_TO_FRAME)

    #model = YOLO(PATH_TO_MODEL)
    #print(model.model.names)
    #{0: 'klt_box_empty', 1: 'klt_box_full', 2: 'rack_1', 3: 'rack_2', 4: 'rack_3'}
    #results = model(PATH_TO_FRAME)
    #detections = Detections(
    #            xyxy=results[0].boxes.xyxy.cpu().numpy(),
    #            confidence=results[0].boxes.conf.cpu().numpy(),
    #            class_id=results[0].boxes.cls.cpu().numpy().astype(int)
    #        )
    #with open("dataset/detections.pkl", "wb") as outp:
    #    pickle.dump(detections, outp, pickle.HIGHEST_PROTOCOL)
    #
#--------------------------------------------------------------------
    with open("dataset/detections.pkl", "rb") as inp:
        detections = pickle.load(inp) 
#        print(detections.xyxy)

    import bytetrack.yolox

    from bytetrack.yolox.tracker.byte_tracker import BYTETracker, STrack
    from onemetric.cv.utils.iou import box_iou_batch
    from dataclasses import dataclass
    from tracking.tracking_utils import detections2boxes, match_detections_with_tracks

    @dataclass(frozen=True)
    class BYTETrackerArgs:
        track_thresh: float = 0.25
        track_buffer: int = 30
        match_thresh: float = 0.8
        aspect_ratio_thresh: float = 3.0
        min_box_area: float = 1.0
        mot20: bool = False
    byte_tracker = BYTETracker(BYTETrackerArgs())
    sys.path.append("bytetrack/")
    scanner = RackScanner(Point(x=300, y=50), 600)
    scanner_annotator = ScannerCounterAnnotator() 
    
    tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape
        )  
    tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
    detections.tracker_id = np.array(tracker_id)
    scanner.update(detections)
    scanner.update(detections)
    scanner.update(detections)
    print("main")
    print(scanner.tracker_state)
    print(scanner.curr_rack)
    print(scanner.rack_detections)
    #print(find_shelve(3, 160, 200))
    
    frame = box_annotator.annotate(frame=frame, detections=detections)
    scanner_annotator.annotate(frame=frame, rack_scanner=scanner)
    #placeholder annotator
    cv2.imshow("frame", frame)
    cv2.waitKey(0)
#--------------------------------------------------------------------
#    print(f"{len(detections)} detections: \n {detections.class_id}")
#
#    #detections.group_racks() 
#
#    ph_detections = detections.get_placeholder_for_rack(21)
#    print(ph_detections.xyxy)
#    labels = [
#                f"# Placeholder {confidence:0.2f}"
#                for _, confidence, class_id, tracker_id
#                in ph_detections
#            ]
#
#    frame = box_annotator.annotate(frame=frame, detections=ph_detections)
#    cv2.imshow("frame", frame)
#    cv2.waitKey(0)

if __name__ == "__main__":
    main()