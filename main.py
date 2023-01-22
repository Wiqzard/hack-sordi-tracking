import sys

from PIL import Image
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

box_annotator = BoxAnnotator(color=ColorPalette(), thickness=2, text_thickness=2, text_scale=1)

PATH_TO_FRAME = "dataset/1.jpg"
PATH_TO_MODEL = "dataset/best.pt"

image = Image.open(PATH_TO_FRAME)
frame = np.asarray(image)


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
with open("dataset/detections.pkl", "rb") as inp:
    detections = pickle.load(inp) 
print(f"{len(detections)} detections: \n {detections.class_id}")

#detections.group_racks() 

ph_detections = detections.get_placeholder_for_rack(0)

labels = [
            f"# Placeholder {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in ph_detections
        ]

frame = box_annotator.annotate(frame=frame, detections=ph_detections)


def main():
    pass

if __name__ == "__main__":
    main()