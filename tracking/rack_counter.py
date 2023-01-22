from typing import Dict, List, Tuple
from dataclasses import dataclass
import cv2
import numpy as np

from draw.color import Color
from geometry.geometry import Point, Rect, Vector, VerticalLine
from detection.detection_tools import Detections
from tracking.tracking_counter import RackDetection

class RackScanner:
    RACK_IDS = [0, 1, 2]
    def __init__(self, start: Point, height: int):
        """
        Initialize a LineCounter object.

        :param start: Point : The starting point of the line.
        :param end: Point : The ending point of the line.
        """
        #self.vector = Vector(start=start, end=end)
        self.scanner = VerticalLine(start, height)
        self.scanned_tracker_states : List[Dict[str, int]] = []

        self.tracker_state: Dict[str, bool] = {}
        self.in_count: int = 0
        self.out_count: int = 0

        self.curr_rack : str = None
        self.rack_detections : List[RackDetection] = []
        

    def update(self, detections: Detections):
        """
        Update the in_count and out_count for the detections that cross the line.

        :param detections: Detections : The detections for which to update the counts.
        """
        for xyxy, confidence, class_id, tracker_id in detections:
            """ 
            1. Get current rack (knowing when it started and ended)
            1.1 Handle IoD of racks, to make sure smt forklist
            2. Based on current rack, know which y-values for shelves
            3. Get all boxes of current rack, via multiple lines based on rack
            4. Append rack_detections a RackDetection when rack ended 
            """ 

            # handle detections with no tracker_id
            if tracker_id is None:
                continue

            # we check if all four anchors of bbox are on the same side of vector
            x1, y1, x2, y2 = xyxy
            anchors = [
                Point(x=x1, y=y1),
                Point(x=x1, y=y2),
                Point(x=x2, y=y1),
                Point(x=x2, y=y2),
            ]

            # number of points right to scanner
            triggers = sum(self.scanner.left_to(anchor) for anchor in anchors)

            # box completely to the right of scanner
            if triggers == 4:
                continue            

            # detection is partially in and partially out
            if triggers == 2:
                if class_id not in self.RACK_IDS:
                    continue
                self.curr_rack = class_id
                self.scanned_tracker_states[tracker_id] = class_id

            # boxes that are completely left to the scanner
            if triggers == 0 and class_id not in self.RACK_IDS: 
                self.tracker_state[tracker_id] = True 
                self.scanned_tracker_states[tracker_id] = class_id

                shelve = find_shelve(self.curr_rack, y1, y2) 
                #save from here
                

RACKS : Dict[str, Dict[str, int]] = {"rack_1" : {"shelve_1" : 3}}
            
CLASSES_ID = {"klt_box_empty" : 1008, 
              "klt_box_full" : 1009,
              "rack_1" : 1200, 
              "rack_2" : 1205, 
              "rack_3" : 1210, 
              "rack_4" : 1215,
              }
NUMBER_BOXES_PER_SHELVE = {"rack_1" : {4 : [6, 2]}, #num_shelves : [hor, ver ] 
                    "rack_2" : {3 : [3, 3], 0 : [4, 3]}, 
                    "rack_3" : {3 : [1, 2]},
                    "rack_4" : {0 : [1, 1], 1 : [2, 2], 2 : [1, 1], 3 : [2, 2], 4 : [2, 2], 5 : [1, 1]}}
                    
RACKS_SHELVE_POSITION = {
    1 : {1 : [177, 270], 2: [290, 420], 3: [445, 560]},
    2 : {1 : [], 2 : [], 3 : [], 4 : []},
    3 : {1 : [] , 2 : [], 3 : [], 4 : []},
    4 : {1 : [] , 2 : [], 3 : [], 4 : []}
}

def find_shelve(rack_id: int, y1: int, y2: int) -> int:
    """returns the shelve of a box for a rack, given the y coordinates of a box"""
    shelves_position = RACKS_SHELVE_POSITION[rack_id]
    return next(
        (
            key
            for key, value in shelves_position.items()
            if value[0] <= y1 <= value[1] and value[0] <= y2 <= value[1]
        ),
        None,
    )


def create_rack_scanner(start : Point, rack_id : int) -> Tuple[VerticalLine]:
    rack_position = None
    x = start.x
    return (VerticalLine(Point(x, shelf_y[0]), shelf_y[1]-shelf_y[0]) for shelf_y in rack_position) 
    
     
    


rack_box = {"label" : "rack_1", "box" : [222, 333, 444, 555]}
           
           
           
           
          
           
"""for annotation:
    draw all bounding bounding boxes for possible boxes as placeholders- 
    relative to rack (measured after line) -
    remove all placeholders with high nms
    
    need placehllders = List[Detection]
""" 
            
@dataclass
class Rack1Grid:
    


    rack_rect = Rect.from_xyxy()