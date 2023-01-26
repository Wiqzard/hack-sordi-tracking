from typing import Dict, List, Tuple
from dataclasses import dataclass
import cv2
import numpy as np

from draw.color import Color
from draw.utils import draw_custom_line
from geometry.geometry import Point, Rect, Vector, VerticalLine
from detection.detection_tools import Detections
from tracking.tracking_counter import RackDetection, RackTracker
from constants.bboxes import CONSTANTS


def find_shelve(class_id: int, y1: int, y2: int) -> int:
    """returns the shelve of a box for a rack, given the y coordinates of a box"""
    assert class_id in CONSTANTS.RACK_IDS, "class is not a rack"
    shelves_position = CONSTANTS.RACKS_SHELVE_POSITION[
        CONSTANTS.CLASS_NAMES_DICT[class_id]
    ]

    # returns shelve position if center of box is inside shelve yy
    center = y1 + (y2 - y1) / 2
    return next(
        (
            key
            for key, value in shelves_position.items()
            if value[0] <= center <= value[1]
        ),
        None,
    )


class RackScanner:
    def __init__(self, start: Point, height: int):
        """
        Initialize a RackScanner object.

        :param start: Point : The starting point of the line.
        :param end: Point : The ending point of the line.
        """
        # self.vector = Vector(start=start, end=end)
        self.scanner = VerticalLine(start, height)
        self.scanned_tracker_states: List[Dict[str, int]] = []

        self.tracker_state: Dict[str, bool] = {}
        self.in_count: int = 0
        self.out_count: int = 0

        self.curr_rack: str = None
        self.curr_rack_conf: float = 0

        self.rack_tracks: List[RackTracker] = []
        self.temp_storage = {}

    def set_curr_rack(self, class_id: int, confidence: float) -> None:
        self.curr_rack = class_id
        self.curr_rack_conf = confidence

    def update(self, detections: Detections):
        """
        Update the in_count and out_count for the detections that cross the line.

        :param detections: Detections : The detections for which to update the counts.
        """
        for xyxy, confidence, class_id, tracker_id in detections:

            # handle detections with no tracker_id
            if tracker_id is None:
                continue

            # we check if all four anchors of bbox are on the same side of vector.
            x1, y1, x2, y2 = xyxy
            anchors = [
                Point(x=x1, y=y1),
                Point(x=x1, y=y2),
                Point(x=x2, y=y1),
                Point(x=x2, y=y2),
            ]

            # number of anchors right to scanner
            triggers = sum(not self.scanner.left_to(anchor) for anchor in anchors)

            if class_id in CONSTANTS.RACK_IDS:
                print(triggers)

            # detection is partially in and partially out, sets current rack
            if triggers == 2:

                # ignore if detection is not a rack
                if class_id not in CONSTANTS.RACK_IDS:
                    continue

                # ignore if rack is already scanned
                if tracker_id in self.tracker_state:
                    continue

                if confidence < 0.7:
                    continue

                # ignore if rack is not the current rack or confidence is lower
                if self.curr_rack is not None:  # and confidence < self.curr_rack_conf:
                    continue

                # scan the rack and set it as current rack
                self.tracker_state[tracker_id] = True
                self.set_curr_rack(class_id, confidence)

                # create new rack tracker
                new_rack = RackTracker(
                    tracker_id=tracker_id, class_id=class_id, rack_conf=confidence
                )
                self.rack_tracks.append(new_rack)

            # unscans rack if it is completely left to scanner
            if triggers == 0 and class_id in CONSTANTS.RACK_IDS:
                if tracker_id not in self.tracker_state:
                    continue
                if not self.tracker_state[tracker_id]:
                    continue

                self.set_curr_rack(None, 0)
                self.tracker_state[tracker_id] = False
                continue

            # boxes are scanned if they are completely left to scanner


#            if triggers == 0 and tracker_id not in self.tracker_state:
#                if self.tracker_state[tracker_id]:
#                    continue
#
#                self.tracker_state[tracker_id] = True
#
#                # if box is scanned before rack, save it and add it as soon as the rack is detected
#                if not self.curr_rack:
#                    """we get a problem here if rack is not properly detected, dynamic programing"""
#                    self.temp_storage[class_id] = [y1, y2]
#                    continue
#
#                # empty the temporary storage
#                for saved_class_id, yy in self.temp_storage.items():
#                    if saved_shelve := find_shelve(self.curr_rack, *yy):
#                        self.rack_tracks[-1].update_shelves(
#                            saved_shelve, saved_class_id
#                        )
#                self.temp_storage = {}
#
#                if shelve := find_shelve(self.curr_rack, y1, y2):
#                    self.rack_tracks[-1].update_shelves(shelve, class_id)
#                else:
#                    print("shelve not found")


class ScannerCounterAnnotator:
    def __init__(
        self,
        thickness: float = 2,
        color: Color = Color.white(),
        text_thickness: float = 2,
        text_color: Color = Color.red(),
        text_scale: float = 0.5,
        text_offset: float = 1.5,
        text_padding: int = 10,
    ):
        """
        Initialize the LineCounterAnnotator object with default values.

        :param thickness: float : The thickness of the line that will be drawn.
        :param color: Color : The color of the line that will be drawn.
        :param text_thickness: float : The thickness of the text that will be drawn.
        :param text_color: Color : The color of the text that will be drawn.
        :param text_scale: float : The scale of the text that will be drawn.
        :param text_offset: float : The offset of the text that will be drawn.
        :param text_padding: int : The padding of the text that will be drawn.
        """
        self.thickness: float = thickness
        self.color: Color = color
        self.text_thickness: float = text_thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_offset: float = text_offset
        self.text_padding: int = text_padding

    def draw_scanner(
        self, scene: np.ndarray, class_id: int, start: Point, height: int
    ) -> None:
        """make sure rack before goin in"""
        assert class_id in CONSTANTS.RACK_IDS, "not a rack"
        shelves_position = CONSTANTS.RACKS_SHELVE_POSITION[
            CONSTANTS.CLASS_NAMES_DICT[class_id]
        ]
        for shelve_id, (y1, y2) in shelves_position.items():
            segment_start = Point(start.x, y1)
            draw_custom_line(
                scene=scene,
                shelve_id=shelve_id,
                start=segment_start,
                height=y2 - y1,
                color=Color.blue().as_bgr(),
                thickness=self.thickness,
            )

    def draw_counter(
        self, scene: np.ndarray, class_id: int, rack_scanner: RackScanner
    ) -> None:
        """Draws a counter displaying information about the current rack in the lower-left corner of the scene."""
        org = (100, 100)
        rack = CONSTANTS.CLASS_NAMES_DICT[rack_scanner.curr_rack]
        text_header = f"{rack}"
        cv2.putText(
            scene,
            text_header,
            org=org,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=Color.red().as_bgr(),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        shelves = CONSTANTS.RACKS_SHELVE_POSITION[rack]
        for idx, (shelve_id, _) in enumerate(shelves.items()):
            detection = rack_scanner.rack_tracks[-1].shelves[shelve_id]
            n_empty = detection["N_empty_KLT"]
            n_full = detection["N_full_KLT"]
            shelve_boxes = CONSTANTS.NUMBER_BOXES_PER_SHELVE[
                CONSTANTS.CLASS_NAMES_DICT[class_id]
            ][shelve_id]
            n_total = shelve_boxes[0] * shelve_boxes[1]
            n_placeholders = n_total - n_empty - n_full
            shelve_text = f"shelve_{shelve_id}: N_empty_KLT: {n_empty} | N_full_KLT: {n_full} | N_placeholder: {n_placeholders}"
            rel_org = (org[0], org[1] + (idx + 1) * 20)
            cv2.putText(
                scene,
                text=shelve_text,
                org=rel_org,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=self.text_color.as_bgr(),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        return scene

    def annotate(self, frame: np.ndarray, rack_scanner: RackScanner) -> np.ndarray:
        """
        Draws the line on the frame using the line_counter provided.

        :param frame: np.ndarray : The image on which the line will be drawn
        :param line_counter: LineCounter : The line counter that will be used to draw the line
        :return: np.ndarray : The image with the line drawn on it
        """
        # print(type(frame))
        if rack_scanner.curr_rack:
            frame = self.draw_scanner(
                scene=frame,
                class_id=rack_scanner.curr_rack,
                start=rack_scanner.scanner.start,
                height=rack_scanner.scanner.height,
            )
            frame = self.draw_counter(
                scene=frame, class_id=rack_scanner.curr_rack, rack_scanner=rack_scanner
            )
        else:
            frame = draw_custom_line(
                scene=frame,
                shelve_id=None,
                start=rack_scanner.scanner.start,
                height=rack_scanner.scanner.height,
                color=self.color.as_bgr(),
                thickness=self.thickness,
            )
        return frame
