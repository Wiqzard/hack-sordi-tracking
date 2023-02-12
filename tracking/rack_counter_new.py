from typing import Dict, List, Deque, NamedTuple, Tuple
import cv2
import numpy as np
from collections import deque, namedtuple

from draw.color import Color
from draw.utils import draw_custom_line
from geometry.geometry import Point, VerticalLine
from detection.detection_tools import Detections
from tracking.tracking_counter import RackTracker
from constants.bboxes import CONSTANTS


def find_shelf(class_id: int, y1: int, y2: int) -> int:
    """Returns the shelf of a box for a rack, given the y coordinates of a box"""

    if not class_id:
        return None
    assert class_id in CONSTANTS.RACK_IDS, "class is not a rack"
    shelves_position = CONSTANTS.RACKS_SHELF_POSITION[
        CONSTANTS.CLASS_NAMES_DICT[class_id]
    ]

    # returns shelf position if center of box is inside shelf yy
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
    def __init__(self, start: Point, height: int) -> None:
        """
        Initialize a RackScanner object.

        :param start: Point : The starting point of the line.
        :param end: Point : The ending point of the line.
        """
        self.__scanner = VerticalLine(start, height)
        self.__scanned_tracks: Dict[str, bool] = {}

        self.__curr_rack: str = None
        self.__curr_rack_conf: float = 0
        self._rack_counter: int = 0

        self.__rack_tracks: Deque[RackTracker] = deque()
        self.__temp_storage: dict = {}

    @property
    def scanner(self) -> VerticalLine:
        return self.__scanner

    @property
    def curr_rack(self) -> str:
        return self.__curr_rack

    @property
    def curr_rack_conf(self) -> float:
        return self.__curr_rack_conf

    @property
    def scanned_tracks(self) -> Dict[str, bool]:
        return self.__scanned_tracks

    @property
    def rack_tracks(self) -> Deque[RackTracker]:
        return self.__rack_tracks

    def _set_curr_rack(
        self, class_id: int, confidence: float, xx: List[float], tracker_id: int
    ) -> None:
        """
        Set the current rack and its confidence together with the x coordinates of the rack"""
        self.__curr_rack = class_id
        self.__curr_rack_conf = confidence
        self.__xx = xx
        self.__curr_rack_tracker_id = tracker_id

    def _process_rack_in_scanner(
        self,
        confidence: float,
        tracker_id: str,
        class_id: int,
        xx: List[float],
    ) -> None:
        """Checks if rack already scanned, if rack confidence is high enough
        and if rack confidence is higher then current rack confidence.
        Scans rack and sets it as current rack. Creates new rack tracker."""
        # ignore if rack is already scanned
        if self.scanned_tracks[tracker_id]:
            return True
        if confidence < 0.85:
            return True

        # ignore if we have a current rack and the confidence of the new rack is lower
        if self.curr_rack is not None and confidence < self.curr_rack_conf:
            return True

        # scan the rack and set it as current rack
        self.__scanned_tracks[tracker_id] = True
        self._set_curr_rack(class_id, confidence, xx, tracker_id)
        self._rack_counter += 1

        # create new rack tracker
        new_rack = RackTracker(
            tracker_id=tracker_id, class_id=class_id, rack_conf=confidence
        )
        self.__rack_tracks.append(new_rack)

    def _process_rack_after_scanner(self, tracker_id: int) -> None:
        """Checks if rack is scanned. Sets rack as not scannerd and resets current rack"""
        if not self.scanned_tracks[tracker_id]:
            return True
        if self.__curr_rack_tracker_id != tracker_id:
            return True
        # self.__scanned_tracks[tracker_id] = False
        self._set_curr_rack(class_id=None, confidence=0, xx=None, tracker_id=None)
        return True

    def _empty_storage(self, tracker_id: int) -> None:
        """Adds all boxes in temp storage to current rack and resets temp storage"""
        for saved_class_id, yy in self.__temp_storage.items():
            if saved_shelve := find_shelf(self.curr_rack, *yy):
                self.__rack_tracks[-1].update_shelves(saved_shelve, saved_class_id)
                self.__scanned_tracks[tracker_id] = True
                return True
        self.__temp_storage = {}

    def _not_in_range(self, x1: int, x2: int) -> bool:
        """Checks if center of detection is outside of current rack range
        or if center of detection is right to scanner or close to image border"""
        center = Point(x=x1 + (x2 - x1) / 2, y=0)
        return (
            center.x < self.__xx[0]
            or center.x > self.__xx[1]
            or center.x < 50
            # or self.scanner.left_to(center)
        )

    def update(self, detections: Detections) -> None:
        """
        Update the in_count and out_count for the detections that cross the line.

        :param detections: Detections : The detections for which to update the counts.
        """
        temp_rack_counter = False
        for xyxy, confidence, class_id, tracker_id in detections:

            # handle detections with no tracker_id
            if tracker_id is None:
                continue

            # register object for scanning
            if tracker_id not in self.scanned_tracks:
                self.__scanned_tracks[tracker_id] = False

            # we check if all four anchors of bbox are on the same side of scanner.
            x1, y1, x2, y2 = xyxy
            anchors = [
                Point(x=x1, y=y1),
                Point(x=x1, y=y2),
                Point(x=x2, y=y1),
                Point(x=x2, y=y2),
            ]
            # number of anchors right to scanner.
            triggers = sum(self.scanner.left_to(anchor) for anchor in anchors)
            if triggers == 4:
                continue

            # detected rack is partially in and partially out, sets current rack
            if triggers == 2 and class_id in CONSTANTS.RACK_IDS:
                self._process_rack_in_scanner(
                    tracker_id=tracker_id,
                    class_id=class_id,
                    confidence=confidence,
                    xx=[x1, x2],
                )
                continue

            # unscans rack if it is completely left to scanner
            if (
                triggers == 0
                and class_id in CONSTANTS.RACK_IDS
                and self._process_rack_after_scanner(tracker_id=tracker_id)
            ):
                # self._set_curr_rack(None, 0, None)
                continue

            if triggers == 2:

                # ignore box that is already scanned
                if self.scanned_tracks[tracker_id]:
                    continue

                # if center! of a box is outside of cur_rack, or right to scanner, ignore it
                if self.curr_rack and self._not_in_range(x1, x2):
                    continue

                # if box is scanned before rack, save it and add it as soon as the rack is detected
                # if not self.curr_rack:
                #    self.__temp_storage[class_id] = [y1, y2]
                #    self.__scanned_tracks[tracker_id] = True
                #    continue

                # empty the temporary storage
                # if self._empty_storage(tracker_id=tracker_id):
                #    continue

                # if box is scanned after rack, add it to the rack
                if shelf := find_shelf(self.curr_rack, y1, y2):
                    if self.curr_rack == self.rack_tracks[-1].class_id:
                        self.__rack_tracks[-1].update_shelves(shelf, class_id)
                        self.__scanned_tracks[tracker_id] = True
                else:
                    self.__scanned_tracks.pop(tracker_id)

        # if not temp_rack_counter and self._rack_counter > 0:
        #    self._set_curr_rack(None, 0, None)


class ScannerCounterAnnotator:
    def __init__(
        self,
        thickness: float = 2,
        color: Color = Color.white(),
        text_thickness: float = 2,
        text_color: Color = Color.red(),
        text_scale: float = 0.6,
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
        assert class_id in CONSTANTS.RACK_IDS, "not a rack"
        shelves_position = CONSTANTS.RACKS_SHELF_POSITION[
            CONSTANTS.CLASS_NAMES_DICT[class_id]
        ]
        for shelf_id, (y1, y2) in shelves_position.items():
            segment_start = Point(start.x, y1)
            draw_custom_line(
                scene=scene,
                shelf_id=shelf_id,
                start=segment_start,
                height=y2 - y1,
                color=Color.blue().as_bgr(),
                thickness=self.thickness,
            )

    def draw_counter(
        self, scene: np.ndarray, class_id: int, rack_tracker: RackTracker
    ) -> None:
        """Draws a counter displaying information about the current rack in the lower-left corner of the scene."""
        org = (690, 560)
        rack = CONSTANTS.CLASS_NAMES_DICT[class_id]
        text_header = f"-----------> scanning {rack} <-----------"

        cv2.rectangle(
            scene,
            (org[0] - 15, org[1] - 20),
            (org[0] + 800, org[1] + 160),
            Color.blue().as_bgr(),
            -1,
        )
        cv2.putText(
            scene,
            text_header,
            org=org,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=self.text_scale,
            color=Color.red().as_bgr(),
            thickness=self.text_thickness,
            lineType=cv2.LINE_AA,
        )

        shelves = CONSTANTS.RACKS_SHELF_POSITION[rack].items()
        for idx, (shelf_id, _) in enumerate(shelves):
            n_empty = rack_tracker.boxes_in_shelf(shelf_id, "empty")
            n_full = rack_tracker.boxes_in_shelf(shelf_id, "full")
            n_placeholders = rack_tracker.boxes_in_shelf(shelf_id, "placeholder")
            shelf_text = f"shelf_{shelf_id}: N_empty_KLT: {n_empty} | N_full_KLT: {n_full} | N_placeholder: {n_placeholders}"

            rel_org = (org[0], org[1] + (idx + 1) * 30)
            cv2.putText(
                scene,
                text=shelf_text,
                org=rel_org,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=self.text_scale,
                color=self.text_color.as_bgr(),
                thickness=self.text_thickness,
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
        if rack_scanner.curr_rack:

            self.draw_scanner(
                scene=frame,
                class_id=rack_scanner.curr_rack,
                start=rack_scanner.scanner.start,
                height=rack_scanner.scanner.height,
            )
            self.draw_counter(
                scene=frame,
                class_id=rack_scanner.curr_rack,
                rack_tracker=rack_scanner.rack_tracks[-1],
            )
        else:
            frame = draw_custom_line(
                scene=frame,
                shelf_id=None,
                start=rack_scanner.scanner.start,
                height=rack_scanner.scanner.height,
                color=self.color.as_bgr(),
                thickness=self.thickness,
            )
        return frame
