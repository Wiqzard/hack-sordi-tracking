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
    """returns the shelf of a box for a rack, given the y coordinates of a box"""
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
        # self.vector = Vector(start=start, end=end)
        self.__scanner = VerticalLine(start, height)
        self.__scanned_tracks: Dict[str, bool] = {}

        self.__curr_rack: str = None
        self.__curr_rack_conf: float = 0

        self.__rack_tracks: Deque[RackTracker] = deque()
        self.__temp_storage = {}

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

    def set_curr_rack(self, class_id: int, confidence: float, xx: List[float]) -> None:
        """
        Set the current rack and its confidence together with the x coordinates of the rack"""
        self.__curr_rack = class_id
        self.__curr_rack_conf = confidence
        self.__xx = xx

    def _process_rack_in_scanner(
        self,
        triggers: int,
        confidence: float,
        tracker_id: str,
        class_id: int,
        xx: List[float],
    ) -> None:
        self.xx = xx
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
        self.__set_curr_rack(class_id, confidence, xx)

        # create new rack tracker
        new_rack = RackTracker(
            tracker_id=tracker_id, class_id=class_id, rack_conf=confidence
        )
        self.rack_tracks.append(new_rack)

    def _process_rack_after_scanner(self, tracker_id: str) -> None:
        if not self.__scanned_tracks[tracker_id]:
            return True

        self.__scanned_tracks[tracker_id] = False
        self.set_curr_rack(None, 0, None)
        return True

    def _process_box_after_scanner(
        self, class_id: int, tracker_id: str, y1: int, y2: int
    ) -> None:
        # ignore box that is already scanned
        if self.__scanned_tracks[tracker_id]:
            return True

        # if box is scanned before rack, save it and add it as soon as the rack is detected
        if not self.__curr_rack:
            """we get a problem here if rack is not properly detected, dynamic programing"""
            self.__temp_storage[class_id] = [y1, y2]
            self.__scanned_tracks[tracker_id] = True
            return True

        # if center of a box is outside of cur_rack, ignore it
        center = x1 + (x2 - x1) / 2
        if center < self.__xx[0] or center > self.__xx[1] or center < 50:
            return True

        # empty the temporary storage
        for saved_class_id, yy in self.__temp_storage.items():
            if saved_shelve := find_shelf(self.__curr_rack, *yy):
                self.__rack_tracks[-1].update_shelves(saved_shelve, saved_class_id)
                self.__scanned_tracks[tracker_id] = True
                return True
        self.temp_storage = {}

        if shelf := find_shelf(self.__curr_rack, y1, y2):
            if self.__curr_rack == self.__rack_tracks[-1].class_id:
                self.__rack_tracks[-1].update_shelves(shelf, class_id)
                self.__scanned_tracks[tracker_id] = True
        else:
            self.scanned_tracks.pop(tracker_id)

    def update(self, detections: Detections) -> None:
        # sourcery skip: merge-nested-ifs
        """
        :param detections: Detections : The detections for which to update the counts.
        """
        for xyxy, confidence, class_id, tracker_id in detections:

            # handle detections with no tracker_id
            if tracker_id is None:
                continue

            # register object for scanning
            if tracker_id not in self.scanned_tracks:
                self.scanned_tracks[tracker_id] = False

            # we check if all four anchors of bbox are on the same side of scanner.
            x1, y1, x2, y2 = xyxy
            anchors = [
                Point(x=x1, y=y1),
                Point(x=x1, y=y2),
                Point(x=x2, y=y1),
                Point(x=x2, y=y2),
            ]
            # number of anchors right to scanner. how many anchors are right to scanner
            triggers = sum(self.scanner.left_to(anchor) for anchor in anchors)

            # detection rack is partially in and partially out, sets current rack
            if triggers == 2 and class_id in CONSTANTS.RACK_IDS:
                if self._process_rack_in_scanner(
                    triggers, confidence, tracker_id, class_id, [x1, x2]
                ):
                    continue

            # unscans rack if it is completely left to scanner
            if triggers == 0 and class_id in CONSTANTS.RACK_IDS:
                if self._process_rack_after_scanner(tracker_id):
                    continue

            # boxes are scanned if they are completely left to scanner
            if triggers == 0:
                if self._process_box_after_scanner(class_id, y1, y2):
                    continue


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
    """returns the shelf of a box for a rack, given the y coordinates of a box"""
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
        # self.vector = Vector(start=start, end=end)
        self.scanner = VerticalLine(start, height)
        self.scanned_tracks: Dict[str, bool] = {}

        self.curr_rack: str = None
        self.curr_rack_conf: float = 0

        self.rack_tracks: Deque[RackTracker] = deque()
        # self.rack_tracks: List[RackTracker] = []
        # self.temp_storage : NamedTuple = namedtuple("yy", [y1, ])
        self.temp_storage = {}

    def set_curr_rack(self, class_id: int, confidence: float, xx: List[float]) -> None:
        self.curr_rack = class_id
        self.curr_rack_conf = confidence
        self.xx = xx

    def process_rack_in_scanner(
        self,
        triggers: int,
        confidence: float,
        tracker_id: str,
        class_id: int,
        xx: List[float],
    ) -> None:
        # if .... :
        # return True or False. If false in main loop if !proces.. continue
        pass

    def process_rack_after_scanner(self, class_id: int, y1: int, y2: int) -> None:
        pass

    def process_box_after_scanner(self, class_id: int, y1: int, y2: int) -> None:
        pass

    def update(self, detections: Detections) -> None:
        """
        Update the in_count and out_count for the detections that cross the line.

        :param detections: Detections : The detections for which to update the counts.
        """
        for xyxy, confidence, class_id, tracker_id in detections:

            # handle detections with no tracker_id
            if tracker_id is None:
                continue

            # register object for scanning
            if tracker_id not in self.scanned_tracks:
                self.scanned_tracks[tracker_id] = False

            # we check if all four anchors of bbox are on the same side of scanner.
            x1, y1, x2, y2 = xyxy
            anchors = [
                Point(x=x1, y=y1),
                Point(x=x1, y=y2),
                Point(x=x2, y=y1),
                Point(x=x2, y=y2),
            ]
            # number of anchors right to scanner. how many anchors are right to scanner
            triggers = sum(self.scanner.left_to(anchor) for anchor in anchors)

            # detection rack is partially in and partially out, sets current rack
            if triggers == 2 and class_id in CONSTANTS.RACK_IDS:
                # self.process_rack_in_scanner(triggers, confidence, tracker_id, class_id, [x1, x2])
                self.xx = [x1, x2]
                # ignore if rack is already scanned
                if self.scanned_tracks[tracker_id]:
                    continue

                if confidence < 0.85:
                    continue

                # ignore if we have a current rack and the confidence of the new rack is lower
                if self.curr_rack is not None and confidence < self.curr_rack_conf:
                    continue

                # scan the rack and set it as current rack
                self.scanned_tracks[tracker_id] = True
                self.set_curr_rack(class_id, confidence, [x1, x2])

                # create new rack tracker
                new_rack = RackTracker(
                    tracker_id=tracker_id, class_id=class_id, rack_conf=confidence
                )
                self.rack_tracks.append(new_rack)

            # unscans rack if it is completely left to scanner
            if triggers == 0 and class_id in CONSTANTS.RACK_IDS:
                if not self.scanned_tracks[tracker_id]:
                    continue

                self.scanned_tracks[tracker_id] = False
                self.set_curr_rack(None, 0, None)
                continue

            # boxes are scanned if they are completely left to scanner
            if triggers == 0:
                # ignore box that is already scanned
                if self.scanned_tracks[tracker_id]:
                    continue

                # if box is scanned before rack, save it and add it as soon as the rack is detected
                if not self.curr_rack:
                    """we get a problem here if rack is not properly detected, dynamic programing"""
                    self.temp_storage[class_id] = [y1, y2]
                    self.scanned_tracks[tracker_id] = True
                    continue

                # if center of a box is outside of cur_rack, ignore it
                center = x1 + (x2 - x1) / 2
                if center < self.xx[0] or center > self.xx[1] or center < 50:
                    continue

                # empty the temporary storage
                for saved_class_id, yy in self.temp_storage.items():
                    if saved_shelve := find_shelf(self.curr_rack, *yy):
                        self.rack_tracks[-1].update_shelves(
                            saved_shelve, saved_class_id
                        )
                        self.scanned_tracks[tracker_id] = True
                        continue
                self.temp_storage = {}

                if shelf := find_shelf(self.curr_rack, y1, y2):
                    if self.curr_rack == self.rack_tracks[-1].class_id:
                        self.rack_tracks[-1].update_shelves(shelf, class_id)
                        self.scanned_tracks[tracker_id] = True
                else:
                    self.scanned_tracks.pop(tracker_id)


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












!pip install Cython
import os
import numpy
import sys

#%cd bbox
#!python3 setup.py build_ext --inplace
#import pyximport
#pyximport.install() 
#pyximport.install(setup_args={"script_args" : ["--verbose"]})
#sys.path.append(f"{HOME}/bbox")
#from bbox import bbox_overlaps
from IPython import display
#display.clear_output()
#%cd ../
#os.getcwd()
%cd {HOME}
!git clone https://github.com/ifzhang/ByteTrack.git
!cd ByteTrack && pip3 install -q -r requirements.txt
!cd ByteTrack && python3 setup.py -q develop
!pip install -q cython_bbox
!pip install -q onemetric

display.clear_output()
import sys
sys.path.append(f"{HOME}/ByteTrack")
import yolox
print("yolox.__version__:", yolox.__version__)
print(numpy.version.version)
sys.path.append(f"{HOME}/ByteTrack")
sys.path.append("/home/5qx9nf8a/.local/bin")
display.clear_output()
!git clone https://github.com/Wiqzard/hack-sordi-tracking.git
#!cd tracking-tools && git pull
sys.path.append(f"{HOME}/tracking-tools")









!git clone https://github.com/PaddlePaddle/PaddleYOLO  # clone
!cd PaddleYOLO
!pip install -r requirements.txt


# 1.训练（单卡/多卡），加 --eval 表示边训边评估，加 --amp 表示混合精度训练
!cd PaddleYOLO && CUDA_VISIBLE_DEVICES=0 python tools/train.py -c PaddleDetection/PaddleYOLO/configs/custom/yolov8_m_500e_coco.yml --eval --amp



#!CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/bytetrack/detector/ppyoloe_crn_l_36e_640x640_mot17half.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/ppyoloe_crn_l_36e_640x640_mot17half.pdparams
%cd PaddleDetection
!CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c /home/5qx9nf8a/team_workspace/PaddleDetection/configs/mot/bytetrack/detector/ppyoloe_plus_l_bytetrack.yml -o weights=/home/5qx9nf8a/team_workspace/PaddleDetection//tracking/model_final.pdparams
    






















