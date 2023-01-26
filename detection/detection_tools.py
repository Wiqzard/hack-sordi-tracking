from __future__ import annotations
from typing import List, Optional, Union, Dict
import cv2
import numpy as np

from draw.color import Color, ColorPalette
from geometry.geometry import Rect, Point
from constants.bboxes import CONSTANTS


class Detections:
    def __init__(
        self,
        xyxy: np.ndarray,
        confidence: np.ndarray,
        class_id: np.ndarray,
        tracker_id: Optional[np.ndarray] = None,
    ):
        """
        Data class containing information about the detections in a video frame.

        :param xyxy: np.ndarray : An array of shape (n, 4) containing the bounding boxes coordinates in format [x1, y1, x2, y2]
        :param confidence: np.ndarray : An array of shape (n,) containing the confidence scores of the detections.
        :param class_id: np.ndarray : An array of shape (n,) containing the class ids of the detections.
        :param tracker_id: Optional[np.ndarray] : An array of shape (n,) containing the tracker ids of the detections.
        """
        self.xyxy: np.ndarray = xyxy
        self.confidence: np.ndarray = confidence
        self.class_id: np.ndarray = class_id
        self.tracker_id: Optional[np.ndarray] = tracker_id

        n = len(self.xyxy)
        validators = [
            (isinstance(self.xyxy, np.ndarray) and self.xyxy.shape == (n, 4)),
            (isinstance(self.confidence, np.ndarray) and self.confidence.shape == (n,)),
            (isinstance(self.class_id, np.ndarray) and self.class_id.shape == (n,)),
            self.tracker_id is None
            or (
                isinstance(self.tracker_id, np.ndarray)
                and self.tracker_id.shape == (n,)
            ),
        ]
        if not all(validators):
            raise ValueError(
                "xyxy must be 2d np.ndarray with (n, 4) shape, "
                "confidence must be 1d np.ndarray with (n,) shape, "
                "class_id must be 1d np.ndarray with (n,) shape, "
                "tracker_id must be None or 1d np.ndarray with (n,) shape"
            )

    def add_placeholders(self) -> None:
        """1. group all racks and boxes with centers inside the racks together
        2. get_placeholders for specific racks
        3. divide into patches if patch does not cotain center of box add placeholder
        4. for security remove placeholders with bigh nms
        """
        """ 1. same
            2. same
            3. remove placeholders with higher IoU 
        """

    def get_placeholder_for_rack(self, rack_detection: int) -> Detections:
        """Given the index of a rack detection, returns all placeholders for that specific rack"""

        assert self.class_id[rack_detection] > 2 and rack_detection < len(
            self
        ), "detection is not a rack"
        x1_rack, y1_rack, x2_rack, y2_rack = self.xyxy[rack_detection, :]
        # relative_boxes = np.array([RACK_1_RELATIVE[1:], RACK_2_RELATIVE[1:],
        #                           RACK_3_RELATIVE[1:], RACK_4_RELATIVE[1:]][self.class_id[rack_detection]-3])
        relative_boxes = np.array(
            CONSTANTS.RELATIVE_RACK_DICT[
                CONSTANTS.CLASS_NAMES_DICT[self.class_id[rack_detection]]
            ][1:]
        )
        # calculate center coordinates of rack
        x_center, y_center = (
            x1_rack + (x2_rack - x1_rack) / 2,
            y1_rack + (y2_rack - y1_rack) / 2,
        )

        # calculate center for all possible placeholders
        placeholder_center_x = x_center + relative_boxes[:, 1] * CONSTANTS.WIDTH
        placeholder_center_y = y_center + relative_boxes[:, 2] * CONSTANTS.HEIGHT
        placeholder_center = np.array([placeholder_center_x, placeholder_center_y]).T
        half_wh = 0.5 * relative_boxes[:, 3:]
        half_wh[:, 0] *= CONSTANTS.WIDTH
        half_wh[:, 1] *= CONSTANTS.HEIGHT
        placeholder_coordinates = np.concatenate(
            (placeholder_center - half_wh, placeholder_center + half_wh), axis=1
        )

        # placeholder_coordinates[:, [0, 2]] = np.clip(placeholder_coordinates[:, [0, 2]], 0, WIDTH)
        # placeholder_coordinates[:, [1, 3]] = np.clip(placeholder_coordinates[:, [1, 3]], 0, HEIGHT)
        # rows_to_delete = np.where(np.sum(placeholder_coordinates == 0, axis=1) >= 2)[0]
        # placeholder_coordinates = np.delete(placeholder_coordinates, rows_to_delete, axis=0)
        # placeholder_coordinates = placeholder_coordinates[np.sum((placeholder_coordinates == 0) | (placeholder_coordinates > [WIDTH, HEIGHT, WIDTH, HEIGHT]), axis=1) < 2]

        # remove placeholders with values outside (0, 1280) x (0, 720)
        placeholder_coordinates = np.minimum(
            np.maximum(placeholder_coordinates, [0, 0, 0, 0]),
            [CONSTANTS.WIDTH, CONSTANTS.HEIGHT, CONSTANTS.WIDTH, CONSTANTS.HEIGHT],
        )
        placeholder_coordinates = placeholder_coordinates[
            np.sum(placeholder_coordinates == 0, axis=1) < 2
        ]

        num_ph = placeholder_coordinates.shape[0]
        return Detections(
            xyxy=placeholder_coordinates,
            confidence=np.ones(num_ph),
            class_id=CONSTANTS.PLACEHOLDER_CLASS_ID * np.ones(num_ph, dtype=np.int8),
            tracker_id=None,
        )

    def group_racks(self) -> Dict[int, List[bool]]:
        """Returns a dictionary of rack detections together with detected boxes as masks with
        center inside the rack of the form {index of rack: [mask]} .
        """
        # get rack and box masks
        rack_id_mask = self.class_mask(CONSTANTS.RACK_IDS)
        box_mask = np.logical_not(rack_id_mask)

        rack_indices = np.flatnonzero(rack_id_mask)

        # transform rack bboxes to Rect
        rack_rects = [
            Rect.from_xyxy(rack_bbox).pad_y(padding=30).pad_x(padding=10)
            for rack_bbox in self.xyxy[rack_id_mask]
        ]

        # for every rack make mask of boxes with center inside Rect of rack
        rack_groups = {}
        for i, rack_index in enumerate(rack_indices):
            boxes_inside = [
                rack_rects[i].contains_point(Point.xyxy_center(xyxy))
                for xyxy in self.xyxy
            ]
            boxes_inside = np.where(rack_id_mask, False, boxes_inside)
            rack_groups[rack_index] = boxes_inside
        return rack_groups

    def class_mask(self, *args) -> np.ndarray:
        return np.isin(self.class_id, args)

    def __len__(self):
        """
        Returns the number of detections in the Detections object.
        """
        return len(self.xyxy)

    def __iter__(self):
        """
        Iterates over the Detections object and yield a tuple of (xyxy, confidence, class_id, tracker_id) for each detection.
        """
        for i in range(len(self.xyxy)):
            yield (
                self.xyxy[i],
                self.confidence[i],
                self.class_id[i],
                self.tracker_id[i] if self.tracker_id is not None else None,
            )

    @classmethod
    def from_yolov5(cls, yolov5_output: np.ndarray):
        """
        Creates a Detections instance from a YOLOv5 output tensor

        :param yolov5_output: np.ndarray : The output tensor from YOLOv5
        :return: Detections : A Detections instance representing the detections in the frame

        Example:
        detections = Detections.from_yolov5(yolov5_output)
        """
        xyxy = yolov5_output[:, :4]
        confidence = yolov5_output[:, 4]
        class_id = yolov5_output[:, 5].astype(int)
        return cls(xyxy, confidence, class_id)

    def filter(self, mask: np.ndarray, inplace: bool = False) -> Optional[np.ndarray]:
        """
        Filter the detections by applying a mask

        :param mask: np.ndarray : A mask of shape (n,) containing a boolean value for each detection indicating if it should be included in the filtered detections
        :param inplace: bool : If True, the original data will be modified and self will be returned.
        :return: Optional[np.ndarray] : A new instance of Detections with the filtered detections, if inplace is set to False. None otherwise.
        """
        if not inplace:
            return Detections(
                xyxy=self.xyxy[mask],
                confidence=self.confidence[mask],
                class_id=self.class_id[mask],
                tracker_id=self.tracker_id[mask]
                if self.tracker_id is not None
                else None,
            )
        self.xyxy = self.xyxy[mask]
        self.confidence = self.confidence[mask]
        self.class_id = self.class_id[mask]
        self.tracker_id = self.tracker_id[mask] if self.tracker_id is not None else None
        return self


class BoxAnnotator:
    def __init__(
        self,
        color: Union[Color, ColorPalette],
        thickness: int = 2,
        text_color: Color = Color.black(),
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
    ):
        """
        A class for drawing bounding boxes on an image using detections provided.

        :param color: Union[Color, ColorPalette] :  The color to draw the bounding box, can be a single color or a color palette
        :param thickness: int :  The thickness of the bounding box lines, default is 2
        :param text_color: Color :  The color of the text on the bounding box, default is white
        :param text_scale: float :  The scale of the text on the bounding box, default is 0.5
        :param text_thickness: int :  The thickness of the text on the bounding box, default is 1
        :param text_padding: int :  The padding around the text on the bounding box, default is 5
        """
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding

    def annotate(
        self,
        frame: np.ndarray,
        detections: Detections,
        labels: Optional[List[str]] = None,
    ) -> np.ndarray:  # sourcery skip: use-assigned-variable
        """
        Draws bounding boxes on the frame using the detections provided.

        :param frame: np.ndarray : The image on which the bounding boxes will be drawn
        :param detections: Detections : The detections for which the bounding boxes will be drawn
        :param labels: Optional[List[str]] :  An optional list of labels corresponding to each detection. If labels is provided, the confidence score of the detection will be replaced with the label.
        :return: np.ndarray : The image with the bounding boxes drawn on it
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, (xyxy, confidence, class_id, tracker_id) in enumerate(detections):
            color = (
                self.color.by_idx(class_id)
                if isinstance(self.color, ColorPalette)
                else self.color
            )
            text = (
                f"{confidence:0.2f}"
                if (labels is None or len(detections) != len(labels))
                else labels[i]
            )

            x1, y1, x2, y2 = xyxy.astype(int)

            text_width, text_height = cv2.getTextSize(
                text=text,
                fontFace=font,
                fontScale=self.text_scale,
                thickness=self.text_thickness,
            )[0]

            text_x = x1 + self.text_padding
            text_y = y1 - self.text_padding

            text_background_x1 = x1
            text_background_y1 = y1 - 2 * self.text_padding - text_height

            text_background_x2 = x1 + 2 * self.text_padding + text_width
            text_background_y2 = y1

            cv2.rectangle(
                img=frame,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color.as_bgr(),
                thickness=self.thickness,
            )
            # cv2.rectangle(
            #    img=frame,
            #    pt1=(text_background_x1, text_background_y1),
            #    pt2=(text_background_x2, text_background_y2),
            #    color=color.as_bgr(),
            #    thickness=cv2.FILLED,
            # )
            cv2.putText(
                img=frame,
                text=text,
                org=(text_x, text_y),
                fontFace=font,
                fontScale=self.text_scale,
                color=self.text_color.as_rgb(),
                thickness=self.text_thickness,
                lineType=cv2.LINE_AA,
            )
        return frame

    def annotate_placeholder(self) -> np.ndarray:
        pass
