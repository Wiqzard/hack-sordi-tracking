from __future__ import annotations
from typing import List, Optional, Union, Dict
import cv2
import numpy as np

from draw.color import Color, ColorPalette
from geometry.geometry import Rect, Point
from constants.bboxes import CONSTANTS

from onemetric.cv.utils.iou import box_iou_batch

PLACEHOLDER_TRACKER_ID = -1


def remove_placeholders_iou(
    detections: Detections, placeholders: Detections
) -> Detections:
    """
    Removes the placeholders from the detections based on the IOU between the detections and the placeholders.

    :param detections: Detections : The detections to remove the placeholders from.
    :param placeholders: Detections : The placeholders to remove from the detections.
    :return: Detections : The detections with the placeholders removed.
    """
    print(detections.xyxy.shape)
    print(placeholders.xyxy.shape)
    detections = detections.filter(detections.xyxy[:, 0] < 300)
    iou = box_iou_batch(detections.xyxy, placeholders.xyxy)
    # print(iou)
    iou_mask = np.max(iou, axis=0) < 0.25
    print(iou_mask.shape)
    return placeholders.filter(iou_mask)


class Detections:
    def __init__(
        self,
        xyxy: np.ndarray = np.empty((0, 4)),
        confidence: np.ndarray = np.empty((0,)),
        class_id: np.ndarray = np.empty((0,)),
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

    def __len__(self) -> int:
        return len(self.xyxy)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"xyxy={self.xyxy}, "
            f"confidence={self.confidence}, "
            f"class_id={self.class_id}, "
            f"tracker_id={self.tracker_id})"
        )

    def __getitem__(self, item: int) -> Detections:
        return Detections(
            self.xyxy[item],
            self.confidence[item],
            self.class_id[item],
            self.tracker_id[item] if self.tracker_id is not None else None,
        )

    @classmethod
    def merge(cls, det1, det2):
        """Merge two Detections objects into one by concatenating their attributes."""
        xyxy = np.concatenate((det1.xyxy, det2.xyxy), axis=0)
        confidence = np.concatenate((det1.confidence, det2.confidence), axis=0)
        class_id = np.concatenate((det1.class_id, det2.class_id), axis=0)
        if det1.tracker_id is not None and det2.tracker_id is not None:
            tracker_id = np.concatenate((det1.tracker_id, det2.tracker_id), axis=0)
        elif det1.tracker_id is not None:
            tracker_id = det1.tracker_id
        else:
            tracker_id = det2.tracker_id
        return cls(xyxy, confidence, class_id, tracker_id)

    @staticmethod
    def get_placeholders_for_racks(
        rack_detections: Detections, scanner_x: float = 300
    ) -> Detections:
        #        placeholders_for_racks: List[Detections] = []
        placeholders_for_racks = Detections()

        for idx in range(len(rack_detections)):
            assert rack_detections.class_id[idx] >= 2, "detection is not a rack"

            x1_rack, y1_rack, x2_rack, y2_rack = rack_detections.xyxy[idx, :]

            # check if rack is not too small
            if x2_rack - x1_rack < 100:
                continue

            relative_boxes = np.array(
                CONSTANTS.RELATIVE_RACK_DICT[
                    CONSTANTS.CLASS_NAMES_DICT[rack_detections.class_id[idx]]
                ][1:][1:]
            )

            # calculate center coordinates of rack
            x_center, y_center = (
                x1_rack + (x2_rack - x1_rack) / 2,
                y1_rack + (y2_rack - y1_rack) / 2,
            )
            # calculate center for all possible placeholders
            placeholder_center_x = x_center + relative_boxes[:, 1] * CONSTANTS.WIDTH
            placeholder_center_y = y_center + relative_boxes[:, 2] * CONSTANTS.HEIGHT
            placeholder_center = np.array(
                [placeholder_center_x, placeholder_center_y]
            ).T

            # transform from xywh relative to xyxy absolute
            half_wh = 0.5 * relative_boxes[:, 3:]
            half_wh[:, 0] *= CONSTANTS.WIDTH
            half_wh[:, 1] *= CONSTANTS.HEIGHT
            placeholder_coordinates = np.concatenate(
                (placeholder_center - half_wh, placeholder_center + half_wh), axis=1
            )
            # remove placeholders that are outside of the rack, scanner and not in image
            mask = np.logical_and(
                np.logical_and(
                    placeholder_coordinates[:, 0] <= scanner_x,
                    placeholder_coordinates[:, 0] > 0,
                ),
                np.logical_and(
                    placeholder_coordinates[:, 0] > x1_rack,
                    placeholder_coordinates[:, 2] < x2_rack,
                ),
            )
            placeholder_coordinates = placeholder_coordinates[mask]

            # if there are no placeholders left, continue
            num_ph = placeholder_coordinates.shape[0]
            if num_ph == 0:
                continue

            # add placeholders to placeholder detections
            placeholders_for_racks = Detections.merge(
                placeholders_for_racks,
                Detections(
                    xyxy=placeholder_coordinates,
                    confidence=np.ones(num_ph),
                    class_id=int(CONSTANTS.PLACEHOLDER_CLASS_ID)
                    * np.ones(num_ph, dtype=np.int8),
                    tracker_id=PLACEHOLDER_TRACKER_ID * np.ones(num_ph, dtype=np.int8),
                ),
            )
        return (
            None
            if placeholders_for_racks.tracker_id is None
            else placeholders_for_racks[1:]
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
            if isinstance(class_id, (np.float32, np.float64)):
                class_id = int(class_id)
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
            cv2.rectangle(
                img=frame,
                pt1=(text_background_x1, text_background_y1),
                pt2=(text_background_x2, text_background_y2),
                color=color.as_bgr(),
                thickness=cv2.FILLED,
            )
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
