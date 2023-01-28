from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
import cv2
import numpy as np

from geometry.geometry import Rect, Color, Point
from detection import Detection


def draw_rect(
    image: np.ndarray, rect: Rect, color: Color, thickness: int = 2
) -> np.ndarray:
    cv2.rectangle(
        image,
        rect.top_left.int_xy_tuple,
        rect.bottom_right.int_xy_tuple,
        color.bgr_tuple,
        thickness,
    )
    return image


def draw_filled_rect(image: np.ndarray, rect: Rect, color: Color) -> np.ndarray:
    cv2.rectangle(
        image,
        rect.top_left.int_xy_tuple,
        rect.bottom_right.int_xy_tuple,
        color.bgr_tuple,
        -1,
    )
    return image


def draw_text(
    image: np.ndarray, anchor: Point, text: str, color: Color, thickness: int = 2
) -> np.ndarray:
    cv2.putText(
        image,
        text,
        anchor.int_xy_tuple,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color.bgr_tuple,
        thickness,
        2,
        False,
    )
    return image


@dataclass
class BaseAnnotator:
    colors: List[Color]
    thickness: int

    def annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated_image = image.copy()
        for detection in detections:
            annotated_image = draw_rect(
                image=image,
                rect=detection.rect,
                color=self.colors[detection.class_id],
                thickness=self.thickness,
            )
        return annotated_image


def calculate_placeholder():
    pass


def draw_placeholder():
    pass


@dataclass
class PlaceholderAnnotator:
    color: Color

    def annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated_image = image.copy()
        for _ in detections:
            annotated_image = draw_placeholder(image=image, color=self.color)
        return annotated_image


# text annotator to display tracker_id
@dataclass
class TextAnnotator:
    background_color: Color
    text_color: Color
    text_thickness: int

    @classmethod
    def annotate_text(self, image: np.ndarray, text: str) -> np.ndarray:
        annotated_image = image.copy()
        size, _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            thickness=self.text_thickness,
        )

    def annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated_image = image.copy()
        for detection in detections:
            # if tracker_id is not assigned skip annotation
            if detection.tracker_id is None:
                continue

            # calculate text dimensions
            size, _ = cv2.getTextSize(
                str(detection.tracker_id),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                thickness=self.text_thickness,
            )
            width, height = size

            # calculate text background position
            center_x, center_y = detection.rect.bottom_center.int_xy_tuple
            x = center_x - width // 2
            y = center_y - height // 2 + 10

            # draw background
            annotated_image = draw_filled_rect(
                image=annotated_image,
                rect=Rect(x=x, y=y, width=width, height=height).pad(padding=5),
                color=self.background_color,
            )

            # draw text
            annotated_image = draw_text(
                image=annotated_image,
                anchor=Point(x=x, y=y + height),
                text=str(detection.tracker_id),
                color=self.text_color,
                thickness=self.text_thickness,
            )
        return annotated_image
