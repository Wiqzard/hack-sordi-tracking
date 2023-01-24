from __future__ import annotations

import cv2
import numpy as np
from geometry.geometry import Point, Rect
from draw.color import Color


def draw_line(
    scene: np.ndarray, start: Point, end: Point, color: Color, thickness: int = 2
) -> np.ndarray:
    """
    Draws a line on a given scene.

    :param scene: np.ndarray : The scene on which the line will be drawn
    :param start: Point : The starting point of the line
    :param end: Point : The end point of the line
    :param color: Color : The color of the line
    :param thickness: int : The thickness of the line
    :return: np.ndarray : The scene with the line drawn on it
    """
    cv2.line(
        scene,
        start.as_xy_int_tuple(),
        end.as_xy_int_tuple(),
        color.as_bgr(),
        thickness=thickness,
    )
    return scene


def draw_rectangle(
    scene: np.ndarray, rect: Rect, color: Color, thickness: int = 2
) -> np.ndarray:
    """
    Draws a rectangle on an image.

    :param scene: np.ndarray : The image on which to draw the rectangle.
    :param rect: Rect : The rectangle to draw.
    :param color: Color : The color of the rectangle.
    :param thickness: int : The thickness of the rectangle border.
    :return: np.ndarray : The image with the rectangle drawn on it.
    """
    cv2.rectangle(
        scene,
        rect.top_left.as_xy_int_tuple(),
        rect.bottom_right.as_xy_int_tuple(),
        color.as_bgr(),
        thickness=thickness,
    )
    return scene


def draw_filled_rectangle(scene: np.ndarray, rect: Rect, color: Color) -> np.ndarray:
    """
    Draws a filled rectangle on the given scene.

    :param scene: np.ndarray : The scene on which to draw the rectangle.
    :param rect: Rect : The rectangle to be drawn.
    :param color: Color : The color of the rectangle.
    :return: np.ndarray : The updated scene with the filled rectangle drawn on it.
    """
    cv2.rectangle(
        scene,
        rect.top_left.as_xy_int_tuple(),
        rect.bottom_right.as_xy_int_tuple(),
        color.as_bgr(),
        -1,
    )
    return scene

def draw_custom_line(scene: np.ndarray, shelve_id: int,start: Point, height: int, color: Color,  thickness: int = 2
) -> np.ndarray:
    """draws a line with 2 dots closing it"""
    start = start.as_xy_int_tuple() 
    end = (start[0], start[1] + height)
    cv2.line(
                scene,
                start,
                end,
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
                shift=0,
            )

    cv2.circle(
                scene,
                start,
                radius=5,
                color=Color.red().as_bgr(),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
    
    cv2.circle(
                scene,
                end,
                radius=5,
                color=Color.red().as_bgr(),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
    
    text_pos = (start[0] - 100, start[1] -15)
    text = f"shelve_{shelve_id}"
    cv2.putText(scene, text, org=text_pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=Color.red().as_bgr(), thickness=1, lineType=cv2.LINE_AA)

    return scene

#draw_cusotm_line(start, height)

        
