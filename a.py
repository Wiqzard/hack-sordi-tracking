import cv2
import numpy as np

from geometry.geometry import Point
from constants.bboxes import CONSTANTS
# Create an image with all pixels set to 0 (black)
img = np.zeros((720, 1280, 3), np.uint8)

# Set the coordinates for the vertical line
x = 300 
y = 50
height = 500

start = Point(x, y)
end = Point(x, y + height)

color1 = (255, 0, 0) # Blue
color2 = (0, 255, 0) # Green
color3 = (0, 0, 255) # Red


def draw_cusotm_line(start: Point, height: int, shelve: str) -> None:
    """draws a line with 2 dots closing it"""
    start = start.as_xy_int_tuple() 
    end = (start[0], start[1] + height)
    cv2.line(
                img,
                start,
                end,
                color=color1,
                thickness=2,
                lineType=cv2.LINE_AA,
                shift=0,
            )

    cv2.circle(
                img,
                start,
                radius=5,
                color=color3,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
    
    cv2.circle(
                img,
                end,
                radius=5,
                color=color3,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
#draw_cusotm_line(start, height)

def draw_scanner(class_id: int, start: Point, height: int) -> None: 
    """make sure rack before goin in"""    
    assert class_id in CONSTANTS.RACK_IDS, 'not a rack' 
    shelves_position = CONSTANTS.RACKS_SHELVE_POSITION[CONSTANTS.CLASS_NAMES_DICT[class_id]]
    for shelve_id, (y1, y2) in shelves_position.items():
        segment_start = Point(start.x, y1)
        draw_cusotm_line(start=segment_start, height=y2-y1, shelve=f"shelve {shelve_id}")
   
draw_scanner(4, start, height)
# Set the coordinates for the segments and their colors

# Draw the line
#cv2.line(img, (x, y), (x, y + height), (255, 255, 255), 2)

cv2.imshow("Image", img)
cv2.waitKey(0)