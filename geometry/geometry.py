from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class Point:
    x: float
    y: float

    def as_xy_int_tuple(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)

    def as_xy_float_tuple(self) -> Tuple[float, float]:
        return self.x, self.y

    @classmethod
    def xyxy_center(cls, xyxy: Tuple[int]) -> Point:
        x1, y1, x2, y2 = xyxy
        center_x = x1 + (x1 + x2) / 2
        center_y = y1 + (y1 + y2) / 2
        return cls(center_x, center_y)


@dataclass
class Vector:
    start: Point
    end: Point

    # negative when point rh, po, but also vertical stuff
    def is_in(self, point: Point) -> bool:
        v1 = Vector(self.start, self.end)
        v2 = Vector(self.start, point)
        cross_product = (v1.end.x - v1.start.x) * (v2.end.y - v2.start.y) - (
            v1.end.y - v1.start.y
        ) * (v2.end.x - v2.start.x)
        return cross_product < 0


@dataclass
class VerticalLine:
    start: Point
    height: int

    @property
    def x(self) -> float:
        """x coordinate of line"""
        return self.start.x

    def vertical_match(self, point: Point) -> bool:
        """True if point lies between the y coordinates of line"""
        return self.start.y < point.y < self.start.y + self.height

    def left_to(self, point: Point) -> bool:
        """True if line lies right to point"""
        return self.start.x < point.x


@dataclass
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def top_left(self) -> Point:
        return Point(x=self.x, y=self.y)

    @property
    def bottom_right(self) -> Point:
        return Point(x=self.x + self.width, y=self.y + self.height)

    @property
    def bottom_center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y + self.height)

    @property
    def top_center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y)

    @property
    def center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y + self.height / 2)

    def pad(self, padding) -> Rect:
        return Rect(
            x=self.x - padding,
            y=self.y - padding,
            width=self.width + 2 * padding,
            height=self.height + 2 * padding,
        )

    def pad_x(self, padding: float) -> Rect:
        return Rect(
            x=self.x, y=self.y, width=self.width + 2 * padding, height=self.height
        )

    def pad_y(self, padding: float) -> Rect:
        return Rect(
            x=self.x, y=self.y, width=self.width, height=self.height + 2 * padding
        )

    def contains_point(self, point: Point) -> bool:
        return (
            self.x - self.width < point.x < self.x + self.width
            and self.y - self.height < point.y < self.y + self.height
        )

    @classmethod
    def from_xyxy(cls, xyxy: Tuple[float]) -> Rect:
        x1, y1, x2, y2 = xyxy
        center = Point.xyxy_center(xyxy)
        width = x2 - x1
        height = y2 - y1
        return Rect(center.x, center.y, width, height)
