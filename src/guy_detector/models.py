from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProximitySource = Literal["depth", "bbox", "lidar"]


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    xyxy: tuple[int, int, int, int]

    @property
    def center_x(self) -> float:
        x1, _, x2, _ = self.xyxy
        return (x1 + x2) / 2

    @property
    def center_y(self) -> float:
        _, y1, _, y2 = self.xyxy
        return (y1 + y2) / 2

    @property
    def width(self) -> int:
        x1, _, x2, _ = self.xyxy
        return max(0, x2 - x1)

    @property
    def height(self) -> int:
        _, y1, _, y2 = self.xyxy
        return max(0, y2 - y1)

    def normalized_x_offset(self, image_width: int) -> float:
        if image_width <= 0:
            raise ValueError("image_width must be positive")
        return (self.center_x - (image_width / 2)) / (image_width / 2)


@dataclass(frozen=True)
class ProximityEstimate:
    distance_m: float | None
    depth_confidence: float
    source: ProximitySource = "depth"
    bbox_height_fraction: float | None = None
    bbox_near: bool = False

    @property
    def is_valid(self) -> bool:
        if self.source == "bbox":
            return self.bbox_near
        return self.distance_m is not None


@dataclass(frozen=True)
class NearPersonEvent:
    timestamp_s: float
    detection: Detection
    proximity: ProximityEstimate
    lateral_offset: float
    shirt_color: str | None = None
