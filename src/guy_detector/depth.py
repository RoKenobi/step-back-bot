from __future__ import annotations

import numpy as np

from guy_detector.models import Detection, ProximityEstimate


def estimate_person_distance(
    depth_image: np.ndarray,
    detection: Detection,
    *,
    min_valid_m: float = 0.2,
    max_valid_m: float = 8.0,
    min_valid_fraction: float = 0.35,
) -> ProximityEstimate:
    crop = _torso_depth_crop(depth_image, detection)
    if crop.size == 0:
        return ProximityEstimate(distance_m=None, depth_confidence=0.0)

    depth_m = depth_to_meters(crop)
    valid = np.isfinite(depth_m) & (depth_m >= min_valid_m) & (depth_m <= max_valid_m)
    valid_fraction = float(valid.sum() / valid.size)
    if valid_fraction < min_valid_fraction:
        return ProximityEstimate(
            distance_m=None,
            depth_confidence=round(valid_fraction, 3),
        )

    median_m = float(np.median(depth_m[valid]))
    return ProximityEstimate(
        distance_m=round(median_m, 3),
        depth_confidence=round(valid_fraction, 3),
    )


def _torso_depth_crop(depth_image: np.ndarray, detection: Detection) -> np.ndarray:
    height, width = depth_image.shape[:2]
    crop_x1, crop_y1, crop_x2, crop_y2 = torso_depth_rect(
        detection,
        image_width=width,
        image_height=height,
    )
    return depth_image[crop_y1:crop_y2, crop_x1:crop_x2]


def torso_depth_rect(
    detection: Detection,
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = detection.xyxy
    box_w = max(0, x2 - x1)
    box_h = max(0, y2 - y1)
    if box_w == 0 or box_h == 0:
        return (0, 0, 0, 0)

    crop_x1 = int(round(x1 + box_w * 0.35))
    crop_x2 = int(round(x1 + box_w * 0.65))
    crop_y1 = int(round(y1 + box_h * 0.25))
    crop_y2 = int(round(y1 + box_h * 0.75))
    crop_x1 = min(max(crop_x1, 0), image_width)
    crop_x2 = min(max(crop_x2, 0), image_width)
    crop_y1 = min(max(crop_y1, 0), image_height)
    crop_y2 = min(max(crop_y2, 0), image_height)
    return (crop_x1, crop_y1, crop_x2, crop_y2)


def depth_to_meters(depth_image: np.ndarray) -> np.ndarray:
    """Convert a depth image to float meters; integer images are assumed to be millimeters."""
    depth = depth_image.astype(np.float32, copy=False)
    if np.issubdtype(depth_image.dtype, np.integer):
        depth = depth / 1000.0
    return depth
