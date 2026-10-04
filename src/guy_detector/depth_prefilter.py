from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from guy_detector.depth import depth_to_meters
from guy_detector.lidar_prefilter import should_run_detector_for_prefilter


@dataclass(frozen=True)
class DepthProximityState:
    timestamp_s: float
    has_close_object: bool
    close_fraction: float
    nearest_distance_m: float | None
    close_xyxy: tuple[int, int, int, int] | None = None
    lateral_offset: float | None = None


def estimate_depth_proximity(
    depth_image: np.ndarray,
    *,
    min_distance_m: float,
    max_distance_m: float,
    min_valid_fraction: float,
    now_s: float,
) -> DepthProximityState:
    if depth_image.size == 0:
        return DepthProximityState(
            timestamp_s=now_s,
            has_close_object=False,
            close_fraction=0.0,
            nearest_distance_m=None,
        )

    depth_m = depth_to_meters(depth_image)
    close = np.isfinite(depth_m) & (depth_m >= min_distance_m) & (depth_m <= max_distance_m)
    close_count = int(close.sum())
    close_fraction = close_count / int(depth_m.size)
    nearest_distance_m = None
    close_xyxy = None
    lateral_offset = None
    if close_count:
        nearest_distance_m = round(float(depth_m[close].min()), 3)
        ys, xs = np.nonzero(close)
        x1 = int(xs.min())
        x2 = int(xs.max()) + 1
        y1 = int(ys.min())
        y2 = int(ys.max()) + 1
        close_xyxy = (x1, y1, x2, y2)
        image_width = int(depth_m.shape[1])
        center_x = float(xs.mean())
        lateral_offset = round((center_x - (image_width / 2)) / (image_width / 2), 3)

    return DepthProximityState(
        timestamp_s=now_s,
        has_close_object=close_fraction >= min_valid_fraction,
        close_fraction=round(close_fraction, 3),
        nearest_distance_m=nearest_distance_m,
        close_xyxy=close_xyxy,
        lateral_offset=lateral_offset,
    )


# The staleness / fail-open rule is identical for every prefilter sensor.
should_run_detector_for_depth_prefilter = should_run_detector_for_prefilter
