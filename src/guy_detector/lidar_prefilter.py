from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

ProximityPrefilterMode = Literal["any", "all"]


@dataclass(frozen=True)
class LidarProximityState:
    timestamp_s: float
    has_close_object: bool
    close_point_count: int
    nearest_distance_m: float | None


def estimate_pointcloud_proximity(
    points: Iterable[tuple[float, float, float]],
    *,
    max_distance_m: float,
    min_points: int,
    now_s: float,
) -> LidarProximityState:
    close_point_count = 0
    nearest_distance_m: float | None = None

    for x, y, z in points:
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            continue
        if x <= 0:
            continue
        distance_m = math.sqrt((x * x) + (y * y) + (z * z))
        if distance_m > max_distance_m:
            continue
        close_point_count += 1
        if nearest_distance_m is None or distance_m < nearest_distance_m:
            nearest_distance_m = distance_m

    nearest = None if nearest_distance_m is None else round(nearest_distance_m, 3)
    return LidarProximityState(
        timestamp_s=now_s,
        has_close_object=close_point_count >= min_points,
        close_point_count=close_point_count,
        nearest_distance_m=nearest,
    )


def should_run_detector_for_prefilter(
    *,
    enabled: bool,
    state: LidarProximityState | None,
    now_s: float,
    stale_seconds: float,
    fail_open: bool,
) -> bool:
    if not enabled:
        return True
    if state is None:
        return fail_open
    if now_s - state.timestamp_s > stale_seconds:
        return fail_open
    return state.has_close_object


def should_run_detector_for_combined_prefilters(
    *,
    enabled_results: list[bool],
    mode: ProximityPrefilterMode,
) -> bool:
    if not enabled_results:
        return True
    if mode == "all":
        return all(enabled_results)
    return any(enabled_results)
