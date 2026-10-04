from __future__ import annotations

from guy_detector.models import Detection, ProximityEstimate


def estimate_bbox_proximity(
    detection: Detection,
    *,
    image_height: int,
    min_height_fraction: float,
) -> ProximityEstimate:
    if image_height <= 0:
        raise ValueError("image_height must be positive")
    height_fraction = detection.height / image_height
    return ProximityEstimate(
        distance_m=None,
        depth_confidence=0.0,
        source="bbox",
        bbox_height_fraction=round(height_fraction, 3),
        bbox_near=height_fraction >= min_height_fraction,
    )
