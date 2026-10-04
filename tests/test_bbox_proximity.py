from __future__ import annotations

from guy_detector.bbox_proximity import estimate_bbox_proximity
from guy_detector.models import Detection


def test_bbox_proximity_marks_tall_person_box_as_near_without_distance() -> None:
    detection = Detection(class_name="person", confidence=0.9, xyxy=(10, 10, 80, 430))

    estimate = estimate_bbox_proximity(
        detection,
        image_height=720,
        min_height_fraction=0.35,
    )

    assert estimate.distance_m is None
    assert estimate.depth_confidence == 0.0
    assert estimate.source == "bbox"
    assert estimate.bbox_height_fraction == 0.583
    assert estimate.is_valid is True


def test_bbox_proximity_rejects_small_person_box() -> None:
    detection = Detection(class_name="person", confidence=0.9, xyxy=(10, 10, 80, 120))

    estimate = estimate_bbox_proximity(
        detection,
        image_height=720,
        min_height_fraction=0.35,
    )

    assert estimate.distance_m is None
    assert estimate.source == "bbox"
    assert estimate.bbox_height_fraction == 0.153
    assert estimate.is_valid is False
