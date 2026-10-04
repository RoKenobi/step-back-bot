import numpy as np

from guy_detector.depth import estimate_person_distance
from guy_detector.models import Detection


def test_estimates_median_depth_from_torso_crop_uint16_mm() -> None:
    depth = np.full((100, 100), 5000, dtype=np.uint16)
    depth[30:70, 42:58] = 1500
    detection = Detection(class_name="person", confidence=0.9, xyxy=(30, 10, 70, 90))

    estimate = estimate_person_distance(depth, detection)

    assert estimate.distance_m == 1.5
    assert estimate.depth_confidence > 0.9


def test_invalid_depth_returns_no_estimate() -> None:
    depth = np.zeros((100, 100), dtype=np.uint16)
    detection = Detection(class_name="person", confidence=0.9, xyxy=(30, 10, 70, 90))

    estimate = estimate_person_distance(depth, detection)

    assert estimate.distance_m is None
    assert estimate.depth_confidence == 0.0


def test_float_depth_in_meters_is_preserved() -> None:
    depth = np.full((80, 80), 2.25, dtype=np.float32)
    detection = Detection(class_name="person", confidence=0.9, xyxy=(20, 5, 60, 75))

    estimate = estimate_person_distance(depth, detection)

    assert estimate.distance_m == 2.25
    assert estimate.depth_confidence == 1.0
