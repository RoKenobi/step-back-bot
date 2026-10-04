from guy_detector.models import Detection, ProximityEstimate


def test_detection_center_normalized_offset() -> None:
    detection = Detection(class_name="person", confidence=0.91, xyxy=(40, 20, 80, 120))

    assert detection.center_x == 60
    assert detection.center_y == 70
    assert detection.width == 40
    assert detection.height == 100
    assert detection.normalized_x_offset(image_width=200) == -0.4


def test_proximity_estimate_marks_missing_depth_invalid() -> None:
    estimate = ProximityEstimate(distance_m=None, depth_confidence=0.0)

    assert estimate.is_valid is False
