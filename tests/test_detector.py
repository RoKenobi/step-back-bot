import numpy as np
import pytest

from guy_detector.detector import (
    FakeDetector,
    UltralyticsYoloDetector,
    filter_person_detections,
    resize_for_detection,
    scale_detection,
)
from guy_detector.models import Detection


def test_fake_detector_returns_configured_detections() -> None:
    detections = [Detection(class_name="person", confidence=0.9, xyxy=(1, 2, 3, 4))]
    detector = FakeDetector(detections)

    assert detector.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == detections


def test_filter_person_detections_applies_class_and_confidence() -> None:
    detections = [
        Detection(class_name="person", confidence=0.44, xyxy=(1, 1, 2, 2)),
        Detection(class_name="person", confidence=0.8, xyxy=(1, 1, 2, 2)),
        Detection(class_name="chair", confidence=0.9, xyxy=(1, 1, 2, 2)),
    ]

    filtered = filter_person_detections(detections, confidence_threshold=0.45)

    assert filtered == [detections[1]]


def test_resize_for_detection_keeps_aspect_ratio_to_max_dimension() -> None:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)

    resized, scale = resize_for_detection(image, max_dimension=640)

    assert resized.shape == (360, 640, 3)
    assert scale == 0.5


def test_resize_for_detection_preserves_small_images() -> None:
    image = np.zeros((320, 480, 3), dtype=np.uint8)

    resized, scale = resize_for_detection(image, max_dimension=640)

    assert resized is image
    assert scale == 1.0


def test_scale_detection_maps_resized_bbox_back_to_original_coordinates() -> None:
    detection = Detection(class_name="person", confidence=0.8, xyxy=(20, 30, 120, 230))

    scaled = scale_detection(detection, scale=0.5)

    assert scaled == Detection(class_name="person", confidence=0.8, xyxy=(40, 60, 240, 460))


def test_ultralytics_detector_requires_explicit_local_model_path() -> None:
    with pytest.raises(FileNotFoundError, match="models/missing.pt"):
        UltralyticsYoloDetector(
            "models/missing.pt",
            device="cpu",
            confidence_threshold=0.45,
        )
