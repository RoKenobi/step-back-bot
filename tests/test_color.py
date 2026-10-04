import numpy as np

from guy_detector.color import estimate_shirt_color
from guy_detector.models import Detection


def test_estimates_red_shirt_from_upper_body_crop() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[25:55, 35:65] = (20, 20, 220)
    detection = Detection(class_name="person", confidence=0.9, xyxy=(30, 10, 70, 90))

    assert estimate_shirt_color(image, detection) == "red"


def test_estimates_white_and_dark() -> None:
    detection = Detection(class_name="person", confidence=0.9, xyxy=(30, 10, 70, 90))
    white = np.full((100, 100, 3), 240, dtype=np.uint8)
    dark = np.full((100, 100, 3), 20, dtype=np.uint8)

    assert estimate_shirt_color(white, detection) == "white"
    assert estimate_shirt_color(dark, detection) == "dark"


def test_returns_none_for_tiny_detection() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    detection = Detection(class_name="person", confidence=0.9, xyxy=(10, 10, 12, 12))

    assert estimate_shirt_color(image, detection) is None
