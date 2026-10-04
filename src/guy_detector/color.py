from __future__ import annotations

import cv2
import numpy as np

from guy_detector.models import Detection


def estimate_shirt_color(bgr_image: np.ndarray, detection: Detection) -> str | None:
    crop = _upper_body_crop(bgr_image, detection)
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None

    pixels = crop.reshape(-1, 3)
    median_bgr = np.median(pixels, axis=0).astype(np.uint8).reshape(1, 1, 3)
    hsv = cv2.cvtColor(median_bgr, cv2.COLOR_BGR2HSV)[0, 0]
    hue, saturation, value = int(hsv[0]), int(hsv[1]), int(hsv[2])

    if value < 50:
        return "dark"
    if saturation < 35 and value > 200:
        return "white"
    if saturation < 45:
        return "gray"
    if hue < 10 or hue >= 170:
        return "red"
    if hue < 25:
        return "yellow"
    if hue < 85:
        return "green"
    if hue < 130:
        return "blue"
    return "dark"


def _upper_body_crop(bgr_image: np.ndarray, detection: Detection) -> np.ndarray:
    height, width = bgr_image.shape[:2]
    x1, y1, x2, y2 = detection.xyxy
    box_w = max(0, x2 - x1)
    box_h = max(0, y2 - y1)
    crop_x1 = int(round(x1 + box_w * 0.25))
    crop_x2 = int(round(x1 + box_w * 0.75))
    crop_y1 = int(round(y1 + box_h * 0.18))
    crop_y2 = int(round(y1 + box_h * 0.48))
    crop_x1 = min(max(crop_x1, 0), width)
    crop_x2 = min(max(crop_x2, 0), width)
    crop_y1 = min(max(crop_y1, 0), height)
    crop_y2 = min(max(crop_y2, 0), height)
    return bgr_image[crop_y1:crop_y2, crop_x1:crop_x2]
