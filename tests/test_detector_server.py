from __future__ import annotations

import json

import cv2
import numpy as np

from guy_detector.detector import FakeDetector
from guy_detector.detector_server import handle_detection_request
from guy_detector.models import Detection


def test_handle_detection_request_scales_decoded_frame_boxes_to_original_size() -> None:
    frame = np.zeros((50, 100, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    detector = FakeDetector(
        [Detection(class_name="person", confidence=0.9, xyxy=(10, 10, 30, 40))]
    )

    status, content_type, payload = handle_detection_request(
        encoded.tobytes(),
        headers={"X-Original-Width": "200", "X-Original-Height": "100"},
        detector=detector,
    )

    assert status == 200
    assert content_type == "application/json"
    body = json.loads(payload)
    assert body["detections"] == [
        {"class_name": "person", "confidence": 0.9, "xyxy": [20, 20, 60, 80]}
    ]
