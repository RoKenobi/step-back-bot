from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from guy_detector.detector import PersonDetector, resize_for_detection
from guy_detector.models import Detection


class RemoteDetectorError(RuntimeError):
    pass


class RemoteHttpDetector(PersonDetector):
    def __init__(
        self,
        url: str,
        *,
        timeout_seconds: float,
        frame_max_dimension: int | None,
        jpeg_quality: int,
    ) -> None:
        self._url = url
        self._timeout_seconds = timeout_seconds
        self._frame_max_dimension = frame_max_dimension
        self._jpeg_quality = jpeg_quality

    def detect(self, bgr_image: np.ndarray) -> list[Detection]:
        payload, headers = encode_jpeg_request(
            bgr_image,
            max_dimension=self._frame_max_dimension,
            jpeg_quality=self._jpeg_quality,
        )
        request = Request(self._url, data=payload, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                body = response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RemoteDetectorError(f"remote detector request failed: {exc}") from exc
        return decode_detection_payload(body)


def encode_jpeg_request(
    bgr_image: np.ndarray,
    *,
    max_dimension: int | None,
    jpeg_quality: int,
) -> tuple[bytes, dict[str, str]]:
    height, width = bgr_image.shape[:2]
    resized, _ = resize_for_detection(bgr_image, max_dimension)
    frame_height, frame_width = resized.shape[:2]
    ok, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise RemoteDetectorError("failed to encode frame as JPEG")
    return (
        encoded.tobytes(),
        {
            "Content-Type": "image/jpeg",
            "X-Original-Width": str(width),
            "X-Original-Height": str(height),
            "X-Frame-Width": str(frame_width),
            "X-Frame-Height": str(frame_height),
        },
    )


def build_detection_payload(detections: list[Detection]) -> bytes:
    return json.dumps(
        {
            "detections": [
                {
                    "class_name": detection.class_name,
                    "confidence": detection.confidence,
                    "xyxy": list(detection.xyxy),
                }
                for detection in detections
            ]
        },
        separators=(",", ":"),
    ).encode("utf-8")


def decode_detection_payload(payload: bytes) -> list[Detection]:
    data = json.loads(payload.decode("utf-8"))
    detections = data.get("detections", [])
    if not isinstance(detections, list):
        raise RemoteDetectorError("remote detector response must contain detections list")
    return [_parse_detection(item) for item in detections]


def _parse_detection(item: object) -> Detection:
    if not isinstance(item, dict):
        raise RemoteDetectorError("remote detector detection must be an object")
    xyxy = item.get("xyxy")
    if not isinstance(xyxy, list) or len(xyxy) != 4:
        raise RemoteDetectorError("remote detector detection must contain xyxy list")
    return Detection(
        class_name=str(item["class_name"]),
        confidence=float(item["confidence"]),
        xyxy=tuple(int(round(float(value))) for value in xyxy),  # type: ignore[arg-type]
    )
