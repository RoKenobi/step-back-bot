from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from guy_detector.models import Detection


class PersonDetector(Protocol):
    def detect(self, bgr_image: np.ndarray) -> list[Detection]:
        ...


class FakeDetector:
    def __init__(self, detections: Sequence[Detection]) -> None:
        self._detections = list(detections)

    def detect(self, bgr_image: np.ndarray) -> list[Detection]:
        return list(self._detections)


class UltralyticsYoloDetector:
    def __init__(
        self,
        model_path: str,
        *,
        device: str,
        confidence_threshold: float,
        max_dimension: int | None = None,
    ) -> None:
        path = Path(model_path)
        if path.parent != Path(".") and not path.exists():
            raise FileNotFoundError(f"Detector model not found: {model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Install guy-detector[detector] to use UltralyticsYoloDetector"
            ) from exc
        self._model = YOLO(model_path)
        self._device = device
        self._confidence_threshold = confidence_threshold
        self._max_dimension = max_dimension

    def detect(self, bgr_image: np.ndarray) -> list[Detection]:
        detector_image, scale = resize_for_detection(bgr_image, self._max_dimension)
        results = self._model.predict(
            source=detector_image,
            device=self._device,
            conf=self._confidence_threshold,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = str(names[class_id])
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]
                detections.append(
                    scale_detection(
                        Detection(
                            class_name=class_name,
                            confidence=confidence,
                            xyxy=(x1, y1, x2, y2),
                        ),
                        scale=scale,
                    )
                )
        return filter_person_detections(detections, confidence_threshold=self._confidence_threshold)


def resize_for_detection(
    bgr_image: np.ndarray,
    max_dimension: int | None,
) -> tuple[np.ndarray, float]:
    if max_dimension is None:
        return bgr_image, 1.0
    height, width = bgr_image.shape[:2]
    largest = max(height, width)
    if largest <= max_dimension:
        return bgr_image, 1.0
    scale = max_dimension / largest
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(bgr_image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def scale_detection(detection: Detection, *, scale: float) -> Detection:
    if scale <= 0:
        raise ValueError("scale must be positive")
    if scale == 1.0:
        return detection
    x1, y1, x2, y2 = detection.xyxy
    inverse = 1.0 / scale
    return Detection(
        class_name=detection.class_name,
        confidence=detection.confidence,
        xyxy=(
            int(round(x1 * inverse)),
            int(round(y1 * inverse)),
            int(round(x2 * inverse)),
            int(round(y2 * inverse)),
        ),
    )


def filter_person_detections(
    detections: Sequence[Detection],
    *,
    confidence_threshold: float,
) -> list[Detection]:
    return [
        detection
        for detection in detections
        if detection.class_name == "person" and detection.confidence >= confidence_threshold
    ]
