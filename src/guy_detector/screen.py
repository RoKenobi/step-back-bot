from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from guy_detector.models import Detection


@dataclass(frozen=True)
class ScreenCaptureConfig:
    local_video_path: Path
    remote_video_path: str
    pc3_target: str
    padding_fraction: float = 0.12
    frame_size: tuple[int, int] = (480, 480)
    duration_seconds: float = 5.0
    fps: int = 10


def crop_detection(
    bgr_image: np.ndarray,
    detection: Detection,
    *,
    padding_fraction: float,
) -> np.ndarray:
    height, width = bgr_image.shape[:2]
    x1, y1, x2, y2 = detection.xyxy
    pad_x = int(round(detection.width * padding_fraction))
    pad_y = int(round(detection.height * padding_fraction))
    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(width, x2 + pad_x)
    crop_y2 = min(height, y2 + pad_y)
    crop_y2 = crop_y1 + max(1, (crop_y2 - crop_y1) // 2)
    return bgr_image[crop_y1:crop_y2, crop_x1:crop_x2]


def write_detection_video(
    bgr_image: np.ndarray,
    detection: Detection,
    config: ScreenCaptureConfig,
) -> Path:
    crop = crop_detection(
        bgr_image,
        detection,
        padding_fraction=config.padding_fraction,
    )
    if crop.size == 0:
        raise ValueError("cannot write screen video for empty crop")

    config.local_video_path.parent.mkdir(parents=True, exist_ok=True)
    frame = _letterbox(crop, config.frame_size)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(config.local_video_path),
        fourcc,
        float(config.fps),
        config.frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {config.local_video_path}")
    try:
        frame_count = max(1, int(round(config.duration_seconds * config.fps)))
        for _ in range(frame_count):
            writer.write(frame)
    finally:
        writer.release()
    return config.local_video_path


def transfer_to_pc3(config: ScreenCaptureConfig, *, timeout_seconds: float = 2.0) -> None:
    subprocess.run(
        build_scp_command(config),
        check=True,
        timeout=timeout_seconds,
    )


def build_scp_command(config: ScreenCaptureConfig) -> list[str]:
    return [
        "scp",
        "-q",
        str(config.local_video_path),
        f"{config.pc3_target}:{config.remote_video_path}",
    ]


def _letterbox(bgr_image: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
    target_width, target_height = frame_size
    height, width = bgr_image.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(bgr_image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x = (target_width - resized_width) // 2
    y = (target_height - resized_height) // 2
    frame[y : y + resized_height, x : x + resized_width] = resized
    return frame
