from __future__ import annotations

from pathlib import Path

import numpy as np

from guy_detector.models import Detection
from guy_detector.screen import ScreenCaptureConfig, build_scp_command, crop_detection


def test_crop_detection_expands_and_clamps_person_box() -> None:
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    detection = Detection(class_name="person", confidence=0.9, xyxy=(10, 20, 50, 90))

    crop = crop_detection(image, detection, padding_fraction=0.1)

    assert crop.shape == (42, 48, 3)


def test_crop_detection_keeps_upper_half_of_padded_person_box() -> None:
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[:55, :] = (0, 255, 0)
    image[55:, :] = (0, 0, 255)
    detection = Detection(class_name="person", confidence=0.9, xyxy=(10, 20, 50, 90))

    crop = crop_detection(image, detection, padding_fraction=0.1)

    assert np.all(crop[:, :, 1] == 255)
    assert np.all(crop[:, :, 2] == 0)


def test_build_scp_command_targets_pc3_path() -> None:
    config = ScreenCaptureConfig(
        local_video_path=Path("/tmp/near_person.mp4"),
        remote_video_path="/agibot/data/home/agi/guy_detector_screen/near_person.mp4",
        pc3_target="agi@agibot-pc3",
    )

    assert build_scp_command(config) == [
        "scp",
        "-q",
        "/tmp/near_person.mp4",
        "agi@agibot-pc3:/agibot/data/home/agi/guy_detector_screen/near_person.mp4",
    ]
