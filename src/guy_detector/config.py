from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

DetectorBackend = Literal["local_yolo", "remote_http"]
ProximityPrefilterMode = Literal["any", "all"]
SensorPrimaryDetectionMode = Literal["any", "all"]


class Settings(BaseModel):
    rgb_topic: str = "/aima/hal/sensor/rgbd_head_front/rgb_image"
    depth_topic: str | None = "/aima/hal/sensor/rgbd_head_front/depth_image"
    camera_info_topic: str = "/aima/hal/sensor/rgbd_head_front/rgb_camera_info"
    tts_service: str = "/aimdk_5Fmsgs/srv/PlayTts"
    play_video_service: str = "/face_ui_proxy/play_video"
    preset_motion_service: str = "/aimdk_5Fmsgs/srv/SetMcPresetMotion"
    mc_action_service: str = "/aimdk_5Fmsgs/srv/GetMcAction"
    head_yaw_command_topic: str = "/aima/hal/joint/head/command"
    detector_backend: DetectorBackend = "local_yolo"
    detector_model_path: str = "models/yolo11n.pt"
    detector_device: str = "cuda:0"
    detector_confidence_threshold: float = Field(default=0.45, ge=0, le=1)
    detector_max_dimension: int | None = Field(default=None, gt=0)
    remote_detector_url: str = "http://127.0.0.1:8766/detect"
    remote_detector_timeout_seconds: float = Field(default=0.5, gt=0)
    remote_frame_max_dimension: int | None = Field(default=640, gt=0)
    remote_jpeg_quality: int = Field(default=70, ge=1, le=100)
    proximity_prefilter_mode: ProximityPrefilterMode = "any"
    proximity_prefilter_enabled: bool = False
    proximity_prefilter_topic: str = "/aima/hal/sensor/lidar_chest_front/lidar_pointcloud"
    proximity_prefilter_max_distance_m: float = Field(default=2.5, gt=0)
    proximity_prefilter_min_points: int = Field(default=20, gt=0)
    proximity_prefilter_stale_seconds: float = Field(default=0.5, gt=0)
    proximity_prefilter_fail_open: bool = True
    depth_prefilter_enabled: bool = False
    depth_prefilter_topic: str = "/aima/hal/sensor/rgbd_head_front/depth_image"
    depth_prefilter_min_distance_m: float = Field(default=0.2, gt=0)
    depth_prefilter_max_distance_m: float = Field(default=2.5, gt=0)
    depth_prefilter_min_valid_fraction: float = Field(default=0.02, gt=0, le=1)
    depth_prefilter_stale_seconds: float = Field(default=0.5, gt=0)
    depth_prefilter_fail_open: bool = True
    sensor_primary_detection_enabled: bool = False
    sensor_primary_detection_use_lidar: bool = True
    sensor_primary_detection_use_depth: bool = True
    sensor_primary_detection_mode: SensorPrimaryDetectionMode = "any"
    sensor_primary_depth_rotate_180: bool = False
    sensor_primary_lateral_offset_gain: float = Field(default=1.0, gt=0)
    sensor_primary_screen_yolo_enabled: bool = False
    proximity_threshold_m: float = Field(default=2.0, gt=0)
    required_consecutive_frames: int = Field(default=3, gt=0)
    speech_cooldown_seconds: float = Field(default=8.0, ge=0)
    tts_enabled: bool = True
    tts_unmute_before_speaking: bool = True
    audio_set_mute_service: str = "/aimdk_5Fmsgs/srv/SetMute"
    screen_enabled: bool = False
    screen_cooldown_seconds: float = Field(default=5.0, ge=0)
    screen_local_video_path: Path = Path("/tmp/guy-detector-screen/near_person.mp4")
    screen_remote_video_path: str = "/agibot/data/home/agi/guy_detector_screen/near_person.mp4"
    screen_pc3_target: str = "agi@agibot-pc3"
    screen_scp_timeout_seconds: float = Field(default=2.0, gt=0)
    screen_video_duration_seconds: float = Field(default=5.0, gt=0)
    screen_video_priority: int = Field(default=5, ge=0)
    screen_video_mode: int = Field(default=1, ge=1, le=2)
    preset_motion_enabled: bool = False
    preset_motion_cooldown_seconds: float = Field(default=8.0, ge=0)
    preset_motion_center_tolerance: float = Field(default=0.2, ge=0, le=1)
    preset_motion_interrupt: bool = False
    shirt_color_enabled: bool = False
    yaw_enabled: bool = False
    max_head_yaw_rad: float = Field(default=0.3, gt=0, le=0.366)
    yaw_stop_tolerance: float = Field(default=0.12, ge=0, le=1)
    yaw_stale_timeout_seconds: float = Field(default=0.5, gt=0)
    locomotion_yaw_enabled: bool = False
    locomotion_yaw_topic: str = "/aima/mc/locomotion/velocity"
    locomotion_yaw_input_source_service: str = "/aimdk_5Fmsgs/srv/SetMcInputSource"
    locomotion_yaw_source: str = "node"
    locomotion_yaw_register_input_source: bool = False
    locomotion_yaw_input_source_priority: int = Field(default=40, ge=0, le=100)
    locomotion_yaw_input_source_timeout_ms: int = Field(default=1000, gt=0)
    locomotion_yaw_max_angular_velocity: float = Field(default=0.3, gt=0, le=1.0)
    locomotion_yaw_min_angular_velocity: float = Field(default=0.1, ge=0, le=1.0)
    locomotion_yaw_stop_tolerance: float = Field(default=0.2, ge=0, le=1)
    locomotion_yaw_invert: bool = False
    locomotion_yaw_publish_hz: float = Field(default=50.0, gt=0)
    locomotion_yaw_command_duration_seconds: float = Field(default=1.0, gt=0)
    locomotion_yaw_post_stop_action_delay_seconds: float = Field(default=0.0, ge=0)
    locomotion_yaw_wait_for_stand_enabled: bool = False
    locomotion_yaw_stand_action_desc: str = "STAND_DEFAULT"
    locomotion_yaw_stand_status_value: int = 100
    locomotion_yaw_stand_wait_timeout_seconds: float = Field(default=3.0, gt=0)
    locomotion_yaw_stand_poll_seconds: float = Field(default=0.2, gt=0)
    bbox_distance_fallback_enabled: bool = False
    bbox_near_height_fraction: float = Field(default=0.35, gt=0, le=1)
    debug_log_detections: bool = False
    debug_server_enabled: bool = False
    debug_frame_max_dimension: int | None = Field(default=None, gt=0)
    debug_server_host: str = "0.0.0.0"
    debug_server_port: int = Field(default=8765, gt=0, le=65535)
    event_jsonl_path: Path = Path("/tmp/guy-detector-events.jsonl")


def load_settings(path: str | Path) -> Settings:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"settings file must contain a mapping: {path}")
    return Settings.model_validate(data)
