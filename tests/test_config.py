from pathlib import Path

from guy_detector.config import Settings, load_settings


def test_default_settings_are_conservative() -> None:
    settings = Settings()

    assert settings.rgb_topic == "/aima/hal/sensor/rgbd_head_front/rgb_image"
    assert settings.depth_topic == "/aima/hal/sensor/rgbd_head_front/depth_image"
    assert settings.detector_model_path == "models/yolo11n.pt"
    assert settings.detector_backend == "local_yolo"
    assert settings.detector_max_dimension is None
    assert settings.remote_detector_url == "http://127.0.0.1:8766/detect"
    assert settings.proximity_prefilter_mode == "any"
    assert settings.proximity_prefilter_enabled is False
    assert (
        settings.proximity_prefilter_topic
        == "/aima/hal/sensor/lidar_chest_front/lidar_pointcloud"
    )
    assert settings.proximity_prefilter_max_distance_m == 2.5
    assert settings.proximity_prefilter_min_points == 20
    assert settings.proximity_prefilter_stale_seconds == 0.5
    assert settings.proximity_prefilter_fail_open is True
    assert settings.depth_prefilter_enabled is False
    assert settings.depth_prefilter_topic == "/aima/hal/sensor/rgbd_head_front/depth_image"
    assert settings.depth_prefilter_min_distance_m == 0.2
    assert settings.depth_prefilter_max_distance_m == 2.5
    assert settings.depth_prefilter_min_valid_fraction == 0.02
    assert settings.depth_prefilter_stale_seconds == 0.5
    assert settings.depth_prefilter_fail_open is True
    assert settings.sensor_primary_detection_enabled is False
    assert settings.sensor_primary_detection_use_lidar is True
    assert settings.sensor_primary_detection_use_depth is True
    assert settings.sensor_primary_detection_mode == "any"
    assert settings.sensor_primary_depth_rotate_180 is False
    assert settings.sensor_primary_lateral_offset_gain == 1.0
    assert settings.sensor_primary_screen_yolo_enabled is False
    assert settings.debug_frame_max_dimension is None
    assert settings.tts_enabled is True
    assert settings.tts_unmute_before_speaking is True
    assert settings.audio_set_mute_service == "/aimdk_5Fmsgs/srv/SetMute"
    assert settings.mc_action_service == "/aimdk_5Fmsgs/srv/GetMcAction"
    assert settings.yaw_enabled is False
    assert settings.locomotion_yaw_enabled is False
    assert settings.locomotion_yaw_topic == "/aima/mc/locomotion/velocity"
    assert settings.locomotion_yaw_input_source_service == "/aimdk_5Fmsgs/srv/SetMcInputSource"
    assert settings.locomotion_yaw_source == "node"
    assert settings.locomotion_yaw_register_input_source is False
    assert settings.locomotion_yaw_max_angular_velocity == 0.3
    assert settings.locomotion_yaw_min_angular_velocity == 0.1
    assert settings.locomotion_yaw_stop_tolerance == 0.2
    assert settings.locomotion_yaw_invert is False
    assert settings.locomotion_yaw_publish_hz == 50.0
    assert settings.locomotion_yaw_command_duration_seconds == 1.0
    assert settings.locomotion_yaw_post_stop_action_delay_seconds == 0.0
    assert settings.locomotion_yaw_wait_for_stand_enabled is False
    assert settings.locomotion_yaw_stand_action_desc == "STAND_DEFAULT"
    assert settings.locomotion_yaw_stand_status_value == 100
    assert settings.locomotion_yaw_stand_wait_timeout_seconds == 3.0
    assert settings.locomotion_yaw_stand_poll_seconds == 0.2
    assert settings.bbox_distance_fallback_enabled is False
    assert settings.bbox_near_height_fraction == 0.35
    assert settings.head_yaw_command_topic == "/aima/hal/joint/head/command"
    assert settings.max_head_yaw_rad <= 0.366


def test_depth_topic_can_be_disabled_for_rgb_only_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        "\n".join(
            [
                'rgb_topic: "/aima/hal/sensor/rgb_head_front_center/rgb_image"',
                "depth_topic: null",
                "bbox_distance_fallback_enabled: true",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.rgb_topic == "/aima/hal/sensor/rgb_head_front_center/rgb_image"
    assert settings.depth_topic is None
    assert settings.bbox_distance_fallback_enabled is True


def test_front_debug_config_uses_live_stereo_left_rgb_only_bbox_fallback() -> None:
    settings = load_settings("config/pc2.front-center-debug.yaml")

    assert settings.rgb_topic == "/aima/hal/sensor/stereo_head_front_left/rgb_image"
    assert settings.depth_topic is None
    assert settings.detector_backend == "remote_http"
    assert settings.remote_detector_url == "http://DETECTOR_HOST:8766/detect"
    assert settings.remote_frame_max_dimension == 640
    assert settings.remote_jpeg_quality == 70
    assert settings.proximity_prefilter_mode == "any"
    assert settings.proximity_prefilter_enabled is False
    assert (
        settings.proximity_prefilter_topic
        == "/aima/hal/sensor/lidar_chest_front/lidar_pointcloud"
    )
    assert settings.proximity_prefilter_max_distance_m == 0.4
    assert settings.depth_prefilter_enabled is False
    assert settings.depth_prefilter_topic == "/aima/hal/sensor/rgbd_head_front/depth_image"
    assert settings.depth_prefilter_max_distance_m == 0.4
    assert settings.detector_max_dimension == 640
    assert settings.debug_frame_max_dimension == 416
    assert settings.bbox_distance_fallback_enabled is True
    assert settings.bbox_near_height_fraction == 0.8
    assert settings.required_consecutive_frames == 1
    assert settings.proximity_threshold_m == 0.4
    assert settings.speech_cooldown_seconds == 13.0
    assert settings.tts_enabled is True
    assert settings.tts_unmute_before_speaking is True
    assert settings.audio_set_mute_service == "/aimdk_5Fmsgs/srv/SetMute"
    assert settings.yaw_enabled is False
    assert settings.locomotion_yaw_enabled is True
    assert settings.locomotion_yaw_register_input_source is True
    assert settings.locomotion_yaw_input_source_timeout_ms == 3000
    assert settings.locomotion_yaw_stop_tolerance == 0.30
    assert settings.locomotion_yaw_command_duration_seconds == 2.5
    assert settings.locomotion_yaw_post_stop_action_delay_seconds == 3.5
    assert settings.locomotion_yaw_stand_poll_seconds == 0.05
    assert settings.screen_enabled is True
    assert settings.screen_pc3_target == "agi@agibot-pc3"
    assert settings.screen_video_duration_seconds == 5.0
    assert settings.preset_motion_enabled is True
    assert settings.preset_motion_service == "/aimdk_5Fmsgs/srv/SetMcPresetMotion"
    assert settings.preset_motion_center_tolerance == 0.45
    assert settings.mc_action_service == "/aimdk_5Fmsgs/srv/GetMcAction"
    assert settings.sensor_primary_detection_enabled is True
    assert settings.sensor_primary_detection_use_lidar is False
    assert settings.sensor_primary_depth_rotate_180 is True
    assert settings.sensor_primary_lateral_offset_gain == 2.0
    assert settings.sensor_primary_screen_yolo_enabled is True
    assert settings.screen_cooldown_seconds == 13.0
    assert settings.preset_motion_cooldown_seconds == 13.0
    assert settings.locomotion_yaw_wait_for_stand_enabled is False
    assert settings.locomotion_yaw_stand_wait_timeout_seconds == 4.0
    assert settings.head_yaw_command_topic == "/aima/hal/joint/head/command"
    assert settings.debug_server_enabled is True


def test_tower_detector_config_uses_gpu_one_and_available_yolo() -> None:
    settings = load_settings("config/tower.detector-server.yaml")

    assert settings.detector_model_path == "models/yolo11s.pt"
    assert settings.detector_device == "cuda:1"
    assert settings.detector_max_dimension == 640


def test_load_settings_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        "\n".join(
            [
                "proximity_threshold_m: 1.75",
                "required_consecutive_frames: 4",
                "screen_video_duration_seconds: 7.5",
                "yaw_enabled: true",
                "event_jsonl_path: /tmp/guy-detector-events.jsonl",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.proximity_threshold_m == 1.75
    assert settings.required_consecutive_frames == 4
    assert settings.screen_video_duration_seconds == 7.5
    assert settings.yaw_enabled is True
    assert settings.event_jsonl_path == Path("/tmp/guy-detector-events.jsonl")
