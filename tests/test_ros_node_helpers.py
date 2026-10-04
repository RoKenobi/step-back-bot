from __future__ import annotations

import struct
from types import SimpleNamespace
from typing import Any

import numpy as np

from guy_detector.depth_prefilter import DepthProximityState
from guy_detector.lidar_prefilter import LidarProximityState
from guy_detector.models import Detection, NearPersonEvent, ProximityEstimate
from guy_detector.ros_node import (
    AgiBotNearPersonNode,
    _build_screen_capture_config,
    _build_sensor_primary_detection,
    _estimate_optional_shirt_color,
    _estimate_proximity,
    _format_detection_log,
    _proximity_is_near,
    _select_locomotion_yaw_target,
    _select_sensor_primary_proximity,
    _sensor_primary_lateral_offset,
    read_pointcloud_xyz,
)


def test_optional_shirt_color_returns_none_when_disabled() -> None:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    detection = Detection(class_name="person", confidence=0.8, xyxy=(10, 10, 60, 70))

    assert _estimate_optional_shirt_color(False, image, detection) is None


def test_optional_shirt_color_uses_estimator_when_enabled() -> None:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[20:40, 20:50] = (0, 0, 255)
    detection = Detection(class_name="person", confidence=0.8, xyxy=(10, 10, 60, 70))

    assert _estimate_optional_shirt_color(True, image, detection) == "red"


def test_debug_detection_tuple_uses_image_width_for_lateral_offset() -> None:
    from guy_detector.ros_node import _debug_detection_tuple

    detection = Detection(class_name="person", confidence=0.8, xyxy=(50, 10, 90, 70))
    proximity = ProximityEstimate(distance_m=1.7, depth_confidence=0.7)

    assert _debug_detection_tuple(detection, proximity, image_width=100) == (
        detection,
        proximity,
        0.4,
    )


def test_estimate_proximity_uses_bbox_fallback_without_depth() -> None:
    detection = Detection(class_name="person", confidence=0.8, xyxy=(20, 20, 80, 460))

    proximity = _estimate_proximity(
        latest_depth=None,
        detection=detection,
        image_height=720,
        bbox_fallback_enabled=True,
        bbox_near_height_fraction=0.35,
    )

    assert proximity.distance_m is None
    assert proximity.source == "bbox"
    assert proximity.is_valid is True


def test_estimate_proximity_is_invalid_without_depth_or_fallback() -> None:
    detection = Detection(class_name="person", confidence=0.8, xyxy=(20, 20, 80, 460))

    proximity = _estimate_proximity(
        latest_depth=None,
        detection=detection,
        image_height=720,
        bbox_fallback_enabled=False,
        bbox_near_height_fraction=0.35,
    )

    assert proximity.distance_m is None
    assert proximity.source == "depth"
    assert proximity.is_valid is False


def test_proximity_is_near_accepts_bbox_near_and_thresholded_depth() -> None:
    assert (
        _proximity_is_near(
            ProximityEstimate(
                distance_m=None,
                depth_confidence=0.0,
                source="bbox",
                bbox_near=True,
            ),
            threshold_m=2.0,
        )
        is True
    )
    assert (
        _proximity_is_near(
            ProximityEstimate(distance_m=1.7, depth_confidence=1.0),
            threshold_m=2.0,
        )
        is True
    )
    assert (
        _proximity_is_near(
            ProximityEstimate(distance_m=2.7, depth_confidence=1.0),
            threshold_m=2.0,
        )
        is False
    )


def test_format_detection_log_includes_near_person_context() -> None:
    detection = Detection(class_name="person", confidence=0.8, xyxy=(10, 20, 50, 120))
    proximity = ProximityEstimate(
        distance_m=None,
        depth_confidence=0.0,
        source="bbox",
        bbox_height_fraction=0.5,
        bbox_near=True,
    )

    line = _format_detection_log(detection, proximity, image_width=100)

    assert line.startswith("near person bbox=[10, 20, 50, 120]")
    assert "lateral=-0.40" in line


def test_select_locomotion_yaw_target_prefers_closest_metric_distance() -> None:
    farther = (
        Detection(class_name="person", confidence=0.8, xyxy=(10, 10, 40, 140)),
        ProximityEstimate(distance_m=1.8, depth_confidence=1.0),
    )
    closer = (
        Detection(class_name="person", confidence=0.8, xyxy=(50, 10, 80, 90)),
        ProximityEstimate(distance_m=1.2, depth_confidence=1.0),
    )

    assert _select_locomotion_yaw_target([farther, closer]) == closer


def test_select_locomotion_yaw_target_uses_largest_bbox_without_metric_distance() -> None:
    small = (
        Detection(class_name="person", confidence=0.8, xyxy=(10, 10, 40, 90)),
        ProximityEstimate(
            distance_m=None,
            depth_confidence=0.0,
            source="bbox",
            bbox_height_fraction=0.2,
            bbox_near=True,
        ),
    )
    large = (
        Detection(class_name="person", confidence=0.8, xyxy=(50, 10, 80, 180)),
        ProximityEstimate(
            distance_m=None,
            depth_confidence=0.0,
            source="bbox",
            bbox_height_fraction=0.5,
            bbox_near=True,
        ),
    )

    assert _select_locomotion_yaw_target([small, large]) == large


def test_select_sensor_primary_proximity_uses_recent_enabled_sensor_state() -> None:
    settings = SimpleNamespace(
        sensor_primary_detection_use_lidar=True,
        sensor_primary_detection_use_depth=True,
        sensor_primary_detection_mode="any",
        proximity_prefilter_stale_seconds=0.5,
        depth_prefilter_stale_seconds=0.5,
    )
    lidar = LidarProximityState(
        timestamp_s=10.0,
        has_close_object=True,
        close_point_count=25,
        nearest_distance_m=1.8,
    )
    depth = DepthProximityState(
        timestamp_s=10.0,
        has_close_object=True,
        close_fraction=0.04,
        nearest_distance_m=1.2,
        close_xyxy=(70, 10, 90, 60),
        lateral_offset=0.6,
    )

    proximity = _select_sensor_primary_proximity(
        settings,
        latest_lidar=lidar,
        latest_depth=depth,
        now_s=10.1,
    )

    assert proximity == ProximityEstimate(
        distance_m=1.2,
        depth_confidence=0.04,
        source="depth",
    )


def test_select_sensor_primary_proximity_requires_all_enabled_sensors_when_configured() -> None:
    settings = SimpleNamespace(
        sensor_primary_detection_use_lidar=True,
        sensor_primary_detection_use_depth=True,
        sensor_primary_detection_mode="all",
        proximity_prefilter_stale_seconds=0.5,
        depth_prefilter_stale_seconds=0.5,
    )
    lidar = LidarProximityState(
        timestamp_s=10.0,
        has_close_object=True,
        close_point_count=25,
        nearest_distance_m=1.8,
    )
    depth = DepthProximityState(
        timestamp_s=10.0,
        has_close_object=False,
        close_fraction=0.0,
        nearest_distance_m=None,
        close_xyxy=None,
        lateral_offset=None,
    )

    assert (
        _select_sensor_primary_proximity(
            settings,
            latest_lidar=lidar,
            latest_depth=depth,
            now_s=10.1,
        )
        is None
    )


def test_build_sensor_primary_detection_covers_current_rgb_frame() -> None:
    detection = _build_sensor_primary_detection(
        image_width=320,
        image_height=240,
        depth_state=None,
    )

    assert detection == Detection(
        class_name="person",
        confidence=1.0,
        xyxy=(0, 0, 320, 240),
    )


def test_build_sensor_primary_detection_uses_depth_region_when_available() -> None:
    depth = DepthProximityState(
        timestamp_s=10.0,
        has_close_object=True,
        close_fraction=0.1,
        nearest_distance_m=1.1,
        close_xyxy=(70, 20, 110, 180),
        lateral_offset=0.6,
    )

    detection = _build_sensor_primary_detection(
        image_width=120,
        image_height=200,
        depth_state=depth,
    )

    assert detection.xyxy == (70, 20, 110, 180)
    assert detection.normalized_x_offset(120) == 0.5


def test_build_sensor_primary_detection_can_rotate_depth_region_180() -> None:
    depth = DepthProximityState(
        timestamp_s=10.0,
        has_close_object=True,
        close_fraction=0.1,
        nearest_distance_m=1.1,
        close_xyxy=(70, 20, 110, 180),
        lateral_offset=0.6,
    )

    detection = _build_sensor_primary_detection(
        image_width=120,
        image_height=200,
        depth_state=depth,
        rotate_depth_180=True,
    )

    assert detection.xyxy == (10, 20, 50, 180)
    assert detection.normalized_x_offset(120) == -0.5


def test_sensor_primary_lateral_offset_uses_depth_estimate_with_rotation_and_gain() -> None:
    depth = DepthProximityState(
        timestamp_s=10.0,
        has_close_object=True,
        close_fraction=0.1,
        nearest_distance_m=1.1,
        close_xyxy=(70, 20, 110, 180),
        lateral_offset=0.35,
    )

    assert _sensor_primary_lateral_offset(
        depth,
        fallback_detection=Detection(
            class_name="person",
            confidence=1.0,
            xyxy=(10, 20, 50, 180),
        ),
        image_width=120,
        rotate_depth_180=True,
        gain=2.0,
    ) == -0.7


def test_build_screen_capture_config_uses_configured_duration() -> None:
    settings = SimpleNamespace(
        screen_local_video_path="/tmp/near_person.mp4",
        screen_remote_video_path="/tmp/remote_near_person.mp4",
        screen_pc3_target="agi@agibot-pc3",
        screen_video_duration_seconds=5.0,
    )

    config = _build_screen_capture_config(settings)

    assert config.duration_seconds == 5.0


def test_read_pointcloud_xyz_reads_float32_fields_by_offset() -> None:
    msg = SimpleNamespace(
        fields=[
            SimpleNamespace(name="x", offset=0, datatype=7),
            SimpleNamespace(name="intensity", offset=4, datatype=7),
            SimpleNamespace(name="y", offset=8, datatype=7),
            SimpleNamespace(name="z", offset=12, datatype=7),
        ],
        is_bigendian=False,
        data=b"".join(
            [
                struct.pack("<ffff", 1.0, 99.0, 2.0, 3.0),
                struct.pack("<ffff", 4.0, 88.0, 5.0, 6.0),
            ]
        ),
        width=2,
        height=1,
        point_step=16,
    )

    assert list(read_pointcloud_xyz(msg)) == [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]


def test_read_pointcloud_xyz_honors_row_step_padding() -> None:
    msg = SimpleNamespace(
        fields=[
            SimpleNamespace(name="x", offset=0, datatype=7),
            SimpleNamespace(name="y", offset=4, datatype=7),
            SimpleNamespace(name="z", offset=8, datatype=7),
        ],
        is_bigendian=False,
        data=b"".join(
            [
                struct.pack("<fff", 1.0, 2.0, 3.0),
                b"PAD!",
                struct.pack("<fff", 4.0, 5.0, 6.0),
                b"PAD!",
            ]
        ),
        width=1,
        height=2,
        point_step=12,
        row_step=16,
    )

    assert list(read_pointcloud_xyz(msg)) == [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]


def test_rgb_callback_skips_rgb_decode_when_prefilter_blocks_detector() -> None:
    class FailingBridge:
        def imgmsg_to_cv2(self, msg: Any, desired_encoding: str) -> np.ndarray:
            raise AssertionError("RGB frame should not be decoded when prefilter blocks")

    class FailingDetector:
        def detect(self, bgr: np.ndarray) -> list[Detection]:
            raise AssertionError("detector should not run when prefilter blocks")

    node = object.__new__(AgiBotNearPersonNode)
    node.settings = SimpleNamespace(bbox_distance_fallback_enabled=True)
    node.bridge = FailingBridge()
    node.detector = FailingDetector()
    node.latest_depth = None
    node._should_run_detector_after_prefilters = lambda *, now_s: False

    node._rgb_callback(SimpleNamespace())


def test_rgb_callback_sensor_primary_skips_detector_and_uses_current_rgb_frame(
    monkeypatch,
) -> None:
    import guy_detector.ros_node as ros_node

    monkeypatch.setattr(ros_node.time, "monotonic", lambda: 10.1)

    class FakeBridge:
        def __init__(self) -> None:
            self.image = np.zeros((80, 120, 3), dtype=np.uint8)

        def imgmsg_to_cv2(self, msg: Any, desired_encoding: str) -> np.ndarray:
            assert desired_encoding == "bgr8"
            return self.image

    class FailingDetector:
        def detect(self, bgr: np.ndarray) -> list[Detection]:
            raise AssertionError("detector should not run in sensor primary mode")

    class FakeEvents:
        def __init__(self) -> None:
            self.written: list[NearPersonEvent] = []

        def write(self, event: NearPersonEvent) -> None:
            self.written.append(event)

    actions: list[tuple[str, NearPersonEvent, np.ndarray, Detection]] = []

    node = object.__new__(AgiBotNearPersonNode)
    node.settings = SimpleNamespace(
        sensor_primary_detection_enabled=True,
        sensor_primary_detection_use_lidar=True,
        sensor_primary_detection_use_depth=False,
        sensor_primary_detection_mode="any",
        proximity_prefilter_stale_seconds=0.5,
        depth_prefilter_stale_seconds=0.5,
        bbox_distance_fallback_enabled=False,
        rgb_topic="/rgb",
        debug_log_detections=False,
        shirt_color_enabled=False,
        locomotion_yaw_enabled=False,
    )
    node.bridge = FakeBridge()
    node.detector = FailingDetector()
    node.gate = SimpleNamespace(
        update=lambda detection,
        proximity,
        *,
        now_s,
        shirt_color,
        image_width,
        lateral_offset_override=None: NearPersonEvent(
            timestamp_s=now_s,
            detection=detection,
            proximity=proximity,
            lateral_offset=(
                lateral_offset_override
                if lateral_offset_override is not None
                else detection.normalized_x_offset(image_width)
            ),
            shirt_color=shirt_color,
        ),
        reset=lambda: None,
    )
    node.events = FakeEvents()
    node.latest_depth = None
    node.latest_depth_proximity = None
    node.latest_lidar_proximity = LidarProximityState(
        timestamp_s=10.0,
        has_close_object=True,
        close_point_count=30,
        nearest_distance_m=1.3,
    )
    node.debug_state = None
    node._logged_first_rgb_frame = True
    node.get_logger = lambda: SimpleNamespace(info=lambda message: None)
    node._run_near_person_immediate_actions = (
        lambda event, bgr, detection, *, now_s, show_screen=True: actions.append(
            ("immediate", event, bgr, detection)
        )
    )
    node._run_near_person_after_yaw_actions = lambda event, *, now_s, preset_lateral_offset: None

    node._rgb_callback(SimpleNamespace())

    assert len(node.events.written) == 1
    assert len(actions) == 1
    _, event, bgr, detection = actions[0]
    assert bgr is node.bridge.image
    assert event.proximity.source == "lidar"
    assert event.proximity.distance_m == 1.3
    assert detection.xyxy == (0, 0, 120, 80)


def test_rgb_callback_sensor_primary_uses_depth_region_for_lateral_offset(
    monkeypatch,
) -> None:
    import guy_detector.ros_node as ros_node

    monkeypatch.setattr(ros_node.time, "monotonic", lambda: 10.1)

    class FakeBridge:
        def __init__(self) -> None:
            self.image = np.zeros((80, 120, 3), dtype=np.uint8)

        def imgmsg_to_cv2(self, msg: Any, desired_encoding: str) -> np.ndarray:
            return self.image

    class FailingDetector:
        def detect(self, bgr: np.ndarray) -> list[Detection]:
            raise AssertionError("detector should not run in sensor primary mode")

    class FakeEvents:
        def __init__(self) -> None:
            self.written: list[NearPersonEvent] = []

        def write(self, event: NearPersonEvent) -> None:
            self.written.append(event)

    actions: list[tuple[NearPersonEvent, Detection]] = []

    node = object.__new__(AgiBotNearPersonNode)
    node.settings = SimpleNamespace(
        sensor_primary_detection_enabled=True,
        sensor_primary_detection_use_lidar=False,
        sensor_primary_detection_use_depth=True,
        sensor_primary_detection_mode="any",
        proximity_prefilter_stale_seconds=0.5,
        depth_prefilter_stale_seconds=0.5,
        bbox_distance_fallback_enabled=False,
        rgb_topic="/rgb",
        debug_log_detections=False,
        shirt_color_enabled=False,
        locomotion_yaw_enabled=False,
    )
    node.bridge = FakeBridge()
    node.detector = FailingDetector()
    node.gate = SimpleNamespace(
        update=lambda detection,
        proximity,
        *,
        now_s,
        shirt_color,
        image_width,
        lateral_offset_override=None: NearPersonEvent(
            timestamp_s=now_s,
            detection=detection,
            proximity=proximity,
            lateral_offset=(
                lateral_offset_override
                if lateral_offset_override is not None
                else detection.normalized_x_offset(image_width)
            ),
            shirt_color=shirt_color,
        ),
        reset=lambda: None,
    )
    node.events = FakeEvents()
    node.latest_depth = None
    node.latest_lidar_proximity = None
    node.latest_depth_proximity = DepthProximityState(
        timestamp_s=10.0,
        has_close_object=True,
        close_fraction=0.1,
        nearest_distance_m=1.2,
        close_xyxy=(70, 10, 110, 70),
        lateral_offset=0.5,
    )
    node.debug_state = None
    node._logged_first_rgb_frame = True
    node.get_logger = lambda: SimpleNamespace(info=lambda message: None)
    node._run_near_person_immediate_actions = (
        lambda event, bgr, detection, *, now_s, show_screen=True: actions.append(
            (event, detection)
        )
    )
    node._run_near_person_after_yaw_actions = lambda event, *, now_s, preset_lateral_offset: None

    node._rgb_callback(SimpleNamespace())

    assert len(actions) == 1
    event, detection = actions[0]
    assert detection.xyxy == (70, 10, 110, 70)
    assert event.lateral_offset == 0.5


def test_rgb_callback_sensor_primary_speaks_before_locomotion_yaw(
    monkeypatch,
) -> None:
    import guy_detector.ros_node as ros_node

    monkeypatch.setattr(ros_node.time, "monotonic", lambda: 10.1)

    class FakeBridge:
        def __init__(self) -> None:
            self.image = np.zeros((80, 120, 3), dtype=np.uint8)

        def imgmsg_to_cv2(self, msg: Any, desired_encoding: str) -> np.ndarray:
            return self.image

    class FakeEvents:
        def write(self, event: NearPersonEvent) -> None:
            return None

    actions: list[str] = []

    node = object.__new__(AgiBotNearPersonNode)
    node.settings = SimpleNamespace(
        sensor_primary_detection_enabled=True,
        sensor_primary_detection_use_lidar=False,
        sensor_primary_detection_use_depth=True,
        sensor_primary_detection_mode="any",
        proximity_prefilter_stale_seconds=0.5,
        depth_prefilter_stale_seconds=0.5,
        bbox_distance_fallback_enabled=False,
        rgb_topic="/rgb",
        debug_log_detections=False,
        shirt_color_enabled=False,
        locomotion_yaw_enabled=True,
    )
    node.bridge = FakeBridge()
    node.gate = SimpleNamespace(
        update=lambda detection,
        proximity,
        *,
        now_s,
        shirt_color,
        image_width,
        lateral_offset_override=None: NearPersonEvent(
            timestamp_s=now_s,
            detection=detection,
            proximity=proximity,
            lateral_offset=(
                lateral_offset_override
                if lateral_offset_override is not None
                else detection.normalized_x_offset(image_width)
            ),
            shirt_color=shirt_color,
        ),
        reset=lambda: None,
    )
    node.events = FakeEvents()
    node.latest_depth = None
    node.latest_lidar_proximity = None
    node.latest_depth_proximity = DepthProximityState(
        timestamp_s=10.0,
        has_close_object=True,
        close_fraction=0.1,
        nearest_distance_m=1.2,
        close_xyxy=(70, 10, 110, 70),
        lateral_offset=0.5,
    )
    node.debug_state = None
    node._logged_first_rgb_frame = True
    node.get_logger = lambda: SimpleNamespace(info=lambda message: None)
    node.locomotion_yaw = SimpleNamespace(
        update=lambda lateral_offset: SimpleNamespace(angular_velocity=-0.2)
    )
    node._run_near_person_immediate_actions = (
        lambda event, bgr, detection, *, now_s, show_screen=True: actions.append("tts")
    )
    node._start_locomotion_yaw = lambda angular_velocity, *, now_s: actions.append("yaw")

    node._rgb_callback(SimpleNamespace())

    assert actions == ["tts", "yaw"]


def test_rgb_callback_sensor_primary_runs_after_yaw_actions_immediately_when_no_turn(
    monkeypatch,
) -> None:
    import guy_detector.ros_node as ros_node

    monkeypatch.setattr(ros_node.time, "monotonic", lambda: 10.1)

    class FakeBridge:
        def __init__(self) -> None:
            self.image = np.zeros((80, 120, 3), dtype=np.uint8)

        def imgmsg_to_cv2(self, msg: Any, desired_encoding: str) -> np.ndarray:
            return self.image

    class FakeEvents:
        def write(self, event: NearPersonEvent) -> None:
            return None

    actions: list[tuple[str, float]] = []
    yaw_calls: list[float] = []

    node = object.__new__(AgiBotNearPersonNode)
    node.settings = SimpleNamespace(
        sensor_primary_detection_enabled=True,
        sensor_primary_detection_use_lidar=False,
        sensor_primary_detection_use_depth=True,
        sensor_primary_detection_mode="any",
        sensor_primary_depth_rotate_180=False,
        sensor_primary_lateral_offset_gain=1.0,
        sensor_primary_screen_yolo_enabled=False,
        proximity_prefilter_stale_seconds=0.5,
        depth_prefilter_stale_seconds=0.5,
        bbox_distance_fallback_enabled=False,
        rgb_topic="/rgb",
        debug_log_detections=False,
        shirt_color_enabled=False,
        locomotion_yaw_enabled=True,
    )
    node.bridge = FakeBridge()
    node.gate = SimpleNamespace(
        update=lambda detection,
        proximity,
        *,
        now_s,
        shirt_color,
        image_width,
        lateral_offset_override=None: NearPersonEvent(
            timestamp_s=now_s,
            detection=detection,
            proximity=proximity,
            lateral_offset=(
                lateral_offset_override
                if lateral_offset_override is not None
                else detection.normalized_x_offset(image_width)
            ),
            shirt_color=shirt_color,
        ),
        reset=lambda: None,
    )
    node.events = FakeEvents()
    node.latest_depth = None
    node.latest_lidar_proximity = None
    node.latest_depth_proximity = DepthProximityState(
        timestamp_s=10.0,
        has_close_object=True,
        close_fraction=0.1,
        nearest_distance_m=1.2,
        close_xyxy=(50, 10, 70, 70),
        lateral_offset=0.05,
    )
    node.debug_state = None
    node._logged_first_rgb_frame = True
    node.get_logger = lambda: SimpleNamespace(info=lambda message: None)
    node.locomotion_yaw = SimpleNamespace(
        update=lambda lateral_offset: SimpleNamespace(angular_velocity=0.0)
    )
    node._pending_after_yaw_action = None
    node._run_near_person_immediate_actions = (
        lambda event, bgr, detection, *, now_s, show_screen=True: actions.append(
            ("immediate", now_s)
        )
    )
    node._run_near_person_after_yaw_actions = (
        lambda event, *, now_s, preset_lateral_offset: actions.append(("after_yaw", now_s))
    )
    node._start_locomotion_yaw = lambda angular_velocity, *, now_s: yaw_calls.append(
        angular_velocity
    )

    node._rgb_callback(SimpleNamespace())

    assert actions == [("immediate", 10.1), ("after_yaw", 10.1)]
    assert yaw_calls == []
    assert node._pending_after_yaw_action is None


def test_rgb_callback_sensor_primary_schedules_yolo_screen_without_blocking(
    monkeypatch,
) -> None:
    import guy_detector.ros_node as ros_node

    monkeypatch.setattr(ros_node.time, "monotonic", lambda: 10.1)

    class FakeBridge:
        def __init__(self) -> None:
            self.image = np.zeros((80, 120, 3), dtype=np.uint8)

        def imgmsg_to_cv2(self, msg: Any, desired_encoding: str) -> np.ndarray:
            return self.image

    class FakeEvents:
        def write(self, event: NearPersonEvent) -> None:
            return None

    class FakeExecutor:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []

        def submit(self, fn: Any, *args: Any, **kwargs: Any) -> None:
            self.calls.append((fn, args, kwargs))

    actions: list[str] = []
    executor = FakeExecutor()

    node = object.__new__(AgiBotNearPersonNode)
    node.settings = SimpleNamespace(
        sensor_primary_detection_enabled=True,
        sensor_primary_detection_use_lidar=False,
        sensor_primary_detection_use_depth=True,
        sensor_primary_detection_mode="any",
        sensor_primary_depth_rotate_180=False,
        sensor_primary_lateral_offset_gain=1.0,
        sensor_primary_screen_yolo_enabled=True,
        proximity_prefilter_stale_seconds=0.5,
        depth_prefilter_stale_seconds=0.5,
        bbox_distance_fallback_enabled=False,
        rgb_topic="/rgb",
        debug_log_detections=False,
        shirt_color_enabled=False,
        locomotion_yaw_enabled=False,
        screen_enabled=True,
        tts_enabled=True,
    )
    node.bridge = FakeBridge()
    node.gate = SimpleNamespace(
        update=lambda detection,
        proximity,
        *,
        now_s,
        shirt_color,
        image_width,
        lateral_offset_override=None: NearPersonEvent(
            timestamp_s=now_s,
            detection=detection,
            proximity=proximity,
            lateral_offset=lateral_offset_override or detection.normalized_x_offset(image_width),
            shirt_color=shirt_color,
        ),
        reset=lambda: None,
    )
    node.events = FakeEvents()
    node.latest_depth = None
    node.latest_lidar_proximity = None
    node.latest_depth_proximity = DepthProximityState(
        timestamp_s=10.0,
        has_close_object=True,
        close_fraction=0.1,
        nearest_distance_m=1.2,
        close_xyxy=(70, 10, 110, 70),
        lateral_offset=0.5,
    )
    node.debug_state = None
    node._logged_first_rgb_frame = True
    node._sensor_primary_screen_executor = executor
    node.get_logger = lambda: SimpleNamespace(
        info=lambda message: None, warning=lambda message: None
    )
    node._speak = lambda text, *, trace_id: actions.append("tts")
    node._show_person_on_screen = lambda bgr, detection, *, now_s: actions.append("screen")
    node._run_near_person_after_yaw_actions = lambda event, *, now_s, preset_lateral_offset: None

    node._rgb_callback(SimpleNamespace())

    assert actions == ["tts"]
    assert len(executor.calls) == 1


def test_sensor_primary_yolo_screen_uses_yolo_detection() -> None:
    class FakeDetector:
        def detect(self, bgr: np.ndarray) -> list[Detection]:
            return [
                Detection(class_name="chair", confidence=0.9, xyxy=(0, 0, 20, 20)),
                Detection(class_name="person", confidence=0.8, xyxy=(30, 5, 70, 75)),
            ]

    image = np.zeros((80, 120, 3), dtype=np.uint8)
    fallback = Detection(class_name="person", confidence=1.0, xyxy=(0, 0, 120, 80))
    shown: list[Detection] = []

    node = object.__new__(AgiBotNearPersonNode)
    node.detector = FakeDetector()
    node.settings = SimpleNamespace(debug_log_detections=False)
    node.get_logger = lambda: SimpleNamespace(
        info=lambda message: None, warning=lambda message: None
    )
    node._show_person_on_screen = lambda bgr, detection, *, now_s: shown.append(detection)

    node._run_sensor_primary_yolo_screen(image, fallback_detection=fallback, now_s=10.1)

    assert shown == [Detection(class_name="person", confidence=0.8, xyxy=(30, 5, 70, 75))]


def test_sensor_primary_yolo_screen_falls_back_when_no_person() -> None:
    class FakeDetector:
        def detect(self, bgr: np.ndarray) -> list[Detection]:
            return [Detection(class_name="chair", confidence=0.9, xyxy=(0, 0, 20, 20))]

    image = np.zeros((80, 120, 3), dtype=np.uint8)
    fallback = Detection(class_name="person", confidence=1.0, xyxy=(0, 0, 120, 80))
    shown: list[Detection] = []

    node = object.__new__(AgiBotNearPersonNode)
    node.detector = FakeDetector()
    node.settings = SimpleNamespace(debug_log_detections=False)
    node.get_logger = lambda: SimpleNamespace(
        info=lambda message: None, warning=lambda message: None
    )
    node._show_person_on_screen = lambda bgr, detection, *, now_s: shown.append(detection)

    node._run_sensor_primary_yolo_screen(image, fallback_detection=fallback, now_s=10.1)

    assert shown == [fallback]


def test_locomotion_yaw_burst_publishes_until_duration_then_stops(monkeypatch) -> None:
    import guy_detector.ros_node as ros_node

    class FakeHeader:
        def __init__(self) -> None:
            self.stamp = None

    class FakeMessageHeader:
        def __init__(self) -> None:
            self.stamp = None

    class FakeVelocity:
        def __init__(self) -> None:
            self.header = None
            self.source = ""
            self.forward_velocity = 0.0
            self.lateral_velocity = 0.0
            self.angular_velocity = 0.0

    class FakeClock:
        def now(self) -> Any:
            return SimpleNamespace(to_msg=lambda: FakeHeader())

    class FakePublisher:
        def __init__(self) -> None:
            self.messages: list[FakeVelocity] = []

        def publish(self, msg: FakeVelocity) -> None:
            self.messages.append(msg)

    monkeypatch.setattr(ros_node, "McLocomotionVelocity", FakeVelocity)
    monkeypatch.setattr(ros_node, "MessageHeader", FakeMessageHeader)

    node = object.__new__(AgiBotNearPersonNode)
    node.settings = SimpleNamespace(
        locomotion_yaw_command_duration_seconds=1.0,
        locomotion_yaw_source="node",
        locomotion_yaw_register_input_source=False,
        debug_log_detections=True,
    )
    node.locomotion_yaw_pub = FakePublisher()
    node._locomotion_yaw_angular_velocity = 0.0
    node._locomotion_yaw_until_s = None
    node._locomotion_yaw_stop_sent = True
    node.get_clock = lambda: FakeClock()
    node._register_locomotion_input_source_if_needed = lambda: None
    node.get_logger = lambda: SimpleNamespace(info=lambda message: None)

    node._start_locomotion_yaw(angular_velocity=-0.2, now_s=10.0)
    node._locomotion_yaw_timer_callback(now_s=10.5)
    node._locomotion_yaw_timer_callback(now_s=11.1)
    node._locomotion_yaw_timer_callback(now_s=11.2)

    assert [msg.angular_velocity for msg in node.locomotion_yaw_pub.messages] == [
        -0.2,
        -0.2,
        0.0,
    ]


def test_locomotion_yaw_start_refreshes_input_source_registration(monkeypatch) -> None:
    import guy_detector.ros_node as ros_node

    class FakeHeader:
        def __init__(self) -> None:
            self.stamp = None

    class FakeMessageHeader:
        def __init__(self) -> None:
            self.stamp = None

    class FakeVelocity:
        def __init__(self) -> None:
            self.header = None
            self.source = ""
            self.forward_velocity = 0.0
            self.lateral_velocity = 0.0
            self.angular_velocity = 0.0

    class FakeClock:
        def now(self) -> Any:
            return SimpleNamespace(to_msg=lambda: FakeHeader())

    class FakePublisher:
        def publish(self, msg: FakeVelocity) -> None:
            return None

    class FakeInputAction:
        def __init__(self) -> None:
            self.value = 0

    class FakeInputSource:
        def __init__(self) -> None:
            self.name = ""
            self.priority = 0
            self.timeout = 0

    class FakeSetMcInputSource:
        class Request:
            def __init__(self) -> None:
                self.request = SimpleNamespace(header=SimpleNamespace(stamp=None))
                self.action = None
                self.input_source = None

    class FakeClient:
        def __init__(self) -> None:
            self.requests: list[Any] = []

        def service_is_ready(self) -> bool:
            return True

        def call_async(self, request: Any) -> None:
            self.requests.append(request)

    monkeypatch.setattr(ros_node, "McLocomotionVelocity", FakeVelocity)
    monkeypatch.setattr(ros_node, "MessageHeader", FakeMessageHeader)
    monkeypatch.setattr(ros_node, "McInputAction", FakeInputAction)
    monkeypatch.setattr(ros_node, "McInputSource", FakeInputSource)
    monkeypatch.setattr(ros_node, "SetMcInputSource", FakeSetMcInputSource)

    node = object.__new__(AgiBotNearPersonNode)
    node.settings = SimpleNamespace(
        locomotion_yaw_command_duration_seconds=2.5,
        locomotion_yaw_source="node",
        locomotion_yaw_register_input_source=True,
        locomotion_yaw_input_source_priority=40,
        locomotion_yaw_input_source_timeout_ms=3000,
        debug_log_detections=False,
    )
    node.locomotion_yaw_pub = FakePublisher()
    node.locomotion_input_source_client = FakeClient()
    node._locomotion_input_source_registered = True
    node._locomotion_yaw_angular_velocity = 0.0
    node._locomotion_yaw_until_s = None
    node._locomotion_yaw_stop_sent = True
    node.get_clock = lambda: FakeClock()
    node.get_logger = lambda: SimpleNamespace(warning=lambda message: None)

    node._start_locomotion_yaw(angular_velocity=-0.7, now_s=10.0)

    assert len(node.locomotion_input_source_client.requests) == 1
    request = node.locomotion_input_source_client.requests[0]
    assert request.input_source.timeout == 3000


def test_tts_and_screen_run_with_locomotion_yaw_and_motion_waits_for_stable_stand(
    monkeypatch,
) -> None:
    import guy_detector.ros_node as ros_node

    class FakeHeader:
        def __init__(self) -> None:
            self.stamp = None

    class FakeMessageHeader:
        def __init__(self) -> None:
            self.stamp = None

    class FakeVelocity:
        def __init__(self) -> None:
            self.header = None
            self.source = ""
            self.forward_velocity = 0.0
            self.lateral_velocity = 0.0
            self.angular_velocity = 0.0

    class FakeClock:
        def now(self) -> Any:
            return SimpleNamespace(to_msg=lambda: FakeHeader())

    class FakePublisher:
        def __init__(self) -> None:
            self.messages: list[FakeVelocity] = []

        def publish(self, msg: FakeVelocity) -> None:
            self.messages.append(msg)

    class FakeCommonRequest:
        def __init__(self) -> None:
            self.header = SimpleNamespace(stamp=None)

    class FakeGetMcAction:
        class Request:
            def __init__(self) -> None:
                self.request = None

    class FakeFuture:
        def __init__(self, response: Any) -> None:
            self._response = response

        def done(self) -> bool:
            return True

        def result(self) -> Any:
            return self._response

    class FakeMcActionClient:
        def __init__(self, responses: list[Any]) -> None:
            self.responses = responses
            self.requests: list[Any] = []

        def service_is_ready(self) -> bool:
            return True

        def call_async(self, request: Any) -> FakeFuture:
            self.requests.append(request)
            return FakeFuture(self.responses.pop(0))

    def mc_action_response(action_desc: str, status: int) -> Any:
        return SimpleNamespace(
            info=SimpleNamespace(
                action_desc=action_desc,
                status=SimpleNamespace(value=status),
            )
        )

    monkeypatch.setattr(ros_node, "McLocomotionVelocity", FakeVelocity)
    monkeypatch.setattr(ros_node, "MessageHeader", FakeMessageHeader)
    monkeypatch.setattr(ros_node, "CommonRequest", FakeCommonRequest)
    monkeypatch.setattr(ros_node, "GetMcAction", FakeGetMcAction)

    detection = Detection(class_name="person", confidence=0.9, xyxy=(60, 10, 100, 90))
    event = NearPersonEvent(
        timestamp_s=10.0,
        detection=detection,
        proximity=ProximityEstimate(distance_m=1.0, depth_confidence=1.0),
        lateral_offset=0.6,
        shirt_color=None,
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    actions: list[str] = []
    motion_offsets: list[float] = []
    motion_times: list[float] = []

    node = object.__new__(AgiBotNearPersonNode)
    node.settings = SimpleNamespace(
        locomotion_yaw_command_duration_seconds=1.0,
        locomotion_yaw_source="node",
        locomotion_yaw_register_input_source=False,
        debug_log_detections=False,
        locomotion_yaw_post_stop_action_delay_seconds=0.5,
        locomotion_yaw_wait_for_stand_enabled=True,
        locomotion_yaw_stand_action_desc="STAND_DEFAULT",
        locomotion_yaw_stand_status_value=100,
        locomotion_yaw_stand_wait_timeout_seconds=3.0,
        locomotion_yaw_stand_poll_seconds=0.1,
        tts_enabled=True,
        screen_enabled=True,
        preset_motion_enabled=True,
        yaw_enabled=False,
    )
    node.locomotion_yaw_pub = FakePublisher()
    node._locomotion_yaw_angular_velocity = 0.0
    node._locomotion_yaw_until_s = None
    node._locomotion_yaw_stop_sent = True
    node._pending_after_yaw_action = None
    node._pending_after_yaw_stand_wait_started_at_s = None
    node._pending_after_yaw_mc_action_future = None
    node._pending_after_yaw_last_mc_action_query_at_s = None
    node.mc_action_client = FakeMcActionClient(
        [
            mc_action_response("LOCOMOTION_DEFAULT", 100),
            mc_action_response("STAND_DEFAULT", 100),
        ]
    )
    node.get_clock = lambda: FakeClock()
    node._register_locomotion_input_source_if_needed = lambda: None
    node.get_logger = lambda: SimpleNamespace(info=lambda message: None)
    node._speak = lambda text, *, trace_id: actions.append("tts")
    node._show_person_on_screen = lambda bgr, detection, *, now_s: actions.append("screen")

    def trigger_preset_motion(lateral_offset: float, *, now_s: float) -> None:
        actions.append("motion")
        motion_offsets.append(lateral_offset)
        motion_times.append(now_s)

    node._trigger_preset_motion = trigger_preset_motion

    node._pending_after_yaw_action = (event, image, detection, 10.0)
    node._start_locomotion_yaw(angular_velocity=-0.2, now_s=10.0)
    node._run_near_person_immediate_actions(event, image, detection, now_s=10.0)

    assert actions == ["tts", "screen"]

    node._locomotion_yaw_timer_callback(now_s=10.5)

    assert actions == ["tts", "screen"]

    node._locomotion_yaw_timer_callback(now_s=11.1)

    assert [msg.angular_velocity for msg in node.locomotion_yaw_pub.messages] == [
        -0.2,
        -0.2,
        0.0,
    ]
    assert actions == ["tts", "screen"]
    assert node._pending_after_yaw_action is not None

    node._locomotion_yaw_timer_callback(now_s=11.5)

    assert actions == ["tts", "screen"]
    assert node._pending_after_yaw_action is not None

    node._locomotion_yaw_timer_callback(now_s=11.6)

    assert actions == ["tts", "screen"]
    assert len(node.mc_action_client.requests) == 1
    assert node._pending_after_yaw_action is not None

    node._locomotion_yaw_timer_callback(now_s=11.71)

    assert actions == ["tts", "screen"]
    assert len(node.mc_action_client.requests) == 2
    assert node._pending_after_yaw_action is not None

    node._locomotion_yaw_timer_callback(now_s=11.82)

    assert actions == ["tts", "screen", "motion"]
    assert len(node.mc_action_client.requests) == 2
    assert motion_offsets == [0.6]
    assert motion_times == [11.82]
    assert node._pending_after_yaw_action is None


def test_speak_unmutes_before_playing_tts(monkeypatch) -> None:
    import guy_detector.ros_node as ros_node

    class FakeSetMute:
        class Request:
            def __init__(self) -> None:
                self.request = SimpleNamespace(header=SimpleNamespace(stamp=None))
                self.is_mute = True

    class FakePlayTts:
        class Request:
            def __init__(self) -> None:
                self.tts_req = SimpleNamespace(
                    text="",
                    domain="",
                    trace_id="",
                    is_interrupted=False,
                    priority_weight=0,
                    priority_level=SimpleNamespace(value=0),
                )

    class FakeClock:
        def now(self) -> Any:
            return SimpleNamespace(to_msg=lambda: "stamp")

    class FakeClient:
        def __init__(self) -> None:
            self.requests: list[Any] = []

        def service_is_ready(self) -> bool:
            return True

        def call_async(self, request: Any) -> None:
            self.requests.append(request)

    monkeypatch.setattr(ros_node, "SetMute", FakeSetMute)
    monkeypatch.setattr(ros_node, "PlayTts", FakePlayTts)

    mute_client = FakeClient()
    tts_client = FakeClient()
    node = object.__new__(AgiBotNearPersonNode)
    node.settings = SimpleNamespace(tts_unmute_before_speaking=True)
    node.mute_client = mute_client
    node.tts_client = tts_client
    node.get_clock = lambda: FakeClock()

    node._speak("Please step back.", trace_id="near-person-test")

    assert len(mute_client.requests) == 1
    assert mute_client.requests[0].is_mute is False
    assert len(tts_client.requests) == 1
    assert tts_client.requests[0].tts_req.text == "Please step back."
