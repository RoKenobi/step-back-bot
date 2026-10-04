from __future__ import annotations

import argparse
import struct
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from guy_detector.bbox_proximity import estimate_bbox_proximity
from guy_detector.color import estimate_shirt_color
from guy_detector.config import Settings, load_settings
from guy_detector.debug_server import DebugDetection, DebugFrameState, DebugHttpServer
from guy_detector.depth import estimate_person_distance
from guy_detector.depth_prefilter import (
    DepthProximityState,
    estimate_depth_proximity,
    should_run_detector_for_depth_prefilter,
)
from guy_detector.detector import UltralyticsYoloDetector
from guy_detector.events import JsonlEventWriter
from guy_detector.gate import ProximityGate
from guy_detector.head_yaw import HeadYawController
from guy_detector.lidar_prefilter import (
    LidarProximityState,
    estimate_pointcloud_proximity,
    should_run_detector_for_combined_prefilters,
    should_run_detector_for_prefilter,
)
from guy_detector.locomotion_yaw import LocomotionYawController
from guy_detector.models import Detection, NearPersonEvent, ProximityEstimate
from guy_detector.preset_motion import PresetMotionSelector
from guy_detector.remote_detector import RemoteDetectorError, RemoteHttpDetector
from guy_detector.screen import ScreenCaptureConfig, transfer_to_pc3, write_detection_video
from guy_detector.speech import build_speech_line, build_tts_fields

try:
    import rclpy
    from aimdk_msgs.msg import (
        CommonRequest,
        JointCommand,
        JointCommandArray,
        McControlArea,
        McInputAction,
        McInputSource,
        McLocomotionVelocity,
        McPresetMotion,
        MessageHeader,
    )
    from aimdk_msgs.srv import (
        GetMcAction,
        PlayTts,
        PlayVideo,
        SetMcInputSource,
        SetMcPresetMotion,
        SetMute,
    )
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import Image, PointCloud2
except ImportError:
    rclpy = None
    Node = object
    Image = object
    PointCloud2 = object
    CvBridge = None
    CommonRequest = None
    GetMcAction = None
    PlayTts = None
    PlayVideo = None
    SetMcPresetMotion = None
    SetMute = None
    JointCommand = None
    JointCommandArray = None
    McControlArea = None
    McInputAction = None
    McInputSource = None
    McLocomotionVelocity = None
    McPresetMotion = None
    MessageHeader = None
    QoSHistoryPolicy = None
    QoSProfile = None
    QoSReliabilityPolicy = None
    SetMcInputSource = None


class AgiBotNearPersonNode(Node):  # type: ignore[misc]
    def __init__(self, settings: Settings) -> None:
        if rclpy is None:
            raise RuntimeError("ROS2 dependencies are required to run AgiBotNearPersonNode")
        super().__init__("guy_detector")
        self.settings = settings
        self.bridge = CvBridge()
        self.detector = _build_detector(settings)
        self.gate = ProximityGate(
            threshold_m=settings.proximity_threshold_m,
            required_consecutive_frames=settings.required_consecutive_frames,
            cooldown_seconds=settings.speech_cooldown_seconds,
        )
        self.head_yaw = HeadYawController(
            max_yaw_rad=settings.max_head_yaw_rad,
            stop_tolerance=settings.yaw_stop_tolerance,
        )
        self.locomotion_yaw = LocomotionYawController(
            max_angular_velocity=settings.locomotion_yaw_max_angular_velocity,
            min_angular_velocity=settings.locomotion_yaw_min_angular_velocity,
            stop_tolerance=settings.locomotion_yaw_stop_tolerance,
            invert=settings.locomotion_yaw_invert,
        )
        self.preset_motion_selector = PresetMotionSelector(
            center_tolerance=settings.preset_motion_center_tolerance,
        )
        self.events = JsonlEventWriter(settings.event_jsonl_path)
        self.screen_config = _build_screen_capture_config(settings)
        self._last_screen_at: float | None = None
        self._sensor_primary_screen_executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1)
            if settings.sensor_primary_screen_yolo_enabled
            else None
        )
        self._last_preset_motion_at: float | None = None
        self._locomotion_yaw_angular_velocity = 0.0
        self._locomotion_yaw_until_s: float | None = None
        self._locomotion_yaw_stop_sent = True
        self._pending_after_yaw_action: (
            tuple[NearPersonEvent, np.ndarray, Any, float] | None
        ) = None
        self._pending_after_yaw_ready_at_s: float | None = None
        self._pending_after_yaw_stand_wait_started_at_s: float | None = None
        self._pending_after_yaw_mc_action_future: Any | None = None
        self._pending_after_yaw_last_mc_action_query_at_s: float | None = None
        self.latest_lidar_proximity: LidarProximityState | None = None
        self.latest_depth_proximity: DepthProximityState | None = None
        self.debug_state: DebugFrameState | None = None
        self.debug_server: DebugHttpServer | None = None
        if settings.debug_server_enabled:
            self.debug_state = DebugFrameState(
                max_dimension=settings.debug_frame_max_dimension,
                depth_near_min_m=settings.depth_prefilter_min_distance_m,
                depth_near_max_m=settings.depth_prefilter_max_distance_m,
            )
            self.debug_server = DebugHttpServer(
                self.debug_state,
                host=settings.debug_server_host,
                port=settings.debug_server_port,
            )
            self.debug_server.start()
            self.get_logger().info(
                f"debug viewer listening on http://{settings.debug_server_host}:{settings.debug_server_port}"
            )
        self.latest_depth: np.ndarray | None = None
        self.latest_debug_depth: np.ndarray | None = None
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        if settings.depth_topic is not None:
            self.create_subscription(Image, settings.depth_topic, self._depth_callback, qos)
        if settings.depth_prefilter_enabled or (
            settings.sensor_primary_detection_enabled
            and settings.sensor_primary_detection_use_depth
        ):
            self.create_subscription(
                Image,
                settings.depth_prefilter_topic,
                self._depth_prefilter_callback,
                qos,
            )
        if settings.proximity_prefilter_enabled or (
            settings.sensor_primary_detection_enabled
            and settings.sensor_primary_detection_use_lidar
        ):
            self.create_subscription(
                PointCloud2,
                settings.proximity_prefilter_topic,
                self._pointcloud_callback,
                qos,
            )
        self.create_subscription(Image, settings.rgb_topic, self._rgb_callback, qos)
        self.tts_client = self.create_client(PlayTts, settings.tts_service)
        self.mute_client = self.create_client(SetMute, settings.audio_set_mute_service)
        self.play_video_client = self.create_client(PlayVideo, settings.play_video_service)
        self.preset_motion_client = self.create_client(
            SetMcPresetMotion,
            settings.preset_motion_service,
        )
        self.locomotion_input_source_client = self.create_client(
            SetMcInputSource,
            settings.locomotion_yaw_input_source_service,
        )
        self.mc_action_client = self.create_client(GetMcAction, settings.mc_action_service)
        self.head_yaw_pub = self.create_publisher(
            JointCommandArray,
            settings.head_yaw_command_topic,
            10,
        )
        self.locomotion_yaw_pub = self.create_publisher(
            McLocomotionVelocity,
            settings.locomotion_yaw_topic,
            10,
        )
        if settings.locomotion_yaw_enabled:
            self.create_timer(
                1.0 / settings.locomotion_yaw_publish_hz,
                self._locomotion_yaw_timer_callback,
            )
        self._locomotion_input_source_registered = False
        self._logged_first_rgb_frame = False
        mode = "RGB-only bbox-proximity" if settings.depth_topic is None else "RGB-D near-person"
        self.get_logger().info(f"guy-detector node started in {mode} mode")
        self.get_logger().info(
            "locomotion yaw config "
            f"enabled={settings.locomotion_yaw_enabled} "
            f"topic={settings.locomotion_yaw_topic} "
            f"source={settings.locomotion_yaw_source} "
            f"register_input_source={settings.locomotion_yaw_register_input_source} "
            f"max_angular={settings.locomotion_yaw_max_angular_velocity:.3f} "
            f"min_angular={settings.locomotion_yaw_min_angular_velocity:.3f} "
            f"stop_tolerance={settings.locomotion_yaw_stop_tolerance:.3f} "
            f"duration={settings.locomotion_yaw_command_duration_seconds:.2f}s"
        )

    def _depth_callback(self, msg: Any) -> None:
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.latest_debug_depth = self.latest_depth

    def _depth_prefilter_callback(self, msg: Any) -> None:
        now_s = time.monotonic()
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.latest_debug_depth = depth
        self.latest_depth_proximity = estimate_depth_proximity(
            depth,
            min_distance_m=self.settings.depth_prefilter_min_distance_m,
            max_distance_m=self.settings.depth_prefilter_max_distance_m,
            min_valid_fraction=self.settings.depth_prefilter_min_valid_fraction,
            now_s=now_s,
        )

    def _debug_depth_image(self) -> np.ndarray | None:
        if self.latest_debug_depth is not None:
            return self.latest_debug_depth
        return self.latest_depth

    def _debug_depth_shape(self) -> tuple[int, ...] | None:
        depth = self._debug_depth_image()
        return depth.shape if depth is not None else None

    def _pointcloud_callback(self, msg: Any) -> None:
        now_s = time.monotonic()
        self.latest_lidar_proximity = estimate_pointcloud_proximity(
            read_pointcloud_xyz(msg),
            max_distance_m=self.settings.proximity_prefilter_max_distance_m,
            min_points=self.settings.proximity_prefilter_min_points,
            now_s=now_s,
        )

    def _rgb_callback(self, msg: Any) -> None:
        now_s = time.monotonic()
        if (
            not getattr(self.settings, "sensor_primary_detection_enabled", False)
            and not self._should_run_detector_after_prefilters(now_s=now_s)
        ):
            return
        if (
            not self.settings.sensor_primary_detection_enabled
            and self.latest_depth is None
            and not self.settings.bbox_distance_fallback_enabled
        ):
            return
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        image_width = int(bgr.shape[1])
        image_height = int(bgr.shape[0])
        if not self._logged_first_rgb_frame:
            self.get_logger().info(
                f"first RGB frame received from {self.settings.rgb_topic}: "
                f"{image_width}x{image_height}"
            )
            self._logged_first_rgb_frame = True
        self._update_debug_view(bgr, [])
        if self.settings.sensor_primary_detection_enabled:
            self._run_sensor_primary_detection(
                bgr,
                now_s=now_s,
                image_width=image_width,
                image_height=image_height,
            )
            return
        try:
            detections = self.detector.detect(bgr)
        except RemoteDetectorError as exc:
            self.get_logger().warning(str(exc))
            return
        debug_detections: list[DebugDetection] = []
        near_targets = []
        last_event = None
        for detection in detections:
            proximity = _estimate_proximity(
                latest_depth=self.latest_depth,
                detection=detection,
                image_height=image_height,
                bbox_fallback_enabled=self.settings.bbox_distance_fallback_enabled,
                bbox_near_height_fraction=self.settings.bbox_near_height_fraction,
            )
            debug_detections.append(
                _debug_detection_tuple(detection, proximity, image_width=image_width)
            )
            is_near = _proximity_is_near(
                proximity,
                threshold_m=self.settings.proximity_threshold_m,
            )
            if self.settings.debug_log_detections and is_near:
                self.get_logger().info(
                    _format_detection_log(detection, proximity, image_width=image_width)
                )
            if is_near:
                near_targets.append((detection, proximity))
            if is_near and not self.settings.locomotion_yaw_enabled:
                if self.settings.debug_log_detections:
                    self.get_logger().info("near person found but locomotion yaw is disabled")
        target = _select_locomotion_yaw_target(near_targets)
        if target is None:
            self.gate.reset()
        else:
            detection, proximity = target
            lateral_offset = detection.normalized_x_offset(image_width)
            if self.settings.debug_log_detections:
                self.get_logger().info(
                    "selected near target "
                    f"bbox={list(detection.xyxy)} "
                    f"distance={proximity.distance_m} "
                    f"bbox_frac={proximity.bbox_height_fraction} "
                    f"lateral={lateral_offset:.2f}"
                )
            shirt_color = _estimate_optional_shirt_color(
                self.settings.shirt_color_enabled,
                bgr,
                detection,
            )
            event = self.gate.update(
                detection,
                proximity,
                now_s=now_s,
                shirt_color=shirt_color,
                image_width=image_width,
            )
            if event is not None:
                last_event = event
                self.events.write(event)
                self._run_near_person_immediate_actions(event, bgr, detection, now_s=now_s)
                self._run_or_defer_after_yaw_actions(event, bgr, detection, now_s=now_s)
        self._update_debug_view(bgr, debug_detections, event=last_event)

    def _run_sensor_primary_detection(
        self,
        bgr: np.ndarray,
        *,
        now_s: float,
        image_width: int,
        image_height: int,
    ) -> None:
        proximity = _select_sensor_primary_proximity(
            self.settings,
            latest_lidar=self.latest_lidar_proximity,
            latest_depth=self.latest_depth_proximity,
            now_s=now_s,
        )
        if proximity is None:
            self.gate.reset()
            self._update_debug_view(bgr, [])
            return

        detection = _build_sensor_primary_detection(
            image_width=image_width,
            image_height=image_height,
            depth_state=self.latest_depth_proximity,
            rotate_depth_180=getattr(self.settings, "sensor_primary_depth_rotate_180", False),
        )
        if self.settings.debug_log_detections:
            self.get_logger().info(
                _format_detection_log(detection, proximity, image_width=image_width)
            )
        event = self.gate.update(
            detection,
            proximity,
            now_s=now_s,
            shirt_color=None,
            image_width=image_width,
            lateral_offset_override=_sensor_primary_lateral_offset(
                self.latest_depth_proximity,
                fallback_detection=detection,
                image_width=image_width,
                rotate_depth_180=getattr(
                    self.settings,
                    "sensor_primary_depth_rotate_180",
                    False,
                ),
                gain=getattr(self.settings, "sensor_primary_lateral_offset_gain", 1.0),
            ),
        )
        debug_detections = [_debug_detection_tuple(detection, proximity, image_width=image_width)]
        if event is None:
            self._update_debug_view(bgr, debug_detections)
            return

        self.events.write(event)
        self._run_near_person_immediate_actions(
            event,
            bgr,
            detection,
            now_s=now_s,
            show_screen=not getattr(self.settings, "sensor_primary_screen_yolo_enabled", False),
        )
        self._schedule_sensor_primary_yolo_screen(bgr, fallback_detection=detection, now_s=now_s)
        self._run_or_defer_after_yaw_actions(event, bgr, detection, now_s=now_s)
        self._update_debug_view(bgr, debug_detections, event=event)

    def _run_or_defer_after_yaw_actions(
        self,
        event: NearPersonEvent,
        bgr: np.ndarray,
        detection: Detection,
        *,
        now_s: float,
    ) -> None:
        """Run after-yaw actions now, or start a body-yaw burst and run them when it ends."""
        angular_velocity = 0.0
        if self.settings.locomotion_yaw_enabled:
            angular_velocity = self.locomotion_yaw.update(
                lateral_offset=event.lateral_offset
            ).angular_velocity
        if angular_velocity == 0.0:
            self._run_near_person_after_yaw_actions(
                event,
                now_s=now_s,
                preset_lateral_offset=event.lateral_offset,
            )
            return
        self._pending_after_yaw_action = (event, bgr.copy(), detection, now_s)
        self._start_locomotion_yaw(angular_velocity, now_s=now_s)

    def _update_debug_view(
        self,
        bgr: np.ndarray,
        detections: list[DebugDetection],
        *,
        event: NearPersonEvent | None = None,
    ) -> None:
        if self.debug_state is None:
            return
        self.debug_state.update(
            bgr,
            detections=detections,
            depth_shape=self._debug_depth_shape(),
            depth_image=self._debug_depth_image(),
            event=event,
        )

    def _speak(self, text: str, *, trace_id: str) -> None:
        if PlayTts is None:
            return
        if not self.tts_client.service_is_ready():
            self.get_logger().warning("TTS service is not ready")
            return
        self._unmute_before_speaking()
        fields = build_tts_fields(text, trace_id=trace_id)
        if getattr(self.settings, "debug_log_detections", False):
            self.get_logger().info(f"sending TTS text={text!r}")
        request = PlayTts.Request()
        request.tts_req.text = str(fields["text"])
        request.tts_req.domain = str(fields["domain"])
        request.tts_req.trace_id = str(fields["trace_id"])
        request.tts_req.is_interrupted = bool(fields["is_interrupted"])
        request.tts_req.priority_weight = int(fields["priority_weight"])
        request.tts_req.priority_level.value = int(fields["priority_level"])
        self.tts_client.call_async(request)

    def _unmute_before_speaking(self) -> None:
        if not self.settings.tts_unmute_before_speaking:
            return
        if SetMute is None:
            return
        if not self.mute_client.service_is_ready():
            self.get_logger().warning("SetMute service is not ready")
            return
        request = SetMute.Request()
        request.request.header.stamp = self.get_clock().now().to_msg()
        request.is_mute = False
        self.mute_client.call_async(request)

    def _publish_head_yaw(self, target_position_rad: float) -> None:
        if JointCommand is None or JointCommandArray is None:
            return
        command = JointCommand()
        command.name = "head_yaw_joint"
        command.position = target_position_rad
        command.velocity = 0.0
        command.effort = 0.0
        command.stiffness = 20.0
        command.damping = 2.0
        msg = JointCommandArray()
        msg.joints.append(command)
        self.head_yaw_pub.publish(msg)

    def _start_locomotion_yaw(self, angular_velocity: float, *, now_s: float) -> None:
        self._locomotion_input_source_registered = False
        self._locomotion_yaw_angular_velocity = angular_velocity
        self._locomotion_yaw_until_s = (
            now_s + self.settings.locomotion_yaw_command_duration_seconds
        )
        self._locomotion_yaw_stop_sent = False
        self._pending_after_yaw_ready_at_s = None
        self._pending_after_yaw_stand_wait_started_at_s = None
        self._pending_after_yaw_mc_action_future = None
        self._pending_after_yaw_last_mc_action_query_at_s = None
        if getattr(self.settings, "debug_log_detections", False):
            self.get_logger().info(
                "starting locomotion yaw "
                f"angular={angular_velocity:.3f} "
                f"duration={self.settings.locomotion_yaw_command_duration_seconds:.2f}s"
            )
        self._publish_locomotion_yaw(angular_velocity)

    def _locomotion_yaw_timer_callback(self, now_s: float | None = None) -> None:
        now_s = time.monotonic() if now_s is None else now_s
        if self._locomotion_yaw_until_s is None:
            self._run_pending_after_yaw_action(now_s=now_s)
            return
        if now_s <= self._locomotion_yaw_until_s:
            self._publish_locomotion_yaw(self._locomotion_yaw_angular_velocity)
            return
        if not self._locomotion_yaw_stop_sent:
            self._publish_locomotion_yaw(0.0)
            self._locomotion_yaw_stop_sent = True
            delay_s = getattr(
                self.settings,
                "locomotion_yaw_post_stop_action_delay_seconds",
                0.0,
            )
            self._pending_after_yaw_ready_at_s = (
                now_s + delay_s
            )
        self._locomotion_yaw_angular_velocity = 0.0
        self._locomotion_yaw_until_s = None
        self._run_pending_after_yaw_action(now_s=now_s)

    def _run_pending_after_yaw_action(self, *, now_s: float) -> None:
        if getattr(self, "_pending_after_yaw_action", None) is None:
            return
        ready_at_s = getattr(self, "_pending_after_yaw_ready_at_s", None)
        if ready_at_s is not None and now_s < ready_at_s:
            return
        if not self._robot_is_ready_for_after_yaw_action(now_s=now_s):
            return
        event, _, _, _ = self._pending_after_yaw_action
        self._pending_after_yaw_action = None
        self._pending_after_yaw_ready_at_s = None
        self._pending_after_yaw_stand_wait_started_at_s = None
        self._pending_after_yaw_mc_action_future = None
        self._pending_after_yaw_last_mc_action_query_at_s = None
        self._run_near_person_after_yaw_actions(
            event,
            now_s=now_s,
            preset_lateral_offset=event.lateral_offset,
        )

    def _robot_is_ready_for_after_yaw_action(self, *, now_s: float) -> bool:
        if not getattr(self.settings, "locomotion_yaw_wait_for_stand_enabled", False):
            return True
        if GetMcAction is None or CommonRequest is None:
            return self._stand_wait_timed_out(now_s=now_s)
        if not getattr(self, "mc_action_client", None):
            return self._stand_wait_timed_out(now_s=now_s)
        if not self.mc_action_client.service_is_ready():
            if getattr(self.settings, "debug_log_detections", False):
                self.get_logger().warning("GetMcAction service is not ready")
            return self._stand_wait_timed_out(now_s=now_s)

        future = getattr(self, "_pending_after_yaw_mc_action_future", None)
        if future is not None:
            if not future.done():
                return self._stand_wait_timed_out(now_s=now_s)
            self._pending_after_yaw_mc_action_future = None
            response = future.result()
            if response is not None and self._mc_action_response_is_stable_stand(response):
                return True

        if self._stand_wait_timed_out(now_s=now_s):
            return True

        last_query_at_s = getattr(self, "_pending_after_yaw_last_mc_action_query_at_s", None)
        poll_s = getattr(self.settings, "locomotion_yaw_stand_poll_seconds", 0.2)
        if last_query_at_s is not None and now_s - last_query_at_s < poll_s:
            return False

        request = GetMcAction.Request()
        request.request = CommonRequest()
        request.request.header.stamp = self.get_clock().now().to_msg()
        self._pending_after_yaw_mc_action_future = self.mc_action_client.call_async(request)
        self._pending_after_yaw_last_mc_action_query_at_s = now_s
        return False

    def _stand_wait_timed_out(self, *, now_s: float) -> bool:
        started_at_s = getattr(self, "_pending_after_yaw_stand_wait_started_at_s", None)
        if started_at_s is None:
            self._pending_after_yaw_stand_wait_started_at_s = now_s
            return False
        timeout_s = getattr(self.settings, "locomotion_yaw_stand_wait_timeout_seconds", 3.0)
        return now_s - started_at_s >= timeout_s

    def _mc_action_response_is_stable_stand(self, response: Any) -> bool:
        info = getattr(response, "info", None)
        action_desc = getattr(info, "action_desc", None)
        status = getattr(info, "status", None)
        status_value = getattr(status, "value", None)
        return (
            action_desc == self.settings.locomotion_yaw_stand_action_desc
            and status_value == self.settings.locomotion_yaw_stand_status_value
        )

    def _run_near_person_immediate_actions(
        self,
        event: NearPersonEvent,
        bgr: np.ndarray,
        detection: Any,
        *,
        now_s: float,
        show_screen: bool = True,
    ) -> None:
        if self.settings.tts_enabled:
            self._speak(build_speech_line(event), trace_id=f"near-person-{now_s:.3f}")
        if show_screen and self.settings.screen_enabled:
            self._show_person_on_screen(bgr, detection, now_s=now_s)

    def _schedule_sensor_primary_yolo_screen(
        self,
        bgr: np.ndarray,
        *,
        fallback_detection: Detection,
        now_s: float,
    ) -> None:
        if not getattr(self.settings, "screen_enabled", False):
            return
        if not getattr(self.settings, "sensor_primary_screen_yolo_enabled", False):
            return
        executor = getattr(self, "_sensor_primary_screen_executor", None)
        if executor is None:
            self._run_sensor_primary_yolo_screen(
                bgr,
                fallback_detection=fallback_detection,
                now_s=now_s,
            )
            return
        executor.submit(
            self._run_sensor_primary_yolo_screen,
            bgr.copy(),
            fallback_detection=fallback_detection,
            now_s=now_s,
        )

    def _run_sensor_primary_yolo_screen(
        self,
        bgr: np.ndarray,
        *,
        fallback_detection: Detection,
        now_s: float,
    ) -> None:
        detection = fallback_detection
        try:
            yolo_detection = _select_screen_yolo_detection(self.detector.detect(bgr))
        except Exception as exc:
            self.get_logger().warning(f"sensor-primary YOLO screen detection failed: {exc}")
        else:
            if yolo_detection is not None:
                detection = yolo_detection
                if getattr(self.settings, "debug_log_detections", False):
                    self.get_logger().info(
                        "sensor-primary YOLO screen detection "
                        f"bbox={list(detection.xyxy)} "
                        f"conf={detection.confidence:.2f}"
                    )
            elif getattr(self.settings, "debug_log_detections", False):
                self.get_logger().info("sensor-primary YOLO found no person for screen")
        self._show_person_on_screen(bgr, detection, now_s=now_s)

    def _run_near_person_after_yaw_actions(
        self,
        event: NearPersonEvent,
        *,
        now_s: float,
        preset_lateral_offset: float,
    ) -> None:
        if self.settings.preset_motion_enabled:
            self._trigger_preset_motion(preset_lateral_offset, now_s=now_s)
        if self.settings.yaw_enabled:
            command = self.head_yaw.update(lateral_offset=event.lateral_offset)
            self._publish_head_yaw(command.target_position_rad)

    def _publish_locomotion_yaw(self, angular_velocity: float) -> None:
        if McLocomotionVelocity is None or MessageHeader is None:
            return
        self._register_locomotion_input_source_if_needed()
        msg = McLocomotionVelocity()
        msg.header = MessageHeader()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.source = self.settings.locomotion_yaw_source
        msg.forward_velocity = 0.0
        msg.lateral_velocity = 0.0
        msg.angular_velocity = angular_velocity
        self.locomotion_yaw_pub.publish(msg)
        if getattr(self.settings, "debug_log_detections", False) and angular_velocity != 0.0:
            self.get_logger().info(f"publishing locomotion yaw angular={angular_velocity:.3f}")

    def _register_locomotion_input_source_if_needed(self) -> None:
        if not self.settings.locomotion_yaw_register_input_source:
            return
        if self._locomotion_input_source_registered:
            return
        if SetMcInputSource is None or McInputAction is None or McInputSource is None:
            return
        if not self.locomotion_input_source_client.service_is_ready():
            self.get_logger().warning("SetMcInputSource service is not ready")
            return

        request = SetMcInputSource.Request()
        request.request.header.stamp = self.get_clock().now().to_msg()
        request.action = McInputAction()
        request.action.value = 1001
        request.input_source = McInputSource()
        request.input_source.name = self.settings.locomotion_yaw_source
        request.input_source.priority = self.settings.locomotion_yaw_input_source_priority
        request.input_source.timeout = self.settings.locomotion_yaw_input_source_timeout_ms
        self.locomotion_input_source_client.call_async(request)
        self._locomotion_input_source_registered = True

    def _show_person_on_screen(self, bgr: np.ndarray, detection: Any, *, now_s: float) -> None:
        if PlayVideo is None:
            return
        if self._last_screen_at is not None:
            if now_s - self._last_screen_at < self.settings.screen_cooldown_seconds:
                return
        if not self.play_video_client.service_is_ready():
            self.get_logger().warning("PlayVideo service is not ready")
            return
        try:
            write_detection_video(bgr, detection, self.screen_config)
            transfer_to_pc3(
                self.screen_config,
                timeout_seconds=self.settings.screen_scp_timeout_seconds,
            )
        except Exception as exc:
            self.get_logger().warning(f"failed to prepare screen video: {exc}")
            return

        request = PlayVideo.Request()
        request.video_path = self.settings.screen_remote_video_path
        request.mode = self.settings.screen_video_mode
        request.priority = self.settings.screen_video_priority
        request.header.header.stamp = self.get_clock().now().to_msg()
        self.play_video_client.call_async(request)
        self._last_screen_at = now_s

    def _trigger_preset_motion(self, lateral_offset: float, *, now_s: float) -> None:
        if SetMcPresetMotion is None or McControlArea is None or McPresetMotion is None:
            if getattr(self.settings, "debug_log_detections", False):
                self.get_logger().warning("preset motion message types are unavailable")
            return
        if self._last_preset_motion_at is not None:
            if now_s - self._last_preset_motion_at < self.settings.preset_motion_cooldown_seconds:
                if getattr(self.settings, "debug_log_detections", False):
                    self.get_logger().info("preset motion skipped by cooldown")
                return
        if not self.preset_motion_client.service_is_ready():
            self.get_logger().warning("SetMcPresetMotion service is not ready")
            return

        command = self.preset_motion_selector.select(lateral_offset=lateral_offset)
        if getattr(self.settings, "debug_log_detections", False):
            self.get_logger().info(
                "sending preset motion "
                f"area={command.area} "
                f"motion={command.motion} "
                f"lateral={lateral_offset:.2f}"
            )
        request = SetMcPresetMotion.Request()
        request.header.stamp = self.get_clock().now().to_msg()
        request.area = McControlArea()
        request.area.value = command.area
        request.motion = McPresetMotion()
        request.motion.value = command.motion
        request.interrupt = self.settings.preset_motion_interrupt
        request.ani_path = ""
        self.preset_motion_client.call_async(request)
        self._last_preset_motion_at = now_s

    def destroy_node(self) -> bool:
        if self.debug_server is not None:
            self.debug_server.stop()
        if self._sensor_primary_screen_executor is not None:
            self._sensor_primary_screen_executor.shutdown(wait=False)
        return super().destroy_node()

    def _should_run_detector_after_prefilters(self, *, now_s: float) -> bool:
        enabled_results = []
        if self.settings.proximity_prefilter_enabled:
            enabled_results.append(
                should_run_detector_for_prefilter(
                    enabled=True,
                    state=self.latest_lidar_proximity,
                    now_s=now_s,
                    stale_seconds=self.settings.proximity_prefilter_stale_seconds,
                    fail_open=self.settings.proximity_prefilter_fail_open,
                )
            )
        if self.settings.depth_prefilter_enabled:
            enabled_results.append(
                should_run_detector_for_depth_prefilter(
                    enabled=True,
                    state=self.latest_depth_proximity,
                    now_s=now_s,
                    stale_seconds=self.settings.depth_prefilter_stale_seconds,
                    fail_open=self.settings.depth_prefilter_fail_open,
                )
            )
        return should_run_detector_for_combined_prefilters(
            enabled_results=enabled_results,
            mode=self.settings.proximity_prefilter_mode,
        )


def _estimate_optional_shirt_color(
    enabled: bool,
    bgr_image: np.ndarray,
    detection: Any,
) -> str | None:
    if not enabled:
        return None
    return estimate_shirt_color(bgr_image, detection)


def _build_detector(settings: Settings) -> Any:
    if settings.detector_backend == "remote_http":
        return RemoteHttpDetector(
            settings.remote_detector_url,
            timeout_seconds=settings.remote_detector_timeout_seconds,
            frame_max_dimension=settings.remote_frame_max_dimension,
            jpeg_quality=settings.remote_jpeg_quality,
        )
    return UltralyticsYoloDetector(
        settings.detector_model_path,
        device=settings.detector_device,
        confidence_threshold=settings.detector_confidence_threshold,
        max_dimension=settings.detector_max_dimension,
    )


def _build_screen_capture_config(settings: Any) -> ScreenCaptureConfig:
    return ScreenCaptureConfig(
        local_video_path=settings.screen_local_video_path,
        remote_video_path=settings.screen_remote_video_path,
        pc3_target=settings.screen_pc3_target,
        duration_seconds=settings.screen_video_duration_seconds,
    )


def _estimate_proximity(
    *,
    latest_depth: np.ndarray | None,
    detection: Any,
    image_height: int,
    bbox_fallback_enabled: bool,
    bbox_near_height_fraction: float,
) -> Any:
    if latest_depth is not None:
        return estimate_person_distance(latest_depth, detection)
    if bbox_fallback_enabled:
        return estimate_bbox_proximity(
            detection,
            image_height=image_height,
            min_height_fraction=bbox_near_height_fraction,
        )
    return ProximityEstimate(distance_m=None, depth_confidence=0.0)


def _proximity_is_near(proximity: Any, *, threshold_m: float) -> bool:
    if proximity.source == "bbox":
        return bool(proximity.is_valid)
    return (
        proximity.is_valid
        and proximity.distance_m is not None
        and proximity.distance_m <= threshold_m
    )


def _format_detection_log(detection: Any, proximity: Any, *, image_width: int) -> str:
    lateral_offset = detection.normalized_x_offset(image_width)
    return (
        f"near person bbox={list(detection.xyxy)} "
        f"conf={detection.confidence:.2f} "
        f"source={proximity.source} "
        f"distance={proximity.distance_m} "
        f"bbox_frac={proximity.bbox_height_fraction} "
        f"depth_conf={proximity.depth_confidence:.2f} "
        f"lateral={lateral_offset:.2f}"
    )


def _select_locomotion_yaw_target(
    candidates: list[tuple[Any, Any]],
) -> tuple[Any, Any] | None:
    if not candidates:
        return None

    with_distance = [
        candidate for candidate in candidates if candidate[1].distance_m is not None
    ]
    if with_distance:
        return min(with_distance, key=lambda candidate: candidate[1].distance_m)

    return max(
        candidates,
        key=lambda candidate: (
            candidate[1].bbox_height_fraction
            if candidate[1].bbox_height_fraction is not None
            else candidate[0].height
        ),
    )


def _select_sensor_primary_proximity(
    settings: Any,
    *,
    latest_lidar: LidarProximityState | None,
    latest_depth: DepthProximityState | None,
    now_s: float,
) -> ProximityEstimate | None:
    candidates: list[ProximityEstimate] = []
    required_results: list[bool] = []

    if getattr(settings, "sensor_primary_detection_use_lidar", False):
        lidar_close = (
            latest_lidar is not None
            and now_s - latest_lidar.timestamp_s <= settings.proximity_prefilter_stale_seconds
            and latest_lidar.has_close_object
        )
        required_results.append(lidar_close)
        if lidar_close:
            candidates.append(
                ProximityEstimate(
                    distance_m=latest_lidar.nearest_distance_m,
                    depth_confidence=1.0,
                    source="lidar",
                )
            )

    if getattr(settings, "sensor_primary_detection_use_depth", False):
        depth_close = (
            latest_depth is not None
            and now_s - latest_depth.timestamp_s <= settings.depth_prefilter_stale_seconds
            and latest_depth.has_close_object
        )
        required_results.append(depth_close)
        if depth_close:
            candidates.append(
                ProximityEstimate(
                    distance_m=latest_depth.nearest_distance_m,
                    depth_confidence=latest_depth.close_fraction,
                    source="depth",
                )
            )

    if not required_results:
        return None
    if settings.sensor_primary_detection_mode == "all" and not all(required_results):
        return None
    if not any(required_results):
        return None

    with_distance = [
        candidate for candidate in candidates if candidate.distance_m is not None
    ]
    if with_distance:
        return min(with_distance, key=lambda candidate: candidate.distance_m or float("inf"))
    return candidates[0] if candidates else None


def _build_sensor_primary_detection(
    *,
    image_width: int,
    image_height: int,
    depth_state: DepthProximityState | None,
    rotate_depth_180: bool = False,
) -> Detection:
    if depth_state is not None and depth_state.has_close_object and depth_state.close_xyxy:
        x1, y1, x2, y2 = depth_state.close_xyxy
        if rotate_depth_180:
            x1, y1, x2, y2 = (
                image_width - x2,
                image_height - y2,
                image_width - x1,
                image_height - y1,
            )
        return Detection(
            class_name="person",
            confidence=1.0,
            xyxy=(
                max(0, min(image_width, x1)),
                max(0, min(image_height, y1)),
                max(0, min(image_width, x2)),
                max(0, min(image_height, y2)),
            ),
        )
    return Detection(
        class_name="person",
        confidence=1.0,
        xyxy=(0, 0, image_width, image_height),
    )


def _sensor_primary_lateral_offset(
    depth_state: DepthProximityState | None,
    *,
    fallback_detection: Detection,
    image_width: int,
    rotate_depth_180: bool,
    gain: float,
) -> float:
    if depth_state is not None and depth_state.lateral_offset is not None:
        lateral_offset = depth_state.lateral_offset
        if rotate_depth_180:
            lateral_offset = -lateral_offset
    else:
        lateral_offset = fallback_detection.normalized_x_offset(image_width)

    lateral_offset *= gain
    return round(max(-1.0, min(1.0, lateral_offset)), 3)


def _select_screen_yolo_detection(detections: list[Detection]) -> Detection | None:
    people = [detection for detection in detections if detection.class_name == "person"]
    if not people:
        return None
    return max(
        people,
        key=lambda detection: (
            detection.confidence,
            detection.width * detection.height,
        ),
    )


def _debug_detection_tuple(
    detection: Any,
    proximity: Any,
    *,
    image_width: int,
) -> DebugDetection:
    return (detection, proximity, detection.normalized_x_offset(image_width))


def read_pointcloud_xyz(msg: Any) -> Iterable[tuple[float, float, float]]:
    fields = {field.name: field for field in msg.fields}
    if not all(name in fields for name in ("x", "y", "z")):
        return ()
    x_field = fields["x"]
    y_field = fields["y"]
    z_field = fields["z"]
    if x_field.datatype != 7 or y_field.datatype != 7 or z_field.datatype != 7:
        return ()

    endian = ">" if msg.is_bigendian else "<"
    unpack_x = struct.Struct(f"{endian}f").unpack_from
    unpack_y = struct.Struct(f"{endian}f").unpack_from
    unpack_z = struct.Struct(f"{endian}f").unpack_from
    data = memoryview(msg.data)
    row_step = int(getattr(msg, "row_step", int(msg.width) * int(msg.point_step)))
    point_step = int(msg.point_step)
    for row in range(int(msg.height)):
        row_offset = row * row_step
        for col in range(int(msg.width)):
            offset = row_offset + (col * point_step)
            if offset + point_step > len(data):
                return
            yield (
                float(unpack_x(data, offset + int(x_field.offset))[0]),
                float(unpack_y(data, offset + int(y_field.offset))[0]),
                float(unpack_z(data, offset + int(z_field.offset))[0]),
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AgiBot near-person detector node.")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    if rclpy is None:
        raise RuntimeError("ROS2 dependencies are required to run guy-detector-node")
    rclpy.init()
    node = AgiBotNearPersonNode(settings)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
