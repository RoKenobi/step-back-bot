from __future__ import annotations

from urllib.request import urlopen

import cv2
import numpy as np

from guy_detector.debug_server import (
    DebugFrameState,
    DebugHttpServer,
    render_debug_html,
    render_depth_debug_frame,
    write_response_body,
)
from guy_detector.models import Detection, ProximityEstimate


def test_debug_frame_state_stores_encoded_overlay_and_stats() -> None:
    state = DebugFrameState(depth_near_min_m=0.2, depth_near_max_m=1.5)
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    depth = np.full((80, 100), 3000, dtype=np.uint16)
    depth[10:20, 30:40] = 1000
    detection = Detection(class_name="person", confidence=0.91, xyxy=(20, 10, 60, 70))
    proximity = ProximityEstimate(distance_m=1.42, depth_confidence=0.83)

    state.update(
        image,
        detections=[(detection, proximity, 0.0)],
        depth_shape=(80, 100),
        depth_image=depth,
    )

    snapshot = state.snapshot()
    assert snapshot is not None
    assert snapshot.jpeg.startswith(b"\xff\xd8")
    assert snapshot.stats["depth_shape"] == [80, 100]
    assert snapshot.stats["detections"][0]["bbox_xyxy"] == [20, 10, 60, 70]
    assert snapshot.stats["detections"][0]["distance_m"] == 1.42
    assert snapshot.stats["detections"][0]["depth_confidence"] == 0.83
    assert snapshot.stats["detections"][0]["lateral_offset"] == 0.0
    assert snapshot.stats["debug_view"] == "depth"
    assert snapshot.stats["depth_near_range_m"] == [0.2, 1.5]


def test_debug_frame_state_downscales_served_jpeg_without_changing_rgb_stats() -> None:
    state = DebugFrameState(max_dimension=50)
    image = np.zeros((80, 100, 3), dtype=np.uint8)

    state.update(image, detections=[], depth_shape=None)

    snapshot = state.snapshot()
    assert snapshot is not None
    assert snapshot.stats["rgb_shape"] == [80, 100]
    assert snapshot.stats["debug_frame_shape"] == [40, 50]
    assert snapshot.stats["debug_view"] == "rgb"


def test_debug_depth_view_highlights_near_pixels_with_distinct_color() -> None:
    state = DebugFrameState(depth_near_min_m=0.2, depth_near_max_m=1.5)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    depth = np.full((4, 4), 3000, dtype=np.uint16)
    depth[1, 1] = 1000

    state.update(image, detections=[], depth_shape=depth.shape, depth_image=depth)

    snapshot = state.snapshot()
    assert snapshot is not None
    decoded = cv2.imdecode(np.frombuffer(snapshot.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    near_pixel = decoded[2, 2].astype(int)
    far_pixel = decoded[0, 0].astype(int)
    assert np.linalg.norm(near_pixel - far_pixel) > 50


def test_debug_depth_view_flips_upside_down_depth_image() -> None:
    depth = np.full((4, 4), 3000, dtype=np.uint16)
    depth[3, 0] = 1000

    rendered = render_depth_debug_frame(depth, near_min_m=0.2, near_max_m=1.5)

    assert tuple(rendered[0, 3]) == (0, 255, 255)
    assert tuple(rendered[3, 0]) != (0, 255, 255)


def test_debug_html_points_browser_at_frame_and_stats_endpoints() -> None:
    html = render_debug_html()

    assert "/frame.jpg" in html
    assert "/stats.json" in html
    assert "AgiBot Near-Person Debug" in html


def test_debug_http_server_accepts_cache_busting_query_strings() -> None:
    state = DebugFrameState()
    state.update(
        np.zeros((20, 20, 3), dtype=np.uint8),
        detections=[],
        depth_shape=(20, 20),
    )
    server = DebugHttpServer(state, host="127.0.0.1", port=0)
    server.start()
    host, port = server.server_address
    try:
        with urlopen(f"http://{host}:{port}/frame.jpg?t=123", timeout=2) as response:
            assert response.status == 200
            assert response.read().startswith(b"\xff\xd8")
        with urlopen(f"http://{host}:{port}/stats.json?t=123", timeout=2) as response:
            assert response.status == 200
            assert b"rgb_shape" in response.read()
    finally:
        server.stop()


def test_write_response_body_ignores_closed_browser_socket() -> None:
    class ClosedSocket:
        def write(self, payload: bytes) -> None:
            raise BrokenPipeError

    write_response_body(ClosedSocket(), b"jpeg bytes")
