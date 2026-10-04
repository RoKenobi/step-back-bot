from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol
from urllib.parse import urlsplit

import cv2
import numpy as np

from guy_detector.depth import depth_to_meters, torso_depth_rect
from guy_detector.detector import resize_for_detection
from guy_detector.events import event_to_dict
from guy_detector.models import Detection, NearPersonEvent, ProximityEstimate

DebugDetection = tuple[Detection, ProximityEstimate, float]


@dataclass(frozen=True)
class DebugSnapshot:
    jpeg: bytes
    stats: dict[str, Any]


class DebugFrameState:
    def __init__(
        self,
        *,
        max_dimension: int | None = None,
        depth_near_min_m: float = 0.2,
        depth_near_max_m: float = 2.0,
    ) -> None:
        self._lock = threading.Lock()
        self._snapshot: DebugSnapshot | None = None
        self._max_dimension = max_dimension
        self._depth_near_min_m = depth_near_min_m
        self._depth_near_max_m = depth_near_max_m

    def update(
        self,
        bgr_image: np.ndarray,
        *,
        detections: list[DebugDetection],
        depth_shape: tuple[int, ...] | None,
        depth_image: np.ndarray | None = None,
        event: NearPersonEvent | None = None,
    ) -> None:
        if depth_image is not None:
            annotated = render_depth_debug_frame(
                depth_image,
                near_min_m=self._depth_near_min_m,
                near_max_m=self._depth_near_max_m,
            )
            debug_view = "depth"
        else:
            annotated = bgr_image.copy()
            debug_view = "rgb"
        image_height, image_width = annotated.shape[:2]
        detection_stats = []
        for detection, proximity, lateral_offset in detections:
            self._draw_detection(
                annotated,
                detection=detection,
                proximity=proximity,
                lateral_offset=lateral_offset,
                image_width=image_width,
                image_height=image_height,
            )
            detection_stats.append(
                {
                    "bbox_xyxy": list(detection.xyxy),
                    "detector_confidence": detection.confidence,
                    "distance_m": proximity.distance_m,
                    "depth_confidence": proximity.depth_confidence,
                    "proximity_source": proximity.source,
                    "bbox_height_fraction": proximity.bbox_height_fraction,
                    "lateral_offset": lateral_offset,
                }
            )

        center_x = image_width // 2
        cv2.line(annotated, (center_x, 0), (center_x, image_height), (255, 255, 0), 1)
        debug_frame = resize_debug_frame(annotated, max_dimension=self._max_dimension)
        debug_height, debug_width = debug_frame.shape[:2]
        ok, encoded = cv2.imencode(".jpg", debug_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return

        stats = {
            "updated_at_s": round(time.time(), 3),
            "rgb_shape": [image_height, image_width],
            "debug_frame_shape": [debug_height, debug_width],
            "depth_shape": list(depth_shape[:2]) if depth_shape is not None else None,
            "debug_view": debug_view,
            "depth_near_range_m": [self._depth_near_min_m, self._depth_near_max_m],
            "detections": detection_stats,
            "last_event": event_to_dict(event) if event is not None else None,
        }
        with self._lock:
            self._snapshot = DebugSnapshot(jpeg=encoded.tobytes(), stats=stats)

    def snapshot(self) -> DebugSnapshot | None:
        with self._lock:
            return self._snapshot

    def _draw_detection(
        self,
        image: np.ndarray,
        *,
        detection: Detection,
        proximity: ProximityEstimate,
        lateral_offset: float,
        image_width: int,
        image_height: int,
    ) -> None:
        x1, y1, x2, y2 = detection.xyxy
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        tx1, ty1, tx2, ty2 = torso_depth_rect(
            detection,
            image_width=image_width,
            image_height=image_height,
        )
        cv2.rectangle(image, (tx1, ty1), (tx2, ty2), (255, 0, 255), 2)
        target_x = int(round((lateral_offset * (image_width / 2)) + (image_width / 2)))
        cv2.line(image, (target_x, 0), (target_x, image_height), (0, 165, 255), 1)
        distance = "none" if proximity.distance_m is None else f"{proximity.distance_m:.2f}m"
        if proximity.source == "bbox":
            label = (
                f"person {detection.confidence:.2f} bbox={proximity.bbox_height_fraction:.2f} "
                f"yaw={lateral_offset:.2f}"
            )
        else:
            label = (
                f"person {detection.confidence:.2f} dist={distance} "
                f"depth={proximity.depth_confidence:.2f} yaw={lateral_offset:.2f}"
            )
        cv2.putText(
            image,
            label,
            (max(0, x1), max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )


class DebugHttpServer:
    def __init__(self, state: DebugFrameState, *, host: str, port: int) -> None:
        self.state = state
        self.host = host
        self.port = port
        self._server = ThreadingHTTPServer((host, port), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)

    @property
    def server_address(self) -> tuple[str, int]:
        host, port = self._server.server_address
        return (str(host), int(port))

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        state = self.state

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                path = urlsplit(self.path).path
                if path in ("/", "/index.html"):
                    self._send_bytes(
                        render_debug_html().encode("utf-8"),
                        "text/html; charset=utf-8",
                    )
                    return
                if path == "/frame.jpg":
                    snapshot = state.snapshot()
                    if snapshot is None:
                        self.send_error(HTTPStatus.NOT_FOUND, "No frame has been received yet")
                        return
                    self._send_bytes(snapshot.jpeg, "image/jpeg", cache=False)
                    return
                if path == "/stats.json":
                    snapshot = state.snapshot()
                    payload = (
                        snapshot.stats
                        if snapshot is not None
                        else {"status": "waiting_for_frame"}
                    )
                    self._send_bytes(
                        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
                        "application/json; charset=utf-8",
                        cache=False,
                    )
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

            def log_message(self, format: str, *args: object) -> None:
                return

            def _send_bytes(self, payload: bytes, content_type: str, *, cache: bool = True) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                if not cache:
                    self.send_header("Cache-Control", "no-store")
                self.end_headers()
                write_response_body(self.wfile, payload)

        return Handler


class Writable(Protocol):
    def write(self, payload: bytes) -> object:
        ...


def write_response_body(wfile: Writable, payload: bytes) -> None:
    try:
        wfile.write(payload)
    except (BrokenPipeError, ConnectionResetError):
        return


def resize_debug_frame(bgr_image: np.ndarray, *, max_dimension: int | None) -> np.ndarray:
    resized, _ = resize_for_detection(bgr_image, max_dimension)
    return resized


def render_depth_debug_frame(
    depth_image: np.ndarray,
    *,
    near_min_m: float,
    near_max_m: float,
) -> np.ndarray:
    depth_m = depth_to_meters(depth_image)
    valid = np.isfinite(depth_m) & (depth_m > 0)
    clipped = np.clip(depth_m, near_min_m, max(near_max_m, near_min_m + 0.001))
    denom = max(near_max_m - near_min_m, 0.001)
    normalized = ((near_max_m - clipped) / denom * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    color[~valid] = (20, 20, 20)

    near = valid & (depth_m >= near_min_m) & (depth_m <= near_max_m)
    color[near] = (0, 255, 255)
    return cv2.rotate(color, cv2.ROTATE_180)


def render_debug_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AgiBot Near-Person Debug</title>
  <style>
    body { margin: 0; font-family: system-ui, sans-serif; background: #111; color: #eee; }
    main { display: grid; grid-template-columns: minmax(0, 1fr) 360px; min-height: 100vh; }
    img { width: 100%; height: 100vh; object-fit: contain; background: #000; }
    aside { border-left: 1px solid #333; padding: 16px; overflow: auto; }
    pre { white-space: pre-wrap; word-break: break-word; font-size: 13px; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } img { height: 70vh; } }
  </style>
</head>
<body>
  <main>
    <img id="frame" src="/frame.jpg" alt="debug frame">
    <aside>
      <h1>AgiBot Near-Person Debug</h1>
      <pre id="stats">waiting...</pre>
    </aside>
  </main>
  <script>
    async function refresh() {
      document.getElementById("frame").src = "/frame.jpg?t=" + Date.now();
      try {
        const response = await fetch("/stats.json?t=" + Date.now(), { cache: "no-store" });
        document.getElementById("stats").textContent =
          JSON.stringify(await response.json(), null, 2);
      } catch (error) {
        document.getElementById("stats").textContent = String(error);
      }
    }
    refresh();
    setInterval(refresh, 300);
  </script>
</body>
</html>
"""
