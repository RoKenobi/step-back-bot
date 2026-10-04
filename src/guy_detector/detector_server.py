from __future__ import annotations

import argparse
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from guy_detector.config import load_settings
from guy_detector.detector import PersonDetector, UltralyticsYoloDetector, scale_detection
from guy_detector.models import Detection
from guy_detector.remote_detector import build_detection_payload


def handle_detection_request(
    jpeg_payload: bytes,
    *,
    headers: Mapping[str, str],
    detector: PersonDetector,
) -> tuple[int, str, bytes]:
    encoded = np.frombuffer(jpeg_payload, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        return (
            HTTPStatus.BAD_REQUEST,
            "application/json",
            b'{"error":"request body must be a JPEG image"}',
        )
    decoded_height, decoded_width = bgr.shape[:2]
    original_width = _parse_positive_int(headers.get("X-Original-Width"), decoded_width)
    original_height = _parse_positive_int(headers.get("X-Original-Height"), decoded_height)
    detections = detector.detect(bgr)
    scaled = _scale_to_original(
        detections,
        decoded_width=decoded_width,
        decoded_height=decoded_height,
        original_width=original_width,
        original_height=original_height,
    )
    return HTTPStatus.OK, "application/json", build_detection_payload(scaled)


def _scale_to_original(
    detections: list[Detection],
    *,
    decoded_width: int,
    decoded_height: int,
    original_width: int,
    original_height: int,
) -> list[Detection]:
    if decoded_width == original_width and decoded_height == original_height:
        return detections
    x_scale = decoded_width / original_width
    y_scale = decoded_height / original_height
    if abs(x_scale - y_scale) < 1e-6:
        return [scale_detection(detection, scale=x_scale) for detection in detections]
    scaled: list[Detection] = []
    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy
        scaled.append(
            Detection(
                class_name=detection.class_name,
                confidence=detection.confidence,
                xyxy=(
                    int(round(x1 / x_scale)),
                    int(round(y1 / y_scale)),
                    int(round(x2 / x_scale)),
                    int(round(y2 / y_scale)),
                ),
            )
        )
    return scaled


def _parse_positive_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


class DetectorHttpServer:
    def __init__(
        self,
        detector: PersonDetector,
        *,
        host: str,
        port: int,
    ) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(handler_self) -> None:  # noqa: N802
                if handler_self.path != "/detect":
                    handler_self.send_error(HTTPStatus.NOT_FOUND)
                    return
                length = int(handler_self.headers.get("Content-Length", "0"))
                status, content_type, payload = handle_detection_request(
                    handler_self.rfile.read(length),
                    headers=handler_self.headers,
                    detector=detector,
                )
                handler_self.send_response(status)
                handler_self.send_header("Content-Type", content_type)
                handler_self.send_header("Content-Length", str(len(payload)))
                handler_self.end_headers()
                handler_self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer((host, port), Handler)

    @property
    def server_address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the tower YOLO detector HTTP server.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--model", default="models/yolo11n.pt")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--max-dimension", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config is not None:
        settings = load_settings(args.config)
        model_path = settings.detector_model_path
        device = settings.detector_device
        confidence = settings.detector_confidence_threshold
        max_dimension = settings.detector_max_dimension
    else:
        model_path = args.model
        device = args.device
        confidence = args.confidence
        max_dimension = args.max_dimension
    detector = UltralyticsYoloDetector(
        str(Path(model_path)),
        device=device,
        confidence_threshold=confidence,
        max_dimension=max_dimension,
    )
    server = DetectorHttpServer(detector, host=args.host, port=args.port)
    host, port = server.server_address
    print(f"guy-detector-server listening on http://{host}:{port}/detect", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
