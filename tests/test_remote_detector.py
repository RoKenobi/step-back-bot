from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from guy_detector.models import Detection
from guy_detector.remote_detector import (
    RemoteHttpDetector,
    build_detection_payload,
    decode_detection_payload,
    encode_jpeg_request,
)


def test_encode_jpeg_request_downscales_before_transport() -> None:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)

    payload, headers = encode_jpeg_request(image, max_dimension=640, jpeg_quality=70)

    assert len(payload) > 0
    assert headers["Content-Type"] == "image/jpeg"
    assert headers["X-Original-Width"] == "1280"
    assert headers["X-Original-Height"] == "720"
    assert headers["X-Frame-Width"] == "640"
    assert headers["X-Frame-Height"] == "360"


def test_detection_payload_round_trips_detections() -> None:
    detections = [Detection(class_name="person", confidence=0.9, xyxy=(1, 2, 3, 4))]

    payload = build_detection_payload(detections)

    assert decode_detection_payload(payload) == detections


def test_remote_http_detector_posts_jpeg_and_reads_detections() -> None:
    received: dict[str, str | int] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            received["path"] = self.path
            received["content_type"] = self.headers["Content-Type"]
            length = int(self.headers["Content-Length"])
            received["length"] = length
            self.rfile.read(length)
            payload = json.dumps(
                {
                    "detections": [
                        {
                            "class_name": "person",
                            "confidence": 0.75,
                            "xyxy": [10, 20, 30, 40],
                        }
                    ]
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        detector = RemoteHttpDetector(
            f"http://{host}:{port}/detect",
            timeout_seconds=1.0,
            frame_max_dimension=64,
            jpeg_quality=70,
        )

        detections = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))

        assert received["path"] == "/detect"
        assert received["content_type"] == "image/jpeg"
        assert received["length"] > 0
        assert detections == [
            Detection(class_name="person", confidence=0.75, xyxy=(10, 20, 30, 40))
        ]
    finally:
        server.shutdown()
        server.server_close()
