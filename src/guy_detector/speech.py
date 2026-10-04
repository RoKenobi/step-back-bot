from __future__ import annotations

from guy_detector.models import NearPersonEvent

GENERIC_LINE = "Please step back and keep a safe distance."


def build_speech_line(event: NearPersonEvent) -> str:
    return GENERIC_LINE


def build_tts_fields(text: str, *, trace_id: str) -> dict[str, object]:
    return {
        "text": text,
        "domain": "guy_detector",
        "trace_id": trace_id,
        "is_interrupted": True,
        "priority_weight": 0,
        "priority_level": 6,
    }
