from guy_detector.models import Detection, NearPersonEvent, ProximityEstimate
from guy_detector.speech import build_speech_line, build_tts_fields


def _event(color: str | None) -> NearPersonEvent:
    return NearPersonEvent(
        timestamp_s=1.0,
        detection=Detection(class_name="person", confidence=0.9, xyxy=(10, 10, 50, 100)),
        proximity=ProximityEstimate(distance_m=1.3, depth_confidence=0.8),
        lateral_offset=0.1,
        shirt_color=color,
    )


def test_builds_generic_line_without_color() -> None:
    assert build_speech_line(_event(None)) == "Please step back and keep a safe distance."


def test_builds_same_line_when_color_is_available() -> None:
    assert build_speech_line(_event("red")) == "Please step back and keep a safe distance."


def test_tts_fields_match_agibot_priority_shape() -> None:
    fields = build_tts_fields("Please step back.", trace_id="near-person-1")

    assert fields["text"] == "Please step back."
    assert fields["domain"] == "guy_detector"
    assert fields["trace_id"] == "near-person-1"
    assert fields["is_interrupted"] is True
    assert fields["priority_level"] == 6
