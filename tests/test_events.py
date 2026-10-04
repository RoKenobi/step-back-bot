import json

from guy_detector.events import JsonlEventWriter, event_to_dict
from guy_detector.models import Detection, NearPersonEvent, ProximityEstimate


def _event() -> NearPersonEvent:
    return NearPersonEvent(
        timestamp_s=1.25,
        detection=Detection(class_name="person", confidence=0.9, xyxy=(10, 20, 30, 80)),
        proximity=ProximityEstimate(distance_m=1.4, depth_confidence=0.75),
        lateral_offset=-0.2,
        shirt_color="blue",
    )


def test_event_to_dict_contains_actionable_fields() -> None:
    payload = event_to_dict(_event())

    assert payload["type"] == "near_person"
    assert payload["distance_m"] == 1.4
    assert payload["depth_confidence"] == 0.75
    assert payload["proximity_source"] == "depth"
    assert payload["bbox_height_fraction"] is None
    assert "valid_depth_fraction" not in payload
    assert "proximity_confidence" not in payload
    assert payload["bbox_xyxy"] == [10, 20, 30, 80]
    assert payload["shirt_color"] == "blue"


def test_jsonl_event_writer_appends_line(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    writer = JsonlEventWriter(path)

    writer.write(_event())

    line = path.read_text(encoding="utf-8").strip()
    assert json.loads(line)["type"] == "near_person"
