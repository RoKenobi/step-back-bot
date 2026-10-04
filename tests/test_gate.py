from guy_detector.gate import ProximityGate
from guy_detector.models import Detection, ProximityEstimate


def _person(distance_m: float | None) -> tuple[Detection, ProximityEstimate]:
    return (
        Detection(class_name="person", confidence=0.9, xyxy=(10, 10, 50, 100)),
        ProximityEstimate(
            distance_m=distance_m,
            depth_confidence=1.0 if distance_m is not None else 0.0,
        ),
    )


def test_gate_requires_consecutive_near_frames() -> None:
    gate = ProximityGate(threshold_m=2.0, required_consecutive_frames=3, cooldown_seconds=5.0)
    detection, proximity = _person(1.5)

    assert gate.update(detection, proximity, now_s=1.0) is None
    assert gate.update(detection, proximity, now_s=1.1) is None
    event = gate.update(detection, proximity, now_s=1.2)

    assert event is not None
    assert event.proximity.distance_m == 1.5


def test_gate_resets_when_person_is_far() -> None:
    gate = ProximityGate(threshold_m=2.0, required_consecutive_frames=2, cooldown_seconds=5.0)
    near_detection, near_proximity = _person(1.5)
    far_detection, far_proximity = _person(2.5)

    assert gate.update(near_detection, near_proximity, now_s=1.0) is None
    assert gate.update(far_detection, far_proximity, now_s=1.1) is None
    assert gate.update(near_detection, near_proximity, now_s=1.2) is None


def test_gate_can_be_reset_when_no_near_target_is_selected() -> None:
    gate = ProximityGate(threshold_m=2.0, required_consecutive_frames=2, cooldown_seconds=5.0)
    detection, proximity = _person(1.5)

    assert gate.update(detection, proximity, now_s=1.0) is None
    gate.reset()
    assert gate.update(detection, proximity, now_s=1.1) is None


def test_gate_suppresses_events_during_cooldown() -> None:
    gate = ProximityGate(threshold_m=2.0, required_consecutive_frames=1, cooldown_seconds=5.0)
    detection, proximity = _person(1.5)

    assert gate.update(detection, proximity, now_s=1.0) is not None
    assert gate.update(detection, proximity, now_s=3.0) is None
    assert gate.update(detection, proximity, now_s=6.1) is not None


def test_gate_ignores_invalid_depth_by_default() -> None:
    gate = ProximityGate(threshold_m=2.0, required_consecutive_frames=1, cooldown_seconds=0.0)
    detection, proximity = _person(None)

    assert gate.update(detection, proximity, now_s=1.0) is None


def test_gate_accepts_bbox_proximity_without_metric_distance() -> None:
    gate = ProximityGate(threshold_m=2.0, required_consecutive_frames=1, cooldown_seconds=0.0)
    detection = Detection(class_name="person", confidence=0.9, xyxy=(10, 10, 50, 100))
    proximity = ProximityEstimate(
        distance_m=None,
        depth_confidence=0.0,
        source="bbox",
        bbox_height_fraction=0.42,
        bbox_near=True,
    )

    event = gate.update(detection, proximity, now_s=1.0)

    assert event is not None
    assert event.proximity.distance_m is None
    assert event.proximity.source == "bbox"


def test_gate_can_use_lateral_offset_override() -> None:
    gate = ProximityGate(threshold_m=2.0, required_consecutive_frames=1, cooldown_seconds=0.0)
    detection, proximity = _person(1.5)

    event = gate.update(
        detection,
        proximity,
        now_s=1.0,
        image_width=100,
        lateral_offset_override=-0.7,
    )

    assert event is not None
    assert event.lateral_offset == -0.7
