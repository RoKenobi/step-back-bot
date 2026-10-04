from guy_detector.preset_motion import PresetMotionSelector


def test_selects_right_hand_clap_for_person_on_right() -> None:
    selector = PresetMotionSelector(center_tolerance=0.2)

    command = selector.select(lateral_offset=0.4)

    assert command.area == 2
    assert command.motion == 1008


def test_selects_left_hand_clap_for_person_on_left() -> None:
    selector = PresetMotionSelector(center_tolerance=0.2)

    command = selector.select(lateral_offset=-0.4)

    assert command.area == 1
    assert command.motion == 1008


def test_selects_center_folded_arms_when_person_is_centered() -> None:
    selector = PresetMotionSelector(center_tolerance=0.2)

    command = selector.select(lateral_offset=0.1)

    assert command.area == 11
    assert command.motion == 3009
