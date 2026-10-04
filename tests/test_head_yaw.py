from guy_detector.head_yaw import HeadYawController


def test_head_yaw_centers_when_target_is_inside_tolerance() -> None:
    controller = HeadYawController(max_yaw_rad=0.3, stop_tolerance=0.12)

    command = controller.update(lateral_offset=0.05)

    assert command.target_position_rad == 0.0
    assert command.should_publish is True


def test_head_yaw_maps_right_side_person_to_negative_yaw() -> None:
    controller = HeadYawController(max_yaw_rad=0.3, stop_tolerance=0.12)

    command = controller.update(lateral_offset=0.8)

    assert command.target_position_rad == -0.24
    assert command.should_publish is True


def test_head_yaw_clamps_to_joint_limit() -> None:
    controller = HeadYawController(max_yaw_rad=0.3, stop_tolerance=0.12)

    command = controller.update(lateral_offset=-2.0)

    assert command.target_position_rad == 0.3
