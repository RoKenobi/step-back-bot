from guy_detector.locomotion_yaw import LocomotionYawController


def test_locomotion_yaw_centers_inside_tolerance() -> None:
    controller = LocomotionYawController(
        max_angular_velocity=0.3,
        min_angular_velocity=0.1,
        stop_tolerance=0.2,
        invert=False,
    )

    command = controller.update(lateral_offset=0.05)

    assert command.angular_velocity == 0.0


def test_locomotion_yaw_maps_right_side_person_to_negative_rotation() -> None:
    controller = LocomotionYawController(
        max_angular_velocity=0.3,
        min_angular_velocity=0.1,
        stop_tolerance=0.2,
        invert=False,
    )

    command = controller.update(lateral_offset=0.8)

    assert command.angular_velocity == -0.24


def test_locomotion_yaw_applies_minimum_start_speed() -> None:
    controller = LocomotionYawController(
        max_angular_velocity=0.3,
        min_angular_velocity=0.1,
        stop_tolerance=0.2,
        invert=False,
    )

    command = controller.update(lateral_offset=0.25)

    assert command.angular_velocity == -0.1


def test_locomotion_yaw_can_invert_sign() -> None:
    controller = LocomotionYawController(
        max_angular_velocity=0.3,
        min_angular_velocity=0.1,
        stop_tolerance=0.2,
        invert=True,
    )

    command = controller.update(lateral_offset=0.8)

    assert command.angular_velocity == 0.24
