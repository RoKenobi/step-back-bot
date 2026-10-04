import math

from guy_detector.lidar_prefilter import (
    LidarProximityState,
    estimate_pointcloud_proximity,
    should_run_detector_for_combined_prefilters,
    should_run_detector_for_prefilter,
)


def test_pointcloud_prefilter_marks_close_when_enough_points_are_near() -> None:
    points = [(1.0, 0.1, 0.2), (1.4, -0.2, 0.0), (2.0, 0.0, 0.5), (3.0, 0.0, 0.0)]

    state = estimate_pointcloud_proximity(
        points,
        max_distance_m=2.5,
        min_points=3,
        now_s=10.0,
    )

    assert state.has_close_object is True
    assert state.close_point_count == 3
    assert state.nearest_distance_m == 1.025
    assert state.timestamp_s == 10.0


def test_pointcloud_prefilter_ignores_far_behind_and_nonfinite_points() -> None:
    points = [
        (2.6, 0.0, 0.0),
        (-0.4, 0.0, 0.0),
        (math.nan, 0.0, 0.0),
        (1.0, math.inf, 0.0),
    ]

    state = estimate_pointcloud_proximity(
        points,
        max_distance_m=2.5,
        min_points=1,
        now_s=10.0,
    )

    assert state.has_close_object is False
    assert state.close_point_count == 0
    assert state.nearest_distance_m is None


def test_prefilter_allows_detector_when_disabled_or_close() -> None:
    far_state = LidarProximityState(
        timestamp_s=9.8,
        has_close_object=False,
        close_point_count=0,
        nearest_distance_m=None,
    )
    close_state = LidarProximityState(
        timestamp_s=9.8,
        has_close_object=True,
        close_point_count=25,
        nearest_distance_m=1.5,
    )

    assert (
        should_run_detector_for_prefilter(
            enabled=False,
            state=far_state,
            now_s=10.0,
            stale_seconds=0.5,
            fail_open=True,
        )
        is True
    )
    assert (
        should_run_detector_for_prefilter(
            enabled=True,
            state=close_state,
            now_s=10.0,
            stale_seconds=0.5,
            fail_open=True,
        )
        is True
    )


def test_prefilter_skips_detector_when_enabled_and_recent_state_is_not_close() -> None:
    state = LidarProximityState(
        timestamp_s=9.8,
        has_close_object=False,
        close_point_count=5,
        nearest_distance_m=1.8,
    )

    assert (
        should_run_detector_for_prefilter(
            enabled=True,
            state=state,
            now_s=10.0,
            stale_seconds=0.5,
            fail_open=True,
        )
        is False
    )


def test_prefilter_fail_open_controls_missing_or_stale_state() -> None:
    stale_state = LidarProximityState(
        timestamp_s=9.0,
        has_close_object=False,
        close_point_count=0,
        nearest_distance_m=None,
    )

    assert (
        should_run_detector_for_prefilter(
            enabled=True,
            state=None,
            now_s=10.0,
            stale_seconds=0.5,
            fail_open=True,
        )
        is True
    )
    assert (
        should_run_detector_for_prefilter(
            enabled=True,
            state=stale_state,
            now_s=10.0,
            stale_seconds=0.5,
            fail_open=False,
        )
        is False
    )


def test_combined_prefilter_any_mode_runs_when_either_sensor_is_close() -> None:
    lidar_far = LidarProximityState(
        timestamp_s=9.8,
        has_close_object=False,
        close_point_count=0,
        nearest_distance_m=None,
    )

    assert (
        should_run_detector_for_combined_prefilters(
            enabled_results=[False, True],
            mode="any",
        )
        is True
    )
    assert (
        should_run_detector_for_combined_prefilters(
            enabled_results=[False, False],
            mode="any",
        )
        is False
    )
    assert (
        should_run_detector_for_combined_prefilters(
            enabled_results=[
                should_run_detector_for_prefilter(
                    enabled=True,
                    state=lidar_far,
                    now_s=10.0,
                    stale_seconds=0.5,
                    fail_open=True,
                ),
                True,
            ],
            mode="any",
        )
        is True
    )


def test_combined_prefilter_all_mode_requires_every_enabled_sensor() -> None:
    assert (
        should_run_detector_for_combined_prefilters(
            enabled_results=[True, False],
            mode="all",
        )
        is False
    )
    assert (
        should_run_detector_for_combined_prefilters(
            enabled_results=[True, True],
            mode="all",
        )
        is True
    )
