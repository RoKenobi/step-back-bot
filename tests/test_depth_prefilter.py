import numpy as np

from guy_detector.depth_prefilter import (
    DepthProximityState,
    estimate_depth_proximity,
    should_run_detector_for_depth_prefilter,
)


def test_depth_prefilter_marks_close_when_enough_pixels_are_near_uint16_mm() -> None:
    depth = np.full((10, 10), 5000, dtype=np.uint16)
    depth[0:2, :] = 1500

    state = estimate_depth_proximity(
        depth,
        min_distance_m=0.2,
        max_distance_m=2.5,
        min_valid_fraction=0.15,
        now_s=10.0,
    )

    assert state.has_close_object is True
    assert state.close_fraction == 0.2
    assert state.nearest_distance_m == 1.5
    assert state.close_xyxy == (0, 0, 10, 2)
    assert state.lateral_offset == -0.1
    assert state.timestamp_s == 10.0


def test_depth_prefilter_ignores_zero_far_and_nonfinite_depth() -> None:
    depth = np.array(
        [
            [0.0, np.nan, 2.6],
            [0.1, np.inf, 8.0],
        ],
        dtype=np.float32,
    )

    state = estimate_depth_proximity(
        depth,
        min_distance_m=0.2,
        max_distance_m=2.5,
        min_valid_fraction=0.01,
        now_s=10.0,
    )

    assert state.has_close_object is False
    assert state.close_fraction == 0.0
    assert state.nearest_distance_m is None
    assert state.close_xyxy is None
    assert state.lateral_offset is None


def test_depth_prefilter_estimates_lateral_offset_from_close_pixels() -> None:
    depth = np.full((4, 10), 5000, dtype=np.uint16)
    depth[1:3, 7:10] = 1200

    state = estimate_depth_proximity(
        depth,
        min_distance_m=0.2,
        max_distance_m=2.5,
        min_valid_fraction=0.1,
        now_s=10.0,
    )

    assert state.has_close_object is True
    assert state.close_xyxy == (7, 1, 10, 3)
    assert state.lateral_offset == 0.6


def test_depth_prefilter_allows_detector_when_disabled_or_close() -> None:
    far_state = DepthProximityState(
        timestamp_s=9.8,
        has_close_object=False,
        close_fraction=0.0,
        nearest_distance_m=None,
        close_xyxy=None,
        lateral_offset=None,
    )
    close_state = DepthProximityState(
        timestamp_s=9.8,
        has_close_object=True,
        close_fraction=0.05,
        nearest_distance_m=1.2,
        close_xyxy=(3, 1, 7, 5),
        lateral_offset=0.0,
    )

    assert (
        should_run_detector_for_depth_prefilter(
            enabled=False,
            state=far_state,
            now_s=10.0,
            stale_seconds=0.5,
            fail_open=True,
        )
        is True
    )
    assert (
        should_run_detector_for_depth_prefilter(
            enabled=True,
            state=close_state,
            now_s=10.0,
            stale_seconds=0.5,
            fail_open=True,
        )
        is True
    )


def test_depth_prefilter_skips_detector_when_recent_state_is_not_close() -> None:
    state = DepthProximityState(
        timestamp_s=9.8,
        has_close_object=False,
        close_fraction=0.01,
        nearest_distance_m=1.8,
        close_xyxy=None,
        lateral_offset=None,
    )

    assert (
        should_run_detector_for_depth_prefilter(
            enabled=True,
            state=state,
            now_s=10.0,
            stale_seconds=0.5,
            fail_open=True,
        )
        is False
    )


def test_depth_prefilter_fail_open_controls_missing_or_stale_state() -> None:
    stale_state = DepthProximityState(
        timestamp_s=9.0,
        has_close_object=False,
        close_fraction=0.0,
        nearest_distance_m=None,
        close_xyxy=None,
        lateral_offset=None,
    )

    assert (
        should_run_detector_for_depth_prefilter(
            enabled=True,
            state=None,
            now_s=10.0,
            stale_seconds=0.5,
            fail_open=True,
        )
        is True
    )
    assert (
        should_run_detector_for_depth_prefilter(
            enabled=True,
            state=stale_state,
            now_s=10.0,
            stale_seconds=0.5,
            fail_open=False,
        )
        is False
    )
