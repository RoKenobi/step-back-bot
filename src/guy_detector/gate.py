from __future__ import annotations

from dataclasses import dataclass

from guy_detector.models import Detection, NearPersonEvent, ProximityEstimate


@dataclass
class ProximityGate:
    threshold_m: float
    required_consecutive_frames: int
    cooldown_seconds: float
    image_width: int = 1
    _near_frames: int = 0
    _last_event_at: float | None = None

    def reset(self) -> None:
        self._near_frames = 0

    def update(
        self,
        detection: Detection,
        proximity: ProximityEstimate,
        *,
        now_s: float,
        shirt_color: str | None = None,
        image_width: int | None = None,
        lateral_offset_override: float | None = None,
    ) -> NearPersonEvent | None:
        if not self._is_near(proximity):
            self._near_frames = 0
            return None

        self._near_frames += 1
        if self._near_frames < self.required_consecutive_frames:
            return None
        if self._last_event_at is not None and now_s - self._last_event_at < self.cooldown_seconds:
            return None

        self._last_event_at = now_s
        width = image_width or self.image_width
        lateral_offset = (
            lateral_offset_override
            if lateral_offset_override is not None
            else detection.normalized_x_offset(width)
        )
        return NearPersonEvent(
            timestamp_s=now_s,
            detection=detection,
            proximity=proximity,
            lateral_offset=lateral_offset,
            shirt_color=shirt_color,
        )

    def _is_near(self, proximity: ProximityEstimate) -> bool:
        if proximity.source == "bbox":
            return proximity.is_valid
        return (
            proximity.is_valid
            and proximity.distance_m is not None
            and proximity.distance_m <= self.threshold_m
        )
