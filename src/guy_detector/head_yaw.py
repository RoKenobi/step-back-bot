from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeadYawCommand:
    target_position_rad: float
    should_publish: bool = True


@dataclass
class HeadYawController:
    max_yaw_rad: float
    stop_tolerance: float

    def update(self, *, lateral_offset: float) -> HeadYawCommand:
        if abs(lateral_offset) <= self.stop_tolerance:
            return HeadYawCommand(target_position_rad=0.0)
        target = -lateral_offset * self.max_yaw_rad
        target = max(-self.max_yaw_rad, min(self.max_yaw_rad, target))
        return HeadYawCommand(target_position_rad=round(target, 3))
