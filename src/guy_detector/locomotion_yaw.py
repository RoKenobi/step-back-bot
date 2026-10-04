from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocomotionYawCommand:
    angular_velocity: float


@dataclass
class LocomotionYawController:
    max_angular_velocity: float
    min_angular_velocity: float
    stop_tolerance: float
    invert: bool = False

    def update(self, *, lateral_offset: float) -> LocomotionYawCommand:
        if abs(lateral_offset) <= self.stop_tolerance:
            return LocomotionYawCommand(angular_velocity=0.0)

        sign = 1.0 if self.invert else -1.0
        angular = sign * lateral_offset * self.max_angular_velocity
        angular = max(-self.max_angular_velocity, min(self.max_angular_velocity, angular))
        if abs(angular) < self.min_angular_velocity:
            angular = self.min_angular_velocity if angular > 0 else -self.min_angular_velocity
        return LocomotionYawCommand(angular_velocity=round(angular, 3))
