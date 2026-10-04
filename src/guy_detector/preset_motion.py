from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PresetMotionCommand:
    area: int
    motion: int


@dataclass(frozen=True)
class PresetMotionSelector:
    center_tolerance: float

    def select(self, *, lateral_offset: float) -> PresetMotionCommand:
        if abs(lateral_offset) <= self.center_tolerance:
            return PresetMotionCommand(area=11, motion=3009)
        if lateral_offset > 0:
            return PresetMotionCommand(area=2, motion=1008)
        return PresetMotionCommand(area=1, motion=1008)
