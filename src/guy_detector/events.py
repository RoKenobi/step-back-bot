from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from guy_detector.models import NearPersonEvent


def event_to_dict(event: NearPersonEvent) -> dict[str, Any]:
    return {
        "type": "near_person",
        "timestamp_s": event.timestamp_s,
        "bbox_xyxy": list(event.detection.xyxy),
        "detector_confidence": event.detection.confidence,
        "distance_m": event.proximity.distance_m,
        "depth_confidence": event.proximity.depth_confidence,
        "proximity_source": event.proximity.source,
        "bbox_height_fraction": event.proximity.bbox_height_fraction,
        "lateral_offset": event.lateral_offset,
        "shirt_color": event.shirt_color,
    }


class JsonlEventWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, event: NearPersonEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event_to_dict(event), sort_keys=True) + "\n")
