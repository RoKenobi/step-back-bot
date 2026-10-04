# Deployment Guide

This guide walks through running Guy Detector on an AgiBot X2 with a separate
GPU machine doing YOLO inference. Replace `<detector-host>` and `<robot-host>`
with the addresses on your network.

## 1. Detector Server (GPU machine)

```bash
uv sync --extra server

# Download YOLO weights once
mkdir -p models
uvx --with ultralytics python -c "from ultralytics import YOLO; YOLO('yolo11s.pt')"
mv yolo11s.pt models/yolo11s.pt

uv run --extra server guy-detector-server --config config/tower.detector-server.yaml
```

The server listens on `0.0.0.0:8766` and accepts JPEG frames on `POST /detect`.
`config/tower.detector-server.yaml` selects the GPU with `detector_device`
(for example `cuda:0` or `cuda:1`).

Check that the GPU is visible:

```bash
uv run --extra server python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Check that the route is reachable from the robot (a non-JPEG body should return
a JSON error, which proves the server is up):

```bash
printf 'not-a-jpeg' | curl -sS -X POST --data-binary @- http://<detector-host>:8766/detect
```

For long-running use, run the server under systemd, Docker, or another
supervisor with automatic restart. Keep it running independently of the robot
node so YOLO is not reloaded every time the node starts.

## 2. Robot Node (PC2)

PC2 only needs the lightweight ROS/client dependencies, not Torch or YOLO:

```bash
uv sync --python /usr/bin/python3
```

Point `remote_detector_url` in `config/pc2.front-center-debug.yaml` at your
detector server, then run:

```bash
.venv/bin/guy-detector-node --config config/pc2.front-center-debug.yaml
tail -f /tmp/guy-detector-events.jsonl
```

## 3. Staged Bring-Up

Enable robot outputs one at a time. Start with an event-only config:

```bash
python3 - <<'PY'
from pathlib import Path
import yaml

data = yaml.safe_load(Path("config/pc2.front-center-debug.yaml").read_text())
for key in ("tts_enabled", "screen_enabled", "yaw_enabled",
            "locomotion_yaw_enabled", "preset_motion_enabled"):
    data[key] = False
out = Path("/tmp/pc2.event-only.yaml")
out.write_text(yaml.safe_dump(data, sort_keys=False))
print(out)
PY
.venv/bin/guy-detector-node --config /tmp/pc2.event-only.yaml
```

Then, in order:

1. **Events only** – confirm `near_person` events appear in the JSONL file.
2. **TTS** – confirm the services exist:
   `ros2 service list | grep -E 'PlayTts|SetMute'`. The node unmutes before
   speaking and respects `speech_cooldown_seconds`.
3. **Head yaw** (`yaw_enabled`) – the head turns toward the person.
4. **Body yaw** (`locomotion_yaw_enabled`) – only after confirming MC input
   source ownership and e-stop behavior.

## 4. Debug Viewer

With `debug_server_enabled: true`, open `http://<robot-host>:8765`. If inbound
connections are blocked, tunnel it:

```bash
ssh -L 8765:localhost:8765 <user>@<robot-host>
# then open http://localhost:8765
```

The viewer shows the YOLO box (green), the torso region used for depth
(magenta), the yaw target line (orange), and live stats. If the magenta box does
not land on the person's torso, the RGB and depth images are misaligned and
`distance_m` will be biased.

## 5. Body Yaw Details

Body yaw publishes `aimdk_msgs/msg/McLocomotionVelocity` on
`/aima/mc/locomotion/velocity` with **forward and lateral velocity always 0.0**.
On each event the node picks one target (closest `distance_m`, otherwise the
tallest bbox), publishes a short angular-velocity burst at
`locomotion_yaw_publish_hz` for `locomotion_yaw_command_duration_seconds`, then
sends one zero-velocity stop command. Preset motion and head yaw run after the
burst ends (optionally after waiting for a stable stand).

If the robot turns away from the person, set `locomotion_yaw_invert: true`.

Manual test of a small right turn:

```bash
ros2 topic pub --once /aima/mc/locomotion/velocity aimdk_msgs/msg/McLocomotionVelocity \
'{source: "node", forward_velocity: 0.0, lateral_velocity: 0.0, angular_velocity: -0.1}'
```

## 6. Optional Proximity Prefilters

To cut network traffic, PC2 can skip sending frames to the detector until a
cheap local sensor sees something close:

```yaml
proximity_prefilter_mode: "any"        # or "all"
proximity_prefilter_enabled: true      # chest LiDAR
depth_prefilter_enabled: true          # head depth camera
proximity_prefilter_fail_open: true
depth_prefilter_fail_open: true
```

With `*_fail_open: true`, a missing or stale sensor stream falls back to normal
RGB detection instead of silently suppressing it.
