# Guy Detector

Near-person detection and deterrence for the **AgiBot X2** humanoid robot.

When someone walks up close to the robot, Guy Detector notices, turns the robot
toward them, asks them to step back, and shows their picture on the robot's face
screen. Every detection is logged as a JSON event.

## How It Works

```
 ┌──────────────── Robot (PC2) ─────────────────┐        ┌──── GPU server ────┐
 │                                               │  JPEG  │                    │
 │  head camera ──► downscale + JPEG ────────────┼───────►│  YOLO (person)     │
 │                                               │◄───────┼─ bounding boxes    │
 │  depth / LiDAR ──► "is something close?"      │  JSON  │                    │
 │                                               │        └────────────────────┘
 │  proximity gate ──► near_person event         │
 │        │                                      │
 │        ├─► JSONL event log                    │
 │        ├─► speech (TTS)                       │
 │        ├─► face-screen video of the person    │
 │        └─► body / head yaw toward the person  │
 └───────────────────────────────────────────────┘
```

- **Detector server** (`guy-detector-server`) runs YOLO on a GPU machine and
  returns person boxes over HTTP. The robot never needs Torch installed.
- **Robot node** (`guy-detector-node`) is a ROS 2 node on the robot's PC2. It
  decides whether a person is *near*, then triggers the robot's actions.

"Near" can be decided in three ways, chosen in the config:

| Mode | How "near" is decided | Key settings |
|------|-----------------------|--------------|
| RGB-D | median depth inside the person's torso box ≤ `proximity_threshold_m` | `depth_topic` |
| RGB-only | person box height ≥ `bbox_near_height_fraction` of the image | `depth_topic: null`, `bbox_distance_fallback_enabled: true` |
| Sensor-primary | enough close depth/LiDAR points, no YOLO needed to trigger | `sensor_primary_detection_enabled: true` |

## Quick Start

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

**Run the tests (any machine):**

```bash
uv sync --extra dev
uv run --extra dev pytest -q
uv run --extra dev ruff check .
```

**Detector server (GPU machine):**

```bash
uv sync --extra server
mkdir -p models
uvx --with ultralytics python -c "from ultralytics import YOLO; YOLO('yolo11s.pt')"
mv yolo11s.pt models/yolo11s.pt
uv run --extra server guy-detector-server --config config/tower.detector-server.yaml
```

**Robot node (on the robot, with ROS 2 and `aimdk_msgs` available):**

```bash
uv sync --python /usr/bin/python3
# set remote_detector_url in the config to your detector server first
.venv/bin/guy-detector-node --config config/pc2.front-center-debug.yaml
```

See **[docs/deployment.md](docs/deployment.md)** for the full setup, a safe
staged bring-up order, the debug viewer, and body-yaw details.

## Configuration

All settings live in one YAML file; every field and its default is defined in
[`src/guy_detector/config.py`](src/guy_detector/config.py).

| File | Purpose |
|------|---------|
| `config/pc2.front-center-debug.yaml` | Main robot config (front-center camera, remote detector) |
| `config/tower.detector-server.yaml` | Detector server (model path, GPU) |
| `config/default.yaml` | RGB-D head camera with defaults |
| `config/pc2.debug.yaml` | RGB-D with local CPU YOLO and the debug viewer |

## Project Layout

```
src/guy_detector/
├── ros_node.py          # ROS 2 node: subscriptions, event handling, robot actions
├── detector_server.py   # HTTP server wrapping YOLO
├── detector.py          # YOLO wrapper + resize helpers
├── remote_detector.py   # HTTP client used by the robot to call the server
├── config.py            # Settings model (pydantic) + YAML loader
├── models.py            # Detection / ProximityEstimate / NearPersonEvent
├── gate.py              # "near for N frames, then cooldown" event gate
├── depth.py             # distance from depth image inside the torso box
├── bbox_proximity.py    # "near" from bounding-box height
├── depth_prefilter.py   # close-object check on the depth image
├── lidar_prefilter.py   # close-object check on the LiDAR point cloud
├── head_yaw.py          # head turn controller
├── locomotion_yaw.py    # body turn controller
├── preset_motion.py     # chooses a preset gesture by person position
├── screen.py            # renders the person crop to video for the face screen
├── speech.py            # TTS line and request fields
├── color.py             # optional shirt-color estimate
├── events.py            # JSONL event writer
└── debug_server.py      # browser debug viewer
```

## Safety

- The node **never** commands forward or lateral motion; body yaw only rotates
  in place with short, time-limited bursts followed by an explicit stop.
- All robot outputs (speech, screen, head yaw, body yaw, preset motion) can be
  switched off independently in the config. Bring them up one at a time.

## Contributing

Issues and pull requests are welcome. Please run `pytest` and `ruff check .`
before opening a PR.

## License

[MIT](LICENSE)
