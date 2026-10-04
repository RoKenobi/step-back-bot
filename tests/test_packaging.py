from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def test_uv_installs_project_entry_point() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["guy-detector-node"] == "guy_detector.ros_node:main"
    assert pyproject["tool"]["uv"]["package"] is True


def test_detector_extra_does_not_install_pypi_torch_on_jetson() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    detector_deps = pyproject["project"]["optional-dependencies"]["detector"]
    lower_deps = [dep.lower() for dep in detector_deps]

    assert not any(dep.startswith("torch") for dep in lower_deps)
    assert not any(dep.startswith("ultralytics") for dep in lower_deps)


def test_server_extra_installs_yolo_runtime() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    server_deps = pyproject["project"]["optional-dependencies"]["server"]

    assert pyproject["project"]["scripts"]["guy-detector-server"] == (
        "guy_detector.detector_server:main"
    )
    assert any(dep.lower().startswith("ultralytics") for dep in server_deps)
