from importlib import import_module


def test_ros_node_module_imports_without_ros_runtime() -> None:
    module = import_module("guy_detector.ros_node")

    assert hasattr(module, "main")
    assert hasattr(module, "AgiBotNearPersonNode")
