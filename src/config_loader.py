"""Load and merge YAML configuration files."""

import os
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str = None) -> dict:
    """Load configuration from robot_config.yaml, with settings.yaml overrides."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'robot_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Apply local settings overrides if present
    settings_path = os.path.join(os.path.dirname(config_path), 'settings.yaml')
    if os.path.exists(settings_path):
        with open(settings_path, 'r') as f:
            overrides = yaml.safe_load(f)
        if overrides:
            _deep_merge(config, overrides)

    return config


def connect_robot(config: dict, safe_mode: bool = False):
    """Create and connect the robot specified by config['robot_type'].

    Args:
        config: Config dict from load_config().
        safe_mode: For arm101, enable reduced torque/speed.

    Returns:
        Connected robot instance (LeRobotArm101 or DobotNova5).
    """
    robot_type = config.get('robot_type', 'nova5')

    if robot_type == 'arm101':
        from robot.lerobot_arm101 import LeRobotArm101
        ac = config.get('arm101', {})
        arm = LeRobotArm101(
            port=ac.get('port', ''),
            baudrate=ac.get('baudrate', 1_000_000),
            motor_ids=ac.get('motor_ids', [1, 2, 3, 4, 5, 6]),
            speed=ac.get('speed', 200),
            safe_mode=safe_mode,
        )
        arm.connect()
        return arm
    elif robot_type == 'nova5':
        from robot.dobot_api import DobotNova5
        rc = config.get('robot', {})
        robot = DobotNova5(
            ip=rc.get('ip', '192.168.5.1'),
            dashboard_port=rc.get('dashboard_port', 29999),
        )
        robot.connect()
        robot.clear_error()
        robot.enable()
        return robot
    else:
        raise ValueError(f"Unknown robot_type: {robot_type!r}. Use 'arm101' or 'nova5'.")
