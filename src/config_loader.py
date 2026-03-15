"""Load and merge YAML configuration files."""

import os
import yaml

from rig_lock import RigLock, RigLockError

# Shared config directory: ~/.config/202602hackathon/
# Falls back to the local config/ directory in the repo if the shared one
# doesn't exist (e.g. fresh clone without setup).
_LOCAL_CONFIG_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'config'))
_SHARED_CONFIG_DIR = os.path.join(
    os.path.expanduser('~'), '.config', '202602hackathon')


def get_config_dir() -> str:
    """Return the active config directory path.

    Prefers ~/.config/202602hackathon/ if it exists, otherwise falls back
    to the local config/ directory next to the repo root.
    """
    if os.path.isdir(_SHARED_CONFIG_DIR):
        return _SHARED_CONFIG_DIR
    return _LOCAL_CONFIG_DIR


def config_path(filename: str) -> str:
    """Return the full path to a config file in the active config directory."""
    return os.path.join(get_config_dir(), filename)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(cfg_path: str = None) -> dict:
    """Load configuration from robot_config.yaml, with settings.yaml overrides."""
    if cfg_path is None:
        cfg_path = os.path.join(get_config_dir(), 'robot_config.yaml')
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    # Apply local settings overrides if present
    settings_path = os.path.join(os.path.dirname(cfg_path), 'settings.yaml')
    if os.path.exists(settings_path):
        with open(settings_path, 'r') as f:
            overrides = yaml.safe_load(f)
        if overrides:
            _deep_merge(config, overrides)

    return config


def acquire_rig_lock(holder: str = "", force: bool = False) -> RigLock:
    """Acquire the host-wide rig lock for exclusive hardware access.

    Call this before connecting to the robot or cameras.  The returned
    RigLock instance must be released when done (or used as a context
    manager).

    Args:
        holder: Optional label (e.g. Forge task ID, user name).
        force:  Steal the lock even if the current holder is alive.

    Returns:
        An acquired RigLock instance.

    Raises:
        RigLockError: If the rig is already locked by another live process.
    """
    lock = RigLock(holder=holder)
    lock.acquire(force=force)
    return lock


def connect_robot(config: dict, safe_mode: bool = False):
    """Create and connect the robot specified by config['robot_type'].

    IMPORTANT: Callers should acquire the rig lock (via ``acquire_rig_lock``
    or ``RigLock``) before calling this function to prevent concurrent
    hardware access from multiple workbenches.

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
        port = ac.get('port', '') or LeRobotArm101.find_port()
        arm = LeRobotArm101(
            port=port,
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
