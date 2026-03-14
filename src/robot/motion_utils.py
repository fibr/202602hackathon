"""Shared Cartesian motion utilities for all robot types.

Provides a unified ``move_to_pose(robot, x, y, z, rx, ry, rz)`` function
that works for both Nova5 (direct firmware IK via MovJ) and arm101 (local
Pinocchio IK + joint commands).

Usage::

    from robot.motion_utils import move_to_pose

    ok = move_to_pose(robot, x=250, y=0, z=150, rx=0, ry=90, rz=0, speed=20)

The IK solver for arm101 is created on first call and cached at module level,
so repeated calls pay no URDF-load overhead.
"""

import logging
import numpy as np
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level IK solver cache (keyed by robot_type string)
# ---------------------------------------------------------------------------
_ik_solvers: dict = {}


def get_robot_type(robot) -> str:
    """Return the robot type string for a connected robot object.

    Returns ``'arm101'`` for :class:`~robot.lerobot_arm101.LeRobotArm101` and
    ``'nova5'`` (default) for :class:`~robot.dobot_api.DobotNova5`.

    The detection relies on the ``robot_type`` class attribute that
    ``LeRobotArm101`` sets to ``'arm101'``.  Any object without that attribute
    is assumed to be a Nova5.
    """
    return getattr(robot, 'robot_type', 'nova5')


def get_ik_solver(robot_type: str = 'arm101'):
    """Return (and cache) a local IK solver for *robot_type*.

    On first call the solver is instantiated (loads the URDF ~10–50 ms).
    Subsequent calls return the cached instance.

    Returns ``None`` for Nova5 because Nova5 uses firmware IK and requires no
    local solver.  Returns ``None`` if no solver is available.

    Args:
        robot_type: ``'arm101'`` or ``'nova5'``.
    """
    if robot_type == 'nova5':
        return None  # Nova5 uses firmware IK — no local solver needed

    if robot_type in _ik_solvers:
        return _ik_solvers[robot_type]

    # Try arm101-specific solver first, then fall back to the generic one.
    solver = None
    try:
        from kinematics.arm101_ik_solver import Arm101IKSolver
        solver = Arm101IKSolver()
        log.debug('move_to_pose: loaded Arm101IKSolver')
    except Exception as exc:
        log.debug(f'Arm101IKSolver unavailable ({exc}), trying generic IKSolver')

    if solver is None:
        try:
            from kinematics.ik_solver import IKSolver
            solver = IKSolver()
            log.debug('move_to_pose: loaded generic IKSolver for arm101')
        except Exception as exc:
            log.error(f'move_to_pose: no IK solver available for arm101: {exc}')

    _ik_solvers[robot_type] = solver  # cache even if None to avoid retries
    return solver


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def move_to_pose(
    robot,
    x: float,
    y: float,
    z: float,
    rx: float,
    ry: float,
    rz: float,
    speed: int = 30,
    timeout: float = 30.0,
    ik=None,
) -> bool:
    """Move the robot TCP to a Cartesian pose, regardless of robot type.

    * **Nova5**: calls ``robot.movj()`` directly (firmware IK via ``MovJ V4``).
    * **arm101**: solves IK locally with Pinocchio, then calls
      ``robot.move_joints()``.

    The IK solver for arm101 is created on first call and cached at module
    level, so repeated calls are efficient (~0.8 ms per IK solve).

    Args:
        robot: Connected robot object (``DobotNova5`` or ``LeRobotArm101``).
        x, y, z: Target TCP position in mm.
        rx, ry, rz: Target TCP orientation in degrees (RPY convention).
        speed: Speed percentage (1–100) passed to ``robot.set_speed()``.
            Applied for Nova5 only; arm101 uses its configured default speed.
        timeout: Maximum time to wait for motion completion, seconds.
            Used by Nova5 only.
        ik: Optional pre-created IK solver instance (e.g. ``Arm101IKSolver``).
            Overrides the module-level cache — useful when the caller already
            holds a solver.  Ignored for Nova5.

    Returns:
        ``True`` if the motion completed successfully, ``False`` otherwise.
    """
    robot_type = get_robot_type(robot)

    if robot_type == 'arm101':
        return _move_arm101(robot, x, y, z, rx, ry, rz, ik=ik)
    else:
        return _move_nova5(robot, x, y, z, rx, ry, rz,
                           speed=speed, timeout=timeout)


# ---------------------------------------------------------------------------
# Robot-type-specific helpers
# ---------------------------------------------------------------------------

def _move_nova5(robot, x: float, y: float, z: float,
                rx: float, ry: float, rz: float,
                speed: int = 30, timeout: float = 30.0) -> bool:
    """Cartesian move for Nova5 via firmware MovJ (V4 syntax)."""
    try:
        robot.set_speed(speed)
    except Exception:
        pass  # set_speed may fail if robot is not yet fully enabled

    if hasattr(robot, 'movj'):
        return robot.movj(x, y, z, rx, ry, rz, timeout=timeout)

    # Fallback: raw V4 command for thin mock/stub objects that lack movj()
    cmd = f'MovJ(pose={{{x:.3f},{y:.3f},{z:.3f},{rx:.3f},{ry:.3f},{rz:.3f}}})'
    resp = getattr(robot, 'send', lambda c: None)(cmd)
    log.debug(f'move_to_pose raw MovJ resp: {resp}')
    return '0,' in (resp or '')


def _move_arm101(robot, x: float, y: float, z: float,
                 rx: float, ry: float, rz: float,
                 ik=None) -> bool:
    """Cartesian move for arm101 via local IK + joint commands."""
    solver = ik or get_ik_solver('arm101')
    if solver is None:
        log.error('move_to_pose: no IK solver available for arm101; cannot move')
        return False

    # Use current joint angles as the IK seed to stay in the same configuration.
    seed: Optional[np.ndarray] = None
    try:
        angles = robot.get_angles()
        if angles:
            seed = np.array(angles[:5], dtype=float)
    except Exception:
        pass

    target_pos = np.array([x, y, z], dtype=float)
    target_rpy = np.array([rx, ry, rz], dtype=float)

    # Pass seed positionally — the keyword arg name differs between IKSolver
    # (seed_joints_deg) and Arm101IKSolver (seed_motor_deg).
    joint_angles = solver.solve_ik(target_pos, target_rpy, seed)
    if joint_angles is None:
        log.warning(
            f'move_to_pose: IK failed for target [{x:.1f}, {y:.1f}, {z:.1f}] '
            f'rx={rx:.1f} ry={ry:.1f} rz={rz:.1f}'
        )
        return False

    return robot.move_joints(list(joint_angles))
