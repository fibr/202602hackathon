"""Trajectory generation for smooth joint-space motion.

Provides quintic polynomial (smoothstep) trajectories that subdivide
large joint moves into small steps with zero velocity/acceleration at
endpoints. Includes retry logic for transient motion failures.
"""

import numpy as np
from logger import get_logger

log = get_logger('trajectory')


def quintic_trajectory(q_start: np.ndarray, q_goal: np.ndarray,
                       max_step_deg: float = 5.0) -> list[np.ndarray]:
    """Generate a quintic-smoothstep trajectory in joint space.

    Uses the polynomial s(t) = 10t^3 - 15t^4 + 6t^5 which has zero
    velocity and zero acceleration at both endpoints, giving smooth
    ramp-up and ramp-down motion.

    Args:
        q_start: Starting joint angles in degrees (6,)
        q_goal: Goal joint angles in degrees (6,)
        max_step_deg: Maximum joint travel per step in degrees.
                      Lower = smoother but more commands sent.

    Returns:
        List of joint angle arrays from q_start to q_goal inclusive.
        Always has at least 2 elements (start and end).
    """
    delta = q_goal - q_start
    max_travel = np.max(np.abs(delta))

    if max_travel < 0.01:
        return [q_start.copy(), q_goal.copy()]

    n_steps = max(2, int(np.ceil(max_travel / max_step_deg)))
    t = np.linspace(0.0, 1.0, n_steps)

    # Quintic smoothstep: s(0)=0, s(1)=1, s'(0)=s'(1)=0, s''(0)=s''(1)=0
    s = 10 * t**3 - 15 * t**4 + 6 * t**5

    trajectory = q_start + np.outer(s, delta)
    return [trajectory[i] for i in range(n_steps)]


def execute_trajectory(robot, q_start: np.ndarray, q_goal: np.ndarray,
                       max_step_deg: float = 5.0,
                       step_timeout: float = 10.0,
                       retries: int = 2) -> bool:
    """Move from q_start to q_goal using a smooth quintic trajectory.

    Subdivides the motion into small steps and sends each as a MovJ(joint=...)
    command. Retries failed steps with error clearing.

    Args:
        robot: DobotNova5 instance
        q_start: Current joint angles in degrees (6,)
        q_goal: Target joint angles in degrees (6,)
        max_step_deg: Maximum joint travel per step
        step_timeout: Timeout per individual step
        retries: Number of retry attempts per failed step

    Returns:
        True if all steps completed successfully.
    """
    import time

    configs = quintic_trajectory(q_start, q_goal, max_step_deg=max_step_deg)
    total = len(configs) - 1

    if total <= 0:
        return True

    max_travel = np.max(np.abs(q_goal - q_start))
    log.debug(f"Trajectory: {total} steps, max_travel={max_travel:.1f}deg, "
              f"max_step={max_step_deg}deg")

    for i, q in enumerate(configs[1:], start=1):  # skip q_start
        for attempt in range(retries + 1):
            ok = robot.movj_joints(*q, timeout=step_timeout)
            if ok:
                break

            if attempt < retries:
                log.warning(f"Step {i}/{total} failed (attempt {attempt+1}), "
                            f"clearing error and retrying...")
                robot.clear_error()
                robot.enable()
                time.sleep(0.5)
            else:
                log.error(f"Step {i}/{total} failed after {retries+1} attempts "
                          f"at q={np.round(q, 1)}")
                return False

    return True
