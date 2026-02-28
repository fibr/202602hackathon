#!/usr/bin/env python3
"""Validate local IK solver against the robot's built-in IK/FK.

Connects to the robot, reads current pose/joints, and compares
local Pinocchio-based FK/IK against the robot's InverseKin/PositiveKin.

Usage: ./run.sh scripts/test_ik.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config_loader import load_config
from robot import DobotNova5
from kinematics import IKSolver


def main():
    config = load_config()
    rc = config.get('robot', {})
    gc = config.get('gripper', {})

    tool_length = gc.get('tool_length_mm', 100.0)
    ik = IKSolver(tool_length_mm=tool_length)

    print(f"Connecting to robot at {rc.get('ip', '192.168.5.1')}...")
    robot = DobotNova5(
        ip=rc.get('ip', '192.168.5.1'),
        dashboard_port=rc.get('dashboard_port', 29999),
        motion_port=rc.get('motion_port', 30003),
    )
    robot.connect()
    robot.enable()
    time.sleep(1)

    print("\n=== Forward Kinematics Comparison ===")
    robot_joints = robot.get_joint_angles()
    robot_pose = robot.get_pose()
    local_pos, local_rpy = ik.forward_kin(robot_joints)

    print(f"  Robot joints:   {np.round(robot_joints, 2)}")
    print(f"  Robot FK pose:  pos={np.round(robot_pose[:3], 1)} rpy={np.round(robot_pose[3:6], 1)}")
    print(f"  Local FK pose:  pos={np.round(local_pos, 1)} rpy={np.round(local_rpy, 1)}")
    pos_err = np.linalg.norm(local_pos - robot_pose[:3])
    rpy_err = np.max(np.abs(local_rpy - robot_pose[3:6]))
    print(f"  Position error: {pos_err:.2f} mm")
    print(f"  RPY error:      {rpy_err:.2f} deg")

    print("\n=== Inverse Kinematics Comparison ===")
    # Use robot's current pose as target
    target_pos = robot_pose[:3]
    target_rpy = robot_pose[3:6]

    robot_ik = robot.inverse_kin(*target_pos, *target_rpy)
    local_ik = ik.solve_ik(np.array(target_pos), np.array(target_rpy),
                            seed_joints_deg=robot_joints)

    print(f"  Target pose:    pos={np.round(target_pos, 1)} rpy={np.round(target_rpy, 1)}")
    print(f"  Robot IK:       {np.round(robot_ik, 2)}")
    if local_ik is not None:
        print(f"  Local IK:       {np.round(local_ik, 2)}")
        # Verify local IK via FK
        verify_pos, verify_rpy = ik.forward_kin(local_ik)
        ik_pos_err = np.linalg.norm(verify_pos - target_pos)
        print(f"  Local IK verify: pos={np.round(verify_pos, 1)} (err={ik_pos_err:.3f}mm)")
    else:
        print("  Local IK:       FAILED")

    print("\n=== IK Speed Benchmark ===")
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        ik.solve_ik(np.array(target_pos), np.array(target_rpy),
                    seed_joints_deg=robot_joints)
        times.append(time.perf_counter() - t0)
    times = np.array(times) * 1000
    print(f"  100 solves: avg={times.mean():.1f}ms, "
          f"median={np.median(times):.1f}ms, max={times.max():.1f}ms")

    print("\n=== Multi-Pose FK Comparison ===")
    test_configs = [
        np.array([0, 0, 0, 0, 0, 0]),
        np.array([30, -20, 45, 10, -60, 15]),
        np.array([-45, 10, -30, 90, -45, -90]),
        robot_joints,
    ]
    max_pos_err = 0
    for q in test_configs:
        robot_fk = robot.forward_kin(*q)
        local_pos, local_rpy = ik.forward_kin(q)
        local_fk = np.concatenate([local_pos, local_rpy])
        err = np.linalg.norm(local_pos - robot_fk[:3])
        max_pos_err = max(max_pos_err, err)
        status = "OK" if err < 5.0 else "WARN"
        print(f"  [{status}] q={np.round(q, 1)} -> pos_err={err:.2f}mm")

    print(f"\n  Max position error: {max_pos_err:.2f} mm")
    if max_pos_err < 5.0:
        print("  RESULT: PASS - all FK matches within 5mm")
    else:
        print("  RESULT: WARN - some FK errors exceed 5mm")
        print("  Check tool_length_mm in config/robot_config.yaml")

    robot.disconnect()


if __name__ == "__main__":
    main()
