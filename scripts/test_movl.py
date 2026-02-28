#!/usr/bin/env python3
"""Test MovL commands: go home, then step in each Cartesian axis and report.

Verifies that MovL(pose={...}) moves the TCP as expected.
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config_loader import load_config
from robot.dobot_api import DobotNova5


def fmt(arr):
    return ', '.join(f'{v:.2f}' for v in arr)


def main():
    config = load_config()
    rc = config.get('robot', {})
    ip = rc.get('ip', '192.168.5.1')

    print(f"Connecting to {ip}...")
    robot = DobotNova5(ip=ip)
    robot.connect()

    print("Enabling...")
    robot.clear_error()
    robot.enable()
    robot.set_speed(20)

    # Read initial state
    pose0 = robot.get_pose()
    joints0 = robot.get_joint_angles()
    print(f"\n=== Initial State ===")
    print(f"  Pose:   [{fmt(pose0)}]  (x, y, z, rx, ry, rz) mm/deg")
    print(f"  Joints: [{fmt(joints0)}] deg")

    # Move to a safe home-ish position using MovJ to current pose (no-op)
    print(f"\n=== MovJ to current pose (no-op test) ===")
    p = pose0
    cmd = f"MovJ(pose={{{p[0]:.2f},{p[1]:.2f},{p[2]:.2f},{p[3]:.2f},{p[4]:.2f},{p[5]:.2f}}})"
    print(f"  Sending: {cmd}")
    resp = robot._send(cmd)
    print(f"  Response: {resp}")
    time.sleep(2)

    pose_after_noop = robot.get_pose()
    print(f"  Pose after: [{fmt(pose_after_noop)}]")
    diff = pose_after_noop - pose0
    print(f"  Diff:       [{fmt(diff)}]")

    # Test each axis with a small step
    STEP_MM = 10.0
    STEP_DEG = 5.0
    AXES = ['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz']

    for axis_idx in range(6):
        step = STEP_MM if axis_idx < 3 else STEP_DEG
        label = AXES[axis_idx]

        # Read current pose
        pose_before = robot.get_pose()
        joints_before = robot.get_joint_angles()

        # Build target: offset one axis
        target = list(pose_before)
        target[axis_idx] += step

        print(f"\n=== Step {label}+ by {step} ===")
        print(f"  Before: [{fmt(pose_before)}]")
        print(f"  Target: [{fmt(target)}]")

        cmd = f"MovL(pose={{{target[0]:.2f},{target[1]:.2f},{target[2]:.2f},{target[3]:.2f},{target[4]:.2f},{target[5]:.2f}}})"
        print(f"  Sending: {cmd}")
        resp = robot._send(cmd)
        code = resp.split(',')[0] if resp else '-1'
        print(f"  Response: {resp}  (code={code})")

        if code != '0':
            print(f"  SKIPPING - command failed")
            continue

        # Wait for motion
        time.sleep(0.5)
        prev = robot.get_joint_angles()
        for _ in range(30):
            time.sleep(0.2)
            cur = robot.get_joint_angles()
            if max(abs(cur[i] - prev[i]) for i in range(6)) < 0.05:
                break
            prev = cur

        pose_after = robot.get_pose()
        joints_after = robot.get_joint_angles()
        pose_diff = pose_after - pose_before
        joint_diff = joints_after - joints_before

        print(f"  After:  [{fmt(pose_after)}]")
        print(f"  P diff: [{fmt(pose_diff)}]")
        print(f"  J diff: [{fmt(joint_diff)}]")

        # Check if the right axis moved
        expected_diff = [0] * 6
        expected_diff[axis_idx] = step
        ok = abs(pose_diff[axis_idx] - step) < 2.0  # within 2mm/deg
        others_ok = all(abs(pose_diff[i]) < 2.0 for i in range(6) if i != axis_idx)
        print(f"  {label} moved by {pose_diff[axis_idx]:.2f} (expected {step:.1f}): {'OK' if ok else 'WRONG'}")
        print(f"  Other axes stable: {'OK' if others_ok else 'DRIFT'}")

        # Move back
        cmd_back = f"MovL(pose={{{pose_before[0]:.2f},{pose_before[1]:.2f},{pose_before[2]:.2f},{pose_before[3]:.2f},{pose_before[4]:.2f},{pose_before[5]:.2f}}})"
        robot._send(cmd_back)
        time.sleep(3)

    print(f"\n=== Final State ===")
    pose_final = robot.get_pose()
    joints_final = robot.get_joint_angles()
    print(f"  Pose:   [{fmt(pose_final)}]")
    print(f"  Joints: [{fmt(joints_final)}]")
    print(f"  Drift from start: [{fmt(pose_final - pose0)}]")

    robot.disable()
    robot.disconnect()
    print("\nDone.")


if __name__ == "__main__":
    main()
