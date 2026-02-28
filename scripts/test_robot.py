#!/usr/bin/env python3
"""Test Dobot Nova5 connection and basic motion commands.

SAFETY: This script moves the robot! Ensure the workspace is clear.
Start with low speed (default 10%) and increase gradually.
"""

import sys
import os
import time
import yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from robot import DobotNova5, Gripper
from main import load_config


def main():
    config = load_config()
    robot_cfg = config.get('robot', {})
    robot = DobotNova5(
        ip=robot_cfg.get('ip', '192.168.5.1'),
        dashboard_port=robot_cfg.get('dashboard_port', 29999),
        motion_port=robot_cfg.get('motion_port', 30003),
        feedback_port=robot_cfg.get('feedback_port', 30004),
    )

    print("=== Dobot Nova5 Connection Test ===")
    print(f"Connecting to {robot.ip}...")

    try:
        robot.connect()
        print("Connected!")

        # Clear errors and enable
        print("Clearing errors...")
        print("  Response:", robot.clear_error())

        print("Enabling robot...")
        print("  Response:", robot.enable())
        time.sleep(1)

        # Set low speed for safety
        print("Setting speed to 10%...")
        robot.set_speed(10)

        # Read current state
        print("\nCurrent pose:", robot.get_pose())
        print("Current joints:", robot.get_joint_angles())

        # Test gripper
        gripper = Gripper(robot, do_port=1)
        print("\nTesting gripper...")
        print("  Opening gripper...")
        gripper.open()
        time.sleep(1)
        print("  Closing gripper...")
        gripper.close()
        time.sleep(1)
        print("  Opening gripper...")
        gripper.open()

        print("\n=== Test Complete ===")
        print("Robot is connected and responding.")
        print("Review the pose and joint values above for sanity.")

    except ConnectionRefusedError:
        print(f"ERROR: Could not connect to robot at {robot.ip}")
        print("Check that:")
        print("  1. Robot is powered on")
        print("  2. Robot is in TCP/IP control mode (check DobotStudio Pro)")
        print(f"  3. Host PC is on the same subnet as {robot.ip}")
        print(f"  4. Robot IP is correct ({robot.ip})")
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        try:
            robot.disable()
        except Exception:
            pass
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
