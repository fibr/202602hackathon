#!/usr/bin/env python3
"""Test Dobot Nova5 connection, jog motion, and gripper.

SAFETY: This script moves the robot! Ensure the workspace is clear.
"""

import sys
import os
import time
import subprocess
import socket
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config_loader import load_config


def ping_host(ip: str) -> bool:
    result = subprocess.run(
        ["ping", "-c", "1", "-W", "2", ip],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def check_port(ip: str, port: int, timeout: float = 2.0) -> bool:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((ip, port))
        sock.close()
        return True
    except (ConnectionRefusedError, socket.timeout, OSError):
        return False


def main():
    config = load_config()
    robot_cfg = config.get('robot', {})

    ip = robot_cfg.get('ip', '192.168.5.1')
    dashboard_port = robot_cfg.get('dashboard_port', 29999)

    print("=== Dobot Nova5 Connection Test ===")
    print(f"  Robot IP:       {ip}")
    print(f"  Dashboard port: {dashboard_port}")
    print()

    # Step 1: Ping
    print(f"[1/4] Pinging {ip}...")
    if ping_host(ip):
        print(f"  OK - {ip} is reachable")
    else:
        print(f"  FAIL - {ip} is not reachable")
        return

    # Step 2: Dashboard port
    print(f"[2/4] Checking dashboard port {dashboard_port}...")
    if not check_port(ip, dashboard_port):
        print(f"  FAIL - port {dashboard_port} is closed")
        return
    print(f"  OK - port {dashboard_port} is open")

    # Step 3: Connect and enable
    print(f"[3/4] Connecting and enabling robot...")
    from robot import DobotNova5, Gripper

    robot = DobotNova5(ip=ip, dashboard_port=dashboard_port)

    try:
        robot.connect()
        print("  Connected!")

        print(f"  Mode: {robot.get_mode()}")
        print(f"  Errors: {robot.get_errors()}")

        print("  Enabling (disable/clear/enable cycle)...")
        ok = robot.enable()
        print(f"  Enable result: {'OK' if ok else 'FAILED'}")
        print(f"  Mode after enable: {robot.get_mode()}")

        robot.set_speed(30)
        print("  Speed set to 30%")

        print()
        print("  Current pose:  ", robot.get_pose())
        print("  Current joints:", robot.get_joint_angles())

        # Step 4: Wiggle and gripper
        print()
        print(f"[4/4] Wiggle test (J1 ±10deg) + gripper...")

        start_angles = robot.get_joint_angles()
        gripper = Gripper(robot)

        print("  Opening gripper...")
        gripper.open()

        print("  Jogging J1+ for 3s...")
        robot.jog_joint(1, "+", 3.0)
        print(f"    J1 now: {robot.get_joint_angles()[0]:.2f}")

        print("  Closing gripper...")
        gripper.close()

        print("  Jogging J1- for 6s...")
        robot.jog_joint(1, "-", 6.0)
        print(f"    J1 now: {robot.get_joint_angles()[0]:.2f}")

        print("  Opening gripper...")
        gripper.open()

        print("  Jogging J1+ for 3s (return)...")
        robot.jog_joint(1, "+", 3.0)
        print(f"    J1 now: {robot.get_joint_angles()[0]:.2f}")

        print()
        print("  Final pose:  ", robot.get_pose())
        print("  Final joints:", robot.get_joint_angles())
        print()
        print("=== Test Complete ===")

    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            robot.disable()
        except Exception:
            pass
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
