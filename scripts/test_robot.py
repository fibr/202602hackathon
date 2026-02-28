#!/usr/bin/env python3
"""Test Dobot Nova5 connection and basic motion commands.

SAFETY: This script moves the robot! Ensure the workspace is clear.
Start with low speed (default 10%) and increase gradually.
"""

import sys
import os
import time
import socket
import subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config_loader import load_config


def ping_host(ip: str) -> bool:
    """Ping the host and return True if reachable."""
    result = subprocess.run(
        ["ping", "-c", "1", "-W", "2", ip],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def check_port(ip: str, port: int, timeout: float = 2.0) -> bool:
    """Check if a TCP port is open."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((ip, port))
        sock.close()
        return True
    except (ConnectionRefusedError, socket.timeout, OSError):
        return False


def send_dashboard_cmd(ip: str, port: int, cmd: str, timeout: float = 5.0) -> str:
    """Send a single command to the dashboard port and return the response."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((ip, port))
    # Read welcome message
    try:
        sock.recv(1024)
    except socket.timeout:
        pass
    sock.send(f"{cmd}\n".encode())
    time.sleep(0.2)
    try:
        response = sock.recv(4096).decode().strip()
    except socket.timeout:
        response = "(no response)"
    sock.close()
    return response


def main():
    config = load_config()
    robot_cfg = config.get('robot', {})

    ip = robot_cfg.get('ip', '192.168.5.1')
    dashboard_port = robot_cfg.get('dashboard_port', 29999)
    motion_port = robot_cfg.get('motion_port', 30003)
    feedback_port = robot_cfg.get('feedback_port', 30004)

    print("=== Dobot Nova5 Connection Test ===")
    print(f"  Robot IP:       {ip}")
    print(f"  Dashboard port: {dashboard_port}")
    print(f"  Motion port:    {motion_port}")
    print(f"  Feedback port:  {feedback_port}")
    print()

    # Step 1: Ping
    print(f"[1/4] Pinging {ip}...")
    if ping_host(ip):
        print(f"  OK - {ip} is reachable")
    else:
        print(f"  FAIL - {ip} is not reachable")
        print("  Check network cable and that host PC is on the same subnet.")
        return

    # Step 2: Check dashboard port
    print(f"[2/4] Checking dashboard port {dashboard_port}...")
    if not check_port(ip, dashboard_port):
        print(f"  FAIL - dashboard port {dashboard_port} is closed")
        print("  Robot may not be fully booted or is in a different control mode.")
        return
    print(f"  OK - dashboard port {dashboard_port} is open")

    # Step 3: Enable robot via dashboard (opens motion/feedback ports)
    print(f"[3/4] Enabling robot via dashboard...")

    print(f"  Sending ClearError()...")
    resp = send_dashboard_cmd(ip, dashboard_port, "ClearError()")
    print(f"    Response: {resp}")

    print(f"  Sending EnableRobot()...")
    resp = send_dashboard_cmd(ip, dashboard_port, "EnableRobot()")
    print(f"    Response: {resp}")

    # Wait for motion port to open
    if not check_port(ip, motion_port):
        print(f"  Motion port {motion_port} not yet open, waiting up to 10s...")
        for i in range(10):
            time.sleep(1)
            if check_port(ip, motion_port):
                print(f"  OK - motion port {motion_port} is now open (after {i+1}s)")
                break
        else:
            print(f"  WARNING - motion port {motion_port} still closed after 10s")
            resp = send_dashboard_cmd(ip, dashboard_port, "RobotMode()")
            print(f"    RobotMode(): {resp}")
            return
    else:
        print(f"  OK - motion port {motion_port} is already open")

    # Check feedback port too
    if check_port(ip, feedback_port):
        print(f"  OK - feedback port {feedback_port} is open")
    else:
        print(f"  WARNING - feedback port {feedback_port} is closed (continuing anyway)")

    # Step 4: Full connection test
    print(f"[4/4] Running full connection test...")
    from robot import DobotNova5, Gripper

    robot = DobotNova5(
        ip=ip,
        dashboard_port=dashboard_port,
        motion_port=motion_port,
        feedback_port=feedback_port,
    )

    try:
        robot.connect()
        print("  Connected to all ports!")

        robot.set_speed(10)
        print("  Speed set to 10%")

        print()
        print("  Current pose:  ", robot.get_pose())
        print("  Current joints:", robot.get_joint_angles())

        # Wiggle test: move +100mm then -100mm on Z from current pose
        pose = robot.get_pose()
        gripper = Gripper(robot, do_port=robot_cfg.get('gripper_do_port', 1))

        if pose is not None and len(pose) >= 6:
            x, y, z, rx, ry, rz = pose[:6]
            print()
            print("  Wiggle test (+/-100mm Z with gripper)...")

            print(f"    Opening gripper...")
            gripper.open()

            print(f"    Moving UP to Z={z + 100:.1f}...")
            robot.move_linear(x, y, z + 100, rx, ry, rz)
            robot.wait_motion_done()

            print(f"    Closing gripper...")
            gripper.close()

            print(f"    Moving DOWN to Z={z - 100:.1f}...")
            robot.move_linear(x, y, z - 100, rx, ry, rz)
            robot.wait_motion_done()

            print(f"    Opening gripper...")
            gripper.open()

            print(f"    Returning to Z={z:.1f}...")
            robot.move_linear(x, y, z, rx, ry, rz)
            robot.wait_motion_done()

            print("  Wiggle done!")

        print()
        print("=== Test Complete ===")

    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
    finally:
        try:
            robot.disable()
        except Exception:
            pass
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
