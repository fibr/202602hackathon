#!/usr/bin/env python3
"""Query robot dashboard for diagnostic info."""

import sys
import os
import socket
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config_loader import load_config


def dashboard_session(ip: str, port: int, timeout: float = 5.0):
    """Open a persistent dashboard connection."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((ip, port))
    # Read welcome banner
    try:
        banner = sock.recv(1024).decode().strip()
        print(f"  Banner: {banner}")
    except socket.timeout:
        print("  (no banner)")
    return sock


def send(sock, cmd: str) -> str:
    sock.send(f"{cmd}\n".encode())
    time.sleep(0.3)
    try:
        resp = sock.recv(4096).decode().strip()
    except socket.timeout:
        resp = "(timeout)"
    return resp


def main():
    config = load_config()
    robot_cfg = config.get('robot', {})
    ip = robot_cfg.get('ip', '192.168.5.1')
    port = robot_cfg.get('dashboard_port', 29999)

    print(f"=== Robot Diagnostics: {ip}:{port} ===")
    print()

    sock = dashboard_session(ip, port)

    cmds = [
        ("RobotMode()",       "Robot mode (1=init, 2=brake_open, 3=disabled, 4=enable, 5=backdrive, 6=running, 7=recording, 8=error, 9=pause, 10=jog)"),
        ("GetErrorID()",      "Error IDs (0 = no error)"),
        ("GetAngle()",        "Current joint angles"),
        ("GetPose()",         "Current TCP pose [x,y,z,rx,ry,rz]"),
        ("SpeedFactor()",     "Current speed factor"),
        ("PayLoad()",         "Current payload setting"),
        ("DI(1)",             "Digital input 1"),
        ("DI(2)",             "Digital input 2"),
        ("DO(1)",             "Digital output 1 (gripper)"),
        ("DO(2)",             "Digital output 2"),
    ]

    for cmd, desc in cmds:
        resp = send(sock, cmd)
        print(f"  {cmd:<20s} -> {resp:<40s}  # {desc}")

    print()
    print("--- Trying to enable ---")
    for cmd in ["ClearError()", "EnableRobot()"]:
        resp = send(sock, cmd)
        print(f"  {cmd:<20s} -> {resp}")

    time.sleep(2)

    print()
    print("--- State after enable ---")
    for cmd in ["RobotMode()", "GetErrorID()"]:
        resp = send(sock, cmd)
        print(f"  {cmd:<20s} -> {resp}")

    # Try a small move command on the motion port
    motion_port = robot_cfg.get('motion_port', 30003)
    print()
    print(f"--- Testing motion port {motion_port} ---")
    try:
        msock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        msock.settimeout(5)
        msock.connect((ip, motion_port))
        try:
            mbanner = msock.recv(1024).decode().strip()
            print(f"  Motion banner: {mbanner}")
        except socket.timeout:
            print("  (no motion banner)")

        # Read current pose, then send a tiny relative move
        pose_resp = send(sock, "GetPose()")
        print(f"  Current pose: {pose_resp}")

        # Send Sync to see if motion port responds
        msock.send(b"Sync()\n")
        time.sleep(0.3)
        try:
            mresp = msock.recv(4096).decode().strip()
            print(f"  Sync() -> {mresp}")
        except socket.timeout:
            print(f"  Sync() -> (timeout)")

        msock.close()
    except (ConnectionRefusedError, socket.timeout, OSError) as e:
        print(f"  Could not connect to motion port: {e}")

    sock.close()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
