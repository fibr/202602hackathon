#!/usr/bin/env python3
"""Query robot dashboard for diagnostic info, then wiggle + gripper test.

Uses only the dashboard port (29999). Motion via MoveJog, gripper via ToolDO.
"""

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


def parse_response(resp: str):
    """Parse Dobot response 'code,{value},cmd;' -> (code, value)."""
    try:
        code = int(resp.split(",")[0])
        inner = resp.split("{")[1].split("}")[0] if "{" in resp else ""
        return code, inner
    except (ValueError, IndexError):
        return None, resp


def main():
    config = load_config()
    robot_cfg = config.get('robot', {})
    ip = robot_cfg.get('ip', '192.168.5.1')
    port = robot_cfg.get('dashboard_port', 29999)

    print(f"=== Robot Diagnostics: {ip}:{port} ===")
    print()

    sock = dashboard_session(ip, port)

    # ── Status dump ──
    cmds = [
        ("RobotMode()",  "Mode (1=init 2=brake 4=disabled 5=enabled 6=backdrive 7=running 9=error 10=pause 11=jog)"),
        ("GetErrorID()", "Errors"),
        ("GetAngle()",   "Joint angles (deg)"),
        ("GetPose()",    "TCP pose [x,y,z,rx,ry,rz] mm/deg"),
    ]
    for cmd, desc in cmds:
        resp = send(sock, cmd)
        code, val = parse_response(resp)
        print(f"  {cmd:<20s} [{code:>6}] {val:<50s}  # {desc}")

    # ── Enable ──
    print()
    print("--- Enable ---")
    for cmd in ["ClearError()", "EnableRobot()"]:
        resp = send(sock, cmd)
        code, val = parse_response(resp)
        print(f"  {cmd:<20s} [{code:>6}]")

    resp = send(sock, "RobotMode()")
    code, val = parse_response(resp)
    print(f"  RobotMode:  {val}")

    # ── Gripper test (ToolDO) ──
    print()
    print("--- Gripper test (ToolDO) ---")
    print("  Closing gripper (ToolDO 1,1)...")
    resp = send(sock, "ToolDO(1,1)")
    code, _ = parse_response(resp)
    print(f"    Response code: {code}")
    time.sleep(1)

    print("  Opening gripper (ToolDO 2,1)...")
    resp = send(sock, "ToolDO(2,1)")
    code, _ = parse_response(resp)
    print(f"    Response code: {code}")
    time.sleep(1)

    # ── Jog wiggle test ──
    print()
    print("--- Wiggle test (MoveJog) ---")
    print("  Setting speed to 10%...")
    send(sock, "SpeedFactor(10)")

    print("  Jogging Z+ for 2s...")
    send(sock, "MoveJog(z+)")
    time.sleep(2)
    send(sock, "MoveJog()")
    time.sleep(0.5)

    print("  Pose after Z+:", send(sock, "GetPose()"))

    print("  Jogging Z- for 2s...")
    send(sock, "MoveJog(z-)")
    time.sleep(2)
    send(sock, "MoveJog()")
    time.sleep(0.5)

    print("  Pose after Z-:", send(sock, "GetPose()"))

    print("  Jogging Z+ for 2s (return)...")
    send(sock, "MoveJog(z+)")
    time.sleep(1)
    send(sock, "MoveJog()")
    time.sleep(0.5)

    print("  Final pose:   ", send(sock, "GetPose()"))

    # ── Final state ──
    print()
    print("--- Final state ---")
    resp = send(sock, "RobotMode()")
    code, val = parse_response(resp)
    print(f"  RobotMode: {val}")
    resp = send(sock, "GetErrorID()")
    code, val = parse_response(resp)
    print(f"  Errors:    {val}")

    sock.close()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
