#!/usr/bin/env python3
"""Interactive control panel for Dobot Nova5.

Hold jog keys for continuous motion, release to stop.
Gripper and other commands are instant.
"""

import sys
import os
import socket
import time
import select
import tty
import termios

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config_loader import load_config


class FastRobot:
    """Minimal low-latency dashboard connection."""

    def __init__(self, ip, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(2)
        self.sock.connect((ip, port))
        try:
            self.sock.recv(1024)
        except socket.timeout:
            pass

    def send(self, cmd):
        self.sock.send(f"{cmd}\n".encode())
        time.sleep(0.05)
        try:
            return self.sock.recv(4096).decode().strip()
        except socket.timeout:
            return ""

    def close(self):
        self.sock.close()


JOG_POS = {'1': 'J1+', '2': 'J2+', '3': 'J3+', '4': 'J4+', '5': 'J5+', '6': 'J6+'}
JOG_NEG = {'!': 'J1-', '@': 'J2-', '#': 'J3-', '$': 'J4-', '%': 'J5-', '^': 'J6-'}
ALL_JOG = {**JOG_POS, **JOG_NEG}

HELP = """\x1b[2J\x1b[H\
=== Dobot Nova5 Control Panel ===

 JOINT JOG (hold to move)     GRIPPER        SETUP
 1/!  J1+ / J1-               c  Close       e  Enable
 2/@  J2+ / J2-               o  Open        d  Disable
 3/#  J3+ / J3-                              x  ClearError
 4/$  J4+ / J4-              STATUS
 5/%  J5+ / J5-               p  Pose        /  Raw command
 6/^  J6+ / J6-               a  Angles      h  Help
                               m  Mode        q  Quit
 SPEED
 [/]  -/+ 10%                 s  Emergency stop
"""


def main():
    config = load_config()
    rc = config.get('robot', {})
    ip = rc.get('ip', '192.168.5.1')
    port = rc.get('dashboard_port', 29999)

    print(f"Connecting to {ip}:{port}...")
    r = FastRobot(ip, port)

    # Enable
    r.send('DisableRobot()')
    time.sleep(1)
    r.send('ClearError()')
    r.send('EnableRobot()')
    time.sleep(1)

    speed = 30
    r.send(f'SpeedFactor({speed})')

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    # Status line helper
    def status(msg):
        # Clear current line, print status, stay on same line
        sys.stdout.write(f"\r\x1b[K  {msg}")
        sys.stdout.flush()

    def status_line():
        status(f"Speed:{speed}%  Mode:{r.send('RobotMode()').split('{')[1].split('}')[0] if '{' in r.send('RobotMode()') else '?'}")

    sys.stdout.write(HELP)
    status(f"Ready. Speed:{speed}%")

    jogging = None      # Currently active jog axis, or None
    last_jog_key = 0.0  # Timestamp of last jog keypress

    # Grace period: keep jogging through key-repeat gaps.
    # Terminal key repeat has ~30-80ms gaps; 150ms covers that comfortably
    # while still stopping quickly on actual release.
    JOG_GRACE = 0.15

    try:
        tty.setcbreak(fd)  # cbreak instead of raw — allows Ctrl+C

        while True:
            ready = select.select([sys.stdin], [], [], 0.02)[0]

            if not ready:
                # No key — stop jog only after grace period expires
                if jogging and (time.time() - last_jog_key) > JOG_GRACE:
                    r.send('MoveJog()')
                    jogging = None
                    status(f"Speed:{speed}%")
                continue

            ch = sys.stdin.read(1)

            # Quit
            if ch == 'q':
                break

            # Jog — continuous while key held, stops after 150ms of no keypress
            elif ch in ALL_JOG:
                axis = ALL_JOG[ch]
                last_jog_key = time.time()
                if jogging != axis:
                    if jogging:
                        r.send('MoveJog()')  # stop previous axis first
                    r.send(f'MoveJog({axis})')
                    jogging = axis
                    status(f"JOG {axis}  Speed:{speed}%")

            elif ch == 's':
                r.send('MoveJog()')
                jogging = None
                status("STOPPED")

            # Gripper
            elif ch == 'c':
                status("Gripper closing...")
                r.send('ToolDOInstant(2,0)')
                r.send('ToolDOInstant(1,1)')
                status("Gripper CLOSED")
            elif ch == 'o':
                status("Gripper opening...")
                r.send('ToolDOInstant(1,0)')
                r.send('ToolDOInstant(2,1)')
                status("Gripper OPEN")

            # Speed
            elif ch == '[':
                speed = max(1, speed - 10)
                r.send(f'SpeedFactor({speed})')
                status(f"Speed:{speed}%")
            elif ch == ']':
                speed = min(100, speed + 10)
                r.send(f'SpeedFactor({speed})')
                status(f"Speed:{speed}%")

            # Status queries
            elif ch == 'p':
                resp = r.send('GetPose()')
                val = resp.split('{')[1].split('}')[0] if '{' in resp else resp
                status(f"Pose: {val}")
            elif ch == 'a':
                resp = r.send('GetAngle()')
                val = resp.split('{')[1].split('}')[0] if '{' in resp else resp
                status(f"Joints: {val}")
            elif ch == 'm':
                mode = r.send('RobotMode()')
                err = r.send('GetErrorID()')
                mv = mode.split('{')[1].split('}')[0] if '{' in mode else mode
                ev = err.split('{')[1].split('}')[0] if '{' in err else err
                status(f"Mode:{mv}  Errors:{ev}")

            # Setup
            elif ch == 'e':
                status("Enabling...")
                r.send('DisableRobot()')
                time.sleep(1)
                r.send('ClearError()')
                r.send('EnableRobot()')
                time.sleep(1)
                status("Enabled")
            elif ch == 'd':
                r.send('DisableRobot()')
                status("Disabled")
            elif ch == 'x':
                r.send('ClearError()')
                status("Errors cleared")

            # Raw command
            elif ch == '/':
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
                sys.stdout.write("\r\n")
                cmd = input("  Command> ")
                if cmd.strip():
                    resp = r.send(cmd)
                    print(f"  -> {resp}")
                tty.setcbreak(fd)
                status(f"Speed:{speed}%")

            # Help
            elif ch == 'h' or ch == '?':
                sys.stdout.write(HELP)
                status(f"Speed:{speed}%")

    except KeyboardInterrupt:
        pass
    finally:
        r.send('MoveJog()')
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print("\r\n  Closing...")
        r.close()
        print("  Done.")


if __name__ == "__main__":
    main()
