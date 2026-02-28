#!/usr/bin/env python3
"""Interactive control panel for Dobot Nova5.

Keyboard-driven: press a key to send a command. See live pose/joint updates.
"""

import sys
import os
import socket
import time
import threading
import select

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config_loader import load_config


class RobotPanel:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5)
        self.sock.connect((self.ip, self.port))
        try:
            self.sock.recv(1024)
        except socket.timeout:
            pass

    def send(self, cmd):
        self.sock.send(f"{cmd}\n".encode())
        time.sleep(0.2)
        try:
            return self.sock.recv(4096).decode().strip()
        except socket.timeout:
            return "(timeout)"

    def close(self):
        if self.sock:
            self.sock.close()


MENU = """
=== Dobot Nova5 Control Panel ===

 SETUP                    JOG JOINTS (hold-style)       GRIPPER (ToolDO)
 e  Enable (full cycle)   1/! J1+ / J1-                 c  ToolDO(1,1) close
 d  Disable               2/@ J2+ / J2-                 o  ToolDO(2,1) open
 x  ClearError            3/# J3+ / J3-                 C  ToolDOInstant(1,1)
                           4/$ J4+ / J4-                 O  ToolDOInstant(2,1)
 SPEED                    5/% J5+ / J5-                 g  ToolDOInstant(1,0) release
 [  Speed down (-10)      6/^ J6+ / J6-                 G  ToolDOInstant(2,0) release
 ]  Speed up (+10)
                           JOG STOP                      TOOL SETUP
 STATUS                   s  Stop jog                    T  SetToolMode(1)
 p  GetPose                                              P  SetToolPower(1)
 a  GetAngle              CUSTOM                         B  SetTool485(115200)
 m  RobotMode             /  Type raw command
 r  GetErrorID

 q  Quit
"""


def main():
    config = load_config()
    robot_cfg = config.get('robot', {})
    ip = robot_cfg.get('ip', '192.168.5.1')
    port = robot_cfg.get('dashboard_port', 29999)

    panel = RobotPanel(ip, port)
    print(f"Connecting to {ip}:{port}...")
    panel.connect()
    print("Connected!")

    speed = 30

    # Key -> command mapping
    jog_pos = {'1': 'J1+', '2': 'J2+', '3': 'J3+', '4': 'J4+', '5': 'J5+', '6': 'J6+'}
    jog_neg = {'!': 'J1-', '@': 'J2-', '#': 'J3-', '$': 'J4-', '%': 'J5-', '^': 'J6-'}

    import tty
    import termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    print(MENU)
    print(f"  Current speed: {speed}%")
    print()

    try:
        tty.setraw(fd)

        while True:
            # Check if key available
            if select.select([sys.stdin], [], [], 0.05)[0]:
                ch = sys.stdin.read(1)
            else:
                continue

            resp = None
            label = None

            # Quit
            if ch == 'q':
                break

            # Setup
            elif ch == 'e':
                label = "Enable (disable/clear/enable)"
                panel.send('DisableRobot()')
                time.sleep(1)
                panel.send('ClearError()')
                resp = panel.send('EnableRobot()')
                time.sleep(1)
            elif ch == 'd':
                label = "DisableRobot"
                resp = panel.send('DisableRobot()')
            elif ch == 'x':
                label = "ClearError"
                resp = panel.send('ClearError()')

            # Speed
            elif ch == '[':
                speed = max(1, speed - 10)
                label = f"SpeedFactor({speed})"
                resp = panel.send(f'SpeedFactor({speed})')
            elif ch == ']':
                speed = min(100, speed + 10)
                label = f"SpeedFactor({speed})"
                resp = panel.send(f'SpeedFactor({speed})')

            # Status
            elif ch == 'p':
                label = "GetPose"
                resp = panel.send('GetPose()')
            elif ch == 'a':
                label = "GetAngle"
                resp = panel.send('GetAngle()')
            elif ch == 'm':
                label = "RobotMode"
                resp = panel.send('RobotMode()')
            elif ch == 'r':
                label = "GetErrorID"
                resp = panel.send('GetErrorID()')

            # Jog
            elif ch in jog_pos:
                axis = jog_pos[ch]
                label = f"MoveJog({axis})"
                resp = panel.send(f'MoveJog({axis})')
            elif ch in jog_neg:
                axis = jog_neg[ch]
                label = f"MoveJog({axis})"
                resp = panel.send(f'MoveJog({axis})')
            elif ch == 's':
                label = "MoveJog() STOP"
                resp = panel.send('MoveJog()')

            # Gripper - ToolDO
            elif ch == 'c':
                label = "ToolDO(1,1) close"
                resp = panel.send('ToolDO(1,1)')
            elif ch == 'o':
                label = "ToolDO(2,1) open"
                resp = panel.send('ToolDO(2,1)')
            elif ch == 'C':
                label = "ToolDOInstant(1,1) close"
                resp = panel.send('ToolDOInstant(1,1)')
            elif ch == 'O':
                label = "ToolDOInstant(2,1) open"
                resp = panel.send('ToolDOInstant(2,1)')
            elif ch == 'g':
                label = "ToolDOInstant(1,0) release"
                resp = panel.send('ToolDOInstant(1,0)')
            elif ch == 'G':
                label = "ToolDOInstant(2,0) release"
                resp = panel.send('ToolDOInstant(2,0)')

            # Tool setup
            elif ch == 'T':
                label = "SetToolMode(1)"
                resp = panel.send('SetToolMode(1)')
            elif ch == 'P':
                label = "SetToolPower(1)"
                resp = panel.send('SetToolPower(1)')
            elif ch == 'B':
                label = "SetTool485(115200)"
                resp = panel.send('SetTool485(115200)')

            # Custom command
            elif ch == '/':
                # Restore terminal for input
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                cmd = input("\r\n  Command> ")
                if cmd.strip():
                    label = cmd
                    resp = panel.send(cmd)
                # Back to raw
                tty.setraw(fd)

            # Help
            elif ch == '?' or ch == 'h':
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                print(MENU)
                print(f"  Current speed: {speed}%")
                tty.setraw(fd)
                continue

            else:
                continue

            if resp is not None:
                # Print on clean line in raw mode
                output = f"\r\n  {label}: {resp}\r\n"
                sys.stdout.write(output)
                sys.stdout.flush()

    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print()
        print("Closing...")
        panel.close()
        print("Done.")


if __name__ == "__main__":
    main()
