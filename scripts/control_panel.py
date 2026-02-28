#!/usr/bin/env python3
"""Interactive control panel for Dobot Nova5.

Keyboard-driven: press a key to send a command. Uses the working command set
discovered on firmware 4.6.2:
  - MoveJog(J1+..J6-) for joint motion (uppercase only)
  - ToolDOInstant for gripper (dual-channel: close=ch1, open=ch2)
"""

import sys
import os
import time
import select

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from robot import DobotNova5, Gripper
from config_loader import load_config


MENU = """
=== Dobot Nova5 Control Panel ===

 SETUP                     JOG JOINTS                 GRIPPER
 e  Enable (full cycle)    1/! J1+ / J1-              c  Close gripper
 d  Disable                2/@ J2+ / J2-              o  Open gripper
 x  ClearError             3/# J3+ / J3-
                            4/$ J4+ / J4-
 SPEED                     5/% J5+ / J5-             STATUS
 [  Speed down (-10)       6/^ J6+ / J6-              p  GetPose
 ]  Speed up (+10)                                     a  GetAngle
                            s  Stop jog                m  RobotMode
 CUSTOM                                                r  GetErrorID
 /  Type raw command

 h  Show this help          q  Quit
"""


def main():
    config = load_config()
    robot_cfg = config.get('robot', {})
    ip = robot_cfg.get('ip', '192.168.5.1')
    dashboard_port = robot_cfg.get('dashboard_port', 29999)

    print(f"Connecting to {ip}:{dashboard_port}...")
    robot = DobotNova5(ip=ip, dashboard_port=dashboard_port)
    robot.connect()
    gripper = Gripper(robot, actuate_delay=0.5)
    print("Connected!")

    speed = 30

    jog_pos = {'1': 'J1+', '2': 'J2+', '3': 'J3+', '4': 'J4+', '5': 'J5+', '6': 'J6+'}
    jog_neg = {'!': 'J1-', '@': 'J2-', '#': 'J3-', '$': 'J4-', '%': 'J5-', '^': 'J6-'}

    import tty
    import termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    print(MENU)
    print(f"  Speed: {speed}%")
    print()

    def output(text):
        sys.stdout.write(f"\r\n  {text}\r\n")
        sys.stdout.flush()

    try:
        tty.setraw(fd)

        while True:
            if select.select([sys.stdin], [], [], 0.05)[0]:
                ch = sys.stdin.read(1)
            else:
                continue

            # Quit
            if ch == 'q':
                break

            # Setup
            elif ch == 'e':
                output("Enabling (disable/clear/enable)...")
                robot.enable()
                output(f"Mode: {robot.get_mode()}")
            elif ch == 'd':
                resp = robot.disable()
                output(f"Disable: {resp}")
            elif ch == 'x':
                resp = robot.clear_error()
                output(f"ClearError: {resp}")

            # Speed
            elif ch == '[':
                speed = max(1, speed - 10)
                robot.set_speed(speed)
                output(f"Speed: {speed}%")
            elif ch == ']':
                speed = min(100, speed + 10)
                robot.set_speed(speed)
                output(f"Speed: {speed}%")

            # Status
            elif ch == 'p':
                output(f"Pose: {robot.get_pose()}")
            elif ch == 'a':
                output(f"Joints: {robot.get_joint_angles()}")
            elif ch == 'm':
                output(f"Mode: {robot.get_mode()}  Errors: {robot.get_errors()}")
            elif ch == 'r':
                output(f"Errors: {robot.get_errors()}")

            # Jog
            elif ch in jog_pos:
                axis = jog_pos[ch]
                robot.jog_start(axis)
                output(f"Jog {axis}")
            elif ch in jog_neg:
                axis = jog_neg[ch]
                robot.jog_start(axis)
                output(f"Jog {axis}")
            elif ch == 's':
                robot.jog_stop()
                output("Jog STOP")

            # Gripper
            elif ch == 'c':
                gripper.close()
                output("Gripper CLOSED")
            elif ch == 'o':
                gripper.open()
                output("Gripper OPENED")

            # Custom command
            elif ch == '/':
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                cmd = input("\r\n  Command> ")
                if cmd.strip():
                    resp = robot._send(cmd)
                    print(f"  -> {resp}")
                tty.setraw(fd)

            # Help
            elif ch == 'h' or ch == '?':
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                print(MENU)
                print(f"  Speed: {speed}%")
                tty.setraw(fd)

    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print()
        print("Closing...")
        robot.jog_stop()
        robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
