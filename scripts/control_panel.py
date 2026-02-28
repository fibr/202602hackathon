#!/usr/bin/env python3
"""Interactive control panel for Dobot Nova5.

Hold jog keys for continuous joint motion, release to stop.
Cartesian keys step via MovL on motion port 30003.
Gripper and other commands are instant.

Requires ROS2 driver: docker compose --profile dobot up -d
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
    """Low-latency dual-port connection: dashboard (29999) + motion (30003)."""

    def __init__(self, ip, dash_port, motion_port):
        self.ip = ip
        self.dash_port = dash_port
        self.motion_port = motion_port
        self.dash_sock = None
        self.motion_sock = None
        self._connect_dash()
        self._connect_motion()

    def _connect_dash(self):
        if self.dash_sock:
            try:
                self.dash_sock.close()
            except Exception:
                pass
        self.dash_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.dash_sock.settimeout(2)
        self.dash_sock.connect((self.ip, self.dash_port))
        try:
            self.dash_sock.recv(1024)
        except socket.timeout:
            pass

    def _connect_motion(self):
        if self.motion_sock:
            try:
                self.motion_sock.close()
            except Exception:
                pass
        self.motion_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.motion_sock.settimeout(2)
        try:
            self.motion_sock.connect((self.ip, self.motion_port))
            try:
                self.motion_sock.recv(1024)
            except socket.timeout:
                pass
        except (ConnectionRefusedError, socket.timeout, OSError):
            self.motion_sock.close()
            self.motion_sock = None

    def send(self, cmd):
        """Send a command on the dashboard port."""
        for attempt in range(3):
            try:
                self.dash_sock.send(f"{cmd}\n".encode())
                time.sleep(0.05)
                try:
                    return self.dash_sock.recv(4096).decode().strip()
                except socket.timeout:
                    return ""
            except (BrokenPipeError, ConnectionResetError, OSError):
                if attempt < 2:
                    print(f"\r\n  Dashboard connection lost, reconnecting ({attempt+2}/3)...")
                    time.sleep(1)
                    try:
                        self._connect_dash()
                    except (ConnectionRefusedError, socket.timeout, OSError):
                        continue
                else:
                    print("\r\n  ERROR: Cannot reach robot after 3 attempts.")
                    raise

    def send_motion(self, cmd):
        """Send a command on the motion port (30003)."""
        if not self.motion_sock:
            return "-1,{},no_motion_port;"
        try:
            self.motion_sock.send(f"{cmd}\n".encode())
            time.sleep(0.05)
            try:
                return self.motion_sock.recv(4096).decode().strip()
            except socket.timeout:
                return ""
        except (BrokenPipeError, ConnectionResetError, OSError):
            return "-1,{},connection_error;"

    def parse_vals(self, resp):
        """Extract comma-separated floats from '{...}' in response."""
        try:
            inner = resp.split('{')[1].split('}')[0]
            return [float(x) for x in inner.split(',')]
        except (IndexError, ValueError):
            return None

    def get_pose(self):
        return self.parse_vals(self.send('GetPose()'))

    def get_angles(self):
        return self.parse_vals(self.send('GetAngle()'))

    def inverse_kin(self, x, y, z, rx, ry, rz):
        resp = self.send(f'InverseKin({x},{y},{z},{rx},{ry},{rz})')
        return self.parse_vals(resp)

    def close(self):
        if self.dash_sock:
            self.dash_sock.close()
        if self.motion_sock:
            self.motion_sock.close()


JOG_JOINT_POS = {'1': 'J1+', '2': 'J2+', '3': 'J3+', '4': 'J4+', '5': 'J5+', '6': 'J6+'}
JOG_JOINT_NEG = {'!': 'J1-', '@': 'J2-', '#': 'J3-', '$': 'J4-', '%': 'J5-', '^': 'J6-'}
ALL_JOINT_JOG = {**JOG_JOINT_POS, **JOG_JOINT_NEG}

# Cartesian step: key -> (axis_index, sign)
# Pose is [x, y, z, rx, ry, rz]
CART_STEP_MM = 10.0    # mm per keypress for translation
CART_STEP_DEG = 5.0    # degrees per keypress for rotation
CART_KEYS = {
    'w': (0, +1), 'W': (0, -1),   # X+ / X-
    'a': (1, +1), 'A': (1, -1),   # Y+ / Y-
    'r': (2, +1), 'f': (2, -1),   # Z+ / Z-
    't': (3, +1), 'T': (3, -1),   # Rx+ / Rx-
    'g': (4, +1), 'G': (4, -1),   # Ry+ / Ry-
    'b': (5, +1), 'B': (5, -1),   # Rz+ / Rz-
}
CART_LABELS = {
    0: 'X', 1: 'Y', 2: 'Z', 3: 'Rx', 4: 'Ry', 5: 'Rz',
}

HELP = """\x1b[2J\x1b[H\
=== Dobot Nova5 Control Panel ===

 JOINT JOG (hold)            CARTESIAN (tap to step)    GRIPPER
 1/!  J1+ / J1-              w/W  X+ / X-  (10mm)       c  Close
 2/@  J2+ / J2-              a/A  Y+ / Y-               o  Open
 3/#  J3+ / J3-              r/f  Z+ / Z-
 4/$  J4+ / J4-              t/T  Rx+ / Rx- (5deg)     SETUP
 5/%  J5+ / J5-              g/G  Ry+ / Ry-             e  Enable
 6/^  J6+ / J6-              b/B  Rz+ / Rz-             d  Disable
                                                         x  ClearError
 SPEED                       STATUS
 [/]  -/+ 10%                p  Pose       s  E-stop    /  Raw command
 {/}  cart step -/+ 5mm      j  Angles     h  Help      q  Quit
                              m  Mode       0  Home
"""


def main():
    config = load_config()
    rc = config.get('robot', {})
    ip = rc.get('ip', '192.168.5.1')
    dash_port = rc.get('dashboard_port', 29999)
    motion_port = rc.get('motion_port', 30003)

    print(f"Connecting to {ip} (dashboard:{dash_port}, motion:{motion_port})...")
    r = FastRobot(ip, dash_port, motion_port)

    if r.motion_sock:
        print("  Motion port connected (MovL/MovJ available)")
    else:
        print("  WARNING: Motion port not available. Cartesian steps and Home disabled.")
        print("  Start ROS2 driver: docker compose --profile dobot up -d")

    # Enable
    r.send('DisableRobot()')
    time.sleep(1)
    r.send('ClearError()')
    r.send('EnableRobot()')
    time.sleep(1)

    speed = 30
    r.send(f'SpeedFactor({speed})')
    cart_step = CART_STEP_MM

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    def status(msg):
        sys.stdout.write(f"\r\x1b[K  {msg}")
        sys.stdout.flush()

    def flush_input():
        """Drain any buffered keystrokes from stdin."""
        while select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.read(1)

    sys.stdout.write(HELP)
    status(f"Ready. Speed:{speed}%  Step:{cart_step}mm")

    jogging = None
    last_jog_key = 0.0
    JOG_GRACE = 0.15

    def do_cart_step(axis_idx, sign):
        """Step in Cartesian space: read pose, offset, MovL to target."""
        if not r.motion_sock:
            status("ERROR: motion port not connected (need ROS2 driver)")
            return

        pose = r.get_pose()
        if not pose or len(pose) < 6:
            status("ERROR: can't read pose")
            return

        step = cart_step if axis_idx < 3 else CART_STEP_DEG
        target_pose = list(pose)
        target_pose[axis_idx] += sign * step

        axis_name = CART_LABELS[axis_idx]
        dir_ch = '+' if sign > 0 else '-'
        cur_val = ','.join(f'{v:.1f}' for v in pose)
        tgt_val = ','.join(f'{v:.1f}' for v in target_pose)
        status(f"{axis_name}{dir_ch}  cur:[{cur_val}] tgt:[{tgt_val}]")

        tp = target_pose
        cmd = f'MovL({tp[0]:.2f},{tp[1]:.2f},{tp[2]:.2f},{tp[3]:.2f},{tp[4]:.2f},{tp[5]:.2f})'
        resp = r.send_motion(cmd)
        code = resp.split(',')[0] if resp else '-1'
        if code != '0':
            status(f"ERROR: {cmd} -> {resp}")
            return

        # Wait for motion to complete (poll joint stability)
        time.sleep(0.3)
        prev = r.get_angles()
        for _ in range(50):  # up to 5s
            time.sleep(0.1)
            cur = r.get_angles()
            if prev and cur:
                if max(abs(cur[i] - prev[i]) for i in range(6)) < 0.05:
                    break
            prev = cur

        new_pose = r.get_pose()
        if new_pose:
            val = ','.join(f'{v:.1f}' for v in new_pose)
            status(f"{axis_name}{dir_ch} done  Pose: {val}")
        else:
            status(f"{axis_name}{dir_ch} done")
        flush_input()

    try:
        tty.setcbreak(fd)

        while True:
            ready = select.select([sys.stdin], [], [], 0.02)[0]

            if not ready:
                if jogging and (time.time() - last_jog_key) > JOG_GRACE:
                    r.send('MoveJog()')
                    jogging = None
                    status(f"Speed:{speed}%  Step:{cart_step}mm")
                continue

            ch = sys.stdin.read(1)

            if ch == 'q':
                break

            # Joint jog
            elif ch in ALL_JOINT_JOG:
                axis = ALL_JOINT_JOG[ch]
                last_jog_key = time.time()
                if jogging != axis:
                    if jogging:
                        r.send('MoveJog()')
                    r.send(f'MoveJog({axis})')
                    jogging = axis
                    status(f"JOG {axis}  Speed:{speed}%")

            # Cartesian step
            elif ch in CART_KEYS:
                # Stop any active jog first
                if jogging:
                    r.send('MoveJog()')
                    jogging = None
                axis_idx, sign = CART_KEYS[ch]
                do_cart_step(axis_idx, sign)

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
                status(f"Speed:{speed}%  Step:{cart_step}mm")
            elif ch == ']':
                speed = min(100, speed + 10)
                r.send(f'SpeedFactor({speed})')
                status(f"Speed:{speed}%  Step:{cart_step}mm")

            # Cartesian step size
            elif ch == '{':
                cart_step = max(1, cart_step - 5)
                status(f"Step:{cart_step}mm  Speed:{speed}%")
            elif ch == '}':
                cart_step = min(50, cart_step + 5)
                status(f"Step:{cart_step}mm  Speed:{speed}%")

            # Status queries
            elif ch == 'p':
                pose = r.get_pose()
                if pose:
                    val = ','.join(f'{v:.1f}' for v in pose)
                    status(f"Pose: {val}")
                else:
                    status("Pose: error")
            elif ch == 'j':
                angles = r.get_angles()
                if angles:
                    val = ','.join(f'{v:.1f}' for v in angles)
                    status(f"Joints: {val}")
                else:
                    status("Joints: error")
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
                flush_input()
            elif ch == 'd':
                r.send('DisableRobot()')
                status("Disabled")
            elif ch == 'x':
                r.send('ClearError()')
                status("Errors cleared")

            # Home
            elif ch == '0':
                if not r.motion_sock:
                    status("ERROR: motion port not connected (need ROS2 driver)")
                    continue
                if jogging:
                    r.send('MoveJog()')
                    jogging = None
                status("Homing...")
                r.send('SpeedFactor(10)')
                resp = r.send_motion('MovJ(43.5,-13.9,-85.4,196.3,-90.0,43.5)')
                code = resp.split(',')[0] if resp else '-1'
                if code != '0':
                    status(f"Home failed: {resp}")
                    r.send(f'SpeedFactor({speed})')
                else:
                    # Wait for motion
                    prev = r.get_angles()
                    for _ in range(150):  # up to 30s
                        time.sleep(0.2)
                        cur = r.get_angles()
                        if prev and cur:
                            if max(abs(cur[i] - prev[i]) for i in range(6)) < 0.05:
                                break
                        prev = cur
                    r.send(f'SpeedFactor({speed})')
                    angles = r.get_angles()
                    if angles:
                        val = ','.join(f'{v:.1f}' for v in angles)
                        status(f"Home done  Joints: {val}")
                    else:
                        status("Home done")
                flush_input()

            # Raw command
            elif ch == '/':
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
                sys.stdout.write("\r\n")
                cmd = input("  Command> ")
                if cmd.strip():
                    resp = r.send(cmd)
                    print(f"  -> {resp}")
                tty.setcbreak(fd)
                status(f"Speed:{speed}%  Step:{cart_step}mm")

            # Help
            elif ch == 'h' or ch == '?':
                sys.stdout.write(HELP)
                status(f"Speed:{speed}%  Step:{cart_step}mm")

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
