#!/usr/bin/env python3
"""Interactive control panel for Dobot Nova5.

Hold jog keys for continuous joint motion, release to stop.
Cartesian keys step via IK + joint jog.
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
        self.sock.close()


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
                              m  Mode
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
    cart_step = CART_STEP_MM

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    def status(msg):
        sys.stdout.write(f"\r\x1b[K  {msg}")
        sys.stdout.flush()

    sys.stdout.write(HELP)
    status(f"Ready. Speed:{speed}%  Step:{cart_step}mm")

    jogging = None
    last_jog_key = 0.0
    JOG_GRACE = 0.15

    def do_cart_step(axis_idx, sign):
        """Step in Cartesian space: read pose, offset, IK, jog to target."""
        pose = r.get_pose()
        if not pose or len(pose) < 6:
            status("ERROR: can't read pose")
            return

        # Apply offset
        step = cart_step if axis_idx < 3 else CART_STEP_DEG
        target_pose = list(pose)
        target_pose[axis_idx] += sign * step

        axis_name = CART_LABELS[axis_idx]
        dir_ch = '+' if sign > 0 else '-'
        status(f"{axis_name}{dir_ch} -> IK solving...")

        # Solve IK
        target_joints = r.inverse_kin(*target_pose)
        if not target_joints or len(target_joints) < 6:
            status(f"ERROR: IK failed for {axis_name}{dir_ch}")
            return

        # Find the joint that needs to move most
        current_joints = r.get_angles()
        if not current_joints:
            status("ERROR: can't read joints")
            return

        # Jog each joint that needs to move (largest error first)
        errors = [(abs(target_joints[i] - current_joints[i]), i) for i in range(6)]
        errors.sort(reverse=True)

        for err, ji in errors:
            if err < 0.5:  # skip joints within 0.5 deg
                continue
            direction = '+' if target_joints[ji] > current_joints[ji] else '-'
            jog_time = min(1.0, err / 15.0)  # rough scaling
            jog_time = max(0.08, jog_time)

            r.send(f'MoveJog(J{ji+1}{direction})')
            time.sleep(jog_time)
            r.send('MoveJog()')
            time.sleep(0.1)

        # Report result
        new_pose = r.get_pose()
        if new_pose:
            val = ','.join(f'{v:.1f}' for v in new_pose)
            status(f"{axis_name}{dir_ch} done  Pose: {val}")
        else:
            status(f"{axis_name}{dir_ch} done")

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
