#!/usr/bin/env python3
"""Interactive control panel for Dobot Nova5.

Hold jog keys for continuous joint motion, release to stop.
Cartesian keys step via local IK + MovJ(joint={...}) on dashboard port.
Gripper and other commands are instant.
"""

import sys
import os
import socket
import time
import select
import tty
import termios
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config_loader import load_config
from kinematics import IKSolver


class FastRobot:
    """Minimal low-latency dashboard connection with auto-reconnect."""

    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = None
        self._connect()

    def _connect(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(2)
        self.sock.connect((self.ip, self.port))
        try:
            self.sock.recv(1024)
        except socket.timeout:
            pass

    def send(self, cmd):
        for attempt in range(3):
            try:
                self.sock.send(f"{cmd}\n".encode())
                time.sleep(0.05)
                try:
                    return self.sock.recv(4096).decode().strip()
                except socket.timeout:
                    return ""
            except (BrokenPipeError, ConnectionResetError, OSError):
                if attempt < 2:
                    print(f"\r\n  Connection lost, reconnecting ({attempt+2}/3)...")
                    time.sleep(1)
                    try:
                        self._connect()
                    except (ConnectionRefusedError, socket.timeout, OSError):
                        continue
                else:
                    print("\r\n  ERROR: Cannot reach robot after 3 attempts.")
                    print("  Check: is another dashboard session open? (only one allowed)")
                    raise

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
=== Dobot Nova5 Control Panel (Local IK) ===

 JOINT JOG (hold)            CARTESIAN (local IK+joints) GRIPPER
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
                              m  Mode       0  Home      l  Log position
"""


def main():
    config = load_config()
    rc = config.get('robot', {})
    gc = config.get('gripper', {})
    ip = rc.get('ip', '192.168.5.1')
    port = rc.get('dashboard_port', 29999)

    print(f"Connecting to {ip}:{port}...")
    r = FastRobot(ip, port)

    # Initialize local IK solver
    tool_length = gc.get('tool_length_mm', 100.0)
    ik = IKSolver(tool_length_mm=tool_length)
    print(f"  IK solver loaded (tool_length={tool_length}mm)")

    # Enable
    r.send('DisableRobot()')
    time.sleep(1)
    r.send('ClearError()')
    r.send('EnableRobot()')
    time.sleep(1)

    speed = 30
    r.send(f'SpeedFactor({speed})')
    cart_step = CART_STEP_MM

    # Position log file
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    pos_log_path = os.path.join(log_dir, f'positions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    with open(pos_log_path, 'w') as f:
        f.write('timestamp,x,y,z,rx,ry,rz,j1,j2,j3,j4,j5,j6,label\n')
    pos_log_count = 0

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
    print(f"  Position log: {os.path.relpath(pos_log_path)}")
    status(f"Ready. Speed:{speed}%  Step:{cart_step}mm")

    jogging = None
    last_jog_key = 0.0
    JOG_GRACE = 0.15

    def do_cart_step(axis_idx, sign):
        """Step in Cartesian space using local IK + joint-angle move."""
        # Get current joint angles and compute current pose via local FK
        angles = r.get_angles()
        if not angles or len(angles) < 6:
            status("ERROR: can't read joint angles")
            return

        current_joints = np.array(angles)
        current_pos, current_rpy = ik.forward_kin(current_joints)

        step = cart_step if axis_idx < 3 else CART_STEP_DEG
        target_pos = current_pos.copy()
        target_rpy = current_rpy.copy()
        if axis_idx < 3:
            target_pos[axis_idx] += sign * step
        else:
            target_rpy[axis_idx - 3] += sign * step

        axis_name = CART_LABELS[axis_idx]
        dir_ch = '+' if sign > 0 else '-'
        status(f"{axis_name}{dir_ch} IK solving...")

        # Solve IK with current joints as seed
        target_joints = ik.solve_ik(target_pos, target_rpy,
                                     seed_joints_deg=current_joints)
        if target_joints is None:
            status(f"ERROR: IK failed for {axis_name}{dir_ch}")
            return

        # Send joint-angle move via V4 syntax on dashboard
        j = target_joints
        cmd = f'MovJ(joint={{{j[0]:.4f},{j[1]:.4f},{j[2]:.4f},{j[3]:.4f},{j[4]:.4f},{j[5]:.4f}}})'
        resp = r.send(cmd)
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

        # Show final pose from local FK
        final_angles = r.get_angles()
        if final_angles:
            final_pos, final_rpy = ik.forward_kin(np.array(final_angles))
            val = ','.join(f'{v:.1f}' for v in np.concatenate([final_pos, final_rpy]))
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

            # Log position
            elif ch == 'l':
                pose = r.get_pose()
                angles = r.get_angles()
                if pose and angles:
                    # Prompt for optional label
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
                    sys.stdout.write("\r\n")
                    label = input("  Label (enter to skip)> ").strip()
                    tty.setcbreak(fd)
                    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    row = ','.join(f'{v:.3f}' for v in pose + angles)
                    with open(pos_log_path, 'a') as f:
                        f.write(f'{ts},{row},{label}\n')
                    pos_log_count += 1
                    pv = ','.join(f'{v:.1f}' for v in pose)
                    status(f"Logged #{pos_log_count}: [{pv}] {label}")
                else:
                    status("Log failed: can't read pose/joints")

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

            # Home (using joint angles via V4 syntax)
            elif ch == '0':
                if jogging:
                    r.send('MoveJog()')
                    jogging = None
                status("Homing (joint angles)...")
                r.send('SpeedFactor(10)')
                resp = r.send('MovJ(joint={43.5,-13.9,-85.4,196.3,-90.0,43.5})')
                code = resp.split(',')[0] if resp else '-1'
                if code != '0':
                    status(f"Home failed: {resp}")
                    r.send(f'SpeedFactor({speed})')
                else:
                    prev = r.get_angles()
                    for _ in range(150):
                        time.sleep(0.2)
                        cur = r.get_angles()
                        if prev and cur:
                            if max(abs(cur[i] - prev[i]) for i in range(6)) < 0.05:
                                break
                        prev = cur
                    r.send(f'SpeedFactor({speed})')
                    final = r.get_angles()
                    if final:
                        val = ','.join(f'{v:.1f}' for v in final)
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
        if pos_log_count:
            print(f"  Saved {pos_log_count} positions to {os.path.relpath(pos_log_path)}")
        print("  Done.")


if __name__ == "__main__":
    main()
