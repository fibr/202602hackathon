#!/usr/bin/env python3
"""Interactive control panel for robot arms with OpenCV GUI.

Supports:
  - Dobot Nova5 (default): TCP/IP dashboard protocol
  - LeRobot arm101 (--arm101): Feetech STS3215 servos over serial

Camera live view on the left, clickable control panel on the right.
GUI controls: XY jog pad, Z up/down, gripper, speed, enable/home.
Keyboard shortcuts: 1-6/!@#$%^ jog, space stop, c/o gripper, [/] speed,
    v enable/torque, s safe mode (arm101), l log position, p pose, Esc quit.

Flags:
    --arm101     Use LeRobot arm101 instead of Dobot Nova5
    --safe       Start in safe mode (reduced torque/speed, arm101 only)
    --no-camera  Run without camera
    --sd         Use 640x480 instead of 1280x720
"""

import sys
import os
import socket
import time
import cv2
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config_loader import load_config
from gui.robot_controls import RobotControlPanel, PANEL_WIDTH


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
                    print(f"  Connection lost, reconnecting ({attempt+2}/3)...")
                    time.sleep(1)
                    try:
                        self._connect()
                    except (ConnectionRefusedError, socket.timeout, OSError):
                        continue
                else:
                    print("  ERROR: Cannot reach robot after 3 attempts.")
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


def connect_dobot(config):
    """Connect to Dobot Nova5 and enable."""
    rc = config.get('robot', {})
    ip = rc.get('ip', '192.168.5.1')
    port = rc.get('dashboard_port', 29999)

    print(f"=== Dobot Nova5 Control Panel ===")
    print(f"Connecting to {ip}:{port}...")
    try:
        r = FastRobot(ip, port)
    except ConnectionRefusedError:
        print(f"  ERROR: Connection refused at {ip}:{port}")
        print(f"  - Is the robot powered on and booted? (takes ~60s after power cycle)")
        print(f"  - Is another dashboard session already connected? (only one allowed)")
        sys.exit(1)
    except socket.timeout:
        print(f"  ERROR: Connection timed out to {ip}:{port}")
        print(f"  - Is the robot on the network? Try: ping {ip}")
        sys.exit(1)
    except OSError as e:
        print(f"  ERROR: Cannot connect to robot: {e}")
        sys.exit(1)
    print("  Connected.")

    # Enable
    r.send('DisableRobot()')
    time.sleep(1)
    r.send('ClearError()')
    r.send('EnableRobot()')
    time.sleep(1)

    speed = 30
    r.send(f'SpeedFactor({speed})')
    return r, speed


def connect_arm101(config, safe_mode=False):
    """Connect to LeRobot arm101 follower.

    Args:
        config: Loaded robot config dict.
        safe_mode: If True, use reduced torque and speed for safety.
    """
    from robot.lerobot_arm101 import LeRobotArm101

    ac = config.get('arm101', {})
    port = ac.get('port', '')
    baudrate = ac.get('baudrate', 1_000_000)
    motor_ids = ac.get('motor_ids', [1, 2, 3, 4, 5, 6])
    speed = ac.get('speed', 200)

    print(f"=== LeRobot arm101 Control Panel ===")
    if safe_mode:
        print("  ** SAFE MODE: reduced torque and speed **")

    if not port:
        print("Auto-detecting serial port...")
        try:
            port = LeRobotArm101.find_port()
            print(f"  Found: {port}")
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            sys.exit(1)

    print(f"Connecting to {port} @ {baudrate}...")
    try:
        arm = LeRobotArm101(port=port, baudrate=baudrate,
                            motor_ids=motor_ids, speed=speed,
                            safe_mode=safe_mode)
        arm.connect()
    except Exception as e:
        print(f"  ERROR: Cannot connect to arm101: {e}")
        sys.exit(1)

    # Enable torque
    print("  Enabling torque...")
    arm.enable_torque()
    print("  Ready.")
    return arm, arm.speed


def main():
    config = load_config()
    use_arm101 = '--arm101' in sys.argv
    safe_mode = '--safe' in sys.argv

    # Camera resolution
    sd = '--sd' in sys.argv
    no_camera = '--no-camera' in sys.argv
    cam_width, cam_height = (640, 480) if sd else (1280, 720)

    # Connect robot
    if use_arm101:
        robot, speed = connect_arm101(config, safe_mode=safe_mode)
    else:
        if safe_mode:
            print("  Note: --safe is only supported for arm101, ignoring.")
        robot, speed = connect_dobot(config)

    # Start camera (optional)
    camera = None
    if not no_camera:
        try:
            from vision import create_camera
            cam_cfg = config.get('camera', {})
            cam_panel_config = dict(config)
            cam_panel_config['camera'] = dict(cam_cfg, width=cam_width, height=cam_height)
            cam_type = cam_cfg.get('type', 'realsense')
            print(f"Starting {cam_type} camera ({cam_width}x{cam_height})...")
            camera = create_camera(cam_panel_config)
            camera.start()
            # Use actual resolution (camera may not support requested size)
            cam_width, cam_height = camera.width, camera.height
            print(f"  Camera started ({cam_width}x{cam_height}).")
        except Exception as e:
            print(f"  WARNING: Camera not available: {e}")
            print("  Running without camera. Use --no-camera to suppress this.")
            no_camera = True

    if no_camera:
        # Create a blank frame instead
        cam_width, cam_height = 640, 480

    # Position log file
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    pos_log_path = os.path.join(log_dir, f'positions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    with open(pos_log_path, 'w') as f:
        f.write('timestamp,j1,j2,j3,j4,j5,j6,x,y,z,rx,ry,rz,label\n')
    pos_log_count = 0

    # GUI panel
    panel = RobotControlPanel(robot, panel_x=cam_width, panel_height=cam_height)
    panel.speed = speed

    robot_name = "arm101" if use_arm101 else "Dobot Nova5"
    print()
    print(f"Robot: {robot_name}")
    if use_arm101:
        safe_str = " [SAFE MODE]" if safe_mode else ""
        print(f"Mode: arm101{safe_str}")
        print("Keyboard: 1-6/!@#$%^ jog, space stop, c/o gripper, [/] speed")
        print("          v torque, s safe mode, l log position, p pose, Esc quit")
    else:
        print("Keyboard: 1-6/!@#$%^ jog, space stop, c/o gripper, [/] speed")
        print("          v enable, l log position, p pose, Esc quit")
    print(f"Position log: {os.path.relpath(pos_log_path)}")

    def on_mouse(event, x, y, flags, param):
        if x >= cam_width:
            panel.handle_mouse(event, x, y, flags)

    cv2.namedWindow('Control Panel')
    cv2.setMouseCallback('Control Panel', on_mouse)

    try:
        while True:
            # Get camera frame or blank
            if camera is not None:
                color_image, depth_image, depth_frame = camera.get_frames()
                if color_image is None:
                    continue
            else:
                color_image = np.zeros((cam_height, cam_width, 3), dtype=np.uint8)
                cv2.putText(color_image, "No Camera", (cam_width // 2 - 80, cam_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)

            # Create canvas: camera on left, panel on right
            canvas = np.zeros((cam_height, cam_width + PANEL_WIDTH, 3), dtype=np.uint8)
            canvas[0:cam_height, 0:cam_width] = color_image

            # Status overlay on camera
            bar_text = f"Spd:{panel.speed}"
            if not use_arm101:
                bar_text += "%"
            if panel.jogging:
                bar_text += f" | JOG {panel.jog_axis}"
            pose = robot.get_pose()
            if pose:
                bar_text += f" | [{pose[0]:.0f},{pose[1]:.0f},{pose[2]:.0f}]mm"
            cv2.putText(canvas, bar_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(canvas, "l log | p pose | Esc quit",
                        (10, cam_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

            # Draw panel
            panel.draw(canvas)

            cv2.imshow('Control Panel', canvas)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # Esc
                break
            try:
                if cv2.getWindowProperty('Control Panel', cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            # Arm control via shared panel
            if key != 255 and panel.handle_key(key):
                pass

            # Log position
            elif key == ord('l'):
                angles = robot.get_angles()
                if angles:
                    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    row = ','.join(f'{v:.3f}' for v in angles)
                    pose = robot.get_pose()
                    if pose:
                        row += ',' + ','.join(f'{v:.3f}' for v in pose)
                    else:
                        row += ',,,,,,'
                    with open(pos_log_path, 'a') as f:
                        f.write(f'{ts},{row},\n')
                    pos_log_count += 1
                    av = ','.join(f'{v:.1f}' for v in angles)
                    panel.status_msg = f"Logged #{pos_log_count}: [{av}]"
                    print(f"  {panel.status_msg}")
                else:
                    panel.status_msg = "Log failed: can't read angles"

            # Print pose
            elif key == ord('p'):
                angles = robot.get_angles()
                if angles:
                    print(f"  Joints: {', '.join(f'{v:.2f}' for v in angles)}")
                    pose = robot.get_pose()
                    if pose:
                        print(f"  Pose:   {', '.join(f'{v:.2f}' for v in pose)}")
                    panel.status_msg = "Pose printed"

    except KeyboardInterrupt:
        pass
    finally:
        if use_arm101:
            robot.close()
        else:
            robot.send('MoveJog()')  # stop any jog
            robot.close()
        if camera is not None:
            camera.stop()
        cv2.destroyAllWindows()
        if pos_log_count:
            print(f"  Saved {pos_log_count} positions to {os.path.relpath(pos_log_path)}")
        print("  Done.")


if __name__ == "__main__":
    main()
