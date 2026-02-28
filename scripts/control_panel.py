#!/usr/bin/env python3
"""Interactive control panel for Dobot Nova5 with OpenCV GUI.

Camera live view on the left, clickable control panel on the right.
GUI controls: XY jog pad, Z up/down, gripper, speed, enable/home.
Keyboard shortcuts: 1-6/!@#$%^ jog, space stop, c/o gripper, [/] speed,
    v enable, l log position, p pose, Esc quit.
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
from vision import RealSenseCamera
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


def main():
    config = load_config()
    rc = config.get('robot', {})
    ip = rc.get('ip', '192.168.5.1')
    port = rc.get('dashboard_port', 29999)

    # Camera resolution
    sd = '--sd' in sys.argv
    cam_width, cam_height = (640, 480) if sd else (1280, 720)

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

    # Start camera
    print(f"Starting camera ({cam_width}x{cam_height})...")
    camera = RealSenseCamera(width=cam_width, height=cam_height, fps=15)
    camera.start()
    print("  Camera started.")

    # Position log file
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    pos_log_path = os.path.join(log_dir, f'positions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    with open(pos_log_path, 'w') as f:
        f.write('timestamp,x,y,z,rx,ry,rz,j1,j2,j3,j4,j5,j6,label\n')
    pos_log_count = 0

    # GUI panel
    panel = RobotControlPanel(r, panel_x=cam_width, panel_height=cam_height)
    panel.speed = speed

    print()
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
            color_image, depth_image, depth_frame = camera.get_frames()
            if color_image is None:
                continue

            # Create canvas: camera on left, panel on right
            canvas = np.zeros((cam_height, cam_width + PANEL_WIDTH, 3), dtype=np.uint8)
            canvas[0:cam_height, 0:cam_width] = color_image

            # Status overlay on camera
            bar_text = f"Spd:{panel.speed}%"
            if panel.jogging:
                bar_text += f" | JOG {panel.jog_axis}"
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
            if cv2.getWindowProperty('Control Panel', cv2.WND_PROP_VISIBLE) < 1:
                break

            # Arm control via shared panel
            if key != 255 and panel.handle_key(key):
                pass

            # Log position
            elif key == ord('l'):
                pose = r.get_pose()
                angles = r.get_angles()
                if pose and angles:
                    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    row = ','.join(f'{v:.3f}' for v in pose + angles)
                    with open(pos_log_path, 'a') as f:
                        f.write(f'{ts},{row},\n')
                    pos_log_count += 1
                    pv = ','.join(f'{v:.1f}' for v in pose)
                    panel.status_msg = f"Logged #{pos_log_count}: [{pv}]"
                    print(f"  {panel.status_msg}")
                else:
                    panel.status_msg = "Log failed: can't read pose"

            # Print pose
            elif key == ord('p'):
                pose = r.get_pose()
                angles = r.get_angles()
                if pose and angles:
                    print(f"  Pose:   {', '.join(f'{v:.2f}' for v in pose)}")
                    print(f"  Joints: {', '.join(f'{v:.2f}' for v in angles)}")
                    panel.status_msg = "Pose printed"

    except KeyboardInterrupt:
        pass
    finally:
        r.send('MoveJog()')  # stop any jog
        camera.stop()
        r.close()
        cv2.destroyAllWindows()
        if pos_log_count:
            print(f"  Saved {pos_log_count} positions to {os.path.relpath(pos_log_path)}")
        print("  Done.")


if __name__ == "__main__":
    main()
