"""Control Panel view: live camera + robot jog controls.

Wraps the existing RobotControlPanel into a unified GUI view.
"""

import os
import time
from datetime import datetime

import cv2
import numpy as np

from gui.views.base import BaseView, ViewRegistry
from gui.robot_controls import RobotControlPanel, PANEL_WIDTH

FONT = cv2.FONT_HERSHEY_SIMPLEX

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))


@ViewRegistry.register
class ControlPanelView(BaseView):
    view_id = 'control'
    view_name = 'Control Panel'
    description = 'Camera + robot jog controls'
    needs_camera = False  # works without (--no-camera)
    needs_robot = True
    headless_ok = False

    def __init__(self, app):
        super().__init__(app)
        self._panel = None
        self._pos_log_path = None
        self._pos_log_count = 0
        self._no_camera = False
        self._cam_width = 640
        self._cam_height = 480
        # Desired panel area width (camera or blank area, excluding robot_controls panel)
        self._panel_area_width = 0

    def setup(self):
        # Connect robot
        self.app.ensure_robot()
        if self.app.robot is None:
            print("  WARNING: No robot connected. Control panel running read-only.")

        # Start camera
        no_cam = getattr(self.app.args, 'no_camera', False)
        if not no_cam:
            self.app.ensure_camera()

        if self.app.camera is not None:
            self._cam_width = self.app.camera.width
            self._cam_height = self.app.camera.height
        else:
            self._no_camera = True
            use_arm101 = self.app.config.get('robot_type') == 'arm101'
            self._cam_height = 800 if use_arm101 else 480

        # Update app view dimensions to fit camera + robot_controls panel
        self._panel_area_width = self._cam_width + PANEL_WIDTH
        self.app.view_width = self._panel_area_width
        self.app.view_height = self._cam_height

        # Create robot control panel
        # The RobotControlPanel draws at panel_x (= camera width)
        self._panel = RobotControlPanel(
            self.app.robot, panel_x=self._cam_width,
            panel_height=self._cam_height)
        speed = self.app.config.get('robot', {}).get('speed_percent', 30)
        if self.app.config.get('robot_type') == 'arm101':
            speed = self.app.config.get('arm101', {}).get('speed', 200)
        self._panel.speed = speed

        # Position log
        log_dir = os.path.join(_PROJECT_ROOT, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._pos_log_path = os.path.join(log_dir, f'positions_{ts}.csv')
        with open(self._pos_log_path, 'w') as f:
            f.write('timestamp,j1,j2,j3,j4,j5,j6,x,y,z,rx,ry,rz,label\n')

    def update(self, canvas):
        # Get camera frame or blank
        if self.app.camera is not None:
            color_image, _, _ = self.app.get_camera_frame()
            if color_image is not None:
                h, w = color_image.shape[:2]
                canvas[0:h, 0:w] = color_image
        elif self._no_camera:
            cv2.putText(canvas, "No Camera", (self._cam_width // 2 - 80,
                        self._cam_height // 2), FONT, 1.0, (100, 100, 100), 2)

        # Status overlay
        if self.app.robot is not None:
            bar_text = f"Spd:{self._panel.speed}"
            if self.app.config.get('robot_type') != 'arm101':
                bar_text += "%"
            if self._panel.jogging:
                bar_text += f" | JOG {self._panel.jog_axis}"
            pose = self.app.robot.get_pose()
            if pose:
                bar_text += f" | [{pose[0]:.0f},{pose[1]:.0f},{pose[2]:.0f}]mm"
            cv2.putText(canvas, bar_text, (10, 25),
                        FONT, 0.45, (0, 255, 0), 1)

        cv2.putText(canvas, "l=log | p=pose",
                    (10, self._cam_height - 10),
                    FONT, 0.38, (200, 200, 200), 1)

        # Draw robot control panel
        self._panel.draw(canvas)

    def handle_key(self, key):
        if self._panel is not None and self._panel.handle_key(key):
            return True

        # Log position
        if key == ord('l'):
            self._log_position()
            return True

        # Print pose
        if key == ord('p'):
            self._print_pose()
            return True

        return False

    def handle_mouse(self, event, x, y, flags):
        if self._panel is not None and x >= self._cam_width:
            return self._panel.handle_mouse(event, x, y, flags)
        return False

    def _log_position(self):
        if self.app.robot is None:
            return
        angles = self.app.robot.get_angles()
        if angles:
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            row = ','.join(f'{v:.3f}' for v in angles)
            pose = self.app.robot.get_pose()
            if pose:
                row += ',' + ','.join(f'{v:.3f}' for v in pose)
            else:
                row += ',,,,,,'
            with open(self._pos_log_path, 'a') as f:
                f.write(f'{ts},{row},\n')
            self._pos_log_count += 1
            av = ','.join(f'{v:.1f}' for v in angles)
            self._panel.status_msg = f"Logged #{self._pos_log_count}: [{av}]"

    def _print_pose(self):
        if self.app.robot is None:
            return
        angles = self.app.robot.get_angles()
        if angles:
            print(f"  Joints: {', '.join(f'{v:.2f}' for v in angles)}")
            pose = self.app.robot.get_pose()
            if pose:
                print(f"  Pose:   {', '.join(f'{v:.2f}' for v in pose)}")
            if self._panel:
                self._panel.status_msg = "Pose printed"

    def cleanup(self):
        if self.app.robot is not None and self.app.config.get('robot_type') != 'arm101':
            try:
                self.app.robot.send('MoveJog()')
            except Exception:
                pass
        if self._pos_log_count:
            print(f"  Saved {self._pos_log_count} positions to {self._pos_log_path}")
