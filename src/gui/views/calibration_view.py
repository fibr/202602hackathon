"""Calibration menu view: navigate to embedded calibration sub-views.

Clicking a button (or pressing the legacy number key) switches directly to
the corresponding embedded view via app.switch_view().  All tools are fully
embedded — no subprocess needed.
"""

import os

import cv2
import numpy as np

from gui.views.base import BaseView, ViewRegistry

FONT = cv2.FONT_HERSHEY_SIMPLEX
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))


@ViewRegistry.register
class CalibrationView(BaseView):
    view_id = 'calibration'
    view_name = 'Calibration'
    description = 'Servo & hand-eye calibration'
    needs_camera = False
    needs_robot = False
    headless_ok = False

    # (view_id, label, description)
    OPTIONS = [
        ('checkerboard', 'Checkerboard Calibration',
         'Intrinsics + click TCP on board, ray-plane solve'),
        ('servo_calib', 'Servo Calibration',
         'Move arm to zero pose, save offsets  (arm101)'),
        ('handeye_yellow', 'Hand-Eye Calibration (Yellow Tape)',
         'Capture FK+pixel poses, joint solve  (arm101)'),
        ('verify_calib', 'Verify Checkerboard',
         'Move arm above board corners to verify calibration'),
        ('servo_direction', 'Servo Direction Auto-Calib',
         'Auto-detect servo signs + offsets via yellow tape (arm101)'),
    ]

    def __init__(self, app):
        super().__init__(app)
        self._buttons = []      # [(x1, y1, x2, y2, view_id), ...]
        self._hover_pos = (-1, -1)

    def setup(self):
        pass

    def update(self, canvas):
        vw = self.app.view_width
        vh = self.app.canvas_height
        canvas[:vh, :vw] = (30, 30, 35)
        self._buttons = []

        cv2.putText(canvas, 'Calibration Tools', (20, 35),
                    FONT, 0.6, (255, 200, 100), 1)
        cv2.putText(canvas, 'Click a tool to open it in the same window',
                    (20, 58), FONT, 0.38, (150, 150, 150), 1)
        cv2.line(canvas, (10, 68), (vw - 10, 68), (60, 60, 70), 1)

        btn_h = 52
        btn_w = vw - 40
        hx, hy = self._hover_pos
        y = 82

        for view_id, label, desc in self.OPTIONS:
            x1, y1 = 20, y
            x2, y2 = x1 + btn_w, y + btn_h

            is_hover = (x1 <= hx <= x2 and y1 <= hy <= y2)

            bg = (55, 50, 65) if is_hover else (40, 40, 50)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
            border_col = (120, 160, 220) if is_hover else (70, 70, 90)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 1)

            text_col = (230, 245, 255) if is_hover else (180, 220, 255)
            cv2.putText(canvas, label, (x1 + 14, y1 + 22),
                        FONT, 0.42, text_col, 1)
            cv2.putText(canvas, desc, (x1 + 14, y1 + 40),
                        FONT, 0.32, (130, 130, 140) if not is_hover else (160, 160, 170), 1)

            self._buttons.append((x1, y1, x2, y2, view_id))
            y += btn_h + 8

    def handle_key(self, key):
        # Legacy number-key shortcuts (kept for backwards compatibility)
        key_map = {
            ord('1'): 'checkerboard',
            ord('2'): 'servo_calib',
            ord('3'): 'handeye_yellow',
            ord('4'): 'verify_calib',
            ord('5'): 'servo_direction',
        }
        if key in key_map:
            self.app.switch_view(key_map[key])
            return True
        return False

    def handle_mouse(self, event, x, y, flags):
        if event == cv2.EVENT_MOUSEMOVE:
            self._hover_pos = (x, y)
            return False
        if event == cv2.EVENT_LBUTTONDOWN:
            for x1, y1, x2, y2, view_id in self._buttons:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.app.switch_view(view_id)
                    return True
        return False

    def cleanup(self):
        pass
