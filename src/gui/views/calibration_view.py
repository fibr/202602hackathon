"""Calibration menu view: navigate to embedded calibration sub-views.

Pressing a number key switches directly to the corresponding embedded view
(servo_calib, handeye_yellow, checkerboard, verify_calib) via
app.switch_view().  All tools are now fully embedded — no subprocess needed.
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

    def __init__(self, app):
        super().__init__(app)

    def setup(self):
        pass

    def update(self, canvas):
        vw = self.app.view_width
        vh = self.app.canvas_height
        canvas[:vh, :vw] = (30, 30, 35)

        cv2.putText(canvas, 'Calibration Tools', (20, 35),
                    FONT, 0.6, (255, 200, 100), 1)
        cv2.putText(canvas, 'Select a tool to open it in the same window',
                    (20, 58), FONT, 0.38, (150, 150, 150), 1)
        cv2.line(canvas, (10, 68), (vw - 10, 68), (60, 60, 70), 1)

        y = 95
        options = [
            ('[1] Checkerboard Calibration',
             'Intrinsics + click TCP on board, ray-plane solve'),
            ('[2] Servo Calibration',
             'Move arm to zero pose, save offsets  (arm101)'),
            ('[3] Hand-Eye Calibration (Yellow Tape)',
             'Capture FK+pixel poses, joint solve  (arm101)'),
            ('[4] Verify Checkerboard',
             'Move arm above board corners to verify calibration'),
            ('[5] Servo Direction Auto-Calib',
             'Auto-detect servo signs + offsets via yellow tape (arm101)'),
        ]
        for label, desc in options:
            cv2.putText(canvas, label, (30, y), FONT, 0.42, (180, 220, 255), 1)
            cv2.putText(canvas, desc, (50, y + 18), FONT, 0.32,
                        (120, 120, 120), 1)
            y += 54

        # Footer hint
        cv2.putText(canvas,
                    '[1-5] open embedded view',
                    (20, vh - 15), FONT, 0.32, (80, 80, 80), 1)

    def handle_key(self, key):
        if key == ord('1'):
            self.app.switch_view('checkerboard')
            return True
        if key == ord('2'):
            self.app.switch_view('servo_calib')
            return True
        if key == ord('3'):
            self.app.switch_view('handeye_yellow')
            return True
        if key == ord('4'):
            self.app.switch_view('verify_calib')
            return True
        if key == ord('5'):
            self.app.switch_view('servo_direction')
            return True
        return False
