"""Extra Scripts view: launcher for miscellaneous utility scripts.

Provides a menu of clickable buttons to launch less-common scripts like
visual servo test, green cube point, visit cubes, ROI selection, etc.
"""

import os
import subprocess
import sys
import threading

import cv2
import numpy as np

from gui.views.base import BaseView, ViewRegistry

FONT = cv2.FONT_HERSHEY_SIMPLEX
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))


@ViewRegistry.register
class ExtrasView(BaseView):
    view_id = 'extras'
    view_name = 'Extra Scripts'
    description = 'Misc utility scripts'
    needs_camera = False
    needs_robot = False
    headless_ok = False

    # Script entries: (label, description, script_path, extra_args)
    SCRIPTS = [
        ('Visual Servo Test', 'Test gripper camera visual servoing',
         'scripts/test_visual_servo.py', []),
        ('Green Cube Point', 'Point robot at green cube',
         'scripts/green_cube_point.py', []),
        ('Visit Cubes', 'Demo: visit colored cubes',
         'scripts/visit_cubes.py', []),
        ('Visit Cubes (calibrated)', 'Calibrated cube visitation',
         'scripts/visit_cubes_calibrated.py', []),
        ('Select ROI', 'Interactive region-of-interest selection',
         'scripts/select_roi.py', []),
        ('Evaluate Dataset', 'Run detection on saved dataset',
         'scripts/eval_dataset.py', []),
        ('Test arm101 FK', 'Forward kinematics validation',
         'scripts/test_arm101_fk.py', []),
    ]

    def __init__(self, app):
        super().__init__(app)
        self._status = 'Ready'
        self._running = False
        self._output = ''
        self._buttons = []      # [(x1, y1, x2, y2, idx), ...]
        self._hover_pos = (-1, -1)

    def setup(self):
        pass

    def update(self, canvas):
        vw = self.app.view_width
        vh = self.app.canvas_height
        canvas[:vh, :vw] = (30, 30, 35)
        self._buttons = []

        cv2.putText(canvas, 'Extra Scripts', (20, 35),
                    FONT, 0.6, (255, 200, 100), 1)
        cv2.putText(canvas, 'Miscellaneous utility and test scripts',
                    (20, 58), FONT, 0.38, (150, 150, 150), 1)
        cv2.line(canvas, (10, 68), (vw - 10, 68), (60, 60, 70), 1)

        btn_h = 44
        btn_w = vw - 40
        hx, hy = self._hover_pos
        y = 78

        for idx, (label, desc, script, _) in enumerate(self.SCRIPTS):
            script_path = os.path.join(_PROJECT_ROOT, script)
            exists = os.path.exists(script_path)
            disabled = self._running or not exists

            x1, y1 = 20, y
            x2, y2 = x1 + btn_w, y + btn_h

            is_hover = (not disabled) and (x1 <= hx <= x2 and y1 <= hy <= y2)

            if disabled:
                bg = (35, 35, 38)
                border_col = (55, 55, 60)
                text_col = (90, 90, 90)
                desc_col = (70, 70, 75)
            elif is_hover:
                bg = (55, 50, 65)
                border_col = (120, 160, 220)
                text_col = (230, 245, 255)
                desc_col = (160, 160, 170)
            else:
                bg = (40, 40, 50)
                border_col = (70, 70, 90)
                text_col = (180, 220, 255)
                desc_col = (110, 110, 120)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 1)

            label_display = label if exists else f'{label} (missing)'
            cv2.putText(canvas, label_display, (x1 + 14, y1 + 18),
                        FONT, 0.4, text_col, 1)
            cv2.putText(canvas, desc, (x1 + 14, y1 + 34),
                        FONT, 0.3, desc_col, 1)

            if not disabled:
                self._buttons.append((x1, y1, x2, y2, idx))
            y += btn_h + 5

        # Status
        y += 6
        status_color = (0, 200, 255) if self._running else (200, 200, 200)
        cv2.putText(canvas, f'Status: {self._status}', (20, y),
                    FONT, 0.4, status_color, 1)

        # Output tail
        y += 22
        lines = self._output.split('\n')[-6:]
        for line in lines:
            y += 14
            if y > vh - 20:
                break
            cv2.putText(canvas, line[:80], (25, y),
                        FONT, 0.28, (150, 150, 150), 1)

        cv2.putText(canvas,
                    'Each tool opens in its own window/terminal.',
                    (20, vh - 15), FONT, 0.32, (80, 80, 80), 1)

    def handle_key(self, key):
        # Legacy single-digit shortcuts kept for backwards compatibility
        if self._running:
            return False
        if ord('1') <= key <= ord('7'):
            idx = key - ord('1')
            if idx < len(self.SCRIPTS):
                label, _desc, script, extra_args = self.SCRIPTS[idx]
                self._launch(label, script, extra_args)
                return True
        return False

    def handle_mouse(self, event, x, y, flags):
        if event == cv2.EVENT_MOUSEMOVE:
            self._hover_pos = (x, y)
            return False
        if event == cv2.EVENT_LBUTTONDOWN:
            for x1, y1, x2, y2, idx in self._buttons:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    label, _desc, script, extra_args = self.SCRIPTS[idx]
                    self._launch(label, script, extra_args)
                    return True
        return False

    def _launch(self, label, script, extra_args):
        script_path = os.path.join(_PROJECT_ROOT, script)
        if not os.path.exists(script_path):
            self._status = f'{label}: script not found'
            return

        self._running = True
        self._status = f'Running {label}...'
        self._output = ''
        args = [sys.executable, script_path] + extra_args

        def _run():
            try:
                result = subprocess.run(
                    args, capture_output=True, text=True, timeout=300,
                    cwd=_PROJECT_ROOT)
                self._output = result.stdout[-2000:] if result.stdout else ''
                if result.stderr:
                    self._output += '\n' + result.stderr[-500:]
                self._status = f'{label} exited (code {result.returncode})'
            except subprocess.TimeoutExpired:
                self._status = f'{label} timed out'
            except Exception as e:
                self._status = f'Error: {e}'
            self._running = False

        threading.Thread(target=_run, daemon=True).start()
