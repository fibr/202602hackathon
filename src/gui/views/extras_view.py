"""Extra Scripts view: launcher for miscellaneous utility scripts.

Provides a menu to launch less-common scripts like visual servo test,
green cube point, visit cubes, ROI selection, etc.
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

    # Script entries: (key, label, description, script_path, extra_args)
    SCRIPTS = [
        ('1', 'Visual Servo Test', 'Test gripper camera visual servoing',
         'scripts/test_visual_servo.py', []),
        ('2', 'Green Cube Point', 'Point robot at green cube',
         'scripts/green_cube_point.py', []),
        ('3', 'Visit Cubes', 'Demo: visit colored cubes',
         'scripts/visit_cubes.py', []),
        ('4', 'Visit Cubes (calibrated)', 'Calibrated cube visitation',
         'scripts/visit_cubes_calibrated.py', []),
        ('5', 'Select ROI', 'Interactive region-of-interest selection',
         'scripts/select_roi.py', []),
        ('6', 'Evaluate Dataset', 'Run detection on saved dataset',
         'scripts/eval_dataset.py', []),
        ('7', 'Test arm101 FK', 'Forward kinematics validation',
         'scripts/test_arm101_fk.py', []),
    ]

    def __init__(self, app):
        super().__init__(app)
        self._status = 'Ready'
        self._running = False
        self._output = ''

    def setup(self):
        pass

    def update(self, canvas):
        vw = self.app.view_width
        vh = self.app.canvas_height
        canvas[:vh, :vw] = (30, 30, 35)

        cv2.putText(canvas, 'Extra Scripts', (20, 35),
                    FONT, 0.6, (255, 200, 100), 1)
        cv2.putText(canvas, 'Miscellaneous utility and test scripts',
                    (20, 58), FONT, 0.38, (150, 150, 150), 1)
        cv2.line(canvas, (10, 68), (vw - 10, 68), (60, 60, 70), 1)

        y = 90
        for key_char, label, desc, script, _ in self.SCRIPTS:
            # Check if script exists
            script_path = os.path.join(_PROJECT_ROOT, script)
            exists = os.path.exists(script_path)
            color = (100, 100, 100) if (self._running or not exists) else (180, 220, 255)
            marker = '' if exists else ' (missing)'

            cv2.putText(canvas, f'[{key_char}] {label}{marker}',
                        (30, y), FONT, 0.4, color, 1)
            cv2.putText(canvas, desc, (50, y + 16),
                        FONT, 0.3, (120, 120, 120), 1)
            y += 40

        # Status
        y += 10
        status_color = (0, 200, 255) if self._running else (200, 200, 200)
        cv2.putText(canvas, f'Status: {self._status}', (20, y),
                    FONT, 0.4, status_color, 1)

        # Output tail
        y += 25
        lines = self._output.split('\n')[-8:]
        for line in lines:
            y += 14
            cv2.putText(canvas, line[:80], (25, y),
                        FONT, 0.28, (150, 150, 150), 1)

        cv2.putText(canvas,
                    'Each tool opens in its own window/terminal.',
                    (20, vh - 15), FONT, 0.32, (80, 80, 80), 1)

    def handle_key(self, key):
        if self._running:
            return False

        for key_char, label, _desc, script, extra_args in self.SCRIPTS:
            if key == ord(key_char):
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
