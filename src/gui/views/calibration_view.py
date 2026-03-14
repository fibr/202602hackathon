"""Calibration view: servo and hand-eye calibration for arm101.

Launches calibration_gui.py in its own window (complex OpenCV GUI that
manages its own event loop). The unified GUI shows a launcher panel.
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
class CalibrationView(BaseView):
    view_id = 'calibration'
    view_name = 'Calibration'
    description = 'Servo & hand-eye calibration'
    needs_camera = False
    needs_robot = False
    headless_ok = False

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

        cv2.putText(canvas, 'Calibration Tools', (20, 35),
                    FONT, 0.6, (255, 200, 100), 1)
        cv2.putText(canvas, 'arm101 servo and hand-eye calibration',
                    (20, 58), FONT, 0.38, (150, 150, 150), 1)
        cv2.line(canvas, (10, 68), (vw - 10, 68), (60, 60, 70), 1)

        y = 95
        options = [
            ('[1] Servo Calibration',
             'Move arm to zero pose, save offsets'),
            ('[2] Hand-Eye Calibration',
             'Capture poses with yellow tape marker'),
            ('[3] Checkerboard Calibration',
             'Interactive hand-eye with checkerboard'),
            ('[4] Verify Checkerboard Calibration',
             'Hover above board corners to verify'),
        ]
        for label, desc in options:
            color = (100, 100, 100) if self._running else (180, 220, 255)
            cv2.putText(canvas, label, (30, y), FONT, 0.42, color, 1)
            cv2.putText(canvas, desc, (50, y + 18), FONT, 0.32, (120, 120, 120), 1)
            y += 48

        # Status
        y += 10
        status_color = (0, 200, 255) if self._running else (200, 200, 200)
        cv2.putText(canvas, f'Status: {self._status}', (20, y),
                    FONT, 0.4, status_color, 1)

        # Output
        y += 30
        lines = self._output.split('\n')[-15:]
        for line in lines:
            y += 16
            cv2.putText(canvas, line[:80], (25, y),
                        FONT, 0.3, (160, 160, 160), 1)

        # Note
        cv2.putText(canvas,
                    'Each tool opens its own window. Close it to return here.',
                    (20, vh - 15), FONT, 0.32, (80, 80, 80), 1)

    def handle_key(self, key):
        if self._running:
            return False

        if key == ord('1'):
            self._launch('calibration_gui.py', [])
            return True
        if key == ord('2'):
            self._launch('calibration_gui.py', ['--handeye'])
            return True
        if key == ord('3'):
            self._launch('detect_checkerboard.py', [])
            return True
        if key == ord('4'):
            self._launch('detect_checkerboard.py', ['--verify'])
            return True
        return False

    def _launch(self, script_name, extra_args):
        """Launch a calibration script in a separate process."""
        self._running = True
        self._status = f'Running {script_name}...'
        self._output = ''
        script = os.path.join(_PROJECT_ROOT, 'scripts', script_name)

        # Pass through common flags
        args = [sys.executable, script] + extra_args
        if getattr(self.app.args, 'safe', False):
            args.append('--safe')
        if getattr(self.app.args, 'sd', False):
            args.append('--sd')

        def _run():
            try:
                result = subprocess.run(
                    args, capture_output=True, text=True, timeout=600,
                    cwd=_PROJECT_ROOT)
                self._output = result.stdout[-2000:] if result.stdout else ''
                if result.stderr:
                    self._output += '\n' + result.stderr[-500:]
                self._status = f'{script_name} exited (code {result.returncode})'
            except subprocess.TimeoutExpired:
                self._status = f'{script_name} timed out'
            except Exception as e:
                self._status = f'Error: {e}'
            self._running = False

        threading.Thread(target=_run, daemon=True).start()
