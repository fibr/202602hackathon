"""Calibration menu view: navigate to embedded calibration sub-views.

Pressing a number key switches directly to the corresponding embedded view
(servo_calib, handeye_yellow, checkerboard) via app.switch_view().  The
subprocess-based launcher is kept only for the --verify mode of
detect_checkerboard.py which uses console input() and is not easily
embeddable in the GUI event loop.
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
        self._verify_status = 'Ready'
        self._verify_running = False
        self._verify_output = ''

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
            ('[1] Servo Calibration',
             'Move arm to zero pose, save offsets  (arm101)'),
            ('[2] Hand-Eye Calibration (Yellow Tape)',
             'Capture FK+pixel poses, joint solve  (arm101)'),
            ('[3] Checkerboard Calibration',
             'Click TCP on board, ray-plane solve'),
            ('[4] Verify Checkerboard (subprocess)',
             'Move arm above board corners to verify'),
        ]
        for i, (label, desc) in enumerate(options):
            # Option 4 is subprocess-only; dim it if subprocess is running
            if i == 3:
                running = self._verify_running
                color = (100, 100, 100) if running else (180, 220, 255)
            else:
                color = (180, 220, 255)
            cv2.putText(canvas, label, (30, y), FONT, 0.42, color, 1)
            cv2.putText(canvas, desc, (50, y + 18), FONT, 0.32,
                        (120, 120, 120), 1)
            y += 54

        # Subprocess status (verify only)
        y += 10
        if self._verify_running or self._verify_output or self._verify_status != 'Ready':
            status_color = (0, 200, 255) if self._verify_running else (200, 200, 200)
            cv2.putText(canvas, f'Verify: {self._verify_status}', (20, y),
                        FONT, 0.4, status_color, 1)
            y += 30
            lines = self._verify_output.split('\n')[-12:]
            for line in lines:
                y += 16
                cv2.putText(canvas, line[:80], (25, y),
                            FONT, 0.3, (160, 160, 160), 1)

        # Footer hint
        cv2.putText(canvas,
                    '[1-3] open embedded view  |  [4] subprocess (console input required)',
                    (20, vh - 15), FONT, 0.32, (80, 80, 80), 1)

    def handle_key(self, key):
        if key == ord('1'):
            self.app.switch_view('servo_calib')
            return True
        if key == ord('2'):
            self.app.switch_view('handeye_yellow')
            return True
        if key == ord('3'):
            self.app.switch_view('checkerboard')
            return True
        if key == ord('4'):
            if not self._verify_running:
                self._launch_verify()
            return True
        return False

    # ------------------------------------------------------------------
    # Subprocess launcher for verify mode (uses console input(), not embeddable)
    # ------------------------------------------------------------------

    def _launch_verify(self):
        """Launch detect_checkerboard.py --verify in a subprocess."""
        self._verify_running = True
        self._verify_status = 'Running verify...'
        self._verify_output = ''
        script = os.path.join(_PROJECT_ROOT, 'scripts', 'detect_checkerboard.py')

        args = [sys.executable, script, '--verify']
        if getattr(self.app.args, 'safe', False):
            args.append('--safe')
        if getattr(self.app.args, 'sd', False):
            args.append('--sd')
        # Pass robot type flag
        if self.app.config.get('robot_type') == 'arm101':
            args.append('--arm101')

        def _run():
            try:
                result = subprocess.run(
                    args, capture_output=True, text=True,
                    timeout=600, cwd=_PROJECT_ROOT)
                self._verify_output = (result.stdout[-2000:]
                                       if result.stdout else '')
                if result.stderr:
                    self._verify_output += '\n' + result.stderr[-500:]
                self._verify_status = (
                    f'verify exited (code {result.returncode})')
            except subprocess.TimeoutExpired:
                self._verify_status = 'verify timed out'
            except Exception as exc:
                self._verify_status = f'Error: {exc}'
            self._verify_running = False

        threading.Thread(target=_run, daemon=True).start()
