"""Dataset Collection view: capture rod detection data.

Launches collect_dataset.py in its own window (complex OpenCV GUI).
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
class DatasetView(BaseView):
    view_id = 'dataset'
    view_name = 'Collect Dataset'
    description = 'Rod detection dataset capture'
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

        cv2.putText(canvas, 'Dataset Collection', (20, 35),
                    FONT, 0.6, (255, 200, 100), 1)
        cv2.putText(canvas, 'Capture rod detection training data',
                    (20, 58), FONT, 0.38, (150, 150, 150), 1)
        cv2.line(canvas, (10, 68), (vw - 10, 68), (60, 60, 70), 1)

        y = 95
        options = [
            ('[1] Live Collection',
             'Camera + robot, capture frames interactively'),
            ('[2] Camera Only (no robot)',
             'Capture frames without robot connection'),
            ('[3] Snapshot Debug',
             'Single-frame detection with 6-stage debug images'),
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
        lines = self._output.split('\n')[-12:]
        for line in lines:
            y += 16
            cv2.putText(canvas, line[:80], (25, y),
                        FONT, 0.3, (160, 160, 160), 1)

        cv2.putText(canvas,
                    'Opens its own window. Close it to return here.',
                    (20, vh - 15), FONT, 0.32, (80, 80, 80), 1)

    def handle_key(self, key):
        if self._running:
            return False
        if key == ord('1'):
            self._launch([])
            return True
        if key == ord('2'):
            self._launch(['--no-robot'])
            return True
        if key == ord('3'):
            self._launch(['--snapshot'])
            return True
        return False

    def _launch(self, extra_args):
        self._running = True
        self._status = 'Running collect_dataset.py...'
        self._output = ''
        script = os.path.join(_PROJECT_ROOT, 'scripts', 'collect_dataset.py')
        args = [sys.executable, script] + extra_args
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
                self._status = f'Exited (code {result.returncode})'
            except subprocess.TimeoutExpired:
                self._status = 'Timed out'
            except Exception as e:
                self._status = f'Error: {e}'
            self._running = False

        threading.Thread(target=_run, daemon=True).start()
