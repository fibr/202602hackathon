"""Main Pipeline view: run the full pick-and-stand pipeline.

Launches src/main.py which runs the state machine:
INIT -> DETECT -> PLAN -> EXECUTE -> DONE
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
class PipelineView(BaseView):
    view_id = 'pipeline'
    view_name = 'Pick & Stand'
    description = 'Full rod pick-and-stand pipeline'
    needs_camera = True
    needs_robot = True
    headless_ok = True

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

        cv2.putText(canvas, 'Pick & Stand Pipeline', (20, 35),
                    FONT, 0.6, (255, 200, 100), 1)
        cv2.putText(canvas, 'INIT -> DETECT -> PLAN -> EXECUTE -> DONE',
                    (20, 58), FONT, 0.38, (150, 150, 150), 1)
        cv2.line(canvas, (10, 68), (vw - 10, 68), (60, 60, 70), 1)

        y = 95
        color = (100, 100, 100) if self._running else (180, 220, 255)
        cv2.putText(canvas, '[ENTER] Run Pipeline', (30, y), FONT, 0.45, color, 1)
        y += 24
        cv2.putText(canvas, 'Detects rod, plans grasp, picks up and stands upright',
                    (50, y), FONT, 0.35, (120, 120, 120), 1)

        # Status
        y += 40
        status_color = (0, 200, 255) if self._running else (200, 200, 200)
        cv2.putText(canvas, f'Status: {self._status}', (20, y),
                    FONT, 0.4, status_color, 1)

        # Output
        y += 30
        lines = self._output.split('\n')[-20:]
        for line in lines:
            y += 16
            if y > vh - 20:
                break
            cv2.putText(canvas, line[:90], (25, y),
                        FONT, 0.3, (160, 160, 160), 1)

    def handle_key(self, key):
        if key == 13 and not self._running:  # Enter
            self._run_pipeline()
            return True
        return False

    def _run_pipeline(self):
        self._running = True
        self._status = 'Running pipeline...'
        self._output = ''
        script = os.path.join(_PROJECT_ROOT, 'src', 'main.py')

        def _run():
            try:
                result = subprocess.run(
                    [sys.executable, script],
                    capture_output=True, text=True, timeout=300,
                    cwd=_PROJECT_ROOT)
                self._output = result.stdout[-3000:] if result.stdout else ''
                if result.stderr:
                    self._output += '\n' + result.stderr[-1000:]
                self._status = f'Pipeline exited (code {result.returncode})'
            except subprocess.TimeoutExpired:
                self._status = 'Pipeline timed out (5 min)'
            except Exception as e:
                self._status = f'Error: {e}'
            self._running = False

        threading.Thread(target=_run, daemon=True).start()

    def run_headless(self):
        script = os.path.join(_PROJECT_ROOT, 'src', 'main.py')
        result = subprocess.run(
            [sys.executable, script], cwd=_PROJECT_ROOT)
        return result.returncode

    def run_direct(self):
        return self.run_headless()
