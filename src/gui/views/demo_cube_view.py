"""Demo Cube view: move robot through random or cube-corner poses.

Supports headless mode (--headless) and direct actuation (--direct).
Wraps scripts/demo_cube.py logic.
"""

import os
import sys
import subprocess

import cv2
import numpy as np

from gui.views.base import BaseView, ViewRegistry

FONT = cv2.FONT_HERSHEY_SIMPLEX
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))


@ViewRegistry.register
class DemoCubeView(BaseView):
    view_id = 'demo_cube'
    view_name = 'Demo Cube'
    description = 'Random/cube-corner motion demo'
    needs_camera = False
    needs_robot = True
    headless_ok = True

    def __init__(self, app):
        super().__init__(app)
        self._status = 'Press [R] to run random, [C] for cube demo'
        self._running = False
        self._thread = None

    def setup(self):
        pass

    def update(self, canvas):
        vw = self.app.view_width
        vh = self.app.canvas_height
        canvas[:vh, :vw] = (30, 30, 35)

        cv2.putText(canvas, 'Demo Cube', (20, 35),
                    FONT, 0.6, (255, 200, 100), 1)
        cv2.putText(canvas, 'Move robot through random or structured poses',
                    (20, 58), FONT, 0.38, (150, 150, 150), 1)
        cv2.line(canvas, (10, 68), (vw - 10, 68), (60, 60, 70), 1)

        # Status
        color = (0, 200, 255) if self._running else (200, 200, 200)
        cv2.putText(canvas, self._status, (20, 100), FONT, 0.42, color, 1)

        # Instructions
        y = 140
        instructions = [
            '[R] Run random poses demo',
            '[C] Run cube corners demo',
            '[S] Stop (if running)',
            '',
            'Robot must be connected.',
            f'Robot: {"connected" if self.app.robot else "NOT connected"}',
        ]
        for line in instructions:
            cv2.putText(canvas, line, (30, y), FONT, 0.38, (180, 180, 180), 1)
            y += 24

    def handle_key(self, key):
        if key == ord('r') and not self._running:
            self._run_demo('random')
            return True
        if key == ord('c') and not self._running:
            self._run_demo('cube')
            return True
        return False

    def _run_demo(self, mode):
        """Launch demo_cube.py as subprocess."""
        import threading
        self._running = True
        self._status = f'Running {mode} demo...'
        script = os.path.join(_PROJECT_ROOT, 'scripts', 'demo_cube.py')
        args = [sys.executable, script, '--mode', mode]

        def _run():
            try:
                result = subprocess.run(
                    args, capture_output=True, text=True, timeout=120,
                    cwd=_PROJECT_ROOT)
                if result.returncode == 0:
                    self._status = f'{mode} demo complete.'
                else:
                    self._status = f'{mode} demo failed (code {result.returncode})'
                    stderr_tail = result.stderr.strip().split('\n')[-3:]
                    for line in stderr_tail:
                        print(f"  demo_cube stderr: {line}")
            except subprocess.TimeoutExpired:
                self._status = f'{mode} demo timed out'
            except Exception as e:
                self._status = f'Error: {e}'
            self._running = False

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def run_headless(self):
        """Run demo in headless mode."""
        script = os.path.join(_PROJECT_ROOT, 'scripts', 'demo_cube.py')
        result = subprocess.run(
            [sys.executable, script], cwd=_PROJECT_ROOT)
        return result.returncode

    def run_direct(self):
        """Direct actuation: run demo immediately."""
        return self.run_headless()
