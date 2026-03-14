"""Discover Cameras view: detect connected cameras and generate config."""

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
class DiscoverCamerasView(BaseView):
    view_id = 'discover'
    view_name = 'Discover Cameras'
    description = 'Detect cameras & write config'
    needs_camera = False
    needs_robot = False
    headless_ok = True

    def __init__(self, app):
        super().__init__(app)
        self._output = ''
        self._running = False
        self._scroll = 0

    def setup(self):
        pass

    def update(self, canvas):
        vw = self.app.view_width
        vh = self.app.canvas_height
        canvas[:vh, :vw] = (30, 30, 35)

        cv2.putText(canvas, 'Discover Cameras', (20, 35),
                    FONT, 0.6, (255, 200, 100), 1)
        cv2.line(canvas, (10, 48), (vw - 10, 48), (60, 60, 70), 1)

        # Buttons
        y = 60
        buttons = [
            ('[D] Detect & write config', 'detect'),
            ('[P] Preview only (dry-run)', 'dryrun'),
            ('[M] Merge with existing', 'merge'),
        ]
        for label, _ in buttons:
            cv2.putText(canvas, label, (30, y + 16), FONT, 0.4, (180, 220, 255), 1)
            y += 28

        if self._running:
            cv2.putText(canvas, 'Running...', (30, y + 20),
                        FONT, 0.4, (0, 200, 255), 1)

        # Output display
        y += 30
        cv2.line(canvas, (10, y), (vw - 10, y), (60, 60, 70), 1)
        y += 5
        lines = self._output.split('\n')
        line_h = 16
        max_lines = (vh - y - 10) // line_h
        start = max(0, len(lines) - max_lines + self._scroll)
        for line in lines[start:start + max_lines]:
            y += line_h
            cv2.putText(canvas, line[:90], (15, y),
                        FONT, 0.32, (200, 200, 200), 1)

    def handle_key(self, key):
        if self._running:
            return False
        if key == ord('d'):
            self._run_discover([])
            return True
        if key == ord('p'):
            self._run_discover(['--dry-run'])
            return True
        if key == ord('m'):
            self._run_discover(['--merge'])
            return True
        return False

    def _run_discover(self, extra_args):
        self._running = True
        self._output = 'Running discover_cameras.py...\n'
        script = os.path.join(_PROJECT_ROOT, 'scripts', 'discover_cameras.py')

        def _run():
            try:
                result = subprocess.run(
                    [sys.executable, script] + extra_args,
                    capture_output=True, text=True, timeout=30,
                    cwd=_PROJECT_ROOT)
                self._output += result.stdout
                if result.stderr:
                    self._output += '\n' + result.stderr
            except Exception as e:
                self._output += f'\nERROR: {e}'
            self._running = False

        threading.Thread(target=_run, daemon=True).start()

    def run_headless(self):
        script = os.path.join(_PROJECT_ROOT, 'scripts', 'discover_cameras.py')
        result = subprocess.run(
            [sys.executable, script], cwd=_PROJECT_ROOT)
        return result.returncode
