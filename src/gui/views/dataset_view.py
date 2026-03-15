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

    OPTIONS = [
        ('Live Collection',
         'Camera + robot, capture frames interactively',
         []),
        ('Camera Only (no robot)',
         'Capture frames without robot connection',
         ['--no-robot']),
        ('Snapshot Debug',
         'Single-frame detection with 6-stage debug images',
         ['--snapshot']),
    ]

    def __init__(self, app):
        super().__init__(app)
        self._status = 'Ready'
        self._running = False
        self._output = ''
        self._buttons = []      # [(x1, y1, x2, y2, option_idx), ...]
        self._hover_pos = (-1, -1)

    def setup(self):
        pass

    def update(self, canvas):
        vw = self.app.view_width
        vh = self.app.canvas_height
        canvas[:vh, :vw] = (30, 30, 35)
        self._buttons = []

        cv2.putText(canvas, 'Dataset Collection', (20, 35),
                    FONT, 0.6, (255, 200, 100), 1)
        cv2.putText(canvas, 'Capture rod detection training data',
                    (20, 58), FONT, 0.38, (150, 150, 150), 1)
        cv2.line(canvas, (10, 68), (vw - 10, 68), (60, 60, 70), 1)

        btn_h = 52
        btn_w = vw - 40
        hx, hy = self._hover_pos
        y = 82

        for idx, (label, desc, _args) in enumerate(self.OPTIONS):
            x1, y1 = 20, y
            x2, y2 = x1 + btn_w, y + btn_h

            disabled = self._running
            is_hover = (not disabled) and (x1 <= hx <= x2 and y1 <= hy <= y2)

            if disabled:
                bg = (35, 35, 38)
                border_col = (55, 55, 60)
                text_col = (100, 100, 100)
                desc_col = (80, 80, 85)
            elif is_hover:
                bg = (55, 50, 65)
                border_col = (120, 160, 220)
                text_col = (230, 245, 255)
                desc_col = (160, 160, 170)
            else:
                bg = (40, 40, 50)
                border_col = (70, 70, 90)
                text_col = (180, 220, 255)
                desc_col = (120, 120, 130)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 1)
            cv2.putText(canvas, label, (x1 + 14, y1 + 22), FONT, 0.42, text_col, 1)
            cv2.putText(canvas, desc, (x1 + 14, y1 + 40), FONT, 0.32, desc_col, 1)

            if not disabled:
                self._buttons.append((x1, y1, x2, y2, idx))
            y += btn_h + 8

        # Status
        y += 6
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
        # Legacy key shortcuts kept for backwards compatibility
        if self._running:
            return False
        if key == ord('1'):
            self._launch(self.OPTIONS[0][2])
            return True
        if key == ord('2'):
            self._launch(self.OPTIONS[1][2])
            return True
        if key == ord('3'):
            self._launch(self.OPTIONS[2][2])
            return True
        return False

    def handle_mouse(self, event, x, y, flags):
        if event == cv2.EVENT_MOUSEMOVE:
            self._hover_pos = (x, y)
            return False
        if event == cv2.EVENT_LBUTTONDOWN:
            for x1, y1, x2, y2, idx in self._buttons:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self._launch(self.OPTIONS[idx][2])
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
