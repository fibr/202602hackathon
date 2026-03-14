"""Digital Twin view: launcher for the Isaac Sim digital twin.

Provides a menu to launch the ARM101 digital twin in Isaac Sim with
different modes (GUI, headless, camera rendering, mirror mode, etc.).

Note: Unlike other views this does NOT run the script inside the unified
GUI process.  Isaac Sim uses an entirely different rendering pipeline and
must be started as an independent process via the Isaac Lab launcher
(scripts/run_digital_twin.sh).  The view only acts as a convenient
launcher and status display.
"""

import os
import subprocess
import threading

import cv2
import numpy as np

from gui.views.base import BaseView, ViewRegistry

FONT = cv2.FONT_HERSHEY_SIMPLEX
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
_LAUNCH_SCRIPT = os.path.join(_PROJECT_ROOT, 'scripts', 'run_digital_twin.sh')
_ISAACLAB_DIR = os.environ.get('ISAACLAB_DIR',
                               os.path.expanduser('~/src/IsaacLab'))


def _isaaclab_available() -> bool:
    """Return True if the Isaac Lab installation looks usable."""
    return (os.path.isdir(_ISAACLAB_DIR) and
            os.path.isfile(os.path.join(_ISAACLAB_DIR, 'isaaclab.sh')))


@ViewRegistry.register
class DigitalTwinView(BaseView):
    view_id = 'digital_twin'
    view_name = 'Digital Twin'
    description = 'Isaac Sim ARM101 digital twin launcher'
    needs_camera = False
    needs_robot = False
    headless_ok = False

    # Launch modes: (key, label, description, extra_args)
    MODES = [
        ('1', 'GUI mode',
         'Interactive Isaac Sim window (default)',
         []),
        ('2', 'Headless',
         'No GUI window, simulation only',
         ['--headless']),
        ('3', 'With cameras',
         'Enable RTX camera rendering',
         ['--enable_cameras']),
        ('4', 'Cameras + save images',
         'Render cameras and save PNGs to disk',
         ['--enable_cameras', '--save_images']),
        ('5', 'Mirror arm',
         'Mirror real ARM101 joint angles in sim (requires HW)',
         ['--mirror']),
        ('6', 'Mirror + cameras',
         'Full digital twin: mirror arm with camera rendering',
         ['--mirror', '--enable_cameras']),
    ]

    def __init__(self, app):
        super().__init__(app)
        self._status = 'Ready'
        self._proc = None          # running Popen, or None
        self._log_lines = []       # last few lines of output
        self._lock = threading.Lock()

    def setup(self):
        self._status = 'Ready'
        self._proc = None
        self._log_lines = []

    def update(self, canvas):
        vw = self.app.view_width
        vh = self.app.canvas_height
        canvas[:vh, :vw] = (30, 30, 35)

        # Title
        cv2.putText(canvas, 'Digital Twin Launcher', (20, 35),
                    FONT, 0.6, (255, 200, 100), 1)
        cv2.putText(canvas, 'ARM101 + Isaac Sim via Isaac Lab',
                    (20, 56), FONT, 0.38, (150, 150, 150), 1)
        cv2.line(canvas, (10, 66), (vw - 10, 66), (60, 60, 70), 1)

        # Isaac Lab availability warning
        y = 84
        if not os.path.exists(_LAUNCH_SCRIPT):
            cv2.putText(canvas, 'WARNING: run_digital_twin.sh not found',
                        (20, y), FONT, 0.38, (0, 80, 220), 1)
            y += 18
        if not _isaaclab_available():
            cv2.putText(canvas,
                        f'Isaac Lab not found at {_ISAACLAB_DIR}',
                        (20, y), FONT, 0.35, (0, 80, 220), 1)
            cv2.putText(canvas,
                        'Set ISAACLAB_DIR env var to your installation path.',
                        (20, y + 16), FONT, 0.32, (100, 100, 100), 1)
            y += 36

        # Mode list
        running = self._is_running()
        for key_char, label, desc, _ in self.MODES:
            color = (100, 100, 100) if running else (180, 220, 255)
            cv2.putText(canvas, f'[{key_char}] {label}',
                        (30, y), FONT, 0.4, color, 1)
            cv2.putText(canvas, desc, (50, y + 16),
                        FONT, 0.3, (120, 120, 120), 1)
            y += 40

        # Stop key
        stop_color = (0, 200, 255) if running else (60, 60, 70)
        cv2.putText(canvas, '[S] Stop / detach from running instance',
                    (30, y), FONT, 0.38, stop_color, 1)
        y += 28

        # Status
        y += 6
        status_color = (0, 200, 255) if running else (200, 200, 200)
        cv2.putText(canvas, f'Status: {self._status}',
                    (20, y), FONT, 0.4, status_color, 1)

        # Log tail
        y += 22
        with self._lock:
            lines = list(self._log_lines[-8:])
        for line in lines:
            y += 14
            if y >= vh - 20:
                break
            cv2.putText(canvas, line[:80], (25, y),
                        FONT, 0.28, (150, 150, 150), 1)

        # Footer
        cv2.putText(canvas,
                    'Isaac Sim opens in its own window; GUI continues running.',
                    (20, vh - 15), FONT, 0.32, (80, 80, 80), 1)

    def handle_key(self, key):
        if key == ord('s') or key == ord('S'):
            self._stop()
            return True

        for key_char, label, _desc, extra_args in self.MODES:
            if key == ord(key_char):
                self._launch(label, extra_args)
                return True
        return False

    def cleanup(self):
        # Don't kill Isaac Sim when leaving the view — it's an independent
        # process the user intentionally started.  Just forget the reference.
        self._proc = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_running(self) -> bool:
        """Return True if the Isaac Sim process is still alive."""
        with self._lock:
            proc = self._proc
        if proc is None:
            return False
        return proc.poll() is None

    def _launch(self, label: str, extra_args: list):
        if self._is_running():
            self._status = 'Already running — press S to detach first'
            return
        if not os.path.exists(_LAUNCH_SCRIPT):
            self._status = 'run_digital_twin.sh not found'
            return
        if not _isaaclab_available():
            self._status = f'Isaac Lab not found at {_ISAACLAB_DIR}'
            return

        cmd = ['/bin/bash', _LAUNCH_SCRIPT] + extra_args
        with self._lock:
            self._log_lines = []

        self._status = f'Starting {label}...'

        def _run():
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=_PROJECT_ROOT,
                    env={**os.environ, 'ISAACLAB_DIR': _ISAACLAB_DIR},
                )
                with self._lock:
                    self._proc = proc

                # Stream output into log_lines so the view can display it
                for raw_line in proc.stdout:
                    line = raw_line.rstrip()
                    with self._lock:
                        self._log_lines.append(line)
                        if len(self._log_lines) > 200:
                            self._log_lines = self._log_lines[-200:]

                proc.wait()
                rc = proc.returncode
                with self._lock:
                    if self._proc is proc:  # still ours
                        self._proc = None
                self._status = f'{label} exited (code {rc})'
            except Exception as exc:
                self._status = f'Error: {exc}'
                with self._lock:
                    self._proc = None

        threading.Thread(target=_run, daemon=True).start()

    def _stop(self):
        """Detach from (forget) the running Isaac Sim process."""
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is not None and proc.poll() is None:
            self._status = 'Detached from Isaac Sim process (still running)'
        else:
            self._status = 'No running instance'
