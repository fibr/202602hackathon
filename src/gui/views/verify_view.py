"""Calibration verify view — click-to-advance instead of console input().

Replaces the subprocess-based ``detect_checkerboard.py --verify`` mode.
Uses the shared camera + robot, so no new subprocess or extra window is
needed.

Workflow
--------
1. **Board detection** — Live camera feed.  Press **'c'** or the
   "Capture Board" button when the checkerboard is clearly visible to
   snapshot its pose.
2. **Target review** — Outer-corner hover targets are computed and shown.
   Press **Space / Enter** or the "Start Moving" button to begin.
   Press **'d'** to toggle dry-run (no arm movement).
3. **Movement** — Arm moves 5 cm above each outer corner in sequence
   (background thread; UI stays live).
4. **At corner** — Results (actual pose + error) are shown.
   Press **Space / Enter**, left-click anywhere in the camera area, or the
   "Next Corner" button to advance to the next corner.
5. **Done** — Summary shown.  **Esc** returns to the Calibration menu.

Keyboard shortcuts
------------------
  c         Capture board (in DETECT state)
  Space / Enter  Confirm / advance (in REVIEW, AT_CORNER states)
  d         Toggle dry-run mode
  Esc       Back to Calibration menu

Supported robots
----------------
* **Nova5** — uses ``robot.movj(x, y, z, rx, ry, rz)`` (V4 firmware).
* **arm101** — uses Arm101IKSolver to convert the Cartesian target to joint
  angles, then ``robot.move_joints(angles)``.  Falls back gracefully if IK
  fails.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from gui.views.base import BaseView, ViewRegistry
from gui.robot_controls import RobotControlPanel, PANEL_WIDTH

FONT = cv2.FONT_HERSHEY_SIMPLEX

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, 'scripts')

VERIFY_HOVER_HEIGHT_M = 0.05   # 5 cm above each corner
VERIFY_SPEED_PERCENT = 20

# View states
_S_DETECT = 'detect'
_S_REVIEW = 'review'
_S_MOVING = 'moving'
_S_AT_CORNER = 'at_corner'
_S_DONE = 'done'
_S_ERROR = 'error'


def _import_dcb():
    """Lazy-import detect_checkerboard helpers."""
    if _SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, _SCRIPTS_DIR)
    import detect_checkerboard as _dcb  # noqa: F401
    return _dcb


# ---------------------------------------------------------------------------

@ViewRegistry.register
class VerifyCalibView(BaseView):
    """Verify calibration by hovering the arm above checkerboard corners."""

    view_id = 'verify_calib'
    view_name = 'Verify Calibration'
    description = 'Move arm above board corners to verify calibration'
    needs_camera = True
    needs_robot = False   # robot optional; no-robot shows dry-run only
    headless_ok = False
    show_in_sidebar = False
    parent_view_id = 'calibration'

    def __init__(self, app):
        super().__init__(app)
        self._dcb = None
        self._panel = None
        self._cam_width = 640
        self._cam_height = 480

        self._state = _S_DETECT
        self._error_msg = ''
        self._dry_run = False
        self._advance_requested = False  # set by click / key, consumed by update

        # Board capture result
        self._T_board_in_cam: Optional[np.ndarray] = None
        self._captured_frame: Optional[np.ndarray] = None

        # Computed targets
        self._targets_base: List[np.ndarray] = []   # hover mm in robot base
        self._target_labels: List[str] = []
        self._robot_orientation: Optional[Tuple[float, float, float]] = None

        # Movement progress
        self._corner_idx = 0
        self._corner_results: List[dict] = []   # {label, target, actual, err}
        self._move_thread: Optional[threading.Thread] = None
        self._move_status = ''     # status line from background thread
        self._move_error = ''      # non-empty if move failed

        # Camera-area detection state (for live overlay in DETECT state)
        self._current_T_board: Optional[np.ndarray] = None
        self._current_corners = None
        self._current_detection = None
        self._current_reproj_err: Optional[float] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self):
        try:
            self._dcb = _import_dcb()
        except ImportError as exc:
            self._state = _S_ERROR
            self._error_msg = f'Import error: {exc}'
            return

        # Initialise board detector from config
        try:
            self._dcb._board_detector = (
                self._dcb.BoardDetector.from_config(self.app.config))
            print(f'  [VerifyCalib] Board: {self._dcb._board_detector.describe()}')
        except Exception as exc:
            print(f'  [VerifyCalib] WARNING: BoardDetector init failed: {exc}')
            self._dcb._board_detector = None

        # Load calibration
        calib_path = os.path.join(_PROJECT_ROOT, 'config', 'calibration.yaml')
        if not os.path.exists(calib_path):
            self._state = _S_ERROR
            self._error_msg = (
                f'No calibration file found at:\n  {calib_path}\n\n'
                'Run Checkerboard Calibration first.')
            return

        try:
            from calibration import CoordinateTransform
            self._transform = CoordinateTransform()
            self._transform.load(calib_path)
            print(f'  [VerifyCalib] Loaded calibration from {calib_path}')
        except Exception as exc:
            self._state = _S_ERROR
            self._error_msg = f'Failed to load calibration:\n  {exc}'
            return

        # Camera (required)
        self.app.ensure_camera()
        if self.app.camera is None:
            self._state = _S_ERROR
            self._error_msg = 'No camera available for calibration verify'
            return

        self._cam_width = self.app.camera.width
        self._cam_height = self.app.camera.height

        # Resize view to camera + control panel
        self.app.view_width = self._cam_width + PANEL_WIDTH
        self.app.view_height = self._cam_height

        # Robot (optional)
        self.app.ensure_robot()
        if self.app.robot is None:
            self._dry_run = True
            print('  [VerifyCalib] No robot — forced dry-run mode')

        # Control panel
        self._panel = RobotControlPanel(
            self.app.robot,
            panel_x=self._cam_width,
            panel_height=self._cam_height)
        self._panel.status_msg = (
            'Aim camera at checkerboard, then press c / Capture')

        # Capture board button
        self._panel.add_button(
            lambda: 'Capture Board',
            self._capture_board,
            color=(0, 120, 80))

        # Start/next button (label changes with state)
        def _start_label():
            if self._state == _S_REVIEW:
                return 'Start Moving' if not self._dry_run else 'Dry-Run Preview'
            if self._state == _S_AT_CORNER:
                n = len(self._targets_base)
                remaining = n - self._corner_idx - 1
                return f'Next Corner ({remaining} left)' if remaining else 'Finish'
            return 'Advance'

        self._panel.add_button(_start_label, self._request_advance,
                               color=(0, 80, 160))

        # Dry-run toggle
        def _dry_run_label():
            return f'Dry-Run: {"ON" if self._dry_run else "off"}'

        self._panel.add_button(_dry_run_label, self._toggle_dry_run,
                               color=(80, 50, 0))

    # ------------------------------------------------------------------
    def update(self, canvas):
        if self._state == _S_ERROR:
            self._draw_error(canvas)
            return

        color_image, _, _ = self.app.get_camera_frame()
        if color_image is None:
            return

        w, h = self._cam_width, self._cam_height

        if self._state == _S_DETECT:
            display = self._update_detect(color_image)
        elif self._state == _S_REVIEW:
            display = self._update_review(color_image)
        elif self._state == _S_MOVING:
            display = self._update_moving(color_image)
        elif self._state == _S_AT_CORNER:
            display = self._update_at_corner(color_image)
        elif self._state == _S_DONE:
            display = self._update_done(color_image)
        else:
            display = color_image.copy()

        # Compose camera area + panel
        canvas[0:h, 0:w] = display
        if self._panel:
            self._panel.draw(canvas)

    # ------------------------------------------------------------------
    # State-specific update methods
    # ------------------------------------------------------------------

    def _update_detect(self, color_image) -> np.ndarray:
        """Phase 1: detect board, wait for capture."""
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        found, corners, detection = self._dcb.detect_corners(gray)

        display = color_image.copy()
        reproj_err = None

        if found:
            self._current_corners = corners
            self._current_detection = detection
            self._current_T_board, _, reproj_err = self._dcb.compute_board_pose(
                corners, self.app.camera.intrinsics, detection)
            self._current_reproj_err = reproj_err
            dcb = self._dcb
            if dcb._board_detector is not None and detection is not None:
                dcb._board_detector.draw_corners(display, detection)
            else:
                cv2.drawChessboardCorners(
                    display,
                    (dcb.BOARD_COLS, dcb.BOARD_ROWS),
                    corners, found)
        else:
            self._current_corners = None
            self._current_T_board = None

        # Top bar
        if found and reproj_err is not None:
            board_txt = f'Board OK | reproj {reproj_err:.2f}px'
            bar_color = (0, 200, 0)
        elif found:
            board_txt = 'Board detected'
            bar_color = (0, 200, 0)
        else:
            board_txt = 'No board detected — move camera over checkerboard'
            bar_color = (0, 0, 220)

        cv2.rectangle(display, (0, 0), (self._cam_width, 32), (0, 0, 0), -1)
        cv2.putText(display, board_txt, (10, 22), FONT, 0.5, bar_color, 1)

        # Bottom help
        dry_tag = '  [DRY-RUN]' if self._dry_run else ''
        help_lines = [
            f'[c] capture board pose  |  board must be clearly visible{dry_tag}',
            '[d] toggle dry-run  |  Esc = back to Calibration menu',
        ]
        self._draw_help_bar(display, help_lines)
        return display

    def _update_review(self, color_image) -> np.ndarray:
        """Phase 2: show targets, wait for confirmation."""
        display = color_image.copy()

        # Overlay board corners
        if self._captured_frame is not None:
            # Use captured frame as background tint
            cv2.addWeighted(self._captured_frame, 0.3, display, 0.7, 0, display)

        # Top bar
        dry_tag = ' (DRY-RUN)' if self._dry_run else ''
        cv2.rectangle(display, (0, 0), (self._cam_width, 32), (0, 0, 0), -1)
        cv2.putText(display, f'Targets ready{dry_tag} — press Space/Enter to start',
                    (10, 22), FONT, 0.5, (0, 200, 255), 1)

        # List targets in the frame
        y = 50
        for i, (label, tgt) in enumerate(zip(self._target_labels,
                                              self._targets_base)):
            cv2.putText(display,
                        f'Corner {i} ({label}): '
                        f'[{tgt[0]:.1f}, {tgt[1]:.1f}, {tgt[2]:.1f}] mm',
                        (15, y), FONT, 0.38, (200, 220, 255), 1)
            y += 20

        # Bottom help
        dry_hint = '[d] toggle dry-run  |  ' if not self._dry_run else '[d] disable dry-run  |  '
        help_lines = [
            f'[Space/Enter] start moving  |  {dry_hint}Esc = back',
        ]
        self._draw_help_bar(display, help_lines)

        # Check advance
        if self._advance_requested:
            self._advance_requested = False
            self._start_moving()

        return display

    def _update_moving(self, color_image) -> np.ndarray:
        """Phase 3: arm is moving — show status, poll thread."""
        display = color_image.copy()
        n = len(self._targets_base)
        idx = self._corner_idx
        label = self._target_labels[idx] if idx < len(self._target_labels) else '?'

        cv2.rectangle(display, (0, 0), (self._cam_width, 32), (0, 0, 0), -1)
        cv2.putText(display,
                    f'Moving to corner {idx}/{n - 1} ({label})...',
                    (10, 22), FONT, 0.5, (0, 200, 255), 1)

        if self._move_status:
            cv2.putText(display, self._move_status,
                        (15, 55), FONT, 0.38, (180, 180, 180), 1)

        # Check if thread finished
        if self._move_thread is not None and not self._move_thread.is_alive():
            self._move_thread = None
            if self._move_error:
                self._state = _S_ERROR
                self._error_msg = self._move_error
            else:
                self._state = _S_AT_CORNER
                if self._panel:
                    idx = self._corner_idx
                    label = self._target_labels[idx]
                    res = self._corner_results[-1] if self._corner_results else {}
                    err = res.get('err_mm')
                    if err is not None:
                        self._panel.status_msg = (
                            f'At corner {idx} ({label}): '
                            f'error = {err:.1f} mm')
                    else:
                        self._panel.status_msg = (
                            f'At corner {idx} ({label}) — move failed')

        self._draw_help_bar(display,
                            ['Moving — please wait...  |  Esc = back'])
        return display

    def _update_at_corner(self, color_image) -> np.ndarray:
        """Phase 4: show results for current corner, wait for advance."""
        display = color_image.copy()
        idx = self._corner_idx
        n = len(self._targets_base)
        label = (self._target_labels[idx]
                 if idx < len(self._target_labels) else '?')
        res = self._corner_results[-1] if self._corner_results else {}

        # Top bar
        cv2.rectangle(display, (0, 0), (self._cam_width, 32), (0, 0, 0), -1)
        cv2.putText(display,
                    f'Corner {idx}/{n - 1} ({label})',
                    (10, 22), FONT, 0.5, (0, 255, 150), 1)

        # Results table
        y = 55
        tgt = res.get('target')
        actual = res.get('actual')
        err = res.get('err_mm')
        if tgt is not None:
            cv2.putText(display,
                        f'Target:  [{tgt[0]:.1f}, {tgt[1]:.1f}, {tgt[2]:.1f}] mm',
                        (15, y), FONT, 0.38, (200, 220, 255), 1)
            y += 20
        if actual is not None:
            cv2.putText(display,
                        f'Actual:  [{actual[0]:.1f}, {actual[1]:.1f}, {actual[2]:.1f}] mm',
                        (15, y), FONT, 0.38, (200, 220, 255), 1)
            y += 20
        if err is not None:
            err_color = (0, 255, 0) if err < 10 else (0, 180, 255) if err < 25 else (0, 80, 255)
            cv2.putText(display,
                        f'Position error: {err:.1f} mm',
                        (15, y), FONT, 0.45, err_color, 1)
            y += 25

        # Summary of completed corners
        if len(self._corner_results) > 1:
            y += 5
            cv2.putText(display, 'Previous corners:', (15, y),
                        FONT, 0.35, (160, 160, 160), 1)
            y += 18
            for r in self._corner_results[:-1]:
                e = r.get('err_mm')
                txt = (f'  {r["label"]}: {e:.1f} mm' if e is not None
                       else f'  {r["label"]}: failed')
                cv2.putText(display, txt, (15, y), FONT, 0.32, (140, 140, 140), 1)
                y += 16

        remaining = n - idx - 1
        if remaining > 0:
            help_lines = [
                f'[Space/Enter/click] → next corner  ({remaining} remaining)  |  Esc = back',
            ]
        else:
            help_lines = [
                '[Space/Enter/click] → finish  |  Esc = back to Calibration menu',
            ]
        self._draw_help_bar(display, help_lines)

        # Handle advance
        if self._advance_requested:
            self._advance_requested = False
            next_idx = idx + 1
            if next_idx < n:
                self._corner_idx = next_idx
                self._start_moving()
            else:
                self._state = _S_DONE
                if self._panel:
                    errs = [r['err_mm'] for r in self._corner_results
                            if r.get('err_mm') is not None]
                    if errs:
                        self._panel.status_msg = (
                            f'Done! mean={np.mean(errs):.1f}mm '
                            f'max={np.max(errs):.1f}mm')
                    else:
                        self._panel.status_msg = 'Done!'

        return display

    def _update_done(self, color_image) -> np.ndarray:
        """Phase 5: show summary."""
        display = color_image.copy()

        cv2.rectangle(display, (0, 0), (self._cam_width, 32), (0, 0, 0), -1)
        cv2.putText(display, 'Verification complete!',
                    (10, 22), FONT, 0.55, (0, 255, 150), 1)

        y = 55
        cv2.putText(display, 'Results:', (15, y), FONT, 0.42, (220, 220, 220), 1)
        y += 22
        errs = []
        for r in self._corner_results:
            e = r.get('err_mm')
            if e is not None:
                errs.append(e)
                err_color = (0, 255, 0) if e < 10 else (0, 180, 255) if e < 25 else (0, 80, 255)
                txt = f'  {r["label"]}: {e:.1f} mm'
            else:
                err_color = (100, 100, 255)
                txt = f'  {r["label"]}: move failed'
            cv2.putText(display, txt, (15, y), FONT, 0.38, err_color, 1)
            y += 18

        if errs:
            y += 8
            cv2.putText(display,
                        f'Mean: {np.mean(errs):.1f} mm   Max: {np.max(errs):.1f} mm',
                        (15, y), FONT, 0.4, (200, 240, 200), 1)

        self._draw_help_bar(display, ['Esc = back to Calibration menu'])
        return display

    # ------------------------------------------------------------------
    # Input handlers
    # ------------------------------------------------------------------

    def handle_key(self, key):
        if key == 27:  # Esc
            self.app.switch_view('calibration')
            return True

        # Panel keys (jogging, speed, gripper)
        if self._panel and self._panel.handle_key(key):
            return True

        if key == ord('d'):
            self._toggle_dry_run()
            return True

        if self._state == _S_DETECT and key == ord('c'):
            self._capture_board()
            return True

        if self._state in (_S_REVIEW, _S_AT_CORNER):
            if key in (ord(' '), 13):  # Space or Enter
                self._request_advance()
                return True

        return False

    def handle_mouse(self, event, x, y, flags):
        # Panel area
        if self._panel and x >= self._cam_width:
            return self._panel.handle_mouse(event, x, y, flags)

        # Camera area: left-click advances in AT_CORNER / REVIEW states
        if event == cv2.EVENT_LBUTTONDOWN:
            if self._state in (_S_REVIEW, _S_AT_CORNER):
                self._request_advance()
                return True

        return False

    # ------------------------------------------------------------------
    def cleanup(self):
        # Stop any ongoing jog for Dobot
        if self.app.robot:
            robot_type = self.app.config.get('robot_type', 'nova5')
            if robot_type != 'arm101':
                try:
                    self.app.robot.send('MoveJog()')
                except Exception:
                    pass

        # Restore default view dimensions
        self.app.view_width = 640
        self.app.view_height = 480

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------

    def _capture_board(self):
        """Capture the current board pose and compute hover targets."""
        if self._state != _S_DETECT:
            return
        if self._current_T_board is None:
            if self._panel:
                self._panel.status_msg = 'No board detected — hold board steady first'
            return

        dcb = self._dcb
        T_board = self._current_T_board

        # Outer corners in camera frame → robot base frame
        corners_cam, labels = dcb._get_board_outer_corners_cam(T_board)
        transform = self._transform

        targets = []
        print('\n=== [VerifyCalib] Checkerboard corners in robot base frame ===')
        for i, (p_cam, label) in enumerate(zip(corners_cam, labels)):
            p_base = transform.camera_to_base(p_cam)
            p_base_mm = p_base * 1000.0
            hover_mm = p_base_mm.copy()
            hover_mm[2] += VERIFY_HOVER_HEIGHT_M * 1000.0
            targets.append(hover_mm)
            print(f'  Corner {i} ({label}):')
            print(f'    Camera:  [{p_cam[0]:.4f}, {p_cam[1]:.4f}, {p_cam[2]:.4f}] m')
            print(f'    Base:    [{p_base_mm[0]:.1f}, {p_base_mm[1]:.1f}, {p_base_mm[2]:.1f}] mm')
            print(f'    Hover:   [{hover_mm[0]:.1f}, {hover_mm[1]:.1f}, {hover_mm[2]:.1f}] mm')

        self._T_board_in_cam = T_board
        self._targets_base = targets
        self._target_labels = labels
        self._captured_frame = self.app.get_camera_frame()[0]

        # Read robot orientation to keep constant across moves
        if self.app.robot:
            try:
                pose = self.app.robot.get_pose()
                if pose:
                    self._robot_orientation = (
                        float(pose[3]), float(pose[4]), float(pose[5]))
                    print(f'  Orientation: rx={pose[3]:.1f}, '
                          f'ry={pose[4]:.1f}, rz={pose[5]:.1f}')
            except Exception:
                pass

        self._corner_results = []
        self._corner_idx = 0
        self._state = _S_REVIEW

        if self._panel:
            self._panel.status_msg = (
                f'{len(targets)} targets computed — press Space/Enter to start')

    def _request_advance(self):
        """Set the advance flag (consumed by update in the next frame)."""
        self._advance_requested = True

    def _toggle_dry_run(self):
        self._dry_run = not self._dry_run
        mode = 'ON' if self._dry_run else 'OFF'
        print(f'  [VerifyCalib] Dry-run: {mode}')
        if self._panel:
            self._panel.status_msg = f'Dry-run: {mode}'

    def _start_moving(self):
        """Launch background thread to move to the current corner."""
        idx = self._corner_idx
        target = self._targets_base[idx]
        label = self._target_labels[idx]

        if self._dry_run:
            # Dry-run: skip movement, store fake result
            print(f'\n[VerifyCalib][DRY-RUN] Corner {idx} ({label}): '
                  f'target=[{target[0]:.1f},{target[1]:.1f},{target[2]:.1f}] mm')
            self._corner_results.append({
                'label': label,
                'target': target,
                'actual': None,
                'err_mm': None,
            })
            self._state = _S_AT_CORNER
            if self._panel:
                self._panel.status_msg = (
                    f'[DRY-RUN] Corner {idx} ({label}) — no movement')
            return

        self._state = _S_MOVING
        self._move_status = f'Moving to {label}...'
        self._move_error = ''

        self._move_thread = threading.Thread(
            target=self._move_worker,
            args=(idx, target, label),
            daemon=True)
        self._move_thread.start()

    def _move_worker(self, idx: int, target_mm: np.ndarray, label: str):
        """Background thread: move arm, record result."""
        robot = self.app.robot
        robot_type = self.app.config.get('robot_type', 'nova5')

        try:
            rx, ry, rz = (self._robot_orientation
                          if self._robot_orientation else (0.0, 0.0, 0.0))

            print(f'\n[VerifyCalib] --- Moving to corner {idx} ({label}) ---')
            print(f'  Target: [{target_mm[0]:.1f}, {target_mm[1]:.1f}, '
                  f'{target_mm[2]:.1f}] mm')

            if robot_type == 'arm101':
                ok = self._move_arm101(robot, target_mm, rx, ry, rz)
            else:
                ok = self._move_nova5(robot, target_mm, rx, ry, rz)

            if not ok:
                self._corner_results.append({
                    'label': label,
                    'target': target_mm,
                    'actual': None,
                    'err_mm': None,
                })
                print(f'  [VerifyCalib] Move failed for corner {idx}')
                return

            # Read actual pose
            time.sleep(0.3)  # settle
            actual_pose = robot.get_pose()
            if actual_pose:
                actual = np.array(actual_pose[:3], dtype=float)
                err = float(np.linalg.norm(actual - target_mm))
                print(f'  Actual: [{actual[0]:.1f}, {actual[1]:.1f}, {actual[2]:.1f}] mm')
                print(f'  Error:  {err:.1f} mm')
                self._corner_results.append({
                    'label': label,
                    'target': target_mm,
                    'actual': actual,
                    'err_mm': err,
                })
                self._move_status = (f'Arrived at {label} — error: {err:.1f} mm')
            else:
                self._corner_results.append({
                    'label': label,
                    'target': target_mm,
                    'actual': None,
                    'err_mm': None,
                })
                self._move_status = f'Arrived at {label} (pose unavailable)'

        except Exception as exc:
            self._move_error = f'Movement error at corner {idx}: {exc}'
            print(f'  [VerifyCalib] ERROR: {exc}')

    def _move_nova5(self, robot, target_mm, rx, ry, rz) -> bool:
        """Move Nova5 to the target pose using MovJ (V4 firmware)."""
        # Set speed
        try:
            robot.set_speed(VERIFY_SPEED_PERCENT)
        except Exception:
            pass

        if hasattr(robot, 'movj'):
            return robot.movj(
                target_mm[0], target_mm[1], target_mm[2],
                rx, ry, rz)
        # Fallback: raw V4 command via send()
        cmd = (f'MovJ(pose={{{target_mm[0]:.3f},{target_mm[1]:.3f},'
               f'{target_mm[2]:.3f},{rx:.3f},{ry:.3f},{rz:.3f}}})')
        resp = robot.send(cmd)
        print(f'  MovJ resp: {resp}')
        return '0,' in (resp or '')

    def _move_arm101(self, robot, target_mm, rx, ry, rz) -> bool:
        """Move arm101 to a Cartesian target via IK + joint move."""
        try:
            from kinematics.arm101_ik_solver import Arm101IKSolver
        except ImportError:
            try:
                from kinematics.ik_solver import IKSolver as Arm101IKSolver
            except ImportError:
                print('  [VerifyCalib] WARNING: No IK solver found for arm101')
                return False

        try:
            ik = Arm101IKSolver(self.app.config)
        except Exception as exc:
            print(f'  [VerifyCalib] IK solver init failed: {exc}')
            return False

        target_rpy = np.array([rx, ry, rz], dtype=float)
        seed = None
        try:
            angles = robot.get_angles()
            if angles:
                seed = np.array(angles, dtype=float)
        except Exception:
            pass

        joint_angles = ik.solve_ik(target_mm, target_rpy, seed_joints=seed)
        if joint_angles is None:
            print(f'  [VerifyCalib] IK failed for target {target_mm}')
            return False

        return robot.move_joints(list(joint_angles))

    # ------------------------------------------------------------------
    # Drawing utilities
    # ------------------------------------------------------------------

    def _draw_error(self, canvas):
        vw = self.app.view_width
        vh = self.app.canvas_height
        canvas[:vh, :vw] = (30, 30, 35)
        cv2.putText(canvas, 'Verify Calibration', (20, 35),
                    FONT, 0.6, (255, 200, 100), 1)
        y = 70
        for line in self._error_msg.split('\n'):
            cv2.putText(canvas, line, (20, y), FONT, 0.42, (80, 100, 255), 1)
            y += 22
        cv2.putText(canvas, 'Press ESC to return to Calibration menu.',
                    (20, y + 10), FONT, 0.38, (130, 130, 130), 1)

    def _draw_help_bar(self, display, lines):
        """Draw a dark help bar at the bottom of display."""
        line_h = 22
        bar_h = line_h * len(lines) + 8
        h = self._cam_height
        w = self._cam_width
        cv2.rectangle(display, (0, h - bar_h), (w, h), (0, 0, 0), -1)
        for i, line in enumerate(lines):
            y = h - bar_h + 4 + line_h * (i + 1)
            cv2.putText(display, line, (10, y), FONT, 0.38, (200, 200, 200), 1)
