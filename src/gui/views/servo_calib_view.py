"""Servo and hand-eye calibration views for arm101 (embedded in unified GUI).

Replaces the subprocess-based launcher for calibration_gui.py.  Both views
use the helper functions from src/calibration/calib_helpers.py directly.

Views registered here:
  servo_calib    — move arm to zero pose and save servo zero offsets
  handeye_yellow — collect FK+pixel correspondences and solve hand-eye
"""

import os
import sys
import time

import cv2
import numpy as np

from gui.views.base import BaseView, ViewRegistry
from calibration.calib_helpers import (
    OFFSET_FILE,
    HANDEYE_FILE,
    read_all_raw,
    find_yellow_tape,
    draw_servo_overlay,
    draw_handeye_overlay,
    load_offsets,
    save_offsets,
    save_offsets_dict,
    joint_solve,
    save_handeye_calibration,
)

FONT = cv2.FONT_HERSHEY_SIMPLEX

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))


# ---------------------------------------------------------------------------
# ServoCalibView
# ---------------------------------------------------------------------------

@ViewRegistry.register
class ServoCalibView(BaseView):
    """Servo calibration: move arm to zero pose by hand, save raw offsets.

    Requires arm101 robot (torque is disabled so you can backdrive the arm).
    Camera is used for live visual feedback but is optional.
    Keys: SPACE = save offsets | R = reload offsets | ESC = back to Calibration
    """

    view_id = 'servo_calib'
    view_name = 'Servo Calib'
    description = 'Zero servo offsets (arm101)'
    needs_camera = False
    needs_robot = True
    headless_ok = False
    show_in_sidebar = False   # reached via the Calibration menu
    parent_view_id = 'calibration'  # ESC / Back button returns here

    def __init__(self, app):
        super().__init__(app)
        self._arm = None
        self._offsets = {}
        self._status_msg = ''
        self._status_time = 0.0
        self._error_msg = ''

    # ------------------------------------------------------------------
    def setup(self):
        # Use the shared robot if it is an arm101
        self.app.ensure_robot()
        robot = self.app.robot
        if robot is None:
            self._error_msg = 'No robot connected'
            return
        if getattr(robot, 'robot_type', None) != 'arm101':
            self._error_msg = 'Servo calibration requires arm101 robot'
            return

        self._arm = robot

        # Disable torque so the user can move the arm by hand
        try:
            self._arm.disable_torque()
            print('  Servo calibration: torque disabled — move arm by hand')
        except Exception as exc:
            print(f'  WARNING: Could not disable torque: {exc}')

        # Ensure camera for live feedback (optional)
        self.app.ensure_camera()

        # Load existing offsets
        self._offsets = load_offsets()
        self._status_msg = 'Move arm to zero pose, then press SPACE'
        self._status_time = time.time()

    # ------------------------------------------------------------------
    def update(self, canvas):
        vw = self.app.view_width
        vh = self.app.canvas_height

        if self._error_msg:
            canvas[:vh, :vw] = (30, 30, 35)
            cv2.putText(canvas, 'Servo Calibration', (20, 35),
                        FONT, 0.6, (255, 200, 100), 1)
            cv2.putText(canvas, self._error_msg, (20, 80),
                        FONT, 0.5, (0, 80, 220), 1)
            cv2.putText(canvas, 'Press ESC to return to Calibration menu.',
                        (20, 110), FONT, 0.38, (130, 130, 130), 1)
            return

        # Grab camera frame (or blank canvas)
        frame = None
        if self.app.camera is not None:
            color, _, _ = self.app.get_camera_frame()
            if color is not None:
                frame = color.copy()
        if frame is None:
            frame = np.zeros((vh, vw, 3), dtype=np.uint8)

        # Read servo state
        raw_positions = {}
        angles = None
        try:
            raw_positions = read_all_raw(self._arm)
            angles = self._arm.get_angles()
        except Exception as exc:
            cv2.putText(frame, f'Read error: {exc}', (10, 50),
                        FONT, 0.4, (0, 80, 220), 1)

        # Store for handle_key
        self._current_raw = raw_positions

        # Draw calibration overlay
        draw_servo_overlay(frame, raw_positions, self._offsets, angles)

        # Breadcrumb / back hint at the bottom of the frame
        fh, fw = frame.shape[:2]
        hint = 'Calibration > Servo Calib  |  ESC = back  |  SPACE = save  |  R = reload'
        cv2.putText(frame, hint, (10, fh - 10),
                    FONT, 0.32, (160, 160, 100), 1)

        # Temporary status message (fades after 4 s)
        if self._status_msg and time.time() - self._status_time < 4.0:
            cv2.putText(frame, self._status_msg,
                        (10, fh - 50),
                        FONT, 0.6, (0, 255, 0), 2)

        # Blit frame onto canvas
        ch = min(fh, vh)
        cw = min(fw, vw)
        canvas[0:ch, 0:cw] = frame[0:ch, 0:cw]

    # ------------------------------------------------------------------
    def handle_key(self, key):
        if key == 27:  # ESC → back to Calibration menu
            self.app.switch_view('calibration')
            return True

        if self._arm is None:
            return False

        if key == ord(' '):
            raw = getattr(self, '_current_raw', {})
            if not raw:
                try:
                    raw = read_all_raw(self._arm)
                except Exception as exc:
                    self._status_msg = f'Read error: {exc}'
                    self._status_time = time.time()
                    return True
            try:
                self._offsets = save_offsets(raw)
                self._status_msg = (
                    f'Saved offsets to '
                    f'{os.path.basename(OFFSET_FILE)}')
                print(f'  {self._status_msg}')
            except Exception as exc:
                self._status_msg = f'Save error: {exc}'
            self._status_time = time.time()
            return True

        if key == ord('r'):
            try:
                self._offsets = load_offsets()
                self._status_msg = 'Reloaded offsets from file'
            except Exception as exc:
                self._status_msg = f'Reload error: {exc}'
            self._status_time = time.time()
            return True

        return False

    # ------------------------------------------------------------------
    def cleanup(self):
        if self._arm is not None:
            try:
                self._arm.enable_torque()
                print('  Servo calibration cleanup: torque re-enabled')
            except Exception:
                pass


# ---------------------------------------------------------------------------
# HandEyeYellowView
# ---------------------------------------------------------------------------

@ViewRegistry.register
class HandEyeYellowView(BaseView):
    """Hand-eye calibration via yellow tape marker (arm101 + camera).

    Move the arm to diverse poses with a yellow tape marker on the TCP.
    Press SPACE to capture FK+pixel correspondence.  Press S to jointly
    optimise servo offsets and extrinsics (≥ 6 points required).

    Keys: SPACE = capture | S = solve | U = undo last | ESC = back to Calibration
    """

    view_id = 'handeye_yellow'
    view_name = 'HandEye Yellow'
    description = 'Hand-eye via yellow tape (arm101)'
    needs_camera = True
    needs_robot = True
    headless_ok = False
    show_in_sidebar = False   # reached via the Calibration menu
    parent_view_id = 'calibration'  # ESC / Back button returns here

    def __init__(self, app):
        super().__init__(app)
        self._arm = None
        self._solver = None
        self._K = None
        self._dist = None
        self._pts_3d = []           # FK TCP positions (mm)
        self._pts_2d = []           # pixel centroids
        self._raw_positions_list = []  # raw servo reads per capture
        self._status_msg = ''
        self._status_time = 0.0
        self._error_msg = ''
        # Per-frame state written by update(), read by handle_key()
        self._current_tcp = None
        self._current_cx = None
        self._current_cy = None
        self._current_raw = None

    # ------------------------------------------------------------------
    def setup(self):
        self.app.ensure_robot()
        robot = self.app.robot
        if robot is None:
            self._error_msg = 'No robot connected'
            return
        if getattr(robot, 'robot_type', None) != 'arm101':
            self._error_msg = 'Hand-eye yellow calibration requires arm101'
            return

        self._arm = robot

        try:
            self._arm.disable_torque()
            print('  Hand-eye (yellow tape): torque disabled, move arm by hand')
        except Exception as exc:
            print(f'  WARNING: Could not disable torque: {exc}')

        # FK solver
        try:
            from kinematics.arm101_ik_solver import Arm101IKSolver
            self._solver = Arm101IKSolver()
        except Exception as exc:
            self._error_msg = f'FK solver not available: {exc}'
            return

        # Camera
        self.app.ensure_camera()

        # Load camera intrinsics
        self._K, self._dist = self._load_intrinsics()

    # ------------------------------------------------------------------
    def _load_intrinsics(self):
        """Load intrinsics from cameras.yaml or use pinhole defaults."""
        import yaml
        cam_cfg = self.app.config.get('camera', {})
        cam_idx = cam_cfg.get('device_index', 4)
        cam_yaml = os.path.join(_PROJECT_ROOT, 'config', 'cameras.yaml')

        if os.path.exists(cam_yaml):
            with open(cam_yaml) as fh:
                cdata = yaml.safe_load(fh)
            for cname, cinfo in cdata.get('cameras', {}).items():
                if cinfo.get('device_index') == cam_idx:
                    intr = cinfo.get('intrinsics', {})
                    K = np.array(
                        intr.get('camera_matrix',
                                 [[554.3, 0, 320], [0, 554.3, 240], [0, 0, 1]]),
                        dtype=np.float64)
                    dist = np.array(
                        intr.get('dist_coeffs', [0, 0, 0, 0, 0]),
                        dtype=np.float64)
                    print(f'  Intrinsics from {cname}: fx={K[0, 0]:.1f}')
                    return K, dist

        K = np.array([[554.3, 0, 320], [0, 554.3, 240], [0, 0, 1]],
                     dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
        print('  Using default intrinsics (estimated)')
        return K, dist

    # ------------------------------------------------------------------
    def update(self, canvas):
        vw = self.app.view_width
        vh = self.app.canvas_height

        if self._error_msg:
            canvas[:vh, :vw] = (30, 30, 35)
            cv2.putText(canvas, 'Hand-Eye Calibration (Yellow Tape)',
                        (20, 35), FONT, 0.6, (255, 200, 100), 1)
            cv2.putText(canvas, self._error_msg, (20, 80),
                        FONT, 0.5, (0, 80, 220), 1)
            cv2.putText(canvas, 'Press ESC to return to Calibration menu.',
                        (20, 110), FONT, 0.38, (130, 130, 130), 1)
            return

        # Get camera frame
        frame = None
        if self.app.camera is not None:
            color, _, _ = self.app.get_camera_frame()
            if color is not None:
                frame = color.copy()
        if frame is None:
            frame = np.zeros((vh, vw, 3), dtype=np.uint8)

        # FK: joint angles → TCP position
        tcp_pos = None
        try:
            angles = self._arm.get_angles()
            if angles is not None and self._solver is not None:
                tcp_pos, _ = self._solver.forward_kin(np.array(angles[:5]))
        except Exception:
            pass

        # Yellow tape detection
        cx, cy, mask = find_yellow_tape(frame)

        # Draw hand-eye overlay
        draw_handeye_overlay(frame, tcp_pos, (cx, cy), len(self._pts_2d))

        # Small mask preview in top-right corner
        fh, fw = frame.shape[:2]
        mask_small = cv2.resize(mask, (160, 120))
        mask_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        frame[0:120, fw - 160:fw] = mask_bgr

        # Breadcrumb / back hint at the bottom of the frame
        hint = ('Calibration > Hand-Eye Yellow  |  '
                'ESC = back  |  SPACE = capture  |  S = solve  |  U = undo')
        cv2.putText(frame, hint, (10, fh - 10),
                    FONT, 0.32, (160, 160, 100), 1)

        # Temporary status message
        if self._status_msg and time.time() - self._status_time < 4.0:
            cv2.putText(frame, self._status_msg,
                        (10, fh - 30), FONT, 0.55, (0, 255, 0), 2)

        # Store per-frame state for handle_key()
        self._current_tcp = tcp_pos
        self._current_cx = cx
        self._current_cy = cy
        try:
            self._current_raw = read_all_raw(self._arm)
        except Exception:
            self._current_raw = None

        # Blit onto canvas
        ch = min(fh, vh)
        cw = min(fw, vw)
        canvas[0:ch, 0:cw] = frame[0:ch, 0:cw]

    # ------------------------------------------------------------------
    def handle_key(self, key):
        if key == 27:  # ESC → back to Calibration menu
            self.app.switch_view('calibration')
            return True

        if self._arm is None:
            return False

        if key == ord(' '):
            tcp = self._current_tcp
            cx = self._current_cx
            raw = self._current_raw
            if tcp is not None and cx is not None and raw is not None:
                cy = self._current_cy
                self._pts_3d.append(tcp.copy())
                self._pts_2d.append([float(cx), float(cy)])
                self._raw_positions_list.append(dict(raw))
                self._status_msg = (
                    f'Captured #{len(self._pts_2d)}: '
                    f'TCP=[{tcp[0]:.0f},{tcp[1]:.0f},{tcp[2]:.0f}] '
                    f'px=({cx},{cy})')
                print(f'  {self._status_msg}')
            else:
                self._status_msg = 'SKIP: no TCP or no yellow tape'
            self._status_time = time.time()
            return True

        if key == ord('s'):
            n = len(self._pts_2d)
            if n < 6:
                self._status_msg = f'Need >= 6 points (have {n})'
                self._status_time = time.time()
            else:
                print(f'\n  Joint solve: {n} points, optimising offsets + extrinsics...')
                opt_offsets, T_c2b = joint_solve(
                    self._raw_positions_list, self._pts_2d,
                    self._K, self._dist, self._solver)
                if opt_offsets is not None:
                    save_offsets_dict(opt_offsets)
                    save_handeye_calibration(T_c2b, HANDEYE_FILE)
                    self._status_msg = 'Joint solve OK — saved offsets + extrinsics'
                else:
                    self._status_msg = 'Joint solve FAILED'
                self._status_time = time.time()
                print(f'  {self._status_msg}')
            return True

        if key == ord('u'):
            if self._pts_2d:
                self._pts_2d.pop()
                self._pts_3d.pop()
                self._raw_positions_list.pop()
                self._status_msg = (
                    f'Undone — {len(self._pts_2d)} points remaining')
                self._status_time = time.time()
            return True

        return False

    # ------------------------------------------------------------------
    def cleanup(self):
        if self._arm is not None:
            try:
                self._arm.enable_torque()
                print('  Hand-eye yellow cleanup: torque re-enabled')
            except Exception:
                pass
