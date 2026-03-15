"""Auto servo calibration: direction (sign), zero offset via ChArUco board.

Uses the gripper-mounted camera looking at a ChArUco board on the table to
determine the correct per-joint sign and zero offset automatically.  The user
moves the arm to diverse poses keeping the board visible, and presses SPACE
to capture.  After enough captures (>= 6), pressing 'S' will:

  1. Try all 2^5 = 32 possible sign combinations.
  2. For each, compute FK for all captures and check whether the board's
     position in the robot base frame is self-consistent (it should be,
     since the board doesn't move).
  3. Pick the combination with the lowest consistency error.
  4. Save the results to servo_offsets.yaml (offsets + signs).

View ID: servo_direction
Reached via calibration menu [5].
"""

import itertools
import os
import time

import cv2
import numpy as np

from config_loader import config_path, load_config
from gui.views.base import BaseView, ViewRegistry
from calibration.calib_helpers import (
    read_all_raw,
    load_offsets,
    save_handeye_calibration,
    HANDEYE_FILE,
)
from vision.board_detector import BoardDetector

FONT = cv2.FONT_HERSHEY_SIMPLEX

MOTOR_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
               'wrist_flex', 'wrist_roll']


def _pose_to_matrix(pos_mm, rpy_deg):
    """Convert position (mm) + RPY (deg) to a 4x4 homogeneous matrix (meters)."""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler('xyz', rpy_deg, degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos_mm / 1000.0  # mm → m
    return T

MIN_CAPTURES = 6   # Minimum for a reliable solve
GOOD_CAPTURES = 10  # Recommended number of captures


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def _brute_force_signs(captures, solver, current_offsets_raw, verbose=True):
    """Try all 32 sign combinations and return the best.

    For each sign combo, compute FK for all captures, transform the board
    corners into base frame, and measure consistency (the board is fixed so
    all captures should agree).

    Each capture has:
        raw: {motor_id: raw_pos}
        T_board_in_cam: 4x4 board pose in gripper-camera frame

    We optimize: 5 offsets + 6 T_cam_in_tcp params (rvec+tvec for the
    gripper camera mount on the TCP).

    The residual: for each pair of captures (i,j), the board origin in base
    frame should match:  T_tcp_i * T_cam_tcp * T_board_cam_i  ≈
                         T_tcp_j * T_cam_tcp * T_board_cam_j
    """
    from scipy.optimize import least_squares

    n = len(captures)
    DEG_PER_POS = 360.0 / 4096.0

    original_signs = solver.signs.copy()
    original_offsets = solver.offsets_deg.copy()

    # Board poses in camera frame (one per capture)
    T_board_in_cam = [c['T_board_in_cam'] for c in captures]

    all_results = []
    sign_options = [+1, -1]
    all_combos = list(itertools.product(sign_options, repeat=5))

    for signs in all_combos:
        signs_arr = np.array(signs, dtype=float)

        def _make_residuals(signs_local):
            def residuals(x):
                offsets_raw = x[:5]
                # T_cam_in_tcp as rvec + tvec
                rvec_ct = x[5:8]
                tvec_ct = x[8:11]
                R_ct, _ = cv2.Rodrigues(rvec_ct)
                T_cam_tcp = np.eye(4)
                T_cam_tcp[:3, :3] = R_ct
                T_cam_tcp[:3, 3] = tvec_ct

                # Compute board origin in base frame for each capture
                board_origins = []
                for cap in captures:
                    angles_deg = np.array([
                        signs_local[j] *
                        (cap['raw'][j + 1] - offsets_raw[j]) * DEG_PER_POS
                        for j in range(5)
                    ])
                    try:
                        solver.signs = np.ones(5)  # signs already applied
                        solver.offsets_deg = np.zeros(5)
                        pos_mm, rpy_deg = solver.forward_kin(angles_deg)
                        T_tcp = _pose_to_matrix(pos_mm, rpy_deg)
                    except Exception:
                        board_origins.append(np.full(3, 1e6))
                        continue
                    T_board_base = T_tcp @ T_cam_tcp @ cap['T_board_in_cam']
                    board_origins.append(T_board_base[:3, 3])

                # Residuals: distance from each board origin to the mean
                mean_origin = np.mean(board_origins, axis=0)
                errs = []
                for bo in board_origins:
                    errs.extend((bo - mean_origin).tolist())
                return np.array(errs)
            return residuals

        # Initial T_cam_in_tcp: camera looking down from gripper
        # Rough estimate: camera ~35mm below TCP, looking down
        rvec_init = np.array([np.pi, 0, 0])  # 180° around X (camera Z down)
        tvec_init = np.array([0.01, -0.02, -0.035])  # from config mount

        x0 = np.concatenate([current_offsets_raw.copy(), rvec_init, tvec_init])
        lb = np.concatenate([np.zeros(5), np.full(6, -np.inf)])
        ub = np.concatenate([np.full(5, 4095.0), np.full(6, np.inf)])

        residuals_fn = _make_residuals(signs_arr.copy())
        try:
            result = least_squares(residuals_fn, x0, method='trf',
                                   bounds=(lb, ub), max_nfev=3000)
            res = result.fun.reshape(-1, 3)
            errs_mm = np.linalg.norm(res, axis=1) * 1000  # m → mm
            mean_err = float(np.mean(errs_mm))
        except Exception:
            mean_err = 9999.0
            errs_mm = np.full(n, 9999.0)
            result = None

        signs_str = ''.join('+' if s > 0 else '-' for s in signs)
        all_results.append({
            'signs': signs_arr.copy(),
            'signs_str': signs_str,
            'mean_err_mm': mean_err,
            'per_point_err': errs_mm.copy(),
            'opt_result': result,
        })

    # Restore solver
    solver.signs = original_signs
    solver.offsets_deg = original_offsets

    all_results.sort(key=lambda r: r['mean_err_mm'])

    best = all_results[0]
    T_cam_tcp = np.eye(4)
    offsets_opt = current_offsets_raw.copy()
    if best['opt_result'] is not None:
        x_opt = best['opt_result'].x
        offsets_opt = x_opt[:5]
        R_ct, _ = cv2.Rodrigues(x_opt[5:8])
        T_cam_tcp[:3, :3] = R_ct
        T_cam_tcp[:3, 3] = x_opt[8:11]

    # Compute T_board_in_base from best result (average across captures)
    T_board_bases = []
    for cap in captures:
        angles_deg = np.array([
            best['signs'][j] *
            (cap['raw'][j + 1] - offsets_opt[j]) * DEG_PER_POS
            for j in range(5)
        ])
        try:
            solver.signs = np.ones(5)
            solver.offsets_deg = np.zeros(5)
            pos_mm, rpy_deg = solver.forward_kin(angles_deg)
            T_tcp = _pose_to_matrix(pos_mm, rpy_deg)
            T_board_bases.append(T_tcp @ T_cam_tcp @ cap['T_board_in_cam'])
        except Exception:
            pass
    solver.signs = original_signs
    solver.offsets_deg = original_offsets

    # Ambiguity check
    threshold = max(best['mean_err_mm'] * 1.5, best['mean_err_mm'] + 5.0)
    near_best = [r for r in all_results if r['mean_err_mm'] <= threshold]
    ambiguous_joints = set()
    for r in near_best[1:]:
        for j in range(5):
            if r['signs'][j] != best['signs'][j]:
                ambiguous_joints.add(j)

    if verbose:
        print(f"\n  === Servo Direction Auto-Calibration Results ===")
        print(f"  Tested all {len(all_combos)} sign combinations with {n} captures")
        print(f"\n  Top 5 results:")
        for i, r in enumerate(all_results[:5]):
            marker = ' <-- BEST' if i == 0 else ''
            print(f"    #{i+1}: signs={r['signs_str']}  "
                  f"err={r['mean_err_mm']:.1f}mm{marker}")

        if not ambiguous_joints:
            print(f"\n  CLEAR winner — all joints unambiguously determined")
        elif len(ambiguous_joints) == 1:
            j = list(ambiguous_joints)[0]
            print(f"\n  NOTE: Joint '{MOTOR_NAMES[j]}' sign is ambiguous.")
        else:
            names = [MOTOR_NAMES[j] for j in sorted(ambiguous_joints)]
            print(f"\n  WARNING: {len(ambiguous_joints)} joints ambiguous: "
                  f"{', '.join(names)}")

        print(f"\n  Best signs:   {best['signs_str']}")
        print(f"  Offsets:      {offsets_opt.astype(int).tolist()}")
        print(f"  Mean error:   {best['mean_err_mm']:.2f}mm")

        from kinematics.arm101_ik_solver import JOINT_SIGNS
        print(f"\n  Per-joint comparison:")
        print(f"    {'Joint':<16} {'Current':>8} {'Found':>8} {'Confidence':>12}")
        for i, name in enumerate(MOTOR_NAMES):
            cur = '+' if JOINT_SIGNS[i] > 0 else '-'
            found = '+' if best['signs'][i] > 0 else '-'
            if i in ambiguous_joints:
                conf = 'AMBIGUOUS'
            elif JOINT_SIGNS[i] == best['signs'][i]:
                conf = 'confirmed'
            else:
                conf = '** CHANGED **'
            print(f"    {name:<16} {cur:>8} {found:>8} {conf:>12}")

    return {
        'signs': best['signs'].copy(),
        'signs_str': best['signs_str'],
        'offsets_raw': offsets_opt.copy(),
        'T_cam_in_tcp': T_cam_tcp,
        'mean_err_mm': best['mean_err_mm'],
        'per_point_err': best['per_point_err'],
        'ambiguous_joints': ambiguous_joints,
        'all_results': all_results,
    }


def save_calibration_results(signs, offsets_raw, T_cam_in_tcp):
    """Save signs + offsets to servo_offsets.yaml."""
    import yaml

    offset_file = config_path('servo_offsets.yaml')
    offsets_dict = {}
    for i, name in enumerate(MOTOR_NAMES):
        offsets_dict[name] = {
            'motor_id': i + 1,
            'zero_raw': int(round(offsets_raw[i])),
        }

    signs_dict = {}
    for i, name in enumerate(MOTOR_NAMES):
        signs_dict[name] = int(signs[i])

    data = {
        'description': 'Servo zero offsets and joint signs for SO-ARM101',
        'zero_offsets': offsets_dict,
        'joint_signs': signs_dict,
        'notes': {
            'usage': 'angle_deg = sign * (raw_position - zero_raw) * 360/4096',
            'default': '2048 (servo center) if no offset defined',
            'signs': '+1 = motor and URDF agree, -1 = inverted',
            'calibrated_by': 'servo_direction auto-calibration (ChArUco)',
            'calibrated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
    }
    os.makedirs(os.path.dirname(offset_file), exist_ok=True)
    with open(offset_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"  Saved offsets + signs to {offset_file}")


# ---------------------------------------------------------------------------
# View
# ---------------------------------------------------------------------------

@ViewRegistry.register
class ServoDirectionCalibView(BaseView):
    """Auto servo calibration using gripper camera + ChArUco board.

    The gripper camera sees the ChArUco board on the table. For each arm
    pose we get the board pose in camera frame (from PnP) and raw servo
    positions. Since the board is fixed, the board position in robot-base
    frame should be consistent across all captures — the solver finds the
    sign/offset combination that minimizes inconsistency.

    Workflow:
      1. Place ChArUco board on the table under the arm
      2. Move arm to diverse poses (keep board visible in gripper camera)
      3. Press SPACE at each pose to capture
      4. Press S to auto-solve (needs >= 6 captures, 10+ recommended)

    Keys: SPACE=capture | S=solve | U=undo | R=reset all | ESC=leave
    """

    view_id = 'servo_direction'
    view_name = 'Servo Direction'
    description = 'Auto-detect servo signs + offsets (arm101)'
    needs_camera = False  # we open our own gripper camera
    needs_robot = True
    headless_ok = False
    show_in_sidebar = False
    parent_view_id = 'calibration'

    def __init__(self, app):
        super().__init__(app)
        self._arm = None
        self._solver = None
        self._board_detector = None
        self._gripper_cap = None  # cv2.VideoCapture for gripper camera
        self._gripper_K = None
        self._gripper_dist = None
        self._captures = []   # list of {raw, T_board_in_cam, corners_px}
        self._status_msg = ''
        self._status_time = 0.0
        self._error_msg = ''
        self._result = None
        # Per-frame state
        self._current_frame = None
        self._current_detection = None
        self._current_T_board = None
        self._current_raw = None

    def setup(self):
        self.app.ensure_robot()
        robot = self.app.robot
        if robot is None:
            reason = getattr(self.app, '_robot_error', None) or 'unknown'
            self._error_msg = f'No robot connected: {reason}'
            return
        if getattr(robot, 'robot_type', None) != 'arm101':
            self._error_msg = 'Servo direction calibration requires arm101'
            return

        self._arm = robot

        try:
            self._arm.disable_torque()
            print('  Servo direction calib: torque disabled, move arm by hand')
        except Exception as exc:
            print(f'  WARNING: Could not disable torque: {exc}')

        # FK solver
        try:
            from kinematics.arm101_ik_solver import Arm101IKSolver
            self._solver = Arm101IKSolver()
        except Exception as exc:
            self._error_msg = f'FK solver not available: {exc}'
            return

        # Board detector from config
        try:
            self._board_detector = BoardDetector.from_config(self.app.config)
        except Exception as exc:
            self._error_msg = f'Board detector error: {exc}'
            return

        # Open gripper camera
        gc = self.app.config.get('gripper_camera', {})
        dev_idx = gc.get('device_index', 0)
        width = gc.get('width', 640)
        height = gc.get('height', 480)
        print(f'  Opening gripper camera /dev/video{dev_idx}...')
        self._gripper_cap = cv2.VideoCapture(dev_idx)
        if not self._gripper_cap.isOpened():
            self._error_msg = f'Cannot open gripper camera (device {dev_idx})'
            return
        self._gripper_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._gripper_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Flush initial frames
        for _ in range(10):
            self._gripper_cap.read()
        print('  Gripper camera ready.')

        # Load gripper camera intrinsics
        self._gripper_K, self._gripper_dist = self._load_gripper_intrinsics()

        self._status_msg = ('Place ChArUco board under arm. '
                            'Move arm to diverse poses, press SPACE.')
        self._status_time = time.time() + 10

    def _load_gripper_intrinsics(self):
        """Load gripper camera intrinsics from cameras.yaml or estimate."""
        import yaml
        import math
        gc = self.app.config.get('gripper_camera', {})
        dev_idx = gc.get('device_index', 0)
        cam_yaml = config_path('cameras.yaml')

        if os.path.exists(cam_yaml):
            with open(cam_yaml) as fh:
                cdata = yaml.safe_load(fh)
            for cname, cinfo in (cdata or {}).get('cameras', {}).items():
                if cinfo.get('device_index') == dev_idx:
                    intr = cinfo.get('intrinsics', {})
                    cm = intr.get('camera_matrix')
                    dc = intr.get('dist_coeffs')
                    if cm is not None:
                        K = np.array(cm, dtype=np.float64)
                        dist = np.array(dc or [0, 0, 0, 0, 0],
                                        dtype=np.float64)
                        print(f'  Gripper intrinsics from {cname}: '
                              f'fx={K[0, 0]:.1f}')
                        return K, dist

        # Estimate from HFOV
        w = gc.get('width', 640)
        h = gc.get('height', 480)
        hfov = gc.get('hfov_deg', 60.0)
        fx = w / (2.0 * math.tan(math.radians(hfov / 2.0)))
        K = np.array([[fx, 0, w / 2.0],
                       [0, fx, h / 2.0],
                       [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
        print(f'  Gripper intrinsics estimated: fx={fx:.1f} '
              f'(hfov={hfov}°)')
        return K, dist

    def update(self, canvas):
        vw = self.app.view_width
        vh = self.app.canvas_height

        if self._error_msg:
            canvas[:vh, :vw] = (30, 30, 35)
            cv2.putText(canvas, 'Servo Direction Calibration',
                        (20, 35), FONT, 0.6, (255, 200, 100), 1)
            cv2.putText(canvas, self._error_msg, (20, 80),
                        FONT, 0.5, (0, 80, 220), 1)
            cv2.putText(canvas, 'Press ESC to return to Calibration menu.',
                        (20, 110), FONT, 0.38, (130, 130, 130), 1)
            return

        # Grab gripper camera frame
        frame = None
        if self._gripper_cap is not None:
            ret, raw_frame = self._gripper_cap.read()
            if ret:
                frame = raw_frame
        if frame is None:
            frame = np.zeros((vh, vw, 3), dtype=np.uint8)

        self._current_frame = frame
        fh, fw = frame.shape[:2]

        # Detect board
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detection = self._board_detector.detect(gray)
        self._current_detection = detection

        # Compute board pose if detected
        self._current_T_board = None
        n_corners = 0
        if detection is not None:
            n_corners = len(detection.corners)
            # Build a CameraIntrinsics-like object for compute_pose
            intr = _SimpleIntrinsics(self._gripper_K, self._gripper_dist)
            T, _, reproj = self._board_detector.compute_pose(detection, intr)
            self._current_T_board = T

            # Draw detected corners
            corners_draw = detection.corners.reshape(-1, 2).astype(int)
            for pt in corners_draw:
                cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

            if T is not None:
                cv2.putText(frame, f'Board: {n_corners} corners, '
                            f'reproj={reproj:.1f}px',
                            (10, fh - 75), FONT, 0.4, (0, 255, 0), 1)

        # Read raw servo positions
        try:
            self._current_raw = read_all_raw(self._arm)
        except Exception:
            self._current_raw = None

        # Draw captured points (show the centroid of corners for each capture)
        for i, cap in enumerate(self._captures):
            cx, cy = int(cap.get('centroid_px', [0, 0])[0]), \
                     int(cap.get('centroid_px', [0, 0])[1])
            cv2.drawMarker(frame, (cx, cy), (0, 200, 0),
                           cv2.MARKER_DIAMOND, 10, 1)
            cv2.putText(frame, str(i + 1), (cx + 8, cy - 4),
                        FONT, 0.3, (0, 200, 0), 1)

        # Header
        n = len(self._captures)
        cv2.putText(frame, 'SERVO DIRECTION CALIBRATION (ChArUco)',
                    (10, 25), FONT, 0.5, (0, 255, 255), 2)

        # Guidance
        if n < MIN_CAPTURES:
            guide = (f'Captures: {n}/{MIN_CAPTURES} (need {MIN_CAPTURES}, '
                     f'{GOOD_CAPTURES}+ recommended)')
            guide_color = (100, 180, 255)
        else:
            guide = f'Captures: {n} - ready to solve! Press S'
            guide_color = (0, 255, 0)
        cv2.putText(frame, guide, (10, 50), FONT, 0.45, guide_color, 1)

        if n < 4:
            tips = [
                'Tip: Move ALL joints between captures',
                'Tip: Include extended, folded, and rotated poses',
                'Tip: Keep ChArUco board visible in gripper camera',
            ]
            cv2.putText(frame, tips[min(n, len(tips) - 1)], (10, 72),
                        FONT, 0.35, (150, 150, 200), 1)

        # Results
        if self._result is not None:
            self._draw_results(frame)

        # Status message
        if self._status_msg and time.time() - self._status_time < 5.0:
            cv2.putText(frame, self._status_msg,
                        (10, fh - 55), FONT, 0.5, (0, 255, 0), 2)

        # Footer
        if detection is None or n_corners < 4:
            cv2.putText(frame, 'No board detected — point gripper camera at board',
                        (10, fh - 30), FONT, 0.4, (0, 0, 255), 1)
        cv2.putText(frame,
                    'Calibration > Servo Direction  |  '
                    'SPACE=capture | S=solve | U=undo | R=reset | ESC=back',
                    (10, fh - 10), FONT, 0.32, (160, 160, 100), 1)

        # Blit onto canvas (resize if needed)
        ch = min(fh, vh)
        cw = min(fw, vw)
        if fh != vh or fw != vw:
            frame = cv2.resize(frame, (vw, vh))
            ch, cw = vh, vw
        canvas[0:ch, 0:cw] = frame[0:ch, 0:cw]

    def _draw_results(self, frame):
        """Draw calibration results on frame."""
        r = self._result
        ambiguous = r.get('ambiguous_joints', set())
        y = 90
        cv2.putText(frame, f'Result: signs={r["signs_str"]}  '
                    f'err={r["mean_err_mm"]:.1f}mm',
                    (10, y), FONT, 0.42, (0, 255, 150), 1)
        y += 20

        from kinematics.arm101_ik_solver import JOINT_SIGNS
        for i, name in enumerate(MOTOR_NAMES):
            old_s = '+' if JOINT_SIGNS[i] > 0 else '-'
            new_s = '+' if r['signs'][i] > 0 else '-'
            if i in ambiguous:
                color = (0, 200, 255)
                label = 'AMBIG'
            elif JOINT_SIGNS[i] != r['signs'][i]:
                color = (0, 0, 255)
                label = 'CHANGED'
            else:
                color = (0, 200, 0)
                label = 'ok'
            offset = int(round(r['offsets_raw'][i]))
            cv2.putText(frame,
                        f'  {name[:14]:<14}  sign: {old_s}->{new_s} [{label}]'
                        f'  offset: {offset}',
                        (10, y), FONT, 0.35, color, 1)
            y += 17

        quality = ('EXCELLENT' if r['mean_err_mm'] < 5 else
                   'GOOD' if r['mean_err_mm'] < 15 else
                   'OK' if r['mean_err_mm'] < 30 else 'POOR')
        qcolor = ((0, 255, 0) if quality in ('EXCELLENT', 'GOOD')
                  else (0, 200, 255) if quality == 'OK'
                  else (0, 0, 255))
        cv2.putText(frame, f'Quality: {quality}', (10, y + 5),
                    FONT, 0.45, qcolor, 1)
        if ambiguous:
            y += 22
            names = [MOTOR_NAMES[j] for j in sorted(ambiguous)]
            cv2.putText(frame,
                        f'Ambiguous: {", ".join(names)} (verify manually)',
                        (10, y + 5), FONT, 0.35, (0, 200, 255), 1)

    def handle_key(self, key):
        if key == 27:
            self.app.switch_view('calibration')
            return True

        if self._arm is None:
            return False

        if key == ord(' '):
            return self._capture()
        if key == ord('s'):
            return self._solve()
        if key == ord('u'):
            if self._captures:
                self._captures.pop()
                self._status_msg = (f'Undone - {len(self._captures)} '
                                    f'captures left')
                self._status_time = time.time()
                self._result = None
            return True
        if key == ord('r'):
            self._captures = []
            self._result = None
            self._status_msg = 'Reset - all captures cleared'
            self._status_time = time.time()
            return True
        return False

    def _capture(self):
        """Capture current pose: raw servos + board pose in camera."""
        raw = self._current_raw
        T_board = self._current_T_board
        detection = self._current_detection

        if T_board is None:
            self._status_msg = 'SKIP: no board detected'
            self._status_time = time.time()
            return True

        if raw is None:
            self._status_msg = 'SKIP: cannot read servo positions'
            self._status_time = time.time()
            return True

        # Duplicate check: any joint must move >= 20 raw (~1.8°)
        cur_raw_vals = [raw[m] for m in range(1, 6)]
        for prev in self._captures:
            prev_raw_vals = [prev['raw'][m] for m in range(1, 6)]
            max_delta = max(abs(a - b) for a, b in
                           zip(cur_raw_vals, prev_raw_vals))
            if max_delta < 20:
                self._status_msg = (f'SKIP: too similar '
                                    f'(max delta={max_delta} raw '
                                    f'≈{max_delta * 360 / 4096:.1f}°)')
                self._status_time = time.time()
                return True

        # Corner centroid for visualization
        corners_2d = detection.corners.reshape(-1, 2)
        centroid = corners_2d.mean(axis=0).tolist()

        self._captures.append({
            'raw': dict(raw),
            'T_board_in_cam': T_board.copy(),
            'centroid_px': centroid,
        })
        n = len(self._captures)
        n_corners = len(detection.corners)
        self._status_msg = (f'Captured #{n}: {n_corners} corners, '
                            f'board at z={T_board[2, 3] * 1000:.0f}mm')
        self._status_time = time.time()
        self._result = None
        print(f'  Captured #{n}: {n_corners} corners  '
              f'raw=[{",".join(str(raw[m]) for m in range(1, 6))}]')
        return True

    def _solve(self):
        """Run brute-force sign detection."""
        n = len(self._captures)
        if n < MIN_CAPTURES:
            self._status_msg = (f'Need >= {MIN_CAPTURES} captures '
                                f'(have {n})')
            self._status_time = time.time()
            return True

        if self._solver is None:
            self._status_msg = 'FK solver not available'
            self._status_time = time.time()
            return True

        self._status_msg = 'Solving... (testing 32 sign combinations)'
        self._status_time = time.time()
        print(f'\n  Starting auto-calibration with {n} captures...')

        offsets = load_offsets()
        current_offsets_raw = np.array([
            offsets.get(name, {}).get('zero_raw', 2048)
            for name in MOTOR_NAMES
        ], dtype=float)

        result = _brute_force_signs(
            self._captures, self._solver, current_offsets_raw, verbose=True)

        self._result = result

        if result['mean_err_mm'] < 30:
            save_calibration_results(
                result['signs'], result['offsets_raw'],
                result.get('T_cam_in_tcp', np.eye(4)))
            self._status_msg = (f'Solved! signs={result["signs_str"]} '
                                f'err={result["mean_err_mm"]:.1f}mm - SAVED')
        else:
            self._status_msg = (f'Solved but error is high '
                                f'({result["mean_err_mm"]:.0f}mm). '
                                f'Collect more diverse poses.')

        self._status_time = time.time()
        return True

    def cleanup(self):
        if self._gripper_cap is not None:
            self._gripper_cap.release()
            self._gripper_cap = None
            print('  Gripper camera released')
        if self._arm is not None:
            try:
                self._arm.enable_torque()
                print('  Servo direction calib cleanup: torque re-enabled')
            except Exception:
                pass


class _SimpleIntrinsics:
    """Minimal intrinsics wrapper for BoardDetector.compute_pose()."""
    def __init__(self, K, dist):
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.ppx = K[0, 2]
        self.ppy = K[1, 2]
        self.coeffs = dist.tolist() if hasattr(dist, 'tolist') else list(dist)
