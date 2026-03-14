"""Auto servo calibration: direction (sign), zero offset, and scale check.

Uses the yellow-tape marker on the gripper and camera to determine the
correct per-joint sign and zero offset automatically.  The user just moves
the arm to diverse poses and presses SPACE to capture.  After enough
captures (>= 8), pressing 'S' will:

  1. Try all 2^5 = 32 possible sign combinations.
  2. For each, jointly optimize 5 zero offsets + 6 camera extrinsic params
     (11 total) by minimizing reprojection error of FK->pixel.
  3. Pick the combination with the lowest reprojection error.
  4. Save the results to servo_offsets.yaml (offsets + signs).

This should be run RIGHT AFTER camera intrinsics calibration, before
hand-eye calibration (which then only needs to refine the extrinsics).

View ID: servo_direction
Reached via calibration menu [5].
"""

import itertools
import os
import sys
import time

import cv2
import numpy as np

from gui.views.base import BaseView, ViewRegistry

FONT = cv2.FONT_HERSHEY_SIMPLEX

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, 'scripts')

MOTOR_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
               'wrist_flex', 'wrist_roll']

MIN_CAPTURES = 8   # Minimum for a reliable solve
GOOD_CAPTURES = 12  # Recommended number of captures


def _get_cg():
    """Lazy-import calibration_gui helpers."""
    if _SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, _SCRIPTS_DIR)
    import calibration_gui as _cg
    return _cg


def _brute_force_signs(raw_positions_list, pts_2d, K, dist_coeffs, solver,
                       current_offsets_raw, verbose=True):
    """Try all 32 sign combinations and return the best.

    For each sign combo, jointly optimize 5 zero offsets + 6 extrinsic
    parameters (rvec + tvec) to minimize reprojection error.

    Args:
        raw_positions_list: List of dicts {motor_id: raw_pos} per capture.
        pts_2d: List of [cx, cy] pixel observations.
        K: 3x3 intrinsic matrix.
        dist_coeffs: Distortion coefficients.
        solver: Arm101IKSolver instance (will be temporarily modified).
        current_offsets_raw: Array of 5 current zero offsets (raw).

    Returns:
        Dict with keys: signs, offsets_raw, T_cam2base, mean_err_px,
        per_point_err, all_results (sorted list of all combos tried).
    """
    from scipy.optimize import least_squares

    n = len(pts_2d)
    pts_2d_arr = np.array(pts_2d, dtype=np.float64)
    DEG_PER_POS = 360.0 / 4096.0

    # Save original signs to restore later
    original_signs = solver.signs.copy()
    original_offsets = solver.offsets_deg.copy()

    all_results = []

    # Generate all 32 sign combinations
    sign_options = [+1, -1]
    all_combos = list(itertools.product(sign_options, repeat=5))

    for combo_idx, signs in enumerate(all_combos):
        signs_arr = np.array(signs, dtype=float)

        # Temporarily set signs on solver
        solver.signs = signs_arr
        solver.offsets_deg = np.zeros(5)

        # Initial extrinsic: use a simple PnP from current offsets
        pts_3d_init = []
        for raw_pos in raw_positions_list:
            angles = np.array([
                (raw_pos[mid] - current_offsets_raw[mid - 1]) * DEG_PER_POS
                for mid in range(1, 6)
            ])
            try:
                tcp, _ = solver.forward_kin(angles)
                pts_3d_init.append(tcp)
            except Exception:
                pts_3d_init.append(np.zeros(3))

        # Simple PnP for extrinsic seed
        obj = np.array(pts_3d_init, dtype=np.float64).reshape(-1, 1, 3)
        img = pts_2d_arr.reshape(-1, 1, 2)
        try:
            ok, rv, tv = cv2.solvePnP(obj, img, K, dist_coeffs,
                                       flags=cv2.SOLVEPNP_EPNP)
            if not ok:
                ok, rv, tv = cv2.solvePnP(obj, img, K, dist_coeffs,
                                           flags=cv2.SOLVEPNP_ITERATIVE)
        except Exception:
            ok = False

        if ok:
            rvec_init = rv.flatten()
            tvec_init = tv.flatten()
        else:
            rvec_init = np.zeros(3)
            tvec_init = np.array([0.0, 0.0, 500.0])

        # Pack: [5 offsets, 3 rvec, 3 tvec]
        x0 = np.concatenate([current_offsets_raw.copy(), rvec_init, tvec_init])

        # Bounds: offsets in [0, 4095], rvec/tvec unbounded
        lb = np.concatenate([np.zeros(5), np.full(6, -np.inf)])
        ub = np.concatenate([np.full(5, 4095.0), np.full(6, np.inf)])

        def _make_residuals(signs_local):
            def residuals(x):
                offsets_raw = x[:5]
                rvec = x[5:8]
                tvec = x[8:11]
                R_b2c, _ = cv2.Rodrigues(rvec)

                errs = []
                for i in range(n):
                    angles = np.array([
                        (raw_positions_list[i][mid] - offsets_raw[mid - 1])
                        * DEG_PER_POS
                        for mid in range(1, 6)
                    ])
                    try:
                        solver.signs = signs_local
                        tcp, _ = solver.forward_kin(angles)
                    except Exception:
                        errs.extend([1000.0, 1000.0])
                        continue

                    p_cam = R_b2c @ tcp + tvec
                    if p_cam[2] <= 0:
                        errs.extend([1000.0, 1000.0])
                        continue
                    px = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
                    py = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
                    errs.append(px - pts_2d_arr[i, 0])
                    errs.append(py - pts_2d_arr[i, 1])
                return np.array(errs)
            return residuals

        residuals_fn = _make_residuals(signs_arr.copy())

        try:
            result = least_squares(residuals_fn, x0, method='trf',
                                   bounds=(lb, ub), max_nfev=2000)
            res = result.fun.reshape(-1, 2)
            errs_px = np.linalg.norm(res, axis=1)
            mean_err = float(np.mean(errs_px))
        except Exception:
            mean_err = 9999.0
            errs_px = np.full(n, 9999.0)
            result = None

        signs_str = ''.join('+' if s > 0 else '-' for s in signs)
        all_results.append({
            'signs': signs_arr.copy(),
            'signs_str': signs_str,
            'mean_err_px': mean_err,
            'per_point_err': errs_px.copy(),
            'opt_result': result,
        })

    # Restore original solver state
    solver.signs = original_signs
    solver.offsets_deg = original_offsets

    # Sort by error
    all_results.sort(key=lambda r: r['mean_err_px'])

    best = all_results[0]
    if best['opt_result'] is not None:
        x_opt = best['opt_result'].x
        offsets_opt = x_opt[:5]
        rvec_opt = x_opt[5:8]
        tvec_opt = x_opt[8:11]

        R_b2c, _ = cv2.Rodrigues(rvec_opt)
        T_b2c = np.eye(4)
        T_b2c[:3, :3] = R_b2c
        T_b2c[:3, 3] = tvec_opt
        T_c2b = np.linalg.inv(T_b2c)
    else:
        offsets_opt = current_offsets_raw.copy()
        T_c2b = np.eye(4)

    if verbose:
        print(f"\n  === Servo Direction Auto-Calibration Results ===")
        print(f"  Tested all {len(all_combos)} sign combinations with {n} captures")
        print(f"\n  Top 5 results:")
        for i, r in enumerate(all_results[:5]):
            marker = ' <-- BEST' if i == 0 else ''
            print(f"    #{i+1}: signs={r['signs_str']}  "
                  f"err={r['mean_err_px']:.1f}px{marker}")

        # Identify per-joint confidence by checking which joints differ
        # among results that are nearly as good as the best.
        # Use max(1.5x best, best + 3px) to catch genuine ambiguity
        # while allowing for small numerical noise.
        threshold = max(best['mean_err_px'] * 1.5,
                        best['mean_err_px'] + 3.0)
        near_best = [r for r in all_results if r['mean_err_px'] <= threshold]
        ambiguous_joints = set()
        for r in near_best[1:]:
            for j in range(5):
                if r['signs'][j] != best['signs'][j]:
                    ambiguous_joints.add(j)

        if not ambiguous_joints:
            print(f"\n  CLEAR winner — all joints unambiguously determined")
        elif len(ambiguous_joints) == 1:
            j = list(ambiguous_joints)[0]
            print(f"\n  NOTE: Joint '{MOTOR_NAMES[j]}' sign is ambiguous "
                  f"(does not affect TCP position much).")
            print(f"  This is expected for wrist_roll. Verify manually if needed.")
        else:
            names = [MOTOR_NAMES[j] for j in sorted(ambiguous_joints)]
            print(f"\n  WARNING: {len(ambiguous_joints)} joints are ambiguous: "
                  f"{', '.join(names)}")
            print(f"  Collect more diverse poses to disambiguate.")

        print(f"\n  Best signs:   {best['signs_str']}")
        print(f"  Offsets:      {offsets_opt.astype(int).tolist()}")
        print(f"  Mean error:   {best['mean_err_px']:.2f}px")

        # Per-joint comparison with current config
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
        'T_cam2base': T_c2b,
        'mean_err_px': best['mean_err_px'],
        'per_point_err': best['per_point_err'],
        'ambiguous_joints': ambiguous_joints,
        'all_results': all_results,
    }


def save_calibration_results(signs, offsets_raw, T_c2b):
    """Save signs + offsets to servo_offsets.yaml and extrinsics to calibration file."""
    import yaml

    # Save offsets + signs
    offset_file = os.path.join(_PROJECT_ROOT, 'config', 'servo_offsets.yaml')
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
            'usage': 'angle_deg = (raw_position - zero_raw) * 360/4096',
            'default': '2048 (servo center) if no offset defined',
            'signs': '+1 = motor and URDF agree, -1 = inverted',
            'calibrated_by': 'servo_direction auto-calibration',
            'calibrated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
    }
    os.makedirs(os.path.dirname(offset_file), exist_ok=True)
    with open(offset_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"  Saved offsets + signs to {offset_file}")

    # Save hand-eye extrinsics
    handeye_file = os.path.join(_PROJECT_ROOT, 'config', 'calibration_arm101.yaml')
    try:
        cg = _get_cg()
        cg.save_handeye_calibration(T_c2b, handeye_file)
    except Exception as exc:
        print(f"  WARNING: Could not save hand-eye calibration: {exc}")


# ---------------------------------------------------------------------------
# View
# ---------------------------------------------------------------------------

@ViewRegistry.register
class ServoDirectionCalibView(BaseView):
    """Auto servo calibration: detect joint signs, offsets, and verify scale.

    Uses yellow tape marker + camera to automatically determine the correct
    sign (direction) and zero offset for each servo by brute-forcing all
    32 possible sign combinations.

    Workflow:
      1. Attach yellow tape to gripper TCP
      2. Move arm to diverse poses (vary ALL joints)
      3. Press SPACE at each pose to capture
      4. Press S to auto-solve (needs >= 8 captures, 12+ recommended)

    Keys: SPACE=capture | S=solve | U=undo | R=reset all | ESC=leave
    """

    view_id = 'servo_direction'
    view_name = 'Servo Direction'
    description = 'Auto-detect servo signs + offsets (arm101)'
    needs_camera = True
    needs_robot = True
    headless_ok = False
    show_in_sidebar = False  # reached via calibration menu

    def __init__(self, app):
        super().__init__(app)
        self._arm = None
        self._solver = None
        self._cg = None
        self._K = None
        self._dist = None
        self._captures = []        # list of {raw: dict, pixel: [cx,cy]}
        self._status_msg = ''
        self._status_time = 0.0
        self._error_msg = ''
        self._result = None        # calibration result after solve
        # Per-frame state
        self._current_cx = None
        self._current_cy = None
        self._current_raw = None

    def setup(self):
        try:
            self._cg = _get_cg()
        except ImportError as exc:
            self._error_msg = f'Import error: {exc}'
            return

        self.app.ensure_robot()
        robot = self.app.robot
        if robot is None:
            self._error_msg = 'No robot connected'
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

        self.app.ensure_camera()
        self._K, self._dist = self._load_intrinsics()

        self._status_msg = ('Attach yellow tape to gripper. '
                            'Move arm to diverse poses, press SPACE to capture.')
        self._status_time = time.time() + 10  # show longer

    def _load_intrinsics(self):
        """Load intrinsics from cameras.yaml or defaults."""
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
                                 [[554.3, 0, 320], [0, 554.3, 240],
                                  [0, 0, 1]]),
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

    def update(self, canvas):
        vw = self.app.view_width
        vh = self.app.canvas_height

        if self._error_msg:
            canvas[:vh, :vw] = (30, 30, 35)
            cv2.putText(canvas, 'Servo Direction Calibration',
                        (20, 35), FONT, 0.6, (255, 200, 100), 1)
            cv2.putText(canvas, self._error_msg, (20, 80),
                        FONT, 0.5, (0, 80, 220), 1)
            cv2.putText(canvas, 'Press ESC or click another view.',
                        (20, 110), FONT, 0.38, (130, 130, 130), 1)
            return

        # Camera frame
        frame = None
        if self.app.camera is not None:
            color, _, _ = self.app.get_camera_frame()
            if color is not None:
                frame = color.copy()
        if frame is None:
            frame = np.zeros((vh, vw, 3), dtype=np.uint8)

        fh, fw = frame.shape[:2]

        # Yellow tape detection
        cx, cy, mask = self._cg.find_yellow_tape(frame)
        self._current_cx = cx
        self._current_cy = cy

        # Read raw positions
        try:
            self._current_raw = self._cg.read_all_raw(self._arm)
        except Exception:
            self._current_raw = None

        # Draw yellow marker
        if cx is not None:
            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
            cv2.circle(frame, (cx, cy), 14, (0, 255, 255), 2)
            cv2.putText(frame, f'({cx},{cy})', (cx + 16, cy - 8),
                        FONT, 0.4, (0, 255, 255), 1)

        # Draw captured points
        for i, cap in enumerate(self._captures):
            px, py = int(cap['pixel'][0]), int(cap['pixel'][1])
            cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
            cv2.putText(frame, str(i + 1), (px + 6, py - 4),
                        FONT, 0.3, (0, 200, 0), 1)

        # Mask preview (top-right)
        mask_small = cv2.resize(mask, (120, 90))
        mask_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        x_off = min(fw, vw) - 125
        if x_off > 0:
            frame[5:95, x_off:x_off + 120] = mask_bgr

        # Header
        n = len(self._captures)
        header_color = (0, 255, 255)
        cv2.putText(frame, 'SERVO DIRECTION CALIBRATION', (10, 25),
                    FONT, 0.55, header_color, 2)

        # Guidance
        if n < MIN_CAPTURES:
            guide = (f'Captures: {n}/{MIN_CAPTURES} (need {MIN_CAPTURES}, '
                     f'{GOOD_CAPTURES}+ recommended)')
            guide_color = (100, 180, 255)
        else:
            guide = (f'Captures: {n} - ready to solve! Press S')
            guide_color = (0, 255, 0)
        cv2.putText(frame, guide, (10, 50), FONT, 0.45, guide_color, 1)

        # Tips for diverse poses
        if n < 4:
            tips = [
                'Tip: Move ALL joints between captures for best results',
                'Tip: Include poses with arm extended, folded, rotated',
                'Tip: Keep yellow tape visible to camera at all times',
            ]
            tip = tips[min(n, len(tips) - 1)]
            cv2.putText(frame, tip, (10, 72), FONT, 0.35,
                        (150, 150, 200), 1)

        # Results display
        if self._result is not None:
            self._draw_results(frame)

        # Status message
        if self._status_msg and time.time() - self._status_time < 5.0:
            cv2.putText(frame, self._status_msg,
                        (10, fh - 55), FONT, 0.5, (0, 255, 0), 2)

        # Footer
        if cx is None:
            cv2.putText(frame, 'No yellow tape detected!',
                        (10, fh - 30), FONT, 0.45, (0, 0, 255), 1)
        cv2.putText(frame, 'SPACE=capture | S=solve | U=undo | R=reset | ESC=back',
                    (10, fh - 10), FONT, 0.35, (150, 150, 150), 1)

        # Blit onto canvas
        ch = min(fh, vh)
        cw = min(fw, vw)
        canvas[0:ch, 0:cw] = frame[0:ch, 0:cw]

    def _draw_results(self, frame):
        """Draw calibration results on frame."""
        r = self._result
        ambiguous = r.get('ambiguous_joints', set())
        y = 90
        cv2.putText(frame, f'Result: signs={r["signs_str"]}  '
                    f'err={r["mean_err_px"]:.1f}px',
                    (10, y), FONT, 0.42, (0, 255, 150), 1)
        y += 20

        # Per-joint signs
        from kinematics.arm101_ik_solver import JOINT_SIGNS
        for i, name in enumerate(MOTOR_NAMES):
            old_s = '+' if JOINT_SIGNS[i] > 0 else '-'
            new_s = '+' if r['signs'][i] > 0 else '-'
            if i in ambiguous:
                color = (0, 200, 255)  # orange-ish for ambiguous
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

        # Quality
        quality = ('EXCELLENT' if r['mean_err_px'] < 3 else
                   'GOOD' if r['mean_err_px'] < 8 else
                   'OK' if r['mean_err_px'] < 20 else 'POOR')
        qcolor = ((0, 255, 0) if quality in ('EXCELLENT', 'GOOD')
                  else (0, 200, 255) if quality == 'OK'
                  else (0, 0, 255))
        cv2.putText(frame, f'Quality: {quality}', (10, y + 5),
                    FONT, 0.45, qcolor, 1)
        if ambiguous:
            y += 22
            names = [MOTOR_NAMES[j] for j in sorted(ambiguous)]
            cv2.putText(frame, f'Ambiguous: {", ".join(names)} (verify manually)',
                        (10, y + 5), FONT, 0.35, (0, 200, 255), 1)

    def handle_key(self, key):
        if self._arm is None or self._cg is None:
            return False

        if key == ord(' '):
            return self._capture()

        if key == ord('s'):
            return self._solve()

        if key == ord('u'):
            if self._captures:
                self._captures.pop()
                self._status_msg = f'Undone - {len(self._captures)} captures left'
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
        """Capture current pose."""
        cx = self._current_cx
        raw = self._current_raw

        if cx is None:
            self._status_msg = 'SKIP: no yellow tape detected'
            self._status_time = time.time()
            return True

        if raw is None:
            self._status_msg = 'SKIP: cannot read servo positions'
            self._status_time = time.time()
            return True

        cy = self._current_cy

        # Check for duplicates (too close to previous capture)
        if self._captures:
            last = self._captures[-1]
            last_raw_vals = [last['raw'][m] for m in range(1, 6)]
            cur_raw_vals = [raw[m] for m in range(1, 6)]
            max_delta = max(abs(a - b) for a, b in
                           zip(cur_raw_vals, last_raw_vals))
            if max_delta < 30:  # ~2.6 degrees
                self._status_msg = 'SKIP: too similar to last capture (move more)'
                self._status_time = time.time()
                return True

        self._captures.append({
            'raw': dict(raw),
            'pixel': [float(cx), float(cy)],
        })
        n = len(self._captures)
        self._status_msg = f'Captured #{n}: px=({cx},{cy})'
        self._status_time = time.time()
        self._result = None  # invalidate previous solve
        print(f'  Captured #{n}: px=({cx},{cy}) '
              f'raw=[{",".join(str(raw[m]) for m in range(1,6))}]')
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

        # Current offsets as seed
        offsets = self._cg.load_offsets()
        current_offsets_raw = np.array([
            offsets.get(name, {}).get('zero_raw', 2048)
            for name in MOTOR_NAMES
        ], dtype=float)

        raw_list = [c['raw'] for c in self._captures]
        px_list = [c['pixel'] for c in self._captures]

        result = _brute_force_signs(
            raw_list, px_list, self._K, self._dist,
            self._solver, current_offsets_raw, verbose=True)

        self._result = result

        if result['mean_err_px'] < 30:
            # Save automatically
            save_calibration_results(
                result['signs'], result['offsets_raw'], result['T_cam2base'])
            self._status_msg = (f'Solved! signs={result["signs_str"]} '
                                f'err={result["mean_err_px"]:.1f}px - SAVED')
        else:
            self._status_msg = (f'Solved but error is high '
                                f'({result["mean_err_px"]:.0f}px). '
                                f'Collect more diverse poses.')

        self._status_time = time.time()
        return True

    def cleanup(self):
        if self._arm is not None:
            try:
                self._arm.enable_torque()
                print('  Servo direction calib cleanup: torque re-enabled')
            except Exception:
                pass
