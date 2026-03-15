"""Brute-force servo sign/offset solver using ChArUco board observations.

Extracted from gui/views/servo_direction_calib_view.py so it can be used
by the PyQt GUI and tests without depending on the old OpenCV GUI framework.

The solver tries all 2^5 = 32 possible joint sign combinations and picks
the one that minimises the inconsistency of the ChArUco board position in the
robot base frame across multiple captures.
"""

import itertools
import os
import time

import cv2
import numpy as np

from calibration.calib_helpers import load_offsets
from config_loader import config_path

MOTOR_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
               'wrist_flex', 'wrist_roll']

MIN_CAPTURES = 6
GOOD_CAPTURES = 10


def _pose_to_matrix(pos_mm, rpy_deg):
    """Convert position (mm) + RPY (deg) to a 4x4 homogeneous matrix (meters)."""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler('xyz', rpy_deg, degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos_mm / 1000.0  # mm -> m
    return T


def _brute_force_signs(captures, solver, current_offsets_raw, verbose=True):
    """Try all 32 sign combinations and return the best.

    Uses a two-phase approach to prevent T_cam_tcp from absorbing sign errors:
      Phase 1: Estimate T_cam_in_tcp using the current signs (6 params only).
      Phase 2: Fix T_cam_tcp, try all 32 sign combos optimizing only offsets (5 params).

    Each capture has:
        raw: {motor_id: raw_pos}
        T_board_in_cam: 4x4 board pose in gripper-camera frame

    The residual: for each pair of captures (i,j), the board origin in base
    frame should match:  T_tcp_i * T_cam_tcp * T_board_cam_i  ~=
                         T_tcp_j * T_cam_tcp * T_board_cam_j
    """
    from scipy.optimize import least_squares

    n = len(captures)
    DEG_PER_POS = 360.0 / 4096.0

    original_signs = solver.signs.copy()
    original_offsets = solver.offsets_deg.copy()

    # Board poses in camera frame (one per capture)
    T_board_in_cam = [c['T_board_in_cam'] for c in captures]

    def _compute_board_origins(captures_list, signs_local, offsets_raw, T_cam_tcp):
        """Compute board origin in base frame for each capture."""
        origins = []
        for cap in captures_list:
            angles_deg = np.array([
                signs_local[j] *
                (cap['raw'][j + 1] - offsets_raw[j]) * DEG_PER_POS
                for j in range(5)
            ])
            try:
                solver.signs = np.ones(5)
                solver.offsets_deg = np.zeros(5)
                pos_mm, rpy_deg = solver.forward_kin(angles_deg)
                T_tcp = _pose_to_matrix(pos_mm, rpy_deg)
            except Exception:
                origins.append(np.full(3, 1e6))
                continue
            T_board_base = T_tcp @ T_cam_tcp @ cap['T_board_in_cam']
            origins.append(T_board_base[:3, 3])
        return origins

    def _consistency_error(origins):
        """Mean distance from each origin to the group mean (meters)."""
        origins_arr = np.array(origins)
        mean_origin = np.mean(origins_arr, axis=0)
        errs_m = np.linalg.norm(origins_arr - mean_origin, axis=1)
        return float(np.mean(errs_m)) * 1000  # mm

    # ---------------------------------------------------------------
    # Phase 1: Estimate T_cam_in_tcp using current signs + fixed offsets
    # ---------------------------------------------------------------
    current_signs = original_signs.copy()

    if verbose:
        print(f"\n  Phase 1: Estimating T_cam_in_tcp (current signs={original_signs.tolist()})...")

    def _make_tcam_residuals(signs_local, offsets_fixed):
        def residuals(x):
            rvec_ct = x[:3]
            tvec_ct = x[3:6]
            R_ct, _ = cv2.Rodrigues(rvec_ct)
            T_cam_tcp = np.eye(4)
            T_cam_tcp[:3, :3] = R_ct
            T_cam_tcp[:3, 3] = tvec_ct

            origins = _compute_board_origins(
                captures, signs_local, offsets_fixed, T_cam_tcp)
            mean_origin = np.mean(origins, axis=0)
            errs = []
            for bo in origins:
                errs.extend((bo - mean_origin).tolist())
            return np.array(errs)
        return residuals

    # Try multiple T_cam_tcp initializations
    rvec_inits = [
        np.array([np.pi, 0, 0]),       # Camera Z down (180° around X)
        np.array([0, np.pi, 0]),        # Camera Z up, flipped
        np.array([np.pi/2, 0, 0]),      # Camera Z forward
        np.array([-np.pi/2, 0, 0]),     # Camera Z backward
    ]
    tvec_inits = [
        np.array([0.0, 0.0, -0.04]),    # ~40mm below TCP
        np.array([0.01, -0.02, -0.035]),
        np.array([0.0, 0.0, -0.06]),
    ]

    best_tcam_err = 1e9
    best_tcam_result = None
    for rvec_init in rvec_inits:
        for tvec_init in tvec_inits:
            x0_tcam = np.concatenate([rvec_init, tvec_init])
            res_fn = _make_tcam_residuals(current_signs, current_offsets_raw)
            try:
                result = least_squares(res_fn, x0_tcam, method='lm', max_nfev=5000)
                res = result.fun.reshape(-1, 3)
                err = float(np.mean(np.linalg.norm(res, axis=1))) * 1000
                if err < best_tcam_err:
                    best_tcam_err = err
                    best_tcam_result = result
            except Exception:
                pass

    if best_tcam_result is None:
        if verbose:
            print(f"  Phase 1 FAILED — falling back to default T_cam_tcp")
        rvec_fixed = np.array([np.pi, 0, 0])
        tvec_fixed = np.array([0.0, 0.0, -0.04])
    else:
        rvec_fixed = best_tcam_result.x[:3]
        tvec_fixed = best_tcam_result.x[3:6]
        if verbose:
            print(f"  Phase 1: T_cam_tcp estimated, consistency={best_tcam_err:.1f}mm")

    R_fixed, _ = cv2.Rodrigues(rvec_fixed)
    T_cam_tcp_fixed = np.eye(4)
    T_cam_tcp_fixed[:3, :3] = R_fixed
    T_cam_tcp_fixed[:3, 3] = tvec_fixed

    # ---------------------------------------------------------------
    # Phase 2: Fix T_cam_tcp, try all 32 sign combos (offsets only)
    # ---------------------------------------------------------------
    if verbose:
        print(f"  Phase 2: Testing 32 sign combos with fixed T_cam_tcp...")

    all_results = []
    sign_options = [+1, -1]
    all_combos = list(itertools.product(sign_options, repeat=5))

    for signs in all_combos:
        signs_arr = np.array(signs, dtype=float)

        def _make_offsets_residuals(signs_local, T_cam_tcp):
            def residuals(x):
                offsets_raw = x[:5]
                origins = _compute_board_origins(
                    captures, signs_local, offsets_raw, T_cam_tcp)
                mean_origin = np.mean(origins, axis=0)
                errs = []
                for bo in origins:
                    errs.extend((bo - mean_origin).tolist())
                return np.array(errs)
            return residuals

        residuals_fn = _make_offsets_residuals(signs_arr.copy(), T_cam_tcp_fixed)
        lb = np.zeros(5)
        ub = np.full(5, 4095.0)
        x0 = current_offsets_raw.copy()

        try:
            result = least_squares(residuals_fn, x0, method='trf',
                                   bounds=(lb, ub), max_nfev=3000)
            res = result.fun.reshape(-1, 3)
            errs_mm = np.linalg.norm(res, axis=1) * 1000
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

    # ---------------------------------------------------------------
    # Phase 3: Refine best with joint T_cam_tcp + offsets optimization
    # ---------------------------------------------------------------
    all_results.sort(key=lambda r: r['mean_err_mm'])
    best_phase2 = all_results[0]

    if verbose:
        print(f"\n  Phase 2 top 5:")
        for i, r in enumerate(all_results[:5]):
            marker = ' <-- BEST' if i == 0 else ''
            print(f"    #{i+1}: signs={r['signs_str']}  "
                  f"err={r['mean_err_mm']:.1f}mm{marker}")

    # Final refinement: jointly optimize offsets + T_cam_tcp for best sign combo
    best_signs = best_phase2['signs'].copy()

    def _make_joint_residuals(signs_local):
        def residuals(x):
            offsets_raw = x[:5]
            rvec_ct = x[5:8]
            tvec_ct = x[8:11]
            R_ct, _ = cv2.Rodrigues(rvec_ct)
            T_cam_tcp = np.eye(4)
            T_cam_tcp[:3, :3] = R_ct
            T_cam_tcp[:3, 3] = tvec_ct

            origins = _compute_board_origins(
                captures, signs_local, offsets_raw, T_cam_tcp)
            mean_origin = np.mean(origins, axis=0)
            errs = []
            for bo in origins:
                errs.extend((bo - mean_origin).tolist())
            return np.array(errs)
        return residuals

    offsets_init = (best_phase2['opt_result'].x[:5]
                    if best_phase2['opt_result'] is not None
                    else current_offsets_raw.copy())
    x0_refine = np.concatenate([offsets_init, rvec_fixed, tvec_fixed])
    lb_refine = np.concatenate([np.zeros(5), np.full(6, -np.inf)])
    ub_refine = np.concatenate([np.full(5, 4095.0), np.full(6, np.inf)])
    res_fn = _make_joint_residuals(best_signs.copy())
    try:
        refined = least_squares(res_fn, x0_refine, method='trf',
                                bounds=(lb_refine, ub_refine), max_nfev=5000)
        res = refined.fun.reshape(-1, 3)
        errs_mm = np.linalg.norm(res, axis=1) * 1000
        refined_err = float(np.mean(errs_mm))
        if verbose:
            print(f"\n  Phase 3 refinement: {refined_err:.1f}mm "
                  f"(was {best_phase2['mean_err_mm']:.1f}mm)")
        # Update best result with refined values
        best_phase2['mean_err_mm'] = refined_err
        best_phase2['per_point_err'] = errs_mm.copy()
        best_phase2['opt_result'] = refined
    except Exception:
        if verbose:
            print(f"  Phase 3 refinement failed, using phase 2 result")

    # Restore solver
    solver.signs = original_signs
    solver.offsets_deg = original_offsets

    all_results.sort(key=lambda r: r['mean_err_mm'])

    best = all_results[0]
    T_cam_tcp = T_cam_tcp_fixed.copy()
    offsets_opt = current_offsets_raw.copy()
    if best['opt_result'] is not None:
        x_opt = best['opt_result'].x
        offsets_opt = x_opt[:5]
        # If the result has 11 params (phase 3 refinement), extract T_cam_tcp
        if len(x_opt) >= 11:
            R_ct, _ = cv2.Rodrigues(x_opt[5:8])
            T_cam_tcp = np.eye(4)
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

    # Ambiguity check — use a tight threshold to avoid false ambiguity.
    # With the two-phase solver, sign combos that differ from the best should
    # show clearly larger errors.  Use 10% relative margin or 1.5mm absolute.
    threshold = max(best['mean_err_mm'] * 1.10, best['mean_err_mm'] + 1.5)
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
