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

    For each sign combo, compute FK for all captures, transform the board
    corners into base frame, and measure consistency (the board is fixed so
    all captures should agree).

    Each capture has:
        raw: {motor_id: raw_pos}
        T_board_in_cam: 4x4 board pose in gripper-camera frame

    We optimize: 5 offsets + 6 T_cam_in_tcp params (rvec+tvec for the
    gripper camera mount on the TCP).

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
        rvec_init = np.array([np.pi, 0, 0])  # 180 deg around X (camera Z down)
        tvec_init = np.array([0.01, -0.02, -0.035])  # from config mount

        x0 = np.concatenate([current_offsets_raw.copy(), rvec_init, tvec_init])
        lb = np.concatenate([np.zeros(5), np.full(6, -np.inf)])
        ub = np.concatenate([np.full(5, 4095.0), np.full(6, np.inf)])

        residuals_fn = _make_residuals(signs_arr.copy())
        try:
            result = least_squares(residuals_fn, x0, method='trf',
                                   bounds=(lb, ub), max_nfev=3000)
            res = result.fun.reshape(-1, 3)
            errs_mm = np.linalg.norm(res, axis=1) * 1000  # m -> mm
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
