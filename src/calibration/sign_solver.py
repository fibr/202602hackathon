"""Brute-force servo sign/offset solver using ChArUco board observations.

Extracted from gui/views/servo_direction_calib_view.py so it can be used
by the PyQt GUI and tests without depending on the old OpenCV GUI framework.

The solver tries all 2^5 = 32 possible joint sign combinations and picks
the one that minimises the inconsistency of the ChArUco board pose (position
AND orientation) in the robot base frame across multiple captures.

Constraints to reduce ambiguity (11 free params → better conditioned):
  1. Orientation consistency: board rotation in base frame must also be
     consistent across captures, not just position (adds 3N residuals).
  2. Offset regularization: penalises large deviations from current offsets,
     preventing the optimizer from drifting to physically implausible values.
  3. T_cam_tcp prior: penalises camera transforms outside physically
     reasonable bounds (distance, orientation relative to gripper).
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

# --- Constraint weights (relative to position consistency in meters) ---
# Orientation weight: rotation error (radians) scaled to be comparable
# to translation error (meters).  A 0.01 rad (~0.6°) orientation error
# should be penalised roughly like a 1mm position error.
ORIENTATION_WEIGHT = 0.1  # 0.1 m/rad → 1 rad error = 100 mm penalty

# Offset regularization: penalty per raw-unit deviation from current offset.
# 1 raw unit ≈ 0.088° — mild penalty keeps offsets near prior values.
OFFSET_REG_WEIGHT = 1e-5  # meters per raw unit deviation

# T_cam_tcp translation prior: penalty for deviating from expected mount
# distance.  Typical gripper camera is 30-60 mm from TCP.
TCAM_TRANSLATION_PRIOR_WEIGHT = 0.01  # meters per meter deviation
TCAM_EXPECTED_DISTANCE_M = 0.045  # ~45 mm from TCP (typical mount)
TCAM_DISTANCE_TOLERANCE_M = 0.025  # ±25 mm before penalty kicks in


def _pose_to_matrix(pos_mm, rpy_deg):
    """Convert position (mm) + RPY (deg) to a 4x4 homogeneous matrix (meters)."""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler('xyz', rpy_deg, degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos_mm / 1000.0  # mm -> m
    return T


def _rotation_error(R1, R2):
    """Geodesic rotation error between two 3x3 rotation matrices (radians).

    Returns the angle of the rotation R1^T @ R2, which is the geodesic
    distance on SO(3).  Clamped to [0, pi].
    """
    R_diff = R1.T @ R2
    # Clamp trace to valid range for arccos
    trace = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
    return np.abs(np.arccos(trace))


def _compute_board_poses(captures_list, signs_local, offsets_raw, T_cam_tcp,
                         solver):
    """Compute full board pose (4x4) in base frame for each capture.

    Returns list of 4x4 matrices (or None for failed FK).
    """
    DEG_PER_POS = 360.0 / 4096.0
    poses = []
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
            poses.append(None)
            continue
        T_board_base = T_tcp @ T_cam_tcp @ cap['T_board_in_cam']
        poses.append(T_board_base)
    return poses


def _build_consistency_residuals(board_poses, orientation_weight):
    """Build position + orientation consistency residuals.

    For N captures, produces:
      - 3*(N-1) position residuals: deviation from mean position (meters)
      - 3*(N-1) orientation residuals: rotation error vs mean orientation
        (radians * weight, converted to meters-equivalent)

    Returns flat residual array.
    """
    valid = [T for T in board_poses if T is not None]
    if len(valid) < 2:
        return np.full(6, 1e6)

    # Position consistency
    origins = np.array([T[:3, 3] for T in valid])
    mean_origin = np.mean(origins, axis=0)
    pos_residuals = (origins - mean_origin).ravel()  # 3*N values in meters

    # Orientation consistency: compute mean rotation, then per-capture
    # rotation error expressed as a 3-vector (axis-angle deviation)
    rotations = [T[:3, :3] for T in valid]

    # Mean rotation via iterative Karcher/Riemannian mean (simple version:
    # use first rotation as reference, average log-map deviations)
    R_ref = rotations[0]
    ori_residuals = []
    for R in rotations:
        R_diff = R_ref.T @ R
        # Convert to axis-angle (Rodrigues)
        rvec, _ = cv2.Rodrigues(R_diff)
        ori_residuals.extend((rvec.flatten() * orientation_weight).tolist())

    # Remove mean from orientation residuals (like position)
    ori_arr = np.array(ori_residuals).reshape(-1, 3)
    mean_ori = np.mean(ori_arr, axis=0)
    ori_centered = (ori_arr - mean_ori).ravel()

    return np.concatenate([pos_residuals, ori_centered])


def _brute_force_signs(captures, solver, current_offsets_raw, verbose=True,
                       orientation_weight=ORIENTATION_WEIGHT,
                       offset_reg_weight=OFFSET_REG_WEIGHT,
                       tcam_prior_weight=TCAM_TRANSLATION_PRIOR_WEIGHT):
    """Try all 32 sign combinations and return the best.

    Uses a constrained multi-phase approach:
      Phase 1: Estimate T_cam_in_tcp using current signs (6 params),
               with orientation + position consistency.
      Phase 2: Fix T_cam_tcp, try all 32 sign combos optimizing offsets (5
               params) with orientation consistency + offset regularization.
      Phase 3: Jointly refine best sign combo (offsets + T_cam_tcp, 11 params)
               with all constraints active.

    Constraints vs prior solver:
      - Orientation consistency prevents T_cam_tcp from absorbing sign errors
        (wrong signs → inconsistent board rotation, not just position)
      - Offset regularization prevents large offset drift that could mask
        sign errors
      - T_cam_tcp translation prior keeps camera transform physically
        reasonable

    Each capture has:
        raw: {motor_id: raw_pos}
        T_board_in_cam: 4x4 board pose in gripper-camera frame

    Args:
        captures: List of capture dicts.
        solver: Arm101IKSolver instance.
        current_offsets_raw: np.ndarray of 5 current raw offsets.
        verbose: Print progress and results.
        orientation_weight: Weight for orientation residuals (m/rad).
        offset_reg_weight: Weight for offset regularization (m/raw-unit).
        tcam_prior_weight: Weight for T_cam_tcp translation prior.

    Returns:
        Dict with signs, offsets, T_cam_in_tcp, errors, diagnostics.
    """
    from scipy.optimize import least_squares

    n = len(captures)
    DEG_PER_POS = 360.0 / 4096.0

    original_signs = solver.signs.copy()
    original_offsets = solver.offsets_deg.copy()

    def _restore_solver():
        solver.signs = original_signs
        solver.offsets_deg = original_offsets

    # ---------------------------------------------------------------
    # Phase 1: Estimate T_cam_in_tcp using current signs + fixed offsets
    # ---------------------------------------------------------------
    current_signs = original_signs.copy()

    if verbose:
        print(f"\n  Phase 1: Estimating T_cam_in_tcp "
              f"(current signs={original_signs.tolist()})...")
        print(f"  Constraints: orientation_weight={orientation_weight}, "
              f"offset_reg={offset_reg_weight}, tcam_prior={tcam_prior_weight}")

    def _make_tcam_residuals(signs_local, offsets_fixed):
        def residuals(x):
            rvec_ct = x[:3]
            tvec_ct = x[3:6]
            R_ct, _ = cv2.Rodrigues(rvec_ct)
            T_cam_tcp = np.eye(4)
            T_cam_tcp[:3, :3] = R_ct
            T_cam_tcp[:3, 3] = tvec_ct

            board_poses = _compute_board_poses(
                captures, signs_local, offsets_fixed, T_cam_tcp, solver)
            consistency = _build_consistency_residuals(
                board_poses, orientation_weight)

            # T_cam_tcp translation prior: penalize if distance from TCP
            # is outside expected range
            dist = np.linalg.norm(tvec_ct)
            dist_dev = max(0.0, abs(dist - TCAM_EXPECTED_DISTANCE_M)
                           - TCAM_DISTANCE_TOLERANCE_M)
            tcam_prior = np.array([dist_dev * tcam_prior_weight])

            return np.concatenate([consistency, tcam_prior])
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
                result = least_squares(res_fn, x0_tcam, method='lm',
                                       max_nfev=5000)
                # Extract just the position part for error reporting
                board_poses = _compute_board_poses(
                    captures, current_signs, current_offsets_raw,
                    _rvec_tvec_to_T(result.x[:3], result.x[3:6]), solver)
                valid_origins = [T[:3, 3] for T in board_poses if T is not None]
                if valid_origins:
                    mean_o = np.mean(valid_origins, axis=0)
                    err = float(np.mean([np.linalg.norm(o - mean_o)
                                         for o in valid_origins])) * 1000
                else:
                    err = 1e9
                if err < best_tcam_err:
                    best_tcam_err = err
                    best_tcam_result = result
            except Exception:
                pass

    _restore_solver()

    if best_tcam_result is None:
        if verbose:
            print(f"  Phase 1 FAILED — falling back to default T_cam_tcp")
        rvec_fixed = np.array([np.pi, 0, 0])
        tvec_fixed = np.array([0.0, 0.0, -0.04])
    else:
        rvec_fixed = best_tcam_result.x[:3]
        tvec_fixed = best_tcam_result.x[3:6]
        if verbose:
            print(f"  Phase 1: T_cam_tcp estimated, "
                  f"consistency={best_tcam_err:.1f}mm")

    T_cam_tcp_fixed = _rvec_tvec_to_T(rvec_fixed, tvec_fixed)

    # ---------------------------------------------------------------
    # Phase 2: Fix T_cam_tcp, try all 32 sign combos (offsets only)
    #          Now with orientation consistency + offset regularization
    # ---------------------------------------------------------------
    if verbose:
        print(f"  Phase 2: Testing 32 sign combos with fixed T_cam_tcp "
              f"(orientation + offset regularization)...")

    all_results = []
    sign_options = [+1, -1]
    all_combos = list(itertools.product(sign_options, repeat=5))

    for signs in all_combos:
        signs_arr = np.array(signs, dtype=float)

        def _make_offsets_residuals(signs_local, T_cam_tcp, offsets_prior):
            def residuals(x):
                offsets_raw = x[:5]
                board_poses = _compute_board_poses(
                    captures, signs_local, offsets_raw, T_cam_tcp, solver)
                consistency = _build_consistency_residuals(
                    board_poses, orientation_weight)

                # Offset regularization: penalize deviation from prior
                offset_dev = (offsets_raw - offsets_prior) * offset_reg_weight
                return np.concatenate([consistency, offset_dev])
            return residuals

        residuals_fn = _make_offsets_residuals(
            signs_arr.copy(), T_cam_tcp_fixed, current_offsets_raw)
        lb = np.zeros(5)
        ub = np.full(5, 4095.0)
        x0 = current_offsets_raw.copy()

        try:
            result = least_squares(residuals_fn, x0, method='trf',
                                   bounds=(lb, ub), max_nfev=3000)
            # Compute position-only error for ranking (comparable metric)
            board_poses = _compute_board_poses(
                captures, signs_arr, result.x[:5], T_cam_tcp_fixed, solver)
            _restore_solver()
            valid_origins = [T[:3, 3] for T in board_poses if T is not None]
            if valid_origins:
                mean_o = np.mean(valid_origins, axis=0)
                errs_mm = np.array([np.linalg.norm(o - mean_o)
                                    for o in valid_origins]) * 1000
                mean_err = float(np.mean(errs_mm))
            else:
                mean_err = 9999.0
                errs_mm = np.full(n, 9999.0)

            # Also compute orientation spread for diagnostics
            valid_rots = [T[:3, :3] for T in board_poses if T is not None]
            if len(valid_rots) >= 2:
                R_ref = valid_rots[0]
                ori_errs_deg = [np.degrees(_rotation_error(R_ref, R))
                                for R in valid_rots[1:]]
                ori_spread_deg = float(np.mean(ori_errs_deg))
            else:
                ori_spread_deg = 999.0
        except Exception:
            mean_err = 9999.0
            errs_mm = np.full(n, 9999.0)
            ori_spread_deg = 999.0
            result = None

        _restore_solver()

        signs_str = ''.join('+' if s > 0 else '-' for s in signs)
        all_results.append({
            'signs': signs_arr.copy(),
            'signs_str': signs_str,
            'mean_err_mm': mean_err,
            'ori_spread_deg': ori_spread_deg,
            'per_point_err': errs_mm.copy() if isinstance(errs_mm, np.ndarray) else np.full(n, errs_mm),
            'opt_result': result,
        })

    # ---------------------------------------------------------------
    # Phase 3: Refine best with joint T_cam_tcp + offsets optimization
    #          All constraints active: orientation, offset reg, T_cam prior
    # ---------------------------------------------------------------
    all_results.sort(key=lambda r: r['mean_err_mm'])
    best_phase2 = all_results[0]

    if verbose:
        print(f"\n  Phase 2 top 5:")
        for i, r in enumerate(all_results[:5]):
            marker = ' <-- BEST' if i == 0 else ''
            print(f"    #{i+1}: signs={r['signs_str']}  "
                  f"pos_err={r['mean_err_mm']:.1f}mm  "
                  f"ori_spread={r['ori_spread_deg']:.2f}°{marker}")

    # Final refinement: jointly optimize offsets + T_cam_tcp for best sign
    best_signs = best_phase2['signs'].copy()

    def _make_joint_residuals(signs_local, offsets_prior):
        def residuals(x):
            offsets_raw = x[:5]
            rvec_ct = x[5:8]
            tvec_ct = x[8:11]
            T_cam_tcp = _rvec_tvec_to_T(rvec_ct, tvec_ct)

            board_poses = _compute_board_poses(
                captures, signs_local, offsets_raw, T_cam_tcp, solver)
            consistency = _build_consistency_residuals(
                board_poses, orientation_weight)

            # Offset regularization
            offset_dev = (offsets_raw - offsets_prior) * offset_reg_weight

            # T_cam_tcp translation prior
            dist = np.linalg.norm(tvec_ct)
            dist_dev = max(0.0, abs(dist - TCAM_EXPECTED_DISTANCE_M)
                           - TCAM_DISTANCE_TOLERANCE_M)
            tcam_prior = np.array([dist_dev * tcam_prior_weight])

            return np.concatenate([consistency, offset_dev, tcam_prior])
        return residuals

    offsets_init = (best_phase2['opt_result'].x[:5]
                    if best_phase2['opt_result'] is not None
                    else current_offsets_raw.copy())
    x0_refine = np.concatenate([offsets_init, rvec_fixed, tvec_fixed])
    lb_refine = np.concatenate([np.zeros(5), np.full(6, -np.inf)])
    ub_refine = np.concatenate([np.full(5, 4095.0), np.full(6, np.inf)])
    res_fn = _make_joint_residuals(best_signs.copy(), current_offsets_raw)
    try:
        refined = least_squares(res_fn, x0_refine, method='trf',
                                bounds=(lb_refine, ub_refine), max_nfev=5000)
        _restore_solver()

        # Compute position error from refined result
        T_cam_refined = _rvec_tvec_to_T(refined.x[5:8], refined.x[8:11])
        board_poses = _compute_board_poses(
            captures, best_signs, refined.x[:5], T_cam_refined, solver)
        _restore_solver()
        valid_origins = [T[:3, 3] for T in board_poses if T is not None]
        if valid_origins:
            mean_o = np.mean(valid_origins, axis=0)
            errs_mm = np.array([np.linalg.norm(o - mean_o)
                                for o in valid_origins]) * 1000
            refined_err = float(np.mean(errs_mm))
        else:
            refined_err = 9999.0
            errs_mm = np.full(n, 9999.0)

        if verbose:
            print(f"\n  Phase 3 refinement: {refined_err:.1f}mm "
                  f"(was {best_phase2['mean_err_mm']:.1f}mm)")
        # Update best result with refined values
        best_phase2['mean_err_mm'] = refined_err
        best_phase2['per_point_err'] = errs_mm.copy()
        best_phase2['opt_result'] = refined
    except Exception:
        _restore_solver()
        if verbose:
            print(f"  Phase 3 refinement failed, using phase 2 result")

    # Restore solver
    _restore_solver()

    all_results.sort(key=lambda r: r['mean_err_mm'])

    best = all_results[0]
    T_cam_tcp = T_cam_tcp_fixed.copy()
    offsets_opt = current_offsets_raw.copy()
    if best['opt_result'] is not None:
        x_opt = best['opt_result'].x
        offsets_opt = x_opt[:5]
        # If the result has 11 params (phase 3 refinement), extract T_cam_tcp
        if len(x_opt) >= 11:
            T_cam_tcp = _rvec_tvec_to_T(x_opt[5:8], x_opt[8:11])

    # Compute T_board_in_base from best result (average across captures)
    T_board_bases = []
    board_poses_final = _compute_board_poses(
        captures, best['signs'], offsets_opt, T_cam_tcp, solver)
    _restore_solver()
    T_board_bases = [T for T in board_poses_final if T is not None]

    # Compute final orientation spread for diagnostics
    if len(T_board_bases) >= 2:
        R_ref = T_board_bases[0][:3, :3]
        final_ori_errs = [np.degrees(_rotation_error(R_ref, T[:3, :3]))
                          for T in T_board_bases[1:]]
        final_ori_spread = float(np.mean(final_ori_errs))
    else:
        final_ori_spread = 999.0

    # Ambiguity check — use combined position + orientation metric.
    # With orientation constraints, ambiguous sign combos show larger
    # orientation spread even if position error is similar.
    threshold = max(best['mean_err_mm'] * 1.10, best['mean_err_mm'] + 1.5)
    near_best = [r for r in all_results if r['mean_err_mm'] <= threshold]
    ambiguous_joints = set()
    for r in near_best[1:]:
        for j in range(5):
            if r['signs'][j] != best['signs'][j]:
                ambiguous_joints.add(j)

    # Secondary ambiguity check using orientation spread
    # If a competitor has much worse orientation spread, it's not truly
    # ambiguous even if position error is close
    if ambiguous_joints and len(near_best) > 1:
        resolved = set()
        for r in near_best[1:]:
            # If orientation spread is >2x worse, this is not a real competitor
            if (r['ori_spread_deg'] > 2.0 * best.get('ori_spread_deg', 999)
                    and best.get('ori_spread_deg', 999) < 5.0):
                for j in range(5):
                    if r['signs'][j] != best['signs'][j]:
                        resolved.add(j)
        ambiguous_joints -= resolved
        if resolved and verbose:
            names = [MOTOR_NAMES[j] for j in sorted(resolved)]
            print(f"\n  Orientation constraint resolved ambiguity for: "
                  f"{', '.join(names)}")

    if verbose:
        print(f"\n  === Servo Direction Auto-Calibration Results ===")
        print(f"  Tested all {len(all_combos)} sign combinations "
              f"with {n} captures")
        print(f"  Constraints: orientation={orientation_weight}, "
              f"offset_reg={offset_reg_weight}, "
              f"tcam_prior={tcam_prior_weight}")
        print(f"\n  Top 5 results:")
        for i, r in enumerate(all_results[:5]):
            marker = ' <-- BEST' if i == 0 else ''
            print(f"    #{i+1}: signs={r['signs_str']}  "
                  f"pos_err={r['mean_err_mm']:.1f}mm  "
                  f"ori={r['ori_spread_deg']:.2f}°{marker}")

        if not ambiguous_joints:
            print(f"\n  CLEAR winner — all joints unambiguously determined")
        elif len(ambiguous_joints) == 1:
            j = list(ambiguous_joints)[0]
            print(f"\n  NOTE: Joint '{MOTOR_NAMES[j]}' sign is ambiguous.")
        else:
            names = [MOTOR_NAMES[j] for j in sorted(ambiguous_joints)]
            print(f"\n  WARNING: {len(ambiguous_joints)} joints ambiguous: "
                  f"{', '.join(names)}")

        print(f"\n  Best signs:     {best['signs_str']}")
        print(f"  Offsets:        {offsets_opt.astype(int).tolist()}")
        print(f"  Mean pos error: {best['mean_err_mm']:.2f}mm")
        print(f"  Ori spread:     {final_ori_spread:.2f}°")

        # Quality warning: high orientation spread suggests unreliable solution
        # (typically caused by too few captures or insufficient pose diversity)
        if final_ori_spread > 20.0:
            print(f"\n  WARNING: Orientation spread {final_ori_spread:.1f}° > 20° "
                  f"— result may be unreliable.")
            print(f"           Collect more captures with greater pose diversity,")
            print(f"           especially varying wrist angles.")

        # T_cam_tcp diagnostics
        tcam_dist_mm = np.linalg.norm(T_cam_tcp[:3, 3]) * 1000
        print(f"  T_cam_tcp dist: {tcam_dist_mm:.1f}mm from TCP")

        try:
            from kinematics.arm101_ik_solver import JOINT_SIGNS
            print(f"\n  Per-joint comparison:")
            print(f"    {'Joint':<16} {'Current':>8} {'Found':>8} "
                  f"{'Confidence':>12}")
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
        except ImportError:
            pass

    return {
        'signs': best['signs'].copy(),
        'signs_str': best['signs_str'],
        'offsets_raw': offsets_opt.copy(),
        'T_cam_in_tcp': T_cam_tcp,
        'mean_err_mm': best['mean_err_mm'],
        'ori_spread_deg': final_ori_spread,
        'per_point_err': best['per_point_err'],
        'ambiguous_joints': ambiguous_joints,
        'all_results': all_results,
        # Quality flag: True when orientation spread is low (< 20°) and
        # there are enough captures for a reliable result.
        'solution_reliable': (final_ori_spread < 20.0 and n >= MIN_CAPTURES),
    }


def _rvec_tvec_to_T(rvec, tvec):
    """Convert Rodrigues rotation vector + translation to 4x4 matrix."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).flatten()
    return T


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
