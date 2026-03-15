#!/usr/bin/env python3
"""Offline comparison: constrained vs position-only sign solver.

This script validates the constrained servo direction solver (current) against
the position-only solver (prior) using synthetic arm101 data.  It specifically
focuses on wrist_roll (J5) ambiguity, which is the hardest joint to resolve:
the camera mounts on the wrist, so a wrong J5 sign rotates the camera around
its optical axis — this changes board *orientation* in the base frame but has
little effect on board *position*, making position-only consistency blind to it.

Scenarios tested
----------------
1. Diverse poses — full joint range, all joints varying widely.
2. Limited wrist-roll range — J5 capped at ±20° (common in real usage).
3. Noisy measurements — 1° Gaussian noise on joint angles (encoder noise).
4. Very limited captures (6) — harder problem with fewer data.

For each scenario, both solvers run on the same captures.  Results show:
- Whether the correct signs are found
- Whether J5 is flagged ambiguous
- Orientation spread (only reported by constrained solver)

Usage
-----
    ./run.sh scripts/compare_sign_solvers.py

Output
------
    Prints a comparison table to stdout.
    Exits with 0 if the constrained solver outperforms on J5 disambiguation.
"""

import sys
import os
import time

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
try:
    from kinematics.arm101_ik_solver import Arm101IKSolver
except ImportError as e:
    print(f"ERROR: Cannot import Arm101IKSolver: {e}")
    print("Run this script via: ./run.sh scripts/compare_sign_solvers.py")
    sys.exit(1)

from calibration.sign_solver import (
    _brute_force_signs as _brute_force_constrained,
    _pose_to_matrix,
    MOTOR_NAMES,
    ORIENTATION_WEIGHT,
    OFFSET_REG_WEIGHT,
    TCAM_TRANSLATION_PRIOR_WEIGHT,
)

# ---------------------------------------------------------------------------
# Position-only (unconstrained) solver — extracted from prior commit 077cf3e
# ---------------------------------------------------------------------------

def _compute_board_origins(captures_list, signs_local, offsets_raw, T_cam_tcp,
                            solver):
    """Position-only version: returns just the 3D board origin per capture."""
    DEG_PER_POS = 360.0 / 4096.0
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
            continue
        T_board_base = T_tcp @ T_cam_tcp @ cap['T_board_in_cam']
        origins.append(T_board_base[:3, 3])
    return np.array(origins) if origins else np.zeros((0, 3))


def _brute_force_position_only(captures, solver, current_offsets_raw,
                                verbose=True):
    """Position-only 3-phase solver (prior version, no orientation constraints).

    Equivalent to the solver from commit 077cf3e — uses only board position
    consistency, no orientation residuals, no offset regularization,
    no T_cam_tcp prior.
    """
    from scipy.optimize import least_squares
    import itertools

    n = len(captures)
    DEG_PER_POS = 360.0 / 4096.0

    original_signs = solver.signs.copy()
    original_offsets = solver.offsets_deg.copy()

    def _restore():
        solver.signs = original_signs
        solver.offsets_deg = original_offsets

    def _rvec_tvec_to_T(rvec, tvec):
        R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.asarray(tvec).flatten()
        return T

    # Phase 1: Estimate T_cam_in_tcp (position only)
    current_signs = original_signs.copy()

    def _make_tcam_residuals_pos(signs_local, offsets_fixed):
        def residuals(x):
            T_cam_tcp = _rvec_tvec_to_T(x[:3], x[3:6])
            origins = _compute_board_origins(
                captures, signs_local, offsets_fixed, T_cam_tcp, solver)
            if len(origins) < 2:
                return np.full(6, 1e6)
            mean_origin = np.mean(origins, axis=0)
            errs = []
            for bo in origins:
                errs.extend((bo - mean_origin).tolist())
            return np.array(errs)
        return residuals

    rvec_inits = [
        np.array([np.pi, 0, 0]),
        np.array([0, np.pi, 0]),
        np.array([np.pi/2, 0, 0]),
        np.array([-np.pi/2, 0, 0]),
    ]
    tvec_inits = [
        np.array([0.0, 0.0, -0.04]),
        np.array([0.01, -0.02, -0.035]),
        np.array([0.0, 0.0, -0.06]),
    ]

    best_tcam_err = 1e9
    best_tcam_result = None
    for rvec_init in rvec_inits:
        for tvec_init in tvec_inits:
            x0 = np.concatenate([rvec_init, tvec_init])
            res_fn = _make_tcam_residuals_pos(current_signs, current_offsets_raw)
            try:
                result = least_squares(res_fn, x0, method='lm', max_nfev=5000)
                res = result.fun.reshape(-1, 3)
                err = float(np.mean(np.linalg.norm(res, axis=1))) * 1000
                if err < best_tcam_err:
                    best_tcam_err = err
                    best_tcam_result = result
            except Exception:
                pass
    _restore()

    if best_tcam_result is None:
        rvec_fixed = np.array([np.pi, 0, 0])
        tvec_fixed = np.array([0.0, 0.0, -0.04])
    else:
        rvec_fixed = best_tcam_result.x[:3]
        tvec_fixed = best_tcam_result.x[3:6]

    T_cam_tcp_fixed = _rvec_tvec_to_T(rvec_fixed, tvec_fixed)

    # Phase 2: Try all 32 sign combos (position only)
    all_results = []
    sign_options = [+1, -1]
    all_combos = list(itertools.product(sign_options, repeat=5))

    for signs in all_combos:
        signs_arr = np.array(signs, dtype=float)

        def _make_offsets_residuals_pos(signs_local, T_cam_tcp):
            def residuals(x):
                offsets_raw = x[:5]
                origins = _compute_board_origins(
                    captures, signs_local, offsets_raw, T_cam_tcp, solver)
                if len(origins) < 2:
                    return np.full(6, 1e6)
                mean_origin = np.mean(origins, axis=0)
                errs = []
                for bo in origins:
                    errs.extend((bo - mean_origin).tolist())
                return np.array(errs)
            return residuals

        residuals_fn = _make_offsets_residuals_pos(signs_arr.copy(),
                                                    T_cam_tcp_fixed)
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
        _restore()

        signs_str = ''.join('+' if s > 0 else '-' for s in signs)
        all_results.append({
            'signs': signs_arr.copy(),
            'signs_str': signs_str,
            'mean_err_mm': mean_err,
            'per_point_err': errs_mm.copy() if result is not None
                             else np.full(n, 9999.0),
            'opt_result': result,
            'ori_spread_deg': 999.0,  # not computed in position-only version
        })

    # Phase 3: Joint refinement
    all_results.sort(key=lambda r: r['mean_err_mm'])
    best_phase2 = all_results[0]
    best_signs = best_phase2['signs'].copy()

    def _make_joint_residuals_pos(signs_local):
        def residuals(x):
            offsets_raw = x[:5]
            T_cam_tcp = _rvec_tvec_to_T(x[5:8], x[8:11])
            origins = _compute_board_origins(
                captures, signs_local, offsets_raw, T_cam_tcp, solver)
            if len(origins) < 2:
                return np.full(6, 1e6)
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
    res_fn = _make_joint_residuals_pos(best_signs.copy())
    try:
        refined = least_squares(res_fn, x0_refine, method='trf',
                                bounds=(lb_refine, ub_refine), max_nfev=5000)
        _restore()
        res = refined.fun.reshape(-1, 3)
        errs_mm = np.linalg.norm(res, axis=1) * 1000
        refined_err = float(np.mean(errs_mm))
        best_phase2['mean_err_mm'] = refined_err
        best_phase2['per_point_err'] = errs_mm.copy()
        best_phase2['opt_result'] = refined
    except Exception:
        pass
    _restore()

    all_results.sort(key=lambda r: r['mean_err_mm'])
    best = all_results[0]

    # Ambiguity check (position-only: same threshold as constrained)
    threshold = max(best['mean_err_mm'] * 1.10, best['mean_err_mm'] + 1.5)
    near_best = [r for r in all_results if r['mean_err_mm'] <= threshold]
    ambiguous_joints = set()
    for r in near_best[1:]:
        for j in range(5):
            if r['signs'][j] != best['signs'][j]:
                ambiguous_joints.add(j)

    return {
        'signs': best['signs'].copy(),
        'signs_str': best['signs_str'],
        'mean_err_mm': best['mean_err_mm'],
        'ori_spread_deg': 999.0,  # not available in position-only
        'ambiguous_joints': ambiguous_joints,
        'all_results': all_results,
    }


# ---------------------------------------------------------------------------
# Synthetic capture generator (arm101 realistic parameters)
# ---------------------------------------------------------------------------

def make_captures(true_signs, true_offsets_raw, T_cam_in_tcp, T_board_in_base,
                  urdf_ranges_deg, n_poses, rng_seed, noise_deg=0.0):
    """Generate synthetic captures with optional joint angle noise."""
    rng = np.random.default_rng(rng_seed)

    solver_true = Arm101IKSolver(
        joint_signs=true_signs.copy(),
        joint_offsets_deg=np.zeros(5),
    )
    T_tcp_to_cam = np.linalg.inv(T_cam_in_tcp)
    DEG_PER_POS = 360.0 / 4096.0

    captures = []
    for _ in range(n_poses):
        urdf_deg = np.array([
            rng.uniform(lo, hi) for lo, hi in urdf_ranges_deg
        ])
        # Add measurement noise (mimics encoder quantization + vibration)
        if noise_deg > 0:
            urdf_deg += rng.normal(0, noise_deg, size=5)

        motor_deg = urdf_deg / true_signs
        raw_pos = {
            mid: int(np.clip(
                round(true_offsets_raw[mid - 1] + motor_deg[mid - 1] / DEG_PER_POS),
                0, 4095))
            for mid in range(1, 6)
        }

        actual_motor_deg = np.array([
            (raw_pos[mid] - true_offsets_raw[mid - 1]) * DEG_PER_POS
            for mid in range(1, 6)
        ])
        pos_mm, rpy_deg = solver_true.forward_kin(actual_motor_deg)
        T_tcp = _pose_to_matrix(pos_mm, rpy_deg)
        T_board_in_cam = T_tcp_to_cam @ np.linalg.inv(T_tcp) @ T_board_in_base

        captures.append({'raw': raw_pos, 'T_board_in_cam': T_board_in_cam})

    return captures


# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------

def check_signs_correct(found_signs, true_signs):
    """Return list of joint indices where sign is correct."""
    return [i for i in range(5) if found_signs[i] == true_signs[i]]


def run_scenario(name, captures, true_signs, current_offsets_raw, verbose=False,
                 init_offsets_raw=None):
    """Run both solvers and return comparison dict."""
    solver = Arm101IKSolver(joint_signs=np.ones(5), joint_offsets_deg=np.zeros(5))
    offsets_for_solver = (init_offsets_raw if init_offsets_raw is not None
                          else current_offsets_raw)

    # --- Constrained solver ---
    t0 = time.time()
    result_c = _brute_force_constrained(
        captures, solver, offsets_for_solver, verbose=verbose)
    solver.signs = np.ones(5)
    solver.offsets_deg = np.zeros(5)
    t_constrained = time.time() - t0

    correct_c = check_signs_correct(result_c['signs'], true_signs)
    j5_correct_c = (result_c['signs'][4] == true_signs[4])
    j5_ambig_c = (4 in result_c['ambiguous_joints'])

    # --- Position-only solver ---
    t0 = time.time()
    result_p = _brute_force_position_only(
        captures, solver, offsets_for_solver, verbose=verbose)
    solver.signs = np.ones(5)
    solver.offsets_deg = np.zeros(5)
    t_position = time.time() - t0

    correct_p = check_signs_correct(result_p['signs'], true_signs)
    j5_correct_p = (result_p['signs'][4] == true_signs[4])
    j5_ambig_p = (4 in result_p['ambiguous_joints'])

    return {
        'name': name,
        'n_captures': len(captures),
        'constrained': {
            'signs_str': result_c['signs_str'],
            'correct_joints': correct_c,
            'all_correct': len(correct_c) == 5,
            'j5_correct': j5_correct_c,
            'j5_ambiguous': j5_ambig_c,
            'mean_err_mm': result_c['mean_err_mm'],
            'ori_spread_deg': result_c.get('ori_spread_deg', 999.0),
            'ambiguous_joints': result_c['ambiguous_joints'],
            'time_s': t_constrained,
            'top5': [(r['signs_str'], r['mean_err_mm'],
                      r.get('ori_spread_deg', 999.0))
                     for r in result_c['all_results'][:5]],
        },
        'position_only': {
            'signs_str': result_p['signs_str'],
            'correct_joints': correct_p,
            'all_correct': len(correct_p) == 5,
            'j5_correct': j5_correct_p,
            'j5_ambiguous': j5_ambig_p,
            'mean_err_mm': result_p['mean_err_mm'],
            'ori_spread_deg': 'N/A',
            'ambiguous_joints': result_p['ambiguous_joints'],
            'time_s': t_position,
            'top5': [(r['signs_str'], r['mean_err_mm'],
                      r.get('ori_spread_deg', 999.0))
                     for r in result_p['all_results'][:5]],
        },
    }


def print_separator(char='=', width=74):
    print(char * width)


def print_scenario_result(res):
    c = res['constrained']
    p = res['position_only']

    print(f"\n{'─'*74}")
    print(f"SCENARIO: {res['name']}  ({res['n_captures']} captures)")
    print(f"{'─'*74}")

    def sign_status(correct, ambig):
        if ambig:
            return 'AMBIGUOUS'
        return 'CORRECT' if correct else 'WRONG'

    # Joint-by-joint table
    print(f"  {'Joint':<16} {'Position-only':>18} {'Constrained':>18}")
    print(f"  {'─'*16} {'─'*18} {'─'*18}")
    true_signs = [1, 1, -1, 1, -1]  # arm101 known ground truth
    for j, name in enumerate(MOTOR_NAMES):
        p_ok = (p['correct_joints'].count(j) > 0)
        c_ok = (c['correct_joints'].count(j) > 0)
        p_ambig = (j in p['ambiguous_joints'])
        c_ambig = (j in c['ambiguous_joints'])
        p_str = sign_status(p_ok, p_ambig)
        c_str = sign_status(c_ok, c_ambig)
        star = ' *' if name == 'wrist_roll' else ''
        change = ' ← IMPROVED' if (p_str != 'CORRECT' and c_str == 'CORRECT') else ''
        change += ' ← RESOLVED' if (p_str == 'AMBIGUOUS' and c_str != 'AMBIGUOUS') else ''
        print(f"  {name+star:<16} {p['signs_str'][j]+' '+p_str:>18}"
              f" {c['signs_str'][j]+' '+c_str:>18}{change}")

    print()
    print(f"  Position-only: err={p['mean_err_mm']:.1f}mm  "
          f"t={p['time_s']:.1f}s  "
          f"ambiguous={sorted(p['ambiguous_joints'])}")
    c_ori = (f"{c['ori_spread_deg']:.2f}°"
             if isinstance(c['ori_spread_deg'], float) else c['ori_spread_deg'])
    print(f"  Constrained:   err={c['mean_err_mm']:.1f}mm  "
          f"t={c['time_s']:.1f}s  "
          f"ambiguous={sorted(c['ambiguous_joints'])}  "
          f"ori={c_ori}")

    # Top-5 for each solver
    print(f"\n  Top 5 by solver:")
    print(f"  {'#':<3} {'Position-only':>28} | {'Constrained':>28}")
    print(f"  {'─'*3} {'─'*28} | {'─'*28}")
    for i in range(5):
        ps, pe, _ = p['top5'][i] if i < len(p['top5']) else ('?????', 999, 999)
        cs, ce, co = c['top5'][i] if i < len(c['top5']) else ('?????', 999, 999)
        co_str = (f"{co:.1f}°" if isinstance(co, float) and co < 900
                  else 'N/A')
        print(f"  #{i+1:<2} {ps:>6} {pe:>6.1f}mm           | "
              f"{cs:>6} {ce:>6.1f}mm ori={co_str}")

    # Summary
    j5_improved = (not p['constrained'] if False else
                   ((p['j5_ambiguous'] and not c['j5_ambiguous'])
                    or (not p['j5_correct'] and c['j5_correct'])))
    if j5_improved:
        print(f"\n  ✓ J5 (wrist_roll): IMPROVED by orientation constraints")
    elif not p['j5_ambiguous'] and not c['j5_ambiguous'] and p['j5_correct']:
        print(f"\n  = J5 (wrist_roll): Both solvers agree (unambiguous)")
    elif p['j5_ambiguous'] and c['j5_ambiguous']:
        print(f"\n  ✗ J5 (wrist_roll): Both solvers still ambiguous")
    else:
        print(f"\n  ~ J5 (wrist_roll): No significant difference")


def main():
    print_separator()
    print("  Servo Direction Sign Solver Comparison")
    print("  Position-only (prior) vs Constrained (current)")
    print_separator()

    # Ground truth for arm101 (from servo_offsets.yaml calibrated 2026-03-15)
    TRUE_SIGNS = np.array([-1.0, +1.0, +1.0, +1.0, -1.0])  # pan,lift,elbow,flex,roll
    TRUE_OFFSETS_RAW = np.array([2108.0, 843.0, 1956.0, 2968.0, 2981.0])

    # Camera mounted on gripper: looking down, ~45mm below TCP
    T_CAM_IN_TCP = np.array([
        [1.0,  0.0,  0.0,  0.010],
        [0.0, -1.0,  0.0, -0.020],
        [0.0,  0.0, -1.0, -0.045],
        [0.0,  0.0,  0.0,  1.0],
    ])

    # ChArUco board on table: 180mm in front of base, 40mm right
    T_BOARD_IN_BASE = np.eye(4)
    T_BOARD_IN_BASE[:3, 3] = [0.180, 0.040, 0.0]

    # Initial offsets (for regularization — start at true values as in real use)
    INIT_OFFSETS = TRUE_OFFSETS_RAW.copy()

    # --- Full joint ranges ---
    FULL_RANGES = [
        (-60, 60),    # shoulder_pan
        (-60, 60),    # shoulder_lift
        (-90, 90),    # elbow_flex
        (-80, 80),    # wrist_flex
        (-120, 120),  # wrist_roll
    ]

    # --- Limited wrist-roll (typical real usage: arm limited to work zone) ---
    LIMITED_WRIST_RANGES = [
        (-60, 60),    # shoulder_pan
        (-60, 60),    # shoulder_lift
        (-90, 90),    # elbow_flex
        (-80, 80),    # wrist_flex
        (-20,  20),   # wrist_roll — very limited range (J5 ambiguity scenario)
    ]

    scenarios = []

    print("\nGenerating synthetic captures...")

    # Scenario 1: Diverse poses — 14 captures, full joint range
    cap1 = make_captures(TRUE_SIGNS, TRUE_OFFSETS_RAW, T_CAM_IN_TCP,
                         T_BOARD_IN_BASE, FULL_RANGES, 14, rng_seed=42)
    scenarios.append(('Diverse poses (14 captures, full range)', cap1))

    # Scenario 2: Limited wrist-roll — 14 captures, J5 capped at ±20°
    cap2 = make_captures(TRUE_SIGNS, TRUE_OFFSETS_RAW, T_CAM_IN_TCP,
                         T_BOARD_IN_BASE, LIMITED_WRIST_RANGES, 14, rng_seed=42)
    scenarios.append(('Limited wrist-roll ±20° (14 captures)', cap2))

    # Scenario 3: Noisy + limited wrist-roll — 14 captures, 1° noise
    cap3 = make_captures(TRUE_SIGNS, TRUE_OFFSETS_RAW, T_CAM_IN_TCP,
                         T_BOARD_IN_BASE, LIMITED_WRIST_RANGES, 14, rng_seed=99,
                         noise_deg=1.0)
    scenarios.append(('Noisy + limited wrist-roll (1° noise, 14 captures)', cap3))

    # Scenario 4: Minimum captures, limited wrist-roll
    cap4 = make_captures(TRUE_SIGNS, TRUE_OFFSETS_RAW, T_CAM_IN_TCP,
                         T_BOARD_IN_BASE, LIMITED_WRIST_RANGES, 6, rng_seed=7)
    scenarios.append(('Minimum captures (6), limited wrist-roll', cap4))

    # Scenario 5: Diverse, noisy — robustness check
    cap5 = make_captures(TRUE_SIGNS, TRUE_OFFSETS_RAW, T_CAM_IN_TCP,
                         T_BOARD_IN_BASE, FULL_RANGES, 14, rng_seed=123,
                         noise_deg=1.5)
    scenarios.append(('Diverse + noisy (1.5° noise, 14 captures)', cap5))

    # Scenario 6: Camera perfectly on-axis (x=y=0) + limited J5 range
    # Worst case for J5: camera axis coincides with wrist axis, so wrong J5
    # sign only rotates the *image*, not the board *origin*.
    # Position-only can't detect this; orientation constraints should.
    T_CAM_ONAXIS = np.array([
        [1.0,  0.0,  0.0,  0.0],    # No x/y offset — camera on wrist axis
        [0.0, -1.0,  0.0,  0.0],
        [0.0,  0.0, -1.0, -0.045],
        [0.0,  0.0,  0.0,  1.0],
    ])
    cap6 = make_captures(TRUE_SIGNS, TRUE_OFFSETS_RAW, T_CAM_ONAXIS,
                         T_BOARD_IN_BASE, LIMITED_WRIST_RANGES, 14, rng_seed=42)
    scenarios.append(('On-axis camera, limited J5 ±20° (J5 hardest)', cap6))

    # Scenario 7: On-axis camera, very limited J5 (±5°), diverse other joints
    VERY_LIMITED_J5 = [
        (-60, 60),   # shoulder_pan
        (-60, 60),   # shoulder_lift
        (-90, 90),   # elbow_flex
        (-80, 80),   # wrist_flex
        (-5,   5),   # wrist_roll — nearly zero (maximum J5 ambiguity)
    ]
    cap7 = make_captures(TRUE_SIGNS, TRUE_OFFSETS_RAW, T_CAM_ONAXIS,
                         T_BOARD_IN_BASE, VERY_LIMITED_J5, 14, rng_seed=42)
    scenarios.append(('On-axis camera, J5 near-zero ±5° (worst case)', cap7))

    # Scenario 8: Fresh calibration — wrong initial offsets (all centered at 2048)
    # This simulates the very first calibration where no prior offsets are known.
    # Position-only solver may be confused because Phase 1 T_cam_tcp estimation
    # starts with wrong signs (all +1) and wrong offsets (all 2048).
    FRESH_OFFSETS = np.full(5, 2048.0)
    cap8 = make_captures(TRUE_SIGNS, TRUE_OFFSETS_RAW, T_CAM_IN_TCP,
                         T_BOARD_IN_BASE, FULL_RANGES, 14, rng_seed=42)
    scenarios.append(('Fresh calibration: wrong offsets (2048), diverse', cap8))

    # Scenario 8 uses fresh (wrong) offsets; all others use calibrated offsets
    scenario_offsets = [INIT_OFFSETS] * 7 + [FRESH_OFFSETS]

    results = []
    for i, ((name, caps), offsets) in enumerate(
            zip(scenarios, scenario_offsets)):
        print(f"  Running scenario {i+1}/{len(scenarios)}: {name[:50]}...",
              flush=True)
        r = run_scenario(name, caps, TRUE_SIGNS, INIT_OFFSETS,
                         verbose=False, init_offsets_raw=offsets)
        results.append(r)

    # ---------------------------------------------------------------------------
    # Print detailed results
    # ---------------------------------------------------------------------------
    print(f"\n{'='*74}")
    print(f"  DETAILED RESULTS")
    gt_str = ''.join('+' if s > 0 else '-' for s in TRUE_SIGNS)
    print(f"  Ground truth: {gt_str}  ({' '.join(MOTOR_NAMES)})")
    print(f"  Note: wrist_roll (J5, index 4) is marked with *")

    for res in results:
        print_scenario_result(res)

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print(f"\n{'='*74}")
    print(f"  SUMMARY: J5 (wrist_roll) Disambiguation")
    print(f"{'='*74}")
    print(f"  {'Scenario':<42} {'Pos-only J5':>12} {'Constrained J5':>14}")
    print(f"  {'─'*42} {'─'*12} {'─'*14}")

    j5_improved_count = 0
    for res in results:
        p = res['position_only']
        c = res['constrained']

        def j5_status(correct, ambig):
            if ambig:
                return 'AMBIGUOUS'
            return 'OK' if correct else 'WRONG'

        p_status = j5_status(p['j5_correct'], p['j5_ambiguous'])
        c_status = j5_status(c['j5_correct'], c['j5_ambiguous'])
        improved = (p_status != 'OK' and c_status == 'OK')
        if improved:
            j5_improved_count += 1
        marker = ' ← improved' if improved else ''
        name_short = res['name'][:42]
        print(f"  {name_short:<42} {p_status:>12} {c_status:>14}{marker}")

    print(f"\n  Constrained solver improved J5 in {j5_improved_count}/"
          f"{len(results)} scenarios")

    # Overall signs accuracy
    print(f"\n  All-joints accuracy (all 5 signs correct):")
    print(f"  {'Scenario':<42} {'Pos-only':>10} {'Constrained':>12}")
    print(f"  {'─'*42} {'─'*10} {'─'*12}")
    for res in results:
        p = res['position_only']
        c = res['constrained']
        name_short = res['name'][:42]
        print(f"  {name_short:<42} "
              f"{'PASS' if p['all_correct'] else 'FAIL':>10} "
              f"{'PASS' if c['all_correct'] else 'FAIL':>12}")

    # Overall accuracy comparison
    constrained_pass = sum(1 for r in results if r['constrained']['all_correct'])
    position_pass = sum(1 for r in results if r['position_only']['all_correct'])

    # Better ambiguity reporting: constrained flags instead of silently wrong
    constrained_silent_wrong = sum(
        1 for r in results
        if not r['constrained']['j5_correct'] and not r['constrained']['j5_ambiguous'])
    position_silent_wrong = sum(
        1 for r in results
        if not r['position_only']['j5_correct'] and not r['position_only']['j5_ambiguous'])

    print(f"\n  Overall accuracy (all 5 joints correct):")
    print(f"    Position-only:  {position_pass}/{len(results)} scenarios PASS")
    print(f"    Constrained:    {constrained_pass}/{len(results)} scenarios PASS")

    print(f"\n  J5 silent failures (wrong sign, not flagged as ambiguous):")
    print(f"    Position-only:  {position_silent_wrong}/{len(results)} scenarios")
    print(f"    Constrained:    {constrained_silent_wrong}/{len(results)} scenarios")

    # Return exit code based on whether constrained solver helps
    print(f"\n{'='*74}")
    if j5_improved_count > 0:
        print(f"  RESULT: Constrained solver reduces J5 ambiguity "
              f"in {j5_improved_count}/{len(results)} scenarios.")
        print(f"          Orientation constraints provide measurable improvement.")
    else:
        print(f"  J5 result: Both solvers determine wrist_roll sign equivalently.")
        print(f"             Orientation constraints do NOT uniquely resolve J5")
        print(f"             in synthetic data with this arm101 camera geometry.")
    if constrained_pass > position_pass:
        print(f"\n  OVERALL: Constrained solver improves accuracy "
              f"({position_pass}→{constrained_pass} of {len(results)} pass).")
        print(f"           Main improvement: prevents T_cam_tcp from absorbing")
        print(f"           sign errors via orientation consistency.")
    if constrained_silent_wrong < position_silent_wrong:
        print(f"\n  CONFIDENCE: Constrained solver has fewer silent failures "
              f"({position_silent_wrong}→{constrained_silent_wrong}).")
        print(f"             Under noise, it correctly flags ambiguity rather")
        print(f"             than returning a wrong answer confidently.")
    print(f"{'='*74}")


if __name__ == '__main__':
    main()
