"""Unit tests for the brute-force servo sign solver (ChArUco-based).

Tests that _brute_force_signs() correctly identifies joint signs (directions)
from synthetic data where the ground-truth signs are known.

The solver receives captures containing raw servo positions and ChArUco board
poses in the gripper camera frame.  Since the board is fixed on the table,
the board position in the robot base frame should be consistent across
captures — the solver finds the sign/offset combination that minimises
inconsistency.

No hardware required.  Skips gracefully if pinocchio is not installed.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

try:
    from kinematics.arm101_ik_solver import Arm101IKSolver
    HAS_PINOCCHIO = True
except (ImportError, FileNotFoundError):
    HAS_PINOCCHIO = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from calibration.sign_solver import (
        _brute_force_signs, _pose_to_matrix)
    HAS_SIGN_SOLVER = True
except Exception:
    HAS_SIGN_SOLVER = False

pytestmark = pytest.mark.skipif(
    not (HAS_PINOCCHIO and HAS_CV2 and HAS_SIGN_SOLVER),
    reason="pinocchio, cv2, or servo_direction_calib_view not available",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEG_PER_POS = 360.0 / 4096.0


# ---------------------------------------------------------------------------
# Synthetic-data generator
# ---------------------------------------------------------------------------

def _make_synthetic_captures(true_signs, true_offsets_raw, T_cam_in_tcp,
                              T_board_in_base, n_poses=12, rng_seed=42):
    """Generate synthetic captures for the ChArUco-based solver.

    For each pose:
      1. Sample random URDF joint angles.
      2. Compute raw encoder positions from the true signs + offsets.
      3. Run FK to get T_tcp_in_base.
      4. Compute T_board_in_cam = inv(T_cam_in_tcp) @ inv(T_tcp_in_base) @ T_board_in_base.

    Returns:
        List of capture dicts {raw, T_board_in_cam}.
    """
    rng = np.random.default_rng(rng_seed)

    solver_true = Arm101IKSolver(
        joint_signs=true_signs.copy(),
        joint_offsets_deg=np.zeros(5),
    )

    T_tcp_to_cam = np.linalg.inv(T_cam_in_tcp)

    urdf_ranges_deg = [
        (-60, 60),    # shoulder_pan
        (-60, 60),    # shoulder_lift
        (-90, 90),    # elbow_flex
        (-80, 80),    # wrist_flex
        (-120, 120),  # wrist_roll
    ]

    captures = []
    for _ in range(n_poses):
        urdf_deg = np.array([
            rng.uniform(lo, hi) for lo, hi in urdf_ranges_deg
        ])

        # URDF → motor → raw
        motor_deg = urdf_deg / true_signs
        raw_pos = {
            mid: int(np.clip(
                round(true_offsets_raw[mid - 1] + motor_deg[mid - 1] / DEG_PER_POS),
                0, 4095))
            for mid in range(1, 6)
        }

        # FK with true params
        actual_motor_deg = np.array([
            (raw_pos[mid] - true_offsets_raw[mid - 1]) * DEG_PER_POS
            for mid in range(1, 6)
        ])
        pos_mm, rpy_deg = solver_true.forward_kin(actual_motor_deg)
        T_tcp = _pose_to_matrix(pos_mm, rpy_deg)

        # Board in camera frame
        T_board_in_cam = T_tcp_to_cam @ np.linalg.inv(T_tcp) @ T_board_in_base

        captures.append({
            'raw': raw_pos,
            'T_board_in_cam': T_board_in_cam,
        })

    return captures


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def synthetic_setup():
    """Return all inputs for _brute_force_signs plus ground truth."""
    true_signs = np.array([+1.0, +1.0, -1.0, +1.0, -1.0])
    true_offsets_raw = np.array([2048.0, 2100.0, 1980.0, 2060.0, 2200.0])

    # Camera mounted on gripper: looking down, 35mm below TCP
    T_cam_in_tcp = np.array([
        [1.0,  0.0,  0.0,  0.010],
        [0.0, -1.0,  0.0, -0.020],
        [0.0,  0.0, -1.0, -0.035],
        [0.0,  0.0,  0.0,  1.0],
    ])

    # Board on the table: 200mm in front, 50mm to the right, at table level
    T_board_in_base = np.eye(4)
    T_board_in_base[:3, 3] = [0.200, 0.050, 0.0]

    captures = _make_synthetic_captures(
        true_signs, true_offsets_raw, T_cam_in_tcp, T_board_in_base,
        n_poses=14)

    assert len(captures) >= 6, (
        f"Need at least 6 synthetic captures but only generated {len(captures)}")

    solver = Arm101IKSolver(
        joint_signs=np.ones(5),
        joint_offsets_deg=np.zeros(5),
    )

    return {
        'true_signs': true_signs,
        'true_offsets_raw': true_offsets_raw,
        'T_cam_in_tcp': T_cam_in_tcp,
        'T_board_in_base': T_board_in_base,
        'solver': solver,
        'captures': captures,
        'current_offsets_raw': true_offsets_raw.copy(),
    }


@pytest.fixture(scope='module')
def solver_result(synthetic_setup):
    """Run _brute_force_signs once and cache the result."""
    s = synthetic_setup
    result = _brute_force_signs(
        captures=s['captures'],
        solver=s['solver'],
        current_offsets_raw=s['current_offsets_raw'],
        verbose=False,
    )
    return result, s['true_signs']


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBruteForceSignSolverSynthetic:
    """Verify _brute_force_signs recovers ground-truth signs from synthetic data."""

    def test_returns_dict_with_required_keys(self, solver_result):
        result, _ = solver_result
        required = {'signs', 'signs_str', 'offsets_raw', 'T_cam_in_tcp',
                    'mean_err_mm', 'per_point_err', 'all_results'}
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - result.keys()}")

    def test_all_32_combinations_tried(self, solver_result):
        result, _ = solver_result
        assert len(result['all_results']) == 32

    def test_best_error_is_low(self, solver_result):
        """With noise-free data the best solution should have low consistency error."""
        result, _ = solver_result
        assert result['mean_err_mm'] < 20.0, (
            f"Expected mean error < 20mm on synthetic data, "
            f"got {result['mean_err_mm']:.2f}mm")

    def test_signs_shape(self, solver_result):
        result, _ = solver_result
        assert result['signs'].shape == (5,)

    def test_all_signs_are_plus_or_minus_one(self, solver_result):
        result, _ = solver_result
        for i, s in enumerate(result['signs']):
            assert s in (1.0, -1.0), f"Sign {i} is {s}, expected ±1"

    @pytest.mark.parametrize("joint_idx", [0, 1, 2, 3])
    def test_joint_sign_correctly_identified(self, solver_result, joint_idx):
        """First 4 joints must match truth (wrist_roll may be ambiguous)."""
        result, true_signs = solver_result
        found = result['signs'][joint_idx]
        expected = true_signs[joint_idx]
        assert found == expected, (
            f"Joint {joint_idx}: expected sign {expected:+.0f}, "
            f"got {found:+.0f} (mean_err={result['mean_err_mm']:.2f}mm)")

    def test_offsets_raw_shape(self, solver_result):
        result, _ = solver_result
        assert result['offsets_raw'].shape == (5,)

    def test_offsets_within_servo_range(self, solver_result):
        result, _ = solver_result
        for i, offset in enumerate(result['offsets_raw']):
            assert 0 <= offset <= 4095, (
                f"Offset {i} = {offset:.0f} outside [0, 4095]")

    def test_T_cam_in_tcp_is_4x4(self, solver_result):
        result, _ = solver_result
        T = result['T_cam_in_tcp']
        assert T.shape == (4, 4)

    def test_T_cam_in_tcp_is_valid_SE3(self, solver_result):
        result, _ = solver_result
        T = result['T_cam_in_tcp']
        R = T[:3, :3]
        I_approx = R.T @ R
        np.testing.assert_allclose(
            I_approx, np.eye(3), atol=1e-4,
            err_msg="Camera rotation matrix is not orthonormal")
        assert abs(np.linalg.det(R) - 1.0) < 1e-4

    def test_results_sorted_by_error(self, solver_result):
        result, _ = solver_result
        errors = [r['mean_err_mm'] for r in result['all_results']]
        assert errors == sorted(errors)

    def test_best_combo_is_first(self, solver_result):
        result, _ = solver_result
        best_err = result['all_results'][0]['mean_err_mm']
        assert result['mean_err_mm'] == pytest.approx(best_err)

    def test_signs_str_matches_signs_array(self, solver_result):
        result, _ = solver_result
        signs = result['signs']
        expected_str = ''.join('+' if s > 0 else '-' for s in signs)
        assert result['signs_str'] == expected_str
