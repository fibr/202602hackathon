"""Unit tests for the brute-force servo sign solver.

Tests that _brute_force_signs() correctly identifies joint signs (directions)
from synthetic data where the ground-truth signs are known.

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
    from gui.views.servo_direction_calib_view import _brute_force_signs
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

DEG_PER_POS = 360.0 / 4096.0   # raw encoder ticks → degrees


# ---------------------------------------------------------------------------
# Synthetic-data generator
# ---------------------------------------------------------------------------

def _make_synthetic_data(true_signs, true_offsets_raw, T_cam2base,
                          K, n_poses=12, rng_seed=42):
    """Generate synthetic (raw_positions, pixels) pairs using ground-truth params.

    For each pose:
      1. Sample random URDF joint angles (diverse, within joint limits).
      2. Compute motor angles: motor_deg = urdf_deg / sign.
      3. Compute raw encoder position: raw = offset + motor_deg / DEG_PER_POS.
      4. Run FK with the true solver to get TCP in robot frame (mm).
      5. Project TCP through T_cam2base and K to get pixel coords.

    Args:
        true_signs: (5,) array of ±1 ground-truth joint signs.
        true_offsets_raw: (5,) array of raw zero-offset values (typically near 2048).
        T_cam2base: 4×4 homogeneous transform, camera frame → robot base frame.
        K: 3×3 camera intrinsic matrix.
        n_poses: number of observations to generate.
        rng_seed: random seed for reproducibility.

    Returns:
        (raw_positions_list, pts_2d): synthetic observations ready for
        _brute_force_signs().
    """
    rng = np.random.default_rng(rng_seed)

    # Create solver with the TRUE signs and zero offsets (offset in degrees = 0)
    solver_true = Arm101IKSolver(
        joint_signs=true_signs.copy(),
        joint_offsets_deg=np.zeros(5),
    )

    # Camera-to-base inverse: base-to-camera transform for projection
    T_base2cam = np.linalg.inv(T_cam2base)
    R_b2c = T_base2cam[:3, :3]
    t_b2c = T_base2cam[:3, 3]

    dist_coeffs = np.zeros(5)   # synthetic data; no distortion

    raw_positions_list = []
    pts_2d = []

    # Use joint limits from the solver URDF (degrees, after sign correction)
    # For diversity we sample URDF angles in a comfortable operating range.
    # arm101 joints can each move ± ~100 °; we restrict a bit for realism.
    urdf_ranges_deg = [
        (-60,  60),   # shoulder_pan
        (-60,  60),   # shoulder_lift
        (-90,  90),   # elbow_flex
        (-80,  80),   # wrist_flex
        (-120, 120),  # wrist_roll
    ]

    for _ in range(n_poses):
        # Sample random URDF angles
        urdf_deg = np.array([
            rng.uniform(lo, hi) for lo, hi in urdf_ranges_deg
        ])

        # Convert URDF angles → motor angles (sign inversion)
        # urdf = motor * sign   →   motor = urdf / sign
        motor_deg = urdf_deg / true_signs

        # Compute raw encoder position for each joint
        raw_pos = {
            mid: int(round(true_offsets_raw[mid - 1] + motor_deg[mid - 1] / DEG_PER_POS))
            for mid in range(1, 6)
        }
        # Clamp to valid servo range [0, 4095]
        raw_pos = {mid: int(np.clip(v, 0, 4095)) for mid, v in raw_pos.items()}

        # Forward kinematics using the true solver
        # We pass motor angles (what _brute_force_signs would compute from raw)
        actual_motor_deg = np.array([
            (raw_pos[mid] - true_offsets_raw[mid - 1]) * DEG_PER_POS
            for mid in range(1, 6)
        ])
        tcp_robot_mm, _ = solver_true.forward_kin(actual_motor_deg)

        # Project TCP into camera frame
        p_cam = R_b2c @ tcp_robot_mm + t_b2c
        if p_cam[2] <= 0:
            continue   # behind camera; skip

        px = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
        py = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]

        raw_positions_list.append(raw_pos)
        pts_2d.append([float(px), float(py)])

    return raw_positions_list, pts_2d


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def synthetic_setup():
    """Return a dict with all inputs for _brute_force_signs plus ground truth."""
    # Ground-truth signs: intentionally mix +1 and -1 for the first 4 joints.
    # Use a config that is NOT the default [+1,-1,-1,-1,-1] to exercise the solver.
    true_signs = np.array([+1.0, +1.0, -1.0, +1.0, -1.0])

    # Slightly off-centre zero offsets (realistic servo variability ±100 ticks)
    true_offsets_raw = np.array([2048.0, 2100.0, 1980.0, 2060.0, 2200.0])

    # Realistic camera intrinsics (640×480 lens)
    K = np.array([
        [554.3,   0.0, 320.0],
        [  0.0, 554.3, 240.0],
        [  0.0,   0.0,   1.0],
    ], dtype=np.float64)
    dist = np.zeros(5)

    # Camera mounted 1200 mm above the robot base, looking straight down.
    # This "top-down" geometry ensures every TCP position (z typically −100 to
    # +400 mm) projects in front of the camera (p_cam[2] > 0).
    #
    # Orientation (OpenCV convention):
    #   camera x  →  base x
    #   camera y  →  −base y  (image y flipped relative to robot y)
    #   camera z  →  −base z  (optical axis points down)
    R_cam2base = np.array([
        [ 1.0,  0.0,  0.0],
        [ 0.0, -1.0,  0.0],
        [ 0.0,  0.0, -1.0],
    ], dtype=np.float64)
    t_cam2base = np.array([0.0, 0.0, 1200.0])  # camera at 1200 mm height
    T_cam2base = np.eye(4)
    T_cam2base[:3, :3] = R_cam2base
    T_cam2base[:3, 3] = t_cam2base

    raw_list, pts_2d = _make_synthetic_data(
        true_signs, true_offsets_raw, T_cam2base, K, n_poses=14)

    assert len(raw_list) >= 8, (
        f"Need at least 8 synthetic poses but only generated {len(raw_list)}; "
        "check the camera placement / joint ranges.")

    # Seed the solver with the DEFAULT signs (all +1 for the test; wrong on purpose)
    solver = Arm101IKSolver(
        joint_signs=np.ones(5),
        joint_offsets_deg=np.zeros(5),
    )

    # Use the true offsets as the starting seed for the brute-force search
    current_offsets_raw = true_offsets_raw.copy()

    return {
        'true_signs': true_signs,
        'true_offsets_raw': true_offsets_raw,
        'K': K,
        'dist': dist,
        'solver': solver,
        'raw_list': raw_list,
        'pts_2d': pts_2d,
        'current_offsets_raw': current_offsets_raw,
    }


@pytest.fixture(scope='module')
def solver_result(synthetic_setup):
    """Run _brute_force_signs once and cache the result for all tests."""
    s = synthetic_setup
    result = _brute_force_signs(
        raw_positions_list=s['raw_list'],
        pts_2d=s['pts_2d'],
        K=s['K'],
        dist_coeffs=s['dist'],
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
        required = {'signs', 'signs_str', 'offsets_raw', 'T_cam2base',
                    'mean_err_px', 'per_point_err', 'all_results'}
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - result.keys()}")

    def test_all_32_combinations_tried(self, solver_result):
        """The solver must try all 2^5 = 32 sign combinations."""
        result, _ = solver_result
        assert len(result['all_results']) == 32

    def test_best_error_is_low(self, solver_result):
        """With noise-free synthetic data the best solution must be near-perfect."""
        result, _ = solver_result
        assert result['mean_err_px'] < 2.0, (
            f"Expected mean error < 2px on synthetic data, "
            f"got {result['mean_err_px']:.2f}px")

    def test_signs_shape(self, solver_result):
        result, _ = solver_result
        assert result['signs'].shape == (5,), (
            f"signs should be a (5,) array, got shape {result['signs'].shape}")

    def test_all_signs_are_plus_or_minus_one(self, solver_result):
        result, _ = solver_result
        for i, s in enumerate(result['signs']):
            assert s in (1.0, -1.0), f"Sign {i} is {s}, expected ±1"

    @pytest.mark.parametrize("joint_idx", [0, 1, 2, 3])
    def test_joint_sign_correctly_identified(self, solver_result, joint_idx):
        """The first 4 joints (shoulder_pan through wrist_flex) must match truth.

        These joints substantially move the TCP and are reliably observable.
        Joint 4 (wrist_roll) rotates around the arm axis and may be ambiguous;
        it is intentionally excluded from this assertion.
        """
        result, true_signs = solver_result
        found = result['signs'][joint_idx]
        expected = true_signs[joint_idx]
        assert found == expected, (
            f"Joint {joint_idx}: expected sign {expected:+.0f}, "
            f"got {found:+.0f} (mean_err={result['mean_err_px']:.2f}px)")

    def test_offsets_raw_shape(self, solver_result):
        result, _ = solver_result
        assert result['offsets_raw'].shape == (5,), (
            f"offsets_raw should be (5,), got {result['offsets_raw'].shape}")

    def test_offsets_within_servo_range(self, solver_result):
        result, _ = solver_result
        for i, offset in enumerate(result['offsets_raw']):
            assert 0 <= offset <= 4095, (
                f"Offset {i} = {offset:.0f} is outside valid [0, 4095] range")

    def test_T_cam2base_is_4x4(self, solver_result):
        result, _ = solver_result
        T = result['T_cam2base']
        assert T.shape == (4, 4), f"T_cam2base should be (4,4), got {T.shape}"

    def test_T_cam2base_is_valid_SE3(self, solver_result):
        """The recovered camera transform must be a valid rotation + translation."""
        result, _ = solver_result
        T = result['T_cam2base']
        R = T[:3, :3]
        # Orthonormality: R^T R ≈ I
        I_approx = R.T @ R
        np.testing.assert_allclose(
            I_approx, np.eye(3), atol=1e-4,
            err_msg="Camera rotation matrix is not orthonormal")
        # Proper rotation: det(R) ≈ +1
        assert abs(np.linalg.det(R) - 1.0) < 1e-4, (
            f"det(R) = {np.linalg.det(R):.4f}, expected +1")

    def test_results_sorted_by_error(self, solver_result):
        """all_results must be sorted ascending by mean_err_px."""
        result, _ = solver_result
        errors = [r['mean_err_px'] for r in result['all_results']]
        assert errors == sorted(errors), "all_results is not sorted by error"

    def test_best_combo_is_first(self, solver_result):
        """The first entry in all_results must be the lowest-error combo."""
        result, _ = solver_result
        best_err = result['all_results'][0]['mean_err_px']
        assert result['mean_err_px'] == pytest.approx(best_err), (
            "result['mean_err_px'] does not match all_results[0]")

    def test_signs_str_matches_signs_array(self, solver_result):
        """signs_str must be a compact string representation of signs."""
        result, _ = solver_result
        signs = result['signs']
        expected_str = ''.join('+' if s > 0 else '-' for s in signs)
        assert result['signs_str'] == expected_str, (
            f"signs_str '{result['signs_str']}' does not match "
            f"signs array {signs} (expected '{expected_str}')")
