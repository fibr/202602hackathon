"""Unit tests for the IK solver (no robot connection needed)."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from kinematics import IKSolver


@pytest.fixture
def ik():
    return IKSolver(tool_length_mm=100.0)


class TestForwardKinematics:
    def test_zero_position(self, ik):
        """FK at zero joint angles should return a valid pose."""
        pos, rpy = ik.forward_kin(np.zeros(6))
        # Should be roughly at the top of the arm's reach
        assert pos[2] > 800, f"Z should be high at zero config, got {pos[2]}"

    def test_fk_deterministic(self, ik):
        """FK should return the same result for the same input."""
        joints = np.array([10.0, -20.0, 30.0, 0.0, 45.0, 0.0])
        pos1, rpy1 = ik.forward_kin(joints)
        pos2, rpy2 = ik.forward_kin(joints)
        np.testing.assert_array_almost_equal(pos1, pos2)
        np.testing.assert_array_almost_equal(rpy1, rpy2)

    def test_fk_varies_with_input(self, ik):
        """Different joint angles should give different poses."""
        pos1, _ = ik.forward_kin(np.zeros(6))
        pos2, _ = ik.forward_kin(np.array([30.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        assert np.linalg.norm(pos1 - pos2) > 1.0, "J1 rotation should move the TCP"


class TestInverseKinematics:
    def test_roundtrip_zero(self, ik):
        """IK(FK(q)) should return ~q for zero config."""
        q_orig = np.zeros(6)
        pos, rpy = ik.forward_kin(q_orig)
        q_solved = ik.solve_ik(pos, rpy, seed_joints_deg=q_orig + 0.5)
        assert q_solved is not None, "IK should converge for zero config"
        # Verify FK of solved joints matches target
        pos2, rpy2 = ik.forward_kin(q_solved)
        np.testing.assert_allclose(pos, pos2, atol=0.1,
                                   err_msg="FK roundtrip position error > 0.1mm")

    def test_roundtrip_arbitrary(self, ik):
        """IK(FK(q)) roundtrip for an arbitrary joint config."""
        q_orig = np.array([20.0, -30.0, 45.0, 10.0, -60.0, 15.0])
        pos, rpy = ik.forward_kin(q_orig)
        q_solved = ik.solve_ik(pos, rpy, seed_joints_deg=q_orig + 1.0)
        assert q_solved is not None, "IK should converge"
        pos2, _ = ik.forward_kin(q_solved)
        err = np.linalg.norm(pos - pos2)
        assert err < 0.5, f"Roundtrip position error {err:.3f}mm > 0.5mm"

    def test_gripper_down_pose(self, ik):
        """IK should solve for a typical gripper-pointing-down pose."""
        target_pos = np.array([300.0, 0.0, 300.0])
        target_rpy = np.array([180.0, 0.0, 0.0])
        joints = ik.solve_ik(target_pos, target_rpy)
        assert joints is not None, "IK should solve gripper-down at (300,0,300)"
        pos, rpy = ik.forward_kin(joints)
        np.testing.assert_allclose(pos, target_pos, atol=0.5,
                                   err_msg="Position error > 0.5mm")

    def test_unreachable_returns_none(self, ik):
        """IK should return None for an unreachable pose."""
        # Way too far away
        target_pos = np.array([2000.0, 2000.0, 2000.0])
        target_rpy = np.array([0.0, 0.0, 0.0])
        result = ik.solve_ik(target_pos, target_rpy, max_iter=50)
        assert result is None, "IK should fail for unreachable pose"

    def test_seed_improves_convergence(self, ik):
        """Using a nearby seed should give better/same results."""
        q_orig = np.array([10.0, -20.0, 30.0, 5.0, -45.0, 10.0])
        pos, rpy = ik.forward_kin(q_orig)

        # Good seed (nearby)
        q_good = ik.solve_ik(pos, rpy, seed_joints_deg=q_orig + 2.0, max_iter=100)
        assert q_good is not None, "IK should converge with good seed"

        # Verify accuracy
        pos2, _ = ik.forward_kin(q_good)
        err = np.linalg.norm(pos - pos2)
        assert err < 0.5, f"Position error with good seed: {err:.3f}mm"


class TestLinearInterpolation:
    def test_straight_line(self, ik):
        """Linear interpolation should produce a sequence of joint configs."""
        start_pos = np.array([300.0, 0.0, 400.0])
        end_pos = np.array([300.0, 0.0, 300.0])  # 100mm straight down
        rpy = np.array([180.0, 0.0, 0.0])

        seed = ik.solve_ik(start_pos, rpy)
        assert seed is not None, "Need valid seed for interpolation test"

        path = ik.interpolate_linear(start_pos, rpy, end_pos, rpy,
                                      seed_joints_deg=seed, step_mm=20.0)
        assert path is not None, "Linear interpolation should succeed"
        assert len(path) >= 5, f"100mm / 20mm step = at least 5 points, got {len(path)}"

        # Verify each waypoint is close to the line
        for i, joints in enumerate(path):
            pos, _ = ik.forward_kin(joints)
            # X and Y should stay near 300.0 and 0.0
            assert abs(pos[0] - 300.0) < 1.0, f"Step {i}: X drift {pos[0]:.1f}"
            assert abs(pos[1] - 0.0) < 1.0, f"Step {i}: Y drift {pos[1]:.1f}"

    def test_short_move(self, ik):
        """Very short move should still produce at least 2 points."""
        pos = np.array([300.0, 0.0, 350.0])
        rpy = np.array([180.0, 0.0, 0.0])
        seed = ik.solve_ik(pos, rpy)
        assert seed is not None

        end_pos = np.array([300.0, 0.0, 349.0])  # 1mm move
        path = ik.interpolate_linear(pos, rpy, end_pos, rpy,
                                      seed_joints_deg=seed, step_mm=5.0)
        assert path is not None
        assert len(path) >= 2


class TestToolLength:
    def test_different_tool_lengths(self):
        """Different tool lengths should produce different TCP positions."""
        ik_short = IKSolver(tool_length_mm=50.0)
        ik_long = IKSolver(tool_length_mm=150.0)

        joints = np.array([10.0, -20.0, 30.0, 0.0, 45.0, 0.0])
        pos_short, _ = ik_short.forward_kin(joints)
        pos_long, _ = ik_long.forward_kin(joints)

        diff = np.linalg.norm(pos_long - pos_short)
        # Should differ by roughly 100mm (the tool length difference)
        assert 80.0 < diff < 120.0, f"Tool length difference effect: {diff:.1f}mm"

    def test_zero_tool_length(self):
        """Zero tool length should work (TCP at flange)."""
        ik = IKSolver(tool_length_mm=0.0)
        pos, rpy = ik.forward_kin(np.zeros(6))
        assert pos is not None


class TestIKSpeed:
    def test_ik_performance(self, ik):
        """IK solve should be fast enough for real-time use (<50ms)."""
        import time
        target_pos = np.array([300.0, 100.0, 300.0])
        target_rpy = np.array([180.0, 0.0, 0.0])
        seed = np.array([20.0, -20.0, -80.0, 0.0, -90.0, 20.0])

        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            ik.solve_ik(target_pos, target_rpy, seed_joints_deg=seed)
            times.append(time.perf_counter() - t0)

        avg_ms = np.mean(times) * 1000
        max_ms = np.max(times) * 1000
        print(f"\n  IK timing: avg={avg_ms:.1f}ms, max={max_ms:.1f}ms")
        assert avg_ms < 50, f"IK too slow: avg {avg_ms:.1f}ms"
