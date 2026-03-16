"""Tests for the self-collision check helpers added to the IK solvers.

These tests exercise:
  * IKSolver.check_self_collision   (Nova5)
  * Arm101IKSolver.check_self_collision  (SO-ARM101)
  * execute_trajectory collision_check_fn callback (trajectory abort)

No robot connection is required.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ---------------------------------------------------------------------------
# Optional-import guards — skip entire module if Pinocchio not installed
# ---------------------------------------------------------------------------
try:
    import pinocchio  # noqa: F401
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False

pytestmark = pytest.mark.skipif(not HAS_PINOCCHIO, reason="pinocchio not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def nova5_ik():
    """IKSolver instance for Nova5."""
    from kinematics import IKSolver
    return IKSolver(tool_length_mm=100.0)


@pytest.fixture(scope='module')
def arm101_ik():
    """Arm101IKSolver instance."""
    from kinematics.arm101_ik_solver import Arm101IKSolver
    return Arm101IKSolver()


# ---------------------------------------------------------------------------
# Nova5 collision check tests
# ---------------------------------------------------------------------------

class TestNova5SelfCollision:
    def test_method_exists(self, nova5_ik):
        """IKSolver must have a check_self_collision method."""
        assert hasattr(nova5_ik, 'check_self_collision')
        assert callable(nova5_ik.check_self_collision)

    def test_returns_bool(self, nova5_ik):
        """check_self_collision must return a plain bool."""
        result = nova5_ik.check_self_collision(np.zeros(6))
        assert isinstance(result, bool)

    def test_zero_config_no_collision(self, nova5_ik):
        """The zero (neutral) configuration should be collision-free."""
        assert not nova5_ik.check_self_collision(np.zeros(6)), (
            "Nova5 zero config must not self-collide"
        )

    def test_home_pose_no_collision(self, nova5_ik):
        """A typical 'ready' pose should be collision-free."""
        home = np.array([0.0, 0.0, -90.0, 0.0, 90.0, 0.0])
        assert not nova5_ik.check_self_collision(home), (
            "Nova5 home pose must not self-collide"
        )

    def test_deterministic(self, nova5_ik):
        """The same configuration must always return the same result."""
        q = np.array([30.0, -20.0, 45.0, 10.0, -60.0, 15.0])
        r1 = nova5_ik.check_self_collision(q)
        r2 = nova5_ik.check_self_collision(q)
        assert r1 == r2

    def test_lazy_init_geom_model(self, nova5_ik):
        """Geometry model should be initialised after first check call."""
        # Call it to ensure lazy init ran
        nova5_ik.check_self_collision(np.zeros(6))
        assert nova5_ik._coll_geom_model is not None
        assert nova5_ik._coll_geom_data is not None
        assert nova5_ik._coll_data is not None

    def test_collision_model_has_active_pairs(self, nova5_ik):
        """At least one collision pair should remain after filtering."""
        nova5_ik.check_self_collision(np.zeros(6))
        n_pairs = len(nova5_ik._coll_geom_model.collisionPairs)
        assert n_pairs > 0, "No active collision pairs after filtering — check disabled lists"

    def test_fk_data_not_corrupted(self, nova5_ik):
        """FK result must not change after check_self_collision is called."""
        q = np.array([10.0, -20.0, 30.0, 0.0, 45.0, 0.0])
        pos_before, rpy_before = nova5_ik.forward_kin(q)
        nova5_ik.check_self_collision(q)
        pos_after, rpy_after = nova5_ik.forward_kin(q)
        np.testing.assert_array_almost_equal(
            pos_before, pos_after,
            err_msg="FK position changed after check_self_collision (data corruption)"
        )


# ---------------------------------------------------------------------------
# SO-ARM101 collision check tests
# ---------------------------------------------------------------------------

class TestArm101SelfCollision:
    def test_method_exists(self, arm101_ik):
        """Arm101IKSolver must have a check_self_collision method."""
        assert hasattr(arm101_ik, 'check_self_collision')
        assert callable(arm101_ik.check_self_collision)

    def test_returns_bool(self, arm101_ik):
        """check_self_collision must return a plain bool."""
        result = arm101_ik.check_self_collision(np.zeros(5))
        assert isinstance(result, bool)

    def test_zero_config_no_collision(self, arm101_ik):
        """The zero motor-angle configuration should be collision-free."""
        assert not arm101_ik.check_self_collision(np.zeros(5)), (
            "ARM101 zero config must not self-collide"
        )

    def test_deterministic(self, arm101_ik):
        """The same configuration must always return the same result."""
        q = np.array([10.0, -20.0, 30.0, 15.0, -45.0])
        r1 = arm101_ik.check_self_collision(q)
        r2 = arm101_ik.check_self_collision(q)
        assert r1 == r2

    def test_lazy_init_geom_model(self, arm101_ik):
        """Geometry model should be initialised after first check call."""
        arm101_ik.check_self_collision(np.zeros(5))
        assert arm101_ik._coll_geom_model is not None
        assert arm101_ik._coll_geom_data is not None
        assert arm101_ik._coll_data is not None

    def test_collision_model_has_active_pairs(self, arm101_ik):
        """At least one collision pair should remain after filtering."""
        arm101_ik.check_self_collision(np.zeros(5))
        n_pairs = len(arm101_ik._coll_geom_model.collisionPairs)
        assert n_pairs > 0, "No active collision pairs after filtering"

    def test_accepts_6_motor_angles(self, arm101_ik):
        """check_self_collision should accept 6-element input (includes gripper)."""
        q6 = np.zeros(6)
        result = arm101_ik.check_self_collision(q6)
        assert isinstance(result, bool)

    def test_fk_data_not_corrupted(self, arm101_ik):
        """FK result must not change after check_self_collision is called."""
        q = np.array([10.0, -20.0, 30.0, 15.0, -45.0])
        pos_before, _ = arm101_ik.forward_kin(q)
        arm101_ik.check_self_collision(q)
        pos_after, _ = arm101_ik.forward_kin(q)
        np.testing.assert_array_almost_equal(
            pos_before, pos_after,
            err_msg="FK position changed after check_self_collision (data corruption)"
        )


# ---------------------------------------------------------------------------
# Trajectory execute_trajectory collision_check_fn tests
# ---------------------------------------------------------------------------

class TestTrajectoryCollisionAbort:
    """Test that execute_trajectory aborts when the collision_check_fn fires."""

    def _make_mock_robot(self, call_log):
        """Return a minimal robot mock that records movj_joints calls."""
        class MockRobot:
            def movj_joints(self, *args, **kwargs):
                call_log.append(args)
                return True
        return MockRobot()

    def test_no_collision_fn_runs_all_steps(self):
        """Without a collision_check_fn, all steps are sent to the robot."""
        from planner.trajectory import execute_trajectory

        calls = []
        robot = self._make_mock_robot(calls)

        q_start = np.zeros(6)
        q_goal = np.array([30.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        ok = execute_trajectory(robot, q_start, q_goal, max_step_deg=10.0)
        assert ok
        assert len(calls) > 0

    def test_collision_fn_never_fires_runs_all_steps(self):
        """A collision_check_fn that always returns False does not abort."""
        from planner.trajectory import execute_trajectory

        calls = []
        robot = self._make_mock_robot(calls)

        q_start = np.zeros(6)
        q_goal = np.array([30.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        ok = execute_trajectory(robot, q_start, q_goal, max_step_deg=10.0,
                                collision_check_fn=lambda q: False)
        assert ok
        assert len(calls) > 0

    def test_collision_fn_always_fires_aborts_immediately(self):
        """A collision_check_fn that always returns True aborts at step 1."""
        from planner.trajectory import execute_trajectory

        calls = []
        robot = self._make_mock_robot(calls)

        q_start = np.zeros(6)
        q_goal = np.array([30.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        ok = execute_trajectory(robot, q_start, q_goal, max_step_deg=10.0,
                                collision_check_fn=lambda q: True)
        assert not ok, "execute_trajectory should return False when collision detected"
        assert len(calls) == 0, "No robot command should be sent before abort"

    def test_collision_fn_fires_mid_trajectory(self):
        """Abort partway through: steps before collision are sent, steps after are not."""
        from planner.trajectory import execute_trajectory
        from planner.trajectory import quintic_trajectory

        calls = []
        robot = self._make_mock_robot(calls)

        q_start = np.zeros(6)
        q_goal = np.array([60.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        configs = quintic_trajectory(q_start, q_goal, max_step_deg=10.0)
        midpoint = len(configs) // 2  # step index where collision will fire

        call_count = [0]

        def collision_fn(q):
            call_count[0] += 1
            return call_count[0] >= midpoint  # True from midpoint onward

        ok = execute_trajectory(robot, q_start, q_goal, max_step_deg=10.0,
                                collision_check_fn=collision_fn)

        assert not ok, "Should abort when collision detected mid-trajectory"
        # Some (but not all) steps were sent
        total_steps = len(configs) - 1
        assert 0 < len(calls) < total_steps, (
            f"Expected partial execution: {len(calls)} commands sent out of {total_steps}"
        )
