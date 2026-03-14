"""Unit tests for robot.motion_utils.move_to_pose().

Covers:
  1. Nova5 path  — robot.movj() called with correct args; set_speed propagated.
  2. arm101 path — IK solved, seed taken from get_angles(), move_joints() called.
  3. IK-unavailable failure case — returns False without calling move_joints().
  4. Seed-passing correctness — first-5-angle slice passed positionally to solve_ik().

All tests use mock robot objects; no hardware, serial port, or URDF required.
"""

import sys
import os
import numpy as np
import pytest
from unittest import mock

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import robot.motion_utils as mu
from robot.motion_utils import move_to_pose, get_robot_type


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nova5_robot(movj_return=True):
    """Return a MagicMock that looks like a Nova5 (no robot_type attr)."""
    r = mock.MagicMock()
    # Remove robot_type so get_robot_type() falls back to 'nova5'
    del r.robot_type
    r.movj.return_value = movj_return
    return r


def _arm101_robot(angles=None, move_joints_return=True):
    """Return a MagicMock that looks like an arm101."""
    r = mock.MagicMock()
    r.robot_type = 'arm101'
    r.get_angles.return_value = angles if angles is not None else [0.0] * 6
    r.move_joints.return_value = move_joints_return
    return r


def _mock_solver(result=None):
    """Return a MagicMock IK solver whose solve_ik() returns *result*.

    If *result* is None the solver signals IK failure.
    """
    s = mock.MagicMock()
    if result is None:
        s.solve_ik.return_value = None
    else:
        s.solve_ik.return_value = result
    return s


def _good_solver():
    """Solver that returns a 5-joint angle array."""
    angles = np.array([10.0, -20.0, 30.0, -45.0, 15.0])
    return _mock_solver(result=angles), angles


# ─────────────────────────────────────────────────────────────────────────────
# 1. get_robot_type() — unit-level helper
# ─────────────────────────────────────────────────────────────────────────────

class TestGetRobotType:
    """Ensure get_robot_type dispatches correctly."""

    def test_defaults_to_nova5_when_no_attribute(self):
        # Plain object with no robot_type attr
        class Plain:
            pass
        assert get_robot_type(Plain()) == 'nova5'

    def test_arm101_detected_from_attribute(self):
        r = mock.MagicMock()
        r.robot_type = 'arm101'
        assert get_robot_type(r) == 'arm101'

    def test_nova5_explicit_attribute(self):
        r = mock.MagicMock()
        r.robot_type = 'nova5'
        assert get_robot_type(r) == 'nova5'

    def test_mock_without_robot_type_is_nova5(self):
        # MagicMock with the attribute deleted → nova5
        r = mock.MagicMock()
        del r.robot_type
        assert get_robot_type(r) == 'nova5'


# ─────────────────────────────────────────────────────────────────────────────
# 2. Nova5 path — movj() dispatch
# ─────────────────────────────────────────────────────────────────────────────

class TestNova5Path:
    """move_to_pose with a Nova5 robot calls movj() with the right arguments."""

    def test_movj_called_once(self):
        robot = _nova5_robot()
        move_to_pose(robot, 100, 200, 300, 0, 90, 0)
        robot.movj.assert_called_once()

    def test_movj_receives_correct_pose(self):
        robot = _nova5_robot()
        move_to_pose(robot, 100, 200, 300, 10, 20, 30)
        robot.movj.assert_called_once_with(100, 200, 300, 10, 20, 30, timeout=30.0)

    def test_returns_true_when_movj_succeeds(self):
        robot = _nova5_robot(movj_return=True)
        assert move_to_pose(robot, 100, 0, 200, 0, 90, 0) is True

    def test_returns_false_when_movj_fails(self):
        robot = _nova5_robot(movj_return=False)
        assert move_to_pose(robot, 100, 0, 200, 0, 90, 0) is False

    def test_set_speed_called_with_default_30(self):
        robot = _nova5_robot()
        move_to_pose(robot, 100, 0, 200, 0, 90, 0)
        robot.set_speed.assert_called_once_with(30)

    def test_set_speed_called_with_custom_value(self):
        robot = _nova5_robot()
        move_to_pose(robot, 100, 0, 200, 0, 90, 0, speed=75)
        robot.set_speed.assert_called_once_with(75)

    def test_timeout_forwarded_to_movj(self):
        robot = _nova5_robot()
        move_to_pose(robot, 100, 0, 200, 0, 90, 0, timeout=15.0)
        _, kwargs = robot.movj.call_args
        assert kwargs['timeout'] == pytest.approx(15.0)

    def test_set_speed_failure_does_not_abort_move(self):
        """set_speed() errors should be swallowed; movj should still run."""
        robot = _nova5_robot()
        robot.set_speed.side_effect = Exception("speed error")
        move_to_pose(robot, 100, 0, 200, 0, 90, 0)
        robot.movj.assert_called_once()

    def test_ik_param_ignored_for_nova5(self):
        """ik= kwarg is silently ignored for Nova5."""
        robot = _nova5_robot()
        fake_solver = _mock_solver(result=np.zeros(5))
        result = move_to_pose(robot, 100, 0, 200, 0, 90, 0, ik=fake_solver)
        robot.movj.assert_called_once()
        fake_solver.solve_ik.assert_not_called()
        assert result is True


# ─────────────────────────────────────────────────────────────────────────────
# 3. Nova5 fallback path — stub with no movj(), uses raw send()
# ─────────────────────────────────────────────────────────────────────────────

class TestNova5FallbackPath:
    """Stub objects without movj() should use robot.send() as a raw fallback."""

    def _stub(self, resp='0,0,MovJ();'):
        # Spec only contains set_speed and send → hasattr(robot, 'movj') is False
        r = mock.MagicMock(spec=['set_speed', 'send'])
        r.send.return_value = resp
        return r

    def test_send_called_when_no_movj(self):
        robot = self._stub()
        move_to_pose(robot, 100, 0, 200, 0, 90, 0)
        robot.send.assert_called_once()

    def test_returns_true_on_0_prefix_response(self):
        assert move_to_pose(self._stub('0,0,MovJ();'), 100, 0, 200, 0, 90, 0) is True

    def test_returns_false_on_error_response(self):
        assert move_to_pose(self._stub('-30001,error'), 100, 0, 200, 0, 90, 0) is False

    def test_returns_false_on_none_response(self):
        robot = self._stub(resp=None)
        assert move_to_pose(robot, 100, 0, 200, 0, 90, 0) is False


# ─────────────────────────────────────────────────────────────────────────────
# 4. arm101 path — IK + move_joints() dispatch
# ─────────────────────────────────────────────────────────────────────────────

class TestArm101Path:
    """move_to_pose with arm101 solves IK locally then calls move_joints()."""

    def test_move_joints_called_on_success(self):
        robot = _arm101_robot()
        solver, _ = _good_solver()
        assert move_to_pose(robot, 100, 0, 200, 0, 90, 0, ik=solver) is True
        robot.move_joints.assert_called_once()

    def test_returns_true_when_ik_and_move_succeed(self):
        robot = _arm101_robot(move_joints_return=True)
        solver, _ = _good_solver()
        assert move_to_pose(robot, 100, 0, 200, 0, 90, 0, ik=solver) is True

    def test_returns_false_when_move_joints_fails(self):
        robot = _arm101_robot(move_joints_return=False)
        solver, _ = _good_solver()
        assert move_to_pose(robot, 100, 0, 200, 0, 90, 0, ik=solver) is False

    def test_returns_false_when_ik_returns_none(self):
        robot = _arm101_robot()
        solver = _mock_solver(result=None)
        assert move_to_pose(robot, 100, 0, 200, 0, 90, 0, ik=solver) is False

    def test_move_joints_not_called_when_ik_fails(self):
        robot = _arm101_robot()
        solver = _mock_solver(result=None)
        move_to_pose(robot, 100, 0, 200, 0, 90, 0, ik=solver)
        robot.move_joints.assert_not_called()

    def test_ik_called_with_correct_position(self):
        robot = _arm101_robot()
        solver, _ = _good_solver()
        move_to_pose(robot, 111.0, 222.0, 333.0, 0, 90, 0, ik=solver)
        pos_arg = solver.solve_ik.call_args[0][0]
        np.testing.assert_array_equal(pos_arg, [111.0, 222.0, 333.0])

    def test_ik_called_with_correct_rpy(self):
        robot = _arm101_robot()
        solver, _ = _good_solver()
        move_to_pose(robot, 100, 0, 200, 10.0, 20.0, 30.0, ik=solver)
        rpy_arg = solver.solve_ik.call_args[0][1]
        np.testing.assert_array_equal(rpy_arg, [10.0, 20.0, 30.0])

    def test_move_joints_receives_ik_result_as_list(self):
        robot = _arm101_robot()
        solver, angles = _good_solver()
        move_to_pose(robot, 100, 0, 200, 0, 90, 0, ik=solver)
        robot.move_joints.assert_called_once_with(list(angles))

    def test_speed_param_ignored_for_arm101(self):
        """arm101 controls its own speed; the speed kwarg should not cause errors."""
        robot = _arm101_robot()
        solver, _ = _good_solver()
        result = move_to_pose(robot, 100, 0, 200, 0, 90, 0, speed=80, ik=solver)
        assert result is True


# ─────────────────────────────────────────────────────────────────────────────
# 5. Seed-passing correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestSeedPassing:
    """Seed derived from robot.get_angles() is passed as third positional arg
    to solver.solve_ik()."""

    def test_seed_is_first_five_angles(self):
        angles = [10.0, 20.0, -30.0, 45.0, 15.0, 5.0]  # 6 joints
        robot = _arm101_robot(angles=angles)
        solver, _ = _good_solver()
        move_to_pose(robot, 100, 0, 200, 0, 90, 0, ik=solver)

        seed_arg = solver.solve_ik.call_args[0][2]
        expected = np.array(angles[:5], dtype=float)
        np.testing.assert_array_almost_equal(seed_arg, expected)

    def test_seed_excludes_sixth_gripper_angle(self):
        angles = [1.0, 2.0, 3.0, 4.0, 5.0, 99.0]  # gripper is index 5
        robot = _arm101_robot(angles=angles)
        solver, _ = _good_solver()
        move_to_pose(robot, 100, 0, 200, 0, 90, 0, ik=solver)

        seed_arg = solver.solve_ik.call_args[0][2]
        assert len(seed_arg) == 5
        # Confirm the gripper angle (99.0) is NOT in the seed
        assert 99.0 not in seed_arg

    def test_seed_is_none_when_get_angles_returns_none(self):
        robot = _arm101_robot(angles=None)
        robot.get_angles.return_value = None
        solver, _ = _good_solver()
        move_to_pose(robot, 100, 0, 200, 0, 90, 0, ik=solver)

        seed_arg = solver.solve_ik.call_args[0][2]
        assert seed_arg is None

    def test_seed_is_none_when_get_angles_returns_empty(self):
        robot = _arm101_robot()
        robot.get_angles.return_value = []
        solver, _ = _good_solver()
        move_to_pose(robot, 100, 0, 200, 0, 90, 0, ik=solver)

        seed_arg = solver.solve_ik.call_args[0][2]
        assert seed_arg is None

    def test_seed_is_none_when_get_angles_raises(self):
        robot = _arm101_robot()
        robot.get_angles.side_effect = Exception("serial error")
        solver, _ = _good_solver()
        move_to_pose(robot, 100, 0, 200, 0, 90, 0, ik=solver)

        seed_arg = solver.solve_ik.call_args[0][2]
        assert seed_arg is None

    def test_seed_is_numpy_array(self):
        angles = [5.0, 10.0, -15.0, 20.0, -25.0, 0.0]
        robot = _arm101_robot(angles=angles)
        solver, _ = _good_solver()
        move_to_pose(robot, 100, 0, 200, 0, 90, 0, ik=solver)

        seed_arg = solver.solve_ik.call_args[0][2]
        assert isinstance(seed_arg, np.ndarray)

    def test_seed_correct_dtype_float(self):
        angles = [1, 2, 3, 4, 5, 6]  # ints from hardware
        robot = _arm101_robot(angles=angles)
        solver, _ = _good_solver()
        move_to_pose(robot, 100, 0, 200, 0, 90, 0, ik=solver)

        seed_arg = solver.solve_ik.call_args[0][2]
        assert seed_arg.dtype == float


# ─────────────────────────────────────────────────────────────────────────────
# 6. IK-unavailable failure case
# ─────────────────────────────────────────────────────────────────────────────

class TestIKUnavailable:
    """When no IK solver is available for arm101, move_to_pose returns False
    and does not call move_joints()."""

    def test_returns_false_when_solver_is_none(self):
        robot = _arm101_robot()
        with mock.patch.object(mu, 'get_ik_solver', return_value=None):
            result = move_to_pose(robot, 100, 0, 200, 0, 90, 0)
        assert result is False

    def test_move_joints_not_called_when_no_solver(self):
        robot = _arm101_robot()
        with mock.patch.object(mu, 'get_ik_solver', return_value=None):
            move_to_pose(robot, 100, 0, 200, 0, 90, 0)
        robot.move_joints.assert_not_called()

    def test_explicit_ik_none_uses_module_solver(self):
        """ik=None explicitly triggers lookup via get_ik_solver(); still returns
        False when that lookup also returns None."""
        robot = _arm101_robot()
        with mock.patch.object(mu, 'get_ik_solver', return_value=None):
            result = move_to_pose(robot, 100, 0, 200, 0, 90, 0, ik=None)
        assert result is False

    def test_ik_param_overrides_module_cache(self):
        """Providing ik= skips get_ik_solver() entirely."""
        robot = _arm101_robot()
        solver, _ = _good_solver()
        # Even if get_ik_solver would return None, injected solver is used
        with mock.patch.object(mu, 'get_ik_solver', return_value=None):
            result = move_to_pose(robot, 100, 0, 200, 0, 90, 0, ik=solver)
        assert result is True
        solver.solve_ik.assert_called_once()

    def test_get_ik_solver_returns_none_for_nova5(self):
        """get_ik_solver('nova5') must return None (Nova5 uses firmware IK)."""
        from robot.motion_utils import get_ik_solver
        assert get_ik_solver('nova5') is None
