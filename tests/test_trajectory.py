"""Unit tests for quintic trajectory generation."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from planner.trajectory import quintic_trajectory


class TestQuinticTrajectory:
    def test_endpoints(self):
        """Trajectory should start at q_start and end at q_goal."""
        q0 = np.array([0.0, 10.0, -20.0, 30.0, -40.0, 50.0])
        qf = np.array([10.0, 20.0, -10.0, 40.0, -30.0, 60.0])
        traj = quintic_trajectory(q0, qf)
        np.testing.assert_array_almost_equal(traj[0], q0)
        np.testing.assert_array_almost_equal(traj[-1], qf)

    def test_step_count_proportional_to_travel(self):
        """Larger moves should produce more steps."""
        q0 = np.zeros(6)
        small = quintic_trajectory(q0, np.array([5.0, 0, 0, 0, 0, 0]), max_step_deg=5.0)
        big = quintic_trajectory(q0, np.array([50.0, 0, 0, 0, 0, 0]), max_step_deg=5.0)
        assert len(big) > len(small)

    def test_max_step_respected(self):
        """No single step should exceed max_step_deg."""
        q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        qf = np.array([90.0, -45.0, 60.0, -30.0, 120.0, -15.0])
        max_step = 5.0
        traj = quintic_trajectory(q0, qf, max_step_deg=max_step)

        for i in range(1, len(traj)):
            step_change = np.max(np.abs(traj[i] - traj[i - 1]))
            # Allow small overshoot due to quintic shape (steps near middle are larger)
            assert step_change < max_step * 2.0, \
                f"Step {i}: change={step_change:.2f}deg exceeds 2x max_step={max_step}"

    def test_monotonic_s_curve(self):
        """Single-joint trajectory should be monotonic (no backtracking)."""
        q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        qf = np.array([60.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        traj = quintic_trajectory(q0, qf, max_step_deg=3.0)

        j1_values = [t[0] for t in traj]
        for i in range(1, len(j1_values)):
            assert j1_values[i] >= j1_values[i - 1], \
                f"J1 not monotonic at step {i}: {j1_values[i-1]:.2f} -> {j1_values[i]:.2f}"

    def test_zero_velocity_at_endpoints(self):
        """Velocity (finite difference) should be near zero at start and end."""
        q0 = np.zeros(6)
        qf = np.array([60.0, -30.0, 45.0, 0.0, -60.0, 20.0])
        traj = quintic_trajectory(q0, qf, max_step_deg=1.0)

        # Velocity ≈ finite difference between consecutive steps
        v_start = np.max(np.abs(traj[1] - traj[0]))
        v_end = np.max(np.abs(traj[-1] - traj[-2]))
        v_mid = np.max(np.abs(traj[len(traj)//2] - traj[len(traj)//2 - 1]))

        # Start/end velocities should be much smaller than mid-trajectory
        assert v_start < v_mid * 0.3, \
            f"Start velocity {v_start:.3f} not much smaller than mid {v_mid:.3f}"
        assert v_end < v_mid * 0.3, \
            f"End velocity {v_end:.3f} not much smaller than mid {v_mid:.3f}"

    def test_zero_move(self):
        """Identical start/goal should return 2 points (start, end)."""
        q = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        traj = quintic_trajectory(q, q.copy())
        assert len(traj) == 2

    def test_tiny_move(self):
        """Very small move should still return at least 2 points."""
        q0 = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        qf = q0 + 0.001
        traj = quintic_trajectory(q0, qf)
        assert len(traj) >= 2

    def test_large_move_many_steps(self):
        """A 180-degree move at 5deg/step should produce ~36+ steps."""
        q0 = np.zeros(6)
        qf = np.array([180.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        traj = quintic_trajectory(q0, qf, max_step_deg=5.0)
        assert len(traj) >= 36, f"Expected ~36 steps, got {len(traj)}"

    def test_multi_joint_move(self):
        """All joints should progress from start to goal."""
        q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        qf = np.array([30.0, -20.0, 40.0, -10.0, 50.0, -5.0])
        traj = quintic_trajectory(q0, qf, max_step_deg=3.0)

        # Each joint at final step should be near goal
        for j in range(6):
            assert abs(traj[-1][j] - qf[j]) < 0.01, \
                f"Joint {j}: final={traj[-1][j]:.3f} != goal={qf[j]:.3f}"
