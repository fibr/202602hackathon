"""Tests for robot overlay: FK chain and projection."""

import os
import sys
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from visualization.robot_overlay import RobotOverlay


class FakeIntrinsics:
    def __init__(self, fx=900.0, fy=900.0, ppx=640.0, ppy=360.0):
        self.fx = fx
        self.fy = fy
        self.ppx = ppx
        self.ppy = ppy


@pytest.fixture
def overlay():
    """Create an overlay with identity transform (camera = base frame)."""
    T = np.eye(4)
    return RobotOverlay(T_camera_to_base=T, tool_length_mm=100.0)


@pytest.fixture
def overlay_with_calibration():
    """Create overlay with actual calibration if available."""
    cal_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'calibration.yaml')
    if not os.path.exists(cal_path):
        pytest.skip("No calibration.yaml present")
    from calibration import CoordinateTransform
    t = CoordinateTransform()
    t.load(cal_path)
    return RobotOverlay(T_camera_to_base=t.T_camera_to_base, tool_length_mm=100.0)


class TestJointPositions:
    def test_zero_config_base_at_origin(self, overlay):
        positions = overlay.compute_joint_positions(np.zeros(6))
        # Base should be at origin
        np.testing.assert_array_almost_equal(positions[0], [0, 0, 0])

    def test_zero_config_has_9_positions(self, overlay):
        positions = overlay.compute_joint_positions(np.zeros(6))
        assert len(positions) == 9  # base + 6 joints + flange + TCP

    def test_j1_above_base(self, overlay):
        """Joint 1 should be above base (positive Z)."""
        positions = overlay.compute_joint_positions(np.zeros(6))
        assert positions[1][2] > 0.2, f"J1 should be ~0.24m above base, got {positions[1][2]}"

    def test_flange_and_tcp_below_j6(self, overlay):
        """Flange and TCP should extend beyond J6."""
        positions = overlay.compute_joint_positions(np.zeros(6))
        # Flange (idx 7) is 100mm from J6 (idx 6)
        flange_dist = np.linalg.norm(positions[7] - positions[6])
        assert 0.08 < flange_dist < 0.12, f"Flange-J6 distance should be ~0.1m, got {flange_dist:.3f}"
        # TCP (idx 8) is tool_length_mm from J6 (default 100mm in test fixture)
        tcp_dist = np.linalg.norm(positions[8] - positions[6])
        assert 0.08 < tcp_dist < 0.12, f"TCP-J6 distance should be ~0.1m, got {tcp_dist:.3f}"

    def test_j1_rotation_moves_positions(self, overlay):
        """Rotating J1 should move all downstream joints."""
        pos_zero = overlay.compute_joint_positions(np.zeros(6))
        pos_rotated = overlay.compute_joint_positions(np.array([90, 0, 0, 0, 0, 0], dtype=float))
        # Base shouldn't move
        np.testing.assert_array_almost_equal(pos_zero[0], pos_rotated[0])
        # TCP should have moved
        diff = np.linalg.norm(pos_zero[8] - pos_rotated[8])
        assert diff > 0.1, f"90deg J1 rotation should move TCP significantly, got {diff:.3f}"

    def test_reasonable_reach(self, overlay):
        """At zero config, TCP should be within the robot's reach (~1m)."""
        positions = overlay.compute_joint_positions(np.zeros(6))
        tcp_dist = np.linalg.norm(positions[8])
        assert tcp_dist < 1.5, f"TCP at zero should be within 1.5m, got {tcp_dist:.3f}"
        assert tcp_dist > 0.3, f"TCP at zero should be at least 0.3m, got {tcp_dist:.3f}"


class TestProjection:
    def test_base_projects_to_valid_pixel(self):
        """Base origin should project to a valid pixel when camera is looking at it."""
        intr = FakeIntrinsics()
        # Camera is 2m in front of base: T_camera_to_base maps camera Z=2 to base origin
        # T_cam_to_base: camera origin at (0, 0, -2) in base frame
        T = np.eye(4)
        T[2, 3] = -2.0  # camera origin at z=-2 in base frame
        ov = RobotOverlay(T_camera_to_base=T, tool_length_mm=100.0)
        # Base at (0,0,0) in base -> (0,0,2) in camera frame -> in front
        pixels = ov.project_to_pixels([np.array([0, 0, 0])], intr)
        assert pixels[0] is not None

    def test_behind_camera_returns_none(self, overlay):
        intr = FakeIntrinsics()
        # Point behind the camera (negative Z in camera frame)
        # With identity transform, a point at z=-1 in base frame is z=-1 in camera
        pixels = overlay.project_to_pixels([np.array([0, 0, -1])], intr)
        assert pixels[0] is None


class TestDrawing:
    def test_draw_base_marker(self, overlay_with_calibration):
        intr = FakeIntrinsics()
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = overlay_with_calibration.draw_base_marker(img, intr)
        assert result.shape == img.shape
        # Should have drawn something (not all black)
        assert result.sum() > 0

    def test_draw_joints(self, overlay_with_calibration):
        intr = FakeIntrinsics()
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        joints = np.zeros(6)
        result = overlay_with_calibration.draw_joints(img, joints, intr)
        assert result.shape == img.shape
        assert result.sum() > 0
