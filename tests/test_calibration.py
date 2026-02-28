"""Tests for calibration coordinate transforms."""

import os
import sys
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from calibration import CoordinateTransform


class TestIdentityTransform:
    def test_identity_preserves_point(self):
        t = CoordinateTransform()
        p = np.array([1.0, 2.0, 3.0])
        result = t.camera_to_base(p)
        np.testing.assert_array_almost_equal(result, p)

    def test_identity_preserves_axis(self):
        t = CoordinateTransform()
        axis = np.array([0.0, 0.0, 1.0])
        result = t.camera_axis_to_base(axis)
        np.testing.assert_array_almost_equal(result, axis)


class TestSetManual:
    def test_pure_translation(self):
        t = CoordinateTransform()
        t.set_manual(
            translation=np.array([1.0, 2.0, 3.0]),
            rotation_matrix=np.eye(3),
        )
        p = np.array([0.0, 0.0, 0.0])
        result = t.camera_to_base(p)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_pure_rotation_90_z(self):
        """90-degree rotation about Z axis."""
        t = CoordinateTransform()
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        t.set_manual(translation=np.zeros(3), rotation_matrix=R)
        p = np.array([1.0, 0.0, 0.0])
        result = t.camera_to_base(p)
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 0.0])

    def test_axis_rotation(self):
        """Axis (direction) should only be rotated, not translated."""
        t = CoordinateTransform()
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        t.set_manual(translation=np.array([100.0, 200.0, 300.0]), rotation_matrix=R)
        axis = np.array([1.0, 0.0, 0.0])
        result = t.camera_axis_to_base(axis)
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 0.0])


class TestSaveLoad:
    def test_roundtrip(self):
        t = CoordinateTransform()
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        t.set_manual(
            translation=np.array([0.5, -0.3, 1.2]),
            rotation_matrix=R,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'cal.yaml')
            t.save(path)

            t2 = CoordinateTransform()
            t2.load(path)

        np.testing.assert_array_almost_equal(
            t.T_camera_to_base, t2.T_camera_to_base)

    def test_load_actual_calibration(self):
        """Load the actual calibration.yaml and verify it's a valid 4x4 matrix."""
        cal_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'calibration.yaml')
        if not os.path.exists(cal_path):
            pytest.skip("No calibration.yaml present")

        t = CoordinateTransform()
        t.load(cal_path)
        T = t.T_camera_to_base
        assert T.shape == (4, 4)
        # Bottom row should be [0, 0, 0, 1]
        np.testing.assert_array_almost_equal(T[3, :], [0, 0, 0, 1])
        # Rotation part should be orthogonal (R^T R = I)
        R = T[:3, :3]
        np.testing.assert_array_almost_equal(R.T @ R, np.eye(3), decimal=5)


class TestTransformMath:
    def test_inverse_roundtrip(self):
        """Applying transform then its inverse should return original point."""
        t = CoordinateTransform()
        cal_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'calibration.yaml')
        if not os.path.exists(cal_path):
            pytest.skip("No calibration.yaml present")

        t.load(cal_path)
        T = t.T_camera_to_base
        T_inv = np.linalg.inv(T)

        p_cam = np.array([0.1, -0.2, 0.8])
        p_base = t.camera_to_base(p_cam)
        # Apply inverse
        p_back = (T_inv @ np.append(p_base, 1.0))[:3]
        np.testing.assert_array_almost_equal(p_cam, p_back, decimal=6)
