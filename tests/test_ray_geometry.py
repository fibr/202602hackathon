"""Tests for ray-plane intersection geometry used in collect_dataset.py."""

import os
import sys
import numpy as np
import pytest

# Import the functions under test from collect_dataset
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from collect_dataset import pixel_to_ray_camera, ray_plane_intersect, compute_hover_pose


class FakeIntrinsics:
    """Mimics RealSense intrinsics for testing."""
    def __init__(self, fx=900.0, fy=900.0, ppx=640.0, ppy=360.0):
        self.fx = fx
        self.fy = fy
        self.ppx = ppx
        self.ppy = ppy


class TestPixelToRay:
    def test_center_pixel_is_forward(self):
        intr = FakeIntrinsics()
        ray = pixel_to_ray_camera(640.0, 360.0, intr)
        # Center pixel should produce a ray straight forward (0, 0, 1) normalized
        np.testing.assert_array_almost_equal(ray, [0.0, 0.0, 1.0], decimal=5)

    def test_right_pixel_has_positive_x(self):
        intr = FakeIntrinsics()
        ray = pixel_to_ray_camera(900.0, 360.0, intr)
        assert ray[0] > 0, "Right of center should have positive x"
        assert abs(ray[1]) < 1e-5, "Y should be ~0 at vertical center"

    def test_bottom_pixel_has_positive_y(self):
        intr = FakeIntrinsics()
        ray = pixel_to_ray_camera(640.0, 600.0, intr)
        assert ray[1] > 0, "Below center should have positive y"
        assert abs(ray[0]) < 1e-5, "X should be ~0 at horizontal center"

    def test_ray_is_unit_vector(self):
        intr = FakeIntrinsics()
        ray = pixel_to_ray_camera(100.0, 200.0, intr)
        np.testing.assert_almost_equal(np.linalg.norm(ray), 1.0)


class TestRayPlaneIntersect:
    def test_vertical_ray_hits_plane(self):
        origin = np.array([0.0, 0.0, 1000.0])
        direction = np.array([0.0, 0.0, -1.0])
        point = ray_plane_intersect(origin, direction, plane_z=0.0)
        assert point is not None
        np.testing.assert_array_almost_equal(point, [0.0, 0.0, 0.0])

    def test_angled_ray(self):
        origin = np.array([0.0, 0.0, 1000.0])
        direction = np.array([1.0, 0.0, -1.0])
        direction /= np.linalg.norm(direction)
        point = ray_plane_intersect(origin, direction, plane_z=0.0)
        assert point is not None
        assert point[0] == pytest.approx(1000.0, abs=0.1)
        assert point[2] == pytest.approx(0.0, abs=0.1)

    def test_ray_at_nonzero_plane(self):
        origin = np.array([0.0, 0.0, 1000.0])
        direction = np.array([0.0, 0.0, -1.0])
        point = ray_plane_intersect(origin, direction, plane_z=13.5)
        assert point is not None
        assert point[2] == pytest.approx(13.5)

    def test_parallel_ray_returns_none(self):
        origin = np.array([0.0, 0.0, 1000.0])
        direction = np.array([1.0, 0.0, 0.0])
        point = ray_plane_intersect(origin, direction, plane_z=0.0)
        assert point is None

    def test_ray_pointing_away_returns_none(self):
        origin = np.array([0.0, 0.0, 1000.0])
        direction = np.array([0.0, 0.0, 1.0])  # pointing up, away from z=0
        point = ray_plane_intersect(origin, direction, plane_z=0.0)
        assert point is None


class TestFindNextIndex:
    def test_empty_dir(self, tmp_path):
        from collect_dataset import find_next_index
        assert find_next_index(str(tmp_path)) == 0

    def test_nonexistent_dir(self, tmp_path):
        from collect_dataset import find_next_index
        assert find_next_index(str(tmp_path / 'nope')) == 0

    def test_with_existing_files(self, tmp_path):
        from collect_dataset import find_next_index
        (tmp_path / '000_color.png').touch()
        (tmp_path / '001_color.png').touch()
        (tmp_path / '002_color.png').touch()
        assert find_next_index(str(tmp_path)) == 3

    def test_gaps_in_indices(self, tmp_path):
        from collect_dataset import find_next_index
        (tmp_path / '000_color.png').touch()
        (tmp_path / '005_color.png').touch()
        assert find_next_index(str(tmp_path)) == 6

    def test_ignores_non_color_files(self, tmp_path):
        from collect_dataset import find_next_index
        (tmp_path / '003_depth.png').touch()
        (tmp_path / '003_depth_vis.png').touch()
        assert find_next_index(str(tmp_path)) == 0
