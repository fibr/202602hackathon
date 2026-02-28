"""Tests for rod detector: ROI mask building and color stats (no FastSAM needed)."""

import os
import sys
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from vision.rod_detector import RodDetector, RodDetection


class TestROIMaskBuilding:
    def test_no_roi_returns_none(self):
        det = RodDetector(workspace_roi=None)
        mask = det._build_roi_mask(480, 640)
        assert mask is None

    def test_rectangle_roi(self):
        roi = {'type': 'rectangle', 'rect': [100, 50, 500, 400],
               'resolution': [640, 480]}
        det = RodDetector(workspace_roi=roi)
        mask = det._build_roi_mask(480, 640)
        assert mask is not None
        assert mask.shape == (480, 640)
        # Inside the rect should be 255
        assert mask[200, 300] == 255
        # Outside should be 0
        assert mask[10, 10] == 0

    def test_polygon_roi(self):
        roi = {'type': 'polygon',
               'points': [[100, 100], [500, 100], [500, 400], [100, 400]],
               'resolution': [640, 480]}
        det = RodDetector(workspace_roi=roi)
        mask = det._build_roi_mask(480, 640)
        assert mask is not None
        assert mask[200, 300] == 255
        assert mask[10, 10] == 0

    def test_roi_resolution_scaling(self):
        """ROI defined at 1280x720 should scale correctly to 640x480."""
        roi = {'type': 'rectangle', 'rect': [200, 100, 1000, 600],
               'resolution': [1280, 720]}
        det = RodDetector(workspace_roi=roi)
        mask = det._build_roi_mask(480, 640)
        assert mask is not None
        assert mask.shape == (480, 640)
        # Center of the scaled rect should be inside
        assert mask[240, 320] == 255

    def test_unknown_roi_type(self):
        roi = {'type': 'circle', 'center': [320, 240], 'radius': 100}
        det = RodDetector(workspace_roi=roi)
        mask = det._build_roi_mask(480, 640)
        assert mask is None


class TestColorStats:
    def test_dark_uniform_region(self):
        """A dark, uniform region should score high on darkness and uniformity."""
        det = RodDetector(max_brightness=120, max_brightness_std=40.0)
        # Create a dark HSV image (V channel = 30)
        hsv = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv[:, :, 2] = 30  # V = 30 (dark)
        mask = np.ones((100, 100), dtype=np.uint8)

        stats = det._color_stats(hsv, mask)
        assert stats['mean_v'] == pytest.approx(30.0)
        assert stats['std_v'] == pytest.approx(0.0)
        assert stats['darkness_score'] == pytest.approx(0.75, abs=0.01)
        assert stats['uniformity_score'] == pytest.approx(1.0, abs=0.01)

    def test_bright_region_scores_low(self):
        det = RodDetector(max_brightness=120, max_brightness_std=40.0)
        hsv = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv[:, :, 2] = 200  # Very bright
        mask = np.ones((100, 100), dtype=np.uint8)

        stats = det._color_stats(hsv, mask)
        assert stats['darkness_score'] == 0.0

    def test_noisy_region_low_uniformity(self):
        det = RodDetector(max_brightness=120, max_brightness_std=40.0)
        hsv = np.zeros((100, 100, 3), dtype=np.uint8)
        # Create checkerboard pattern with high variance
        hsv[::2, ::2, 2] = 10
        hsv[1::2, 1::2, 2] = 200
        mask = np.ones((100, 100), dtype=np.uint8)

        stats = det._color_stats(hsv, mask)
        assert stats['uniformity_score'] < 0.3

    def test_empty_mask(self):
        det = RodDetector()
        hsv = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)

        stats = det._color_stats(hsv, mask)
        assert stats['darkness_score'] == 0.0
        assert stats['uniformity_score'] == 0.0


class TestRodDetection:
    def test_detection_dataclass(self):
        d = RodDetection(
            center_3d=np.array([0.1, 0.2, 0.3]),
            axis_3d=np.array([1.0, 0.0, 0.0]),
            center_2d=(320, 240),
            contour=np.array([[[0, 0]], [[100, 0]], [[100, 50]], [[0, 50]]]),
            confidence=0.85,
        )
        assert d.center_2d == (320, 240)
        assert d.confidence == 0.85
        assert d.mask is None
