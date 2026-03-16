"""Unit tests for src/cube_face_aligner.py — CubeFaceAligner.

Covers:
- _normalize_45 helper: 4-fold symmetry normalization
- _wrap_nearest helper: angle wrapping to nearest reference
- CubeFaceAligner.compute_alignment basic cases
- Symmetry: ±45 deg yaw gives equivalent result
- Zero yaw gives zero delta
- All candidate deltas are within ±45 for mid-range J5
- Joint limit handling: plans are valid even near limits
- Deadband: small yaw produces "already aligned" status
- batch_alignment_test formatting
- from_config factory
"""

import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cube_face_aligner import (
    CubeFaceAligner,
    FaceAlignmentPlan,
    _normalize_45,
    _wrap_nearest,
    batch_alignment_test,
)


# ---------------------------------------------------------------------------
# _normalize_45 tests
# ---------------------------------------------------------------------------

class TestNormalize45:
    def test_zero(self):
        assert abs(_normalize_45(0.0)) < 1e-9

    def test_positive_in_range(self):
        assert abs(_normalize_45(30.0) - 30.0) < 1e-9

    def test_negative_in_range(self):
        assert abs(_normalize_45(-30.0) - (-30.0)) < 1e-9

    def test_at_45(self):
        # 45 wraps to -45 (half-open interval [-45, +45))
        result = _normalize_45(45.0)
        assert abs(result - (-45.0)) < 1e-9

    def test_at_90(self):
        result = _normalize_45(90.0)
        assert abs(result) < 1e-9

    def test_at_135(self):
        result = _normalize_45(135.0)
        assert abs(result - (-45.0)) < 1e-6 or abs(result - 45.0) < 1e-6

    def test_negative_90(self):
        result = _normalize_45(-90.0)
        assert abs(result) < 1e-9

    def test_large_positive(self):
        result = _normalize_45(370.0)  # 370 = 10 mod 90 = 10
        assert -45.0 <= result < 45.0

    def test_large_negative(self):
        result = _normalize_45(-370.0)
        assert -45.0 <= result < 45.0


# ---------------------------------------------------------------------------
# _wrap_nearest tests
# ---------------------------------------------------------------------------

class TestWrapNearest:
    def test_same(self):
        assert abs(_wrap_nearest(0.0, 0.0)) < 1e-9

    def test_small_delta(self):
        result = _wrap_nearest(10.0, 5.0)
        assert abs(result - 10.0) < 1e-9

    def test_wrap_positive(self):
        # 350 is closer to 0 than staying at 350 (delta = -10 vs +350)
        result = _wrap_nearest(350.0, 0.0)
        assert abs(result - (-10.0)) < 1e-9

    def test_wrap_negative(self):
        result = _wrap_nearest(-350.0, 0.0)
        assert abs(result - 10.0) < 1e-9


# ---------------------------------------------------------------------------
# CubeFaceAligner.compute_alignment tests
# ---------------------------------------------------------------------------

class TestComputeAlignment:
    def setup_method(self):
        self.aligner = CubeFaceAligner(mount_angle_deg=0.0, deadband_deg=2.0)

    def test_zero_yaw_zero_j5(self):
        """Zero yaw with J5=0 should produce delta near zero."""
        plan = self.aligner.compute_alignment(0.0, 0.0)
        assert plan.valid
        assert abs(plan.delta_deg) < 2.1  # within deadband

    def test_positive_yaw(self):
        """Positive yaw should produce positive delta."""
        plan = self.aligner.compute_alignment(20.0, 0.0)
        assert plan.valid
        assert abs(plan.delta_deg - 20.0) < 1e-6

    def test_negative_yaw(self):
        """Negative yaw should produce negative delta."""
        plan = self.aligner.compute_alignment(-20.0, 0.0)
        assert plan.valid
        assert abs(plan.delta_deg - (-20.0)) < 1e-6

    def test_delta_within_45_midrange(self):
        """For J5=0, all yaws in [-45,45) should give |delta| <= 45."""
        for yaw in range(-44, 45):
            plan = self.aligner.compute_alignment(float(yaw), 0.0)
            assert plan.valid
            assert abs(plan.delta_deg) <= 45.01, f'yaw={yaw}: delta={plan.delta_deg}'

    def test_four_candidates(self):
        """Should always produce 4 candidate J5 values, 90 degrees apart."""
        plan = self.aligner.compute_alignment(15.0, 0.0)
        assert len(plan.candidate_j5_degs) == 4

    def test_candidates_90_apart(self):
        """Candidates should be 90 degrees apart (modulo wrapping)."""
        plan = self.aligner.compute_alignment(15.0, 0.0)
        cands = sorted(plan.candidate_j5_degs)
        # After sorting, consecutive differences should be ~90
        diffs = [cands[i+1] - cands[i] for i in range(3)]
        for d in diffs:
            assert abs(d - 90.0) < 1e-6 or abs(d + 270.0) < 1e-6, f'diff={d}'

    def test_symmetry_45(self):
        """yaw=+45 and yaw=-45 should give same absolute delta."""
        p1 = self.aligner.compute_alignment(44.0, 0.0)
        p2 = self.aligner.compute_alignment(-44.0, 0.0)
        assert abs(abs(p1.delta_deg) - abs(p2.delta_deg)) < 1e-6

    def test_deadband(self):
        """Yaw within deadband should report 'already aligned'."""
        plan = self.aligner.compute_alignment(1.0, 0.0)
        assert plan.valid
        assert 'already aligned' in plan.status

    def test_various_j5_starts(self):
        """Algorithm should work from any valid J5 starting position."""
        for j5 in [-100, -50, 0, 50, 100]:
            plan = self.aligner.compute_alignment(25.0, float(j5))
            assert plan.valid, f'J5={j5}: plan invalid'

    def test_selected_within_limits(self):
        """Selected J5 should always be within joint limits."""
        for j5 in [-140, -100, 0, 100, 140]:
            for yaw in [-40, 0, 40]:
                plan = self.aligner.compute_alignment(float(yaw), float(j5))
                if plan.valid:
                    assert self.aligner.j5_min_deg <= plan.selected_j5_deg <= self.aligner.j5_max_deg

    def test_mount_angle(self):
        """Mount angle should offset the robot yaw."""
        a = CubeFaceAligner(mount_angle_deg=10.0)
        plan = a.compute_alignment(20.0, 0.0)
        assert plan.valid
        assert abs(plan.robot_yaw_deg - 30.0) < 1e-6

    def test_cam_flip(self):
        """Camera flip should negate detected yaw."""
        a = CubeFaceAligner(cam_flip=True)
        plan = a.compute_alignment(20.0, 0.0)
        assert plan.valid
        assert abs(plan.robot_yaw_deg - (-20.0)) < 1e-6

    def test_plan_summary(self):
        """summary() should return a non-empty string."""
        plan = self.aligner.compute_alignment(15.0, 0.0)
        s = plan.summary()
        assert len(s) > 10
        assert '15.0' in s


# ---------------------------------------------------------------------------
# from_config tests
# ---------------------------------------------------------------------------

class TestFromConfig:
    def test_default_config(self):
        a = CubeFaceAligner.from_config({})
        assert a.mount_angle_deg == 0.0
        assert a.cam_flip is False

    def test_with_mount_angle(self):
        cfg = {'visual_servoing': {'mount_angle_deg': 15.0}}
        a = CubeFaceAligner.from_config(cfg)
        assert a.mount_angle_deg == 15.0

    def test_with_cam_flip(self):
        cfg = {'visual_servoing': {'cam_flip_x': True}}
        a = CubeFaceAligner.from_config(cfg)
        assert a.cam_flip is True


# ---------------------------------------------------------------------------
# batch_alignment_test formatting
# ---------------------------------------------------------------------------

class TestBatchAlignment:
    def test_returns_string(self):
        a = CubeFaceAligner()
        result = batch_alignment_test(a, [0.0, 15.0, -15.0])
        assert isinstance(result, str)
        assert 'Yaw' in result  # header present
        assert len(result.split('\n')) >= 5  # header + separator + 3 rows
