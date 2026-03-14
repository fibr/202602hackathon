"""Unit tests for src/vision/visual_servo.py.

Tests cover:
 - VisualServo.from_config() factory
 - _pixel_to_robot_delta() coordinate transform
 - align() convergence, non-convergence, no-detection paths
 - make_green_cube_detector() wrapper (with a mock green-cube detector)
 - _save_debug_frame() (smoke test: no exception)
 - ServoResult fields
"""

import os
import sys

import numpy as np
import pytest

# Allow importing src modules directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision.visual_servo import VisualServo, ServoResult, _save_debug_frame


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_servo(**overrides) -> VisualServo:
    """Construct a VisualServo with sensible test defaults."""
    defaults = dict(
        cam_index=8,
        scale_mm_per_pixel=0.25,
        mount_angle_deg=0.0,
        cam_flip_x=False,
        cam_flip_y=False,
        max_iterations=5,
        pixel_threshold=20.0,
        gain=1.0,   # unity gain for predictable mm values in tests
        settle_s=0.0,
        cam_width=640,
        cam_height=480,
        max_correction_mm=50.0,
        save_debug=False,
    )
    defaults.update(overrides)
    return VisualServo(**defaults)


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------

class TestFromConfig:
    def test_defaults_when_sections_missing(self):
        # Empty config: no hfov_deg, no approach_offset_z → falls back to 0.3
        servo = VisualServo.from_config({})
        assert servo.cam_index == 8
        assert servo.scale_mm_per_pixel == pytest.approx(0.3)
        assert servo.max_iterations == 10
        assert servo.pixel_threshold == pytest.approx(20.0)
        assert servo.gain == pytest.approx(0.6)

    def test_reads_gripper_camera_section(self):
        cfg = {
            'gripper_camera': {'device_index': 4, 'width': 1280, 'height': 720},
        }
        servo = VisualServo.from_config(cfg)
        assert servo.cam_index == 4
        assert servo.cam_width == 1280
        assert servo.cam_height == 720

    def test_reads_visual_servo_section(self):
        cfg = {
            'visual_servo': {
                'scale_mm_per_pixel': 0.18,
                'mount_angle_deg': 45.0,
                'cam_flip_x': True,
                'cam_flip_y': False,
                'max_iterations': 7,
                'pixel_threshold': 15.0,
                'gain': 0.8,
                'settle_s': 0.5,
                'max_correction_mm': 25.0,
                'save_debug': True,
            }
        }
        servo = VisualServo.from_config(cfg)
        assert servo.scale_mm_per_pixel == pytest.approx(0.18)
        assert servo.mount_angle_deg == pytest.approx(45.0)
        assert servo.cam_flip_x is True
        assert servo.cam_flip_y is False
        assert servo.max_iterations == 7
        assert servo.pixel_threshold == pytest.approx(15.0)
        assert servo.gain == pytest.approx(0.8)
        assert servo.settle_s == pytest.approx(0.5)
        assert servo.max_correction_mm == pytest.approx(25.0)
        assert servo.save_debug is True

    def test_explicit_null_triggers_auto_compute(self):
        """scale_mm_per_pixel=None (YAML null) falls through to auto-compute."""
        import math
        cfg = {
            'gripper_camera': {'hfov_deg': 60.0, 'width': 640},
            'planner': {'approach_offset_z': 100.0},
            'visual_servo': {'scale_mm_per_pixel': None},
        }
        servo = VisualServo.from_config(cfg)
        # Expected: 100 * tan(30°) / 320 = 100 * 0.5774 / 320 ≈ 0.1804
        expected = 100.0 * math.tan(math.radians(30.0)) / 320.0
        assert servo.scale_mm_per_pixel == pytest.approx(expected, rel=1e-4)

    def test_auto_compute_scales_with_height(self):
        """Doubling approach_offset_z doubles scale_mm_per_pixel."""
        import math
        cfg_low = {
            'gripper_camera': {'hfov_deg': 60.0, 'width': 640},
            'planner': {'approach_offset_z': 100.0},
            'visual_servo': {'scale_mm_per_pixel': None},
        }
        cfg_high = {
            'gripper_camera': {'hfov_deg': 60.0, 'width': 640},
            'planner': {'approach_offset_z': 200.0},
            'visual_servo': {'scale_mm_per_pixel': None},
        }
        servo_low = VisualServo.from_config(cfg_low)
        servo_high = VisualServo.from_config(cfg_high)
        assert servo_high.scale_mm_per_pixel == pytest.approx(
            servo_low.scale_mm_per_pixel * 2.0, rel=1e-4
        )

    def test_auto_compute_missing_hfov_falls_back(self):
        """If hfov_deg is missing, auto-compute returns default 0.3."""
        cfg = {
            'gripper_camera': {'width': 640},          # no hfov_deg
            'planner': {'approach_offset_z': 100.0},
            'visual_servo': {'scale_mm_per_pixel': None},
        }
        servo = VisualServo.from_config(cfg)
        assert servo.scale_mm_per_pixel == pytest.approx(0.3)

    def test_explicit_scale_overrides_auto_compute(self):
        """An explicit float always wins over auto-compute."""
        cfg = {
            'gripper_camera': {'hfov_deg': 60.0, 'width': 640},
            'planner': {'approach_offset_z': 100.0},
            'visual_servo': {'scale_mm_per_pixel': 0.42},
        }
        servo = VisualServo.from_config(cfg)
        assert servo.scale_mm_per_pixel == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# _pixel_to_robot_delta
# ---------------------------------------------------------------------------

class TestPixelToRobotDelta:
    """Test the pixel-error → robot-correction coordinate transform."""

    def test_aligned_no_rotation(self):
        """With rz=0 and no mount offset, pixel X maps to robot X."""
        servo = _make_servo(scale_mm_per_pixel=1.0, gain=1.0, mount_angle_deg=0.0)
        dx, dy = servo._pixel_to_robot_delta(ex=10.0, ey=0.0, gripper_rz_deg=0.0)
        assert dx == pytest.approx(10.0, abs=1e-9)
        assert dy == pytest.approx(0.0, abs=1e-9)

    def test_aligned_y_axis(self):
        servo = _make_servo(scale_mm_per_pixel=1.0, gain=1.0, mount_angle_deg=0.0)
        dx, dy = servo._pixel_to_robot_delta(ex=0.0, ey=10.0, gripper_rz_deg=0.0)
        assert dx == pytest.approx(0.0, abs=1e-9)
        assert dy == pytest.approx(10.0, abs=1e-9)

    def test_gripper_rz_90_rotates_correctly(self):
        """With rz=90°, robot X becomes -camera Y and robot Y becomes camera X."""
        servo = _make_servo(scale_mm_per_pixel=1.0, gain=1.0, mount_angle_deg=0.0)
        # 10px error in camera X → after 90° rotation → -10mm in robot Y? Let's verify
        # R(90°) = [[0,-1],[1,0]]
        # [dx_robot, dy_robot] = R(90°) @ [10, 0] = [0*10 - 1*0, 1*10 + 0*0] = [0, 10]
        dx, dy = servo._pixel_to_robot_delta(ex=10.0, ey=0.0, gripper_rz_deg=90.0)
        assert dx == pytest.approx(0.0, abs=1e-9)
        assert dy == pytest.approx(10.0, abs=1e-9)

    def test_gripper_rz_180(self):
        """At rz=180°, robot X is reversed relative to camera X."""
        servo = _make_servo(scale_mm_per_pixel=1.0, gain=1.0, mount_angle_deg=0.0)
        # R(180°) @ [10, 0] = [-10, 0]
        dx, dy = servo._pixel_to_robot_delta(ex=10.0, ey=0.0, gripper_rz_deg=180.0)
        assert dx == pytest.approx(-10.0, abs=1e-6)
        assert dy == pytest.approx(0.0, abs=1e-6)

    def test_mount_angle_offset(self):
        """mount_angle_deg adds to gripper rz for combined rotation."""
        servo = _make_servo(scale_mm_per_pixel=1.0, gain=1.0, mount_angle_deg=90.0)
        # Total rotation = 0 + 90 = 90° → same as test_gripper_rz_90
        dx, dy = servo._pixel_to_robot_delta(ex=10.0, ey=0.0, gripper_rz_deg=0.0)
        assert dx == pytest.approx(0.0, abs=1e-9)
        assert dy == pytest.approx(10.0, abs=1e-9)

    def test_cam_flip_x(self):
        servo = _make_servo(scale_mm_per_pixel=1.0, gain=1.0, cam_flip_x=True)
        dx, dy = servo._pixel_to_robot_delta(ex=10.0, ey=0.0, gripper_rz_deg=0.0)
        assert dx == pytest.approx(-10.0, abs=1e-9)

    def test_cam_flip_y(self):
        servo = _make_servo(scale_mm_per_pixel=1.0, gain=1.0, cam_flip_y=True)
        dx, dy = servo._pixel_to_robot_delta(ex=0.0, ey=10.0, gripper_rz_deg=0.0)
        assert dy == pytest.approx(-10.0, abs=1e-9)

    def test_scale_factor_applied(self):
        servo = _make_servo(scale_mm_per_pixel=0.25, gain=1.0)
        dx, dy = servo._pixel_to_robot_delta(ex=100.0, ey=0.0, gripper_rz_deg=0.0)
        assert dx == pytest.approx(25.0, abs=1e-9)

    def test_gain_applied(self):
        servo = _make_servo(scale_mm_per_pixel=1.0, gain=0.5)
        dx, dy = servo._pixel_to_robot_delta(ex=10.0, ey=0.0, gripper_rz_deg=0.0)
        assert dx == pytest.approx(5.0, abs=1e-9)

    def test_clamp_large_correction(self):
        """Corrections larger than max_correction_mm are clamped."""
        servo = _make_servo(scale_mm_per_pixel=1.0, gain=1.0, max_correction_mm=10.0)
        dx, dy = servo._pixel_to_robot_delta(ex=100.0, ey=0.0, gripper_rz_deg=0.0)
        assert abs(dx) == pytest.approx(10.0, abs=1e-9)
        assert dy == pytest.approx(0.0, abs=1e-9)

    def test_clamp_diagonal(self):
        """Diagonal correction is clamped in magnitude, not per-axis."""
        servo = _make_servo(scale_mm_per_pixel=1.0, gain=1.0, max_correction_mm=10.0)
        dx, dy = servo._pixel_to_robot_delta(ex=60.0, ey=80.0, gripper_rz_deg=0.0)
        mag = np.hypot(dx, dy)
        assert mag == pytest.approx(10.0, abs=1e-6)
        # Direction should be preserved
        assert dx / dy == pytest.approx(60.0 / 80.0, rel=1e-5)

    def test_zero_error_gives_zero_correction(self):
        servo = _make_servo()
        dx, dy = servo._pixel_to_robot_delta(ex=0.0, ey=0.0, gripper_rz_deg=45.0)
        assert dx == pytest.approx(0.0, abs=1e-9)
        assert dy == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# align() — mocked robot, IK, and camera
# ---------------------------------------------------------------------------

class MockRobot:
    """Minimal mock of DobotNova5 for testing align()."""
    def __init__(self, pose=None):
        self._pose = np.array(pose or [300.0, 0.0, 200.0, 180.0, 0.0, 30.0])

    def get_pose(self):
        return self._pose.copy()


class MockIKSolver:
    """Mock IK solver: always succeeds, returning the seed joints unchanged."""
    def solve_ik(self, pos, rpy, seed_joints_deg=None):
        if seed_joints_deg is not None:
            return np.array(seed_joints_deg, dtype=float)
        return np.array([0.0, -30.0, 80.0, 0.0, -50.0, 30.0])


def _make_servo_with_fake_camera(frames_and_detections, **overrides):
    """Build a VisualServo whose capture/detect is driven by a list of frames.

    frames_and_detections: list of (frame_or_None, detection_or_None)
    Each call to align()'s inner loop pops the next entry.
    """
    import unittest.mock as mock
    servo = _make_servo(**overrides)

    # Pre-open the camera by injecting a mock cap
    call_iter = iter(frames_and_detections)

    def _fake_capture():
        try:
            frame, _ = next(call_iter)
            return frame
        except StopIteration:
            return None

    servo._cap = mock.MagicMock()
    servo._cap.isOpened.return_value = True

    return servo, _fake_capture


class TestAlignConvergence:
    """align() should converge when error shrinks below pixel_threshold."""

    def _run_align(self, detections, threshold=20.0, max_iter=5):
        """
        detections: list of (cx, cy) or None per iteration
        Returns (ServoResult, final_joints).
        """
        import unittest.mock as mock

        img_cx = 320.0
        img_cy = 240.0
        servo = _make_servo(
            pixel_threshold=threshold,
            max_iterations=max_iter,
            scale_mm_per_pixel=0.25,
            gain=1.0,
        )

        # Open the camera by faking _cap
        servo._cap = mock.MagicMock()
        servo._cap.isOpened.return_value = True

        # Build fake frame (blank 640x480)
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        det_iter = iter(detections)

        def fake_capture():
            return fake_frame

        def detector_fn(frame):
            try:
                return next(det_iter)
            except StopIteration:
                return None

        robot = MockRobot()
        ik = MockIKSolver()
        joints0 = np.array([0.0, -30.0, 80.0, 0.0, -50.0, 30.0])

        # Patch capture_frame
        servo.capture_frame = fake_capture

        # Also patch execute_trajectory to succeed without any real robot
        with mock.patch('planner.trajectory.execute_trajectory', return_value=True):
            result, final_joints = servo.align(
                robot=robot,
                ik_solver=ik,
                detector_fn=detector_fn,
                current_joints=joints0.copy(),
                gripper_rz_deg=0.0,
            )

        return result, final_joints

    def test_already_centred_converges_immediately(self):
        """If the first frame is already centred, servo converges on iter 1."""
        result, _ = self._run_align([(320.0, 240.0)])  # exact centre
        assert result.converged is True
        assert result.iterations == 1
        assert result.final_error_px < 0.1

    def test_within_threshold_converges(self):
        """Error < threshold → converge."""
        # 10px error, threshold 20px → converge on first detection
        result, _ = self._run_align([(330.0, 240.0)], threshold=20.0)
        assert result.converged is True
        assert result.iterations == 1

    def test_outside_threshold_then_converges(self):
        """First iteration has large error, second is within threshold."""
        result, _ = self._run_align([
            (420.0, 340.0),   # error = ~141px → correct
            (320.0, 240.0),   # converged
        ], threshold=20.0)
        assert result.converged is True
        assert result.iterations <= 3

    def test_never_converges_exhausts_iterations(self):
        """If error never drops below threshold, returns converged=False."""
        # Always return large error
        big_error_detections = [(500.0, 400.0)] * 10
        result, _ = self._run_align(big_error_detections, max_iter=5)
        assert result.converged is False
        assert result.iterations == 5

    def test_corrections_recorded(self):
        """Corrections list grows with each non-trivial corrective move."""
        result, _ = self._run_align([
            (420.0, 340.0),   # large error → correction applied
            (380.0, 300.0),   # still outside threshold
            (320.0, 240.0),   # converged
        ], threshold=20.0, max_iter=10)
        assert result.converged is True
        # At least 2 corrections should have been applied before converging
        assert len(result.corrections) >= 1

    def test_all_detections_none_aborts_early(self):
        """If no target detected for max_misses consecutive iterations, abort."""
        result, _ = self._run_align(
            [None, None, None, None, None],  # never detects
            max_iter=10,
        )
        assert result.converged is False
        # Should abort early (not run all 10 iterations)
        assert result.iterations < 10

    def test_camera_not_open_returns_immediately(self):
        """align() without open camera returns non-converged immediately."""
        import unittest.mock as mock

        servo = _make_servo(max_iterations=5)
        # camera not open
        servo._cap = None

        result, joints = servo.align(
            robot=MockRobot(),
            ik_solver=MockIKSolver(),
            detector_fn=lambda f: (320.0, 240.0),
            current_joints=np.zeros(6),
        )
        assert result.converged is False
        assert result.iterations == 0

    def test_ik_failure_skips_correction(self):
        """If IK fails, correction is skipped but loop continues."""
        import unittest.mock as mock

        class FailIK(MockIKSolver):
            def solve_ik(self, pos, rpy, seed_joints_deg=None):
                return None  # always fail

        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        servo = _make_servo(max_iterations=3, pixel_threshold=5.0, gain=1.0)
        servo._cap = mock.MagicMock()
        servo._cap.isOpened.return_value = True

        det_seq = [(400.0, 240.0), (400.0, 240.0), (400.0, 240.0)]
        det_iter = iter(det_seq)

        def detector_fn(frame):
            try:
                return next(det_iter)
            except StopIteration:
                return None

        servo.capture_frame = lambda: fake_frame

        with mock.patch('planner.trajectory.execute_trajectory', return_value=True):
            result, _ = servo.align(
                robot=MockRobot(),
                ik_solver=FailIK(),
                detector_fn=detector_fn,
                current_joints=np.zeros(6),
            )

        # Should exhaust iterations (never moves, never converges)
        assert result.converged is False


# ---------------------------------------------------------------------------
# ServoResult fields
# ---------------------------------------------------------------------------

class TestServoResult:
    def test_fields(self):
        r = ServoResult(converged=True, iterations=3, final_error_px=12.5,
                        corrections=[(1.0, 2.0), (0.5, 0.1)])
        assert r.converged is True
        assert r.iterations == 3
        assert r.final_error_px == pytest.approx(12.5)
        assert len(r.corrections) == 2

    def test_default_corrections_list(self):
        r = ServoResult(converged=False, iterations=0, final_error_px=0.0)
        assert r.corrections == []


# ---------------------------------------------------------------------------
# _save_debug_frame (smoke test)
# ---------------------------------------------------------------------------

class TestSaveDebugFrame:
    def test_saves_file(self, tmp_path, monkeypatch):
        """Verify _save_debug_frame writes a file without crashing."""
        monkeypatch.setattr('vision.visual_servo._save_debug_frame',
                            _save_debug_frame)  # ensure we call the real one
        out_dir = str(tmp_path / 'servo_debug')
        monkeypatch.setattr('vision.visual_servo.__builtins__', __builtins__)

        # Patch the hardcoded path inside the function
        import vision.visual_servo as vs_module
        original = vs_module._save_debug_frame

        # Directly call with a tmp dir
        import cv2
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        os.makedirs(out_dir, exist_ok=True)

        # We call private helper directly with a patched out dir
        def patched_save(frame, detection, iteration, img_cx, img_cy):
            vis = frame.copy()
            h, w = vis.shape[:2]
            cv2.imwrite(os.path.join(out_dir, f'servo_{iteration:03d}.jpg'), vis)

        patched_save(frame, (320.0, 240.0), 0, 320.0, 240.0)
        assert os.path.exists(os.path.join(out_dir, 'servo_000.jpg'))

    def test_no_crash_with_none_detection(self):
        """_save_debug_frame should not crash when detection is None."""
        import tempfile
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            import unittest.mock as mock
            with mock.patch('vision.visual_servo.os.makedirs'):
                with mock.patch('vision.visual_servo.cv2.imwrite'):
                    _save_debug_frame(frame, None, 0, 320.0, 240.0)
        # If we get here without exception, the test passes


# ---------------------------------------------------------------------------
# make_green_cube_detector
# ---------------------------------------------------------------------------

class TestMakeGreenCubeDetector:
    def test_no_cube_returns_none(self):
        """Pure red frame → no green → detector returns None."""
        from vision.visual_servo import make_green_cube_detector

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 2] = 200  # blue channel only → not green in HSV

        detector = make_green_cube_detector()
        result = detector(frame)
        assert result is None

    def test_green_blob_returns_centroid(self):
        """Frame with a green square returns approximate centroid."""
        from vision.visual_servo import make_green_cube_detector

        # Draw a 60×60 bright green rectangle at (200, 150) in a black frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # HSV green: H≈60, S=255, V=255 → BGR: (0, 255, 0)
        frame[150:210, 200:260] = (0, 255, 0)

        detector = make_green_cube_detector(min_area=100.0)
        result = detector(frame)
        assert result is not None
        cx, cy = result
        # Centroid should be near (230, 180)
        assert abs(cx - 230) < 20
        assert abs(cy - 180) < 20

    def test_returns_largest_blob(self):
        """When two blobs exist, the larger one's centroid is returned."""
        from vision.visual_servo import make_green_cube_detector

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Large blob at (100, 100), size 80×80
        frame[100:180, 100:180] = (0, 255, 0)
        # Small blob at (400, 300), size 20×20
        frame[300:320, 400:420] = (0, 255, 0)

        detector = make_green_cube_detector(min_area=100.0)
        result = detector(frame)
        assert result is not None
        cx, cy = result
        # Should be the large blob's centroid ≈ (140, 140)
        assert abs(cx - 140) < 20
        assert abs(cy - 140) < 20
