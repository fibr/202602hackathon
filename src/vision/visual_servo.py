"""Closed-loop visual servoing using a gripper-mounted downward camera.

Implements image-based visual servoing (IBVS) to precisely center the
robot end-effector over a detected target by reading from a camera mounted
on the gripper (cam_8 = /dev/video8).

The control law is proportional:

    delta_robot_xy = gain * scale_mm_per_px * R(rz + mount_angle) @ pixel_error

where:
  - pixel_error = (target_cx - img_cx, target_cy - img_cy) in pixels
  - R(angle) is a 2D rotation by the combined gripper rz + camera mount offset
  - scale_mm_per_px converts pixels to mm at the current working height

Usage pattern (in main.py):
    servo = VisualServo.from_config(config)
    servo.open_camera()
    try:
        result, joints = servo.align(robot, ik_solver, detector_fn,
                                     current_joints, gripper_rz_deg)
        if not result.converged:
            log.warning("Servo did not converge; proceeding with best estimate")
    finally:
        servo.close_camera()
"""

import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from logger import get_logger

log = get_logger('visual_servo')


@dataclass
class ServoResult:
    """Result of a single visual servo run."""
    converged: bool           # True if final pixel error < pixel_threshold
    iterations: int           # Total iterations executed
    final_error_px: float     # Final pixel error magnitude (Euclidean)
    corrections: list = field(default_factory=list)  # [(dx_mm, dy_mm), ...]


class VisualServo:
    """Image-based visual servo using a downward gripper camera.

    Opens /dev/video{cam_index} and repeatedly:
      1. Captures a frame
      2. Calls detector_fn(frame) -> (cx, cy) or None
      3. Computes pixel error from image centre
      4. Converts error to robot-base-frame XY correction
      5. Executes a small IK-based corrective move
      6. Repeats until error < pixel_threshold or max_iterations

    Coordinate conventions
    ----------------------
    Image frame : X right, Y down  (standard OpenCV / pixel convention)
    Camera frame: same as image but scaled to mm, centred at (0,0)
    Gripper frame: rotates with joint 6 (rz).  When rz=0 and
        mount_angle_deg=0, camera-X == robot-X and camera-Y == robot-Y.
        Adjust mount_angle_deg if the camera is physically rotated relative
        to the gripper.
    Sign flips (cam_flip_x / cam_flip_y): account for mirror mounting.

    Args:
        cam_index: V4L2 device index (e.g. 8 for /dev/video8).
        scale_mm_per_pixel: mm per pixel at the current working height.
            Typical values: ~0.18 mm/px at 100 mm, ~0.36 mm/px at 200 mm
            (for a 60° HFOV camera at 640×480).
        mount_angle_deg: Angle (degrees) the camera X-axis makes with the
            gripper X-axis (CCW positive when viewed from above).  0 = aligned.
        cam_flip_x: Mirror the X component of the pixel error before
            converting to mm (set True if camera is physically mirrored).
        cam_flip_y: Mirror the Y component of the pixel error.
        max_iterations: Upper bound on servo iterations.
        pixel_threshold: Convergence criterion (pixels, Euclidean).
        gain: Proportional gain in (0, 1].  Lower → more stable, slower.
        settle_s: Delay after each corrective move (seconds).
        cam_width: Capture width in pixels.
        cam_height: Capture height in pixels.
        max_correction_mm: Per-iteration XY correction is clipped to this
            magnitude (mm). Prevents runaway from bad detections.
        save_debug: If True, annotated frames are written to
            /tmp/servo_debug/servo_NNN.jpg.
    """

    def __init__(
        self,
        cam_index: int = 8,
        scale_mm_per_pixel: float = 0.3,
        mount_angle_deg: float = 0.0,
        cam_flip_x: bool = False,
        cam_flip_y: bool = False,
        max_iterations: int = 10,
        pixel_threshold: float = 20.0,
        gain: float = 0.6,
        settle_s: float = 0.3,
        cam_width: int = 640,
        cam_height: int = 480,
        max_correction_mm: float = 30.0,
        save_debug: bool = False,
        autofocus: bool = False,
    ):
        self.cam_index = cam_index
        self.scale_mm_per_pixel = scale_mm_per_pixel
        self.mount_angle_deg = mount_angle_deg
        self.cam_flip_x = cam_flip_x
        self.cam_flip_y = cam_flip_y
        self.max_iterations = max_iterations
        self.pixel_threshold = pixel_threshold
        self.gain = gain
        self.settle_s = settle_s
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.max_correction_mm = max_correction_mm
        self.save_debug = save_debug
        self.autofocus = autofocus

        self._cap: Optional[cv2.VideoCapture] = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def _compute_scale_from_config(cls, config: dict) -> float:
        """Estimate scale_mm_per_pixel from camera FOV and pre-grasp height.

        Uses the geometry of a pinhole camera looking straight down:

            scale = Z_mm * tan(hfov / 2) / (img_width / 2)

        where:
          - Z_mm      = ``planner.approach_offset_z`` (mm above the target)
          - hfov      = ``gripper_camera.hfov_deg`` converted to radians
          - img_width = ``gripper_camera.width`` in pixels

        Falls back to 0.3 mm/px if neither the planner height nor the
        camera FOV can be determined from config.

        Args:
            config: Full config dict returned by load_config().

        Returns:
            Estimated scale_mm_per_pixel (float, mm per pixel).
        """
        gc = config.get('gripper_camera', {})
        planner = config.get('planner', {})

        z_mm = planner.get('approach_offset_z')
        hfov_deg = gc.get('hfov_deg')
        width = gc.get('width', 640)

        if z_mm is None or hfov_deg is None:
            log.debug(
                "[SERVO] scale_mm_per_pixel: missing approach_offset_z or hfov_deg "
                "in config — falling back to default 0.3 mm/px"
            )
            return 0.3

        scale = z_mm * math.tan(math.radians(hfov_deg / 2.0)) / (width / 2.0)
        log.info(
            f"[SERVO] Auto-computed scale_mm_per_pixel={scale:.4f} "
            f"(Z={z_mm}mm, hfov={hfov_deg}°, width={width}px)"
        )
        return scale

    @classmethod
    def from_config(cls, config: dict) -> "VisualServo":
        """Construct a VisualServo from the project config dict.

        Reads the ``visual_servo`` and ``gripper_camera`` sections from
        ``config/robot_config.yaml`` (merged into *config* by load_config).

        ``scale_mm_per_pixel`` is resolved in priority order:

        1. Explicit float in ``visual_servo.scale_mm_per_pixel`` — used as-is.
        2. ``null`` / missing in config — auto-computed from
           ``gripper_camera.hfov_deg`` and ``planner.approach_offset_z``
           using the pinhole formula
           ``scale = Z_mm * tan(hfov/2) / (width/2)``.
        3. Neither available — hard-coded fallback of 0.3 mm/px.

        Args:
            config: Full config dict returned by load_config().

        Returns:
            VisualServo instance.
        """
        gc = config.get('gripper_camera', {})
        vs = config.get('visual_servo', {})

        # Resolve scale: explicit value takes priority; None/missing → auto
        raw_scale = vs.get('scale_mm_per_pixel')
        if raw_scale is None:
            scale = cls._compute_scale_from_config(config)
        else:
            scale = float(raw_scale)

        return cls(
            cam_index=gc.get('device_index', 8),
            scale_mm_per_pixel=scale,
            mount_angle_deg=vs.get('mount_angle_deg', 0.0),
            cam_flip_x=vs.get('cam_flip_x', False),
            cam_flip_y=vs.get('cam_flip_y', False),
            max_iterations=vs.get('max_iterations', 10),
            pixel_threshold=vs.get('pixel_threshold', 20.0),
            gain=vs.get('gain', 0.6),
            settle_s=vs.get('settle_s', 0.3),
            cam_width=gc.get('width', 640),
            cam_height=gc.get('height', 480),
            max_correction_mm=vs.get('max_correction_mm', 30.0),
            save_debug=vs.get('save_debug', False),
        )

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------

    def open_camera(self):
        """Open the gripper camera device.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open gripper camera /dev/video{self.cam_index}"
            )
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if self.autofocus else 0)
        # Warm up auto-exposure
        for _ in range(8):
            cap.read()
        self._cap = cap
        log.info(
            f"Gripper camera opened: /dev/video{self.cam_index} "
            f"{self.cam_width}x{self.cam_height}"
        )

    def close_camera(self):
        """Release the gripper camera device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            log.debug("Gripper camera closed")

    @property
    def camera_open(self) -> bool:
        """True if the gripper camera is currently open."""
        return self._cap is not None and self._cap.isOpened()

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture one BGR frame from the gripper camera.

        Returns:
            BGR frame array, or None on failure.
        """
        if self._cap is None:
            return None
        for _ in range(3):
            ret, frame = self._cap.read()
            if ret and frame is not None:
                return frame
        log.warning("Failed to capture frame from gripper camera")
        return None

    # ------------------------------------------------------------------
    # Coordinate transform
    # ------------------------------------------------------------------

    def _pixel_to_robot_delta(
        self,
        ex: float,
        ey: float,
        gripper_rz_deg: float,
    ) -> Tuple[float, float]:
        """Convert a pixel error to a robot-base-frame XY correction (mm).

        Steps:
          1. Apply sign flips for camera mirror mounting.
          2. Scale from pixels to mm (camera frame).
          3. Rotate by (gripper_rz + mount_angle) to align with robot base.
          4. Apply proportional gain and clamp to max_correction_mm.

        Args:
            ex: Pixel error X = target_cx - image_cx (positive = target right).
            ey: Pixel error Y = target_cy - image_cy (positive = target below).
            gripper_rz_deg: Current gripper rz joint angle in degrees.

        Returns:
            (dx_mm, dy_mm): Required TCP correction in robot base frame (mm).
        """
        # Apply optional axis flips (for mirrored or inverted mounts)
        if self.cam_flip_x:
            ex = -ex
        if self.cam_flip_y:
            ey = -ey

        # Scale to mm in camera/gripper frame
        dx_cam = ex * self.scale_mm_per_pixel * self.gain
        dy_cam = ey * self.scale_mm_per_pixel * self.gain

        # Rotate from gripper frame to robot base frame
        # The gripper rz describes how the gripper (and camera) are rotated
        # relative to the robot base.
        angle_rad = np.radians(gripper_rz_deg + self.mount_angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        dx_robot = cos_a * dx_cam - sin_a * dy_cam
        dy_robot = sin_a * dx_cam + cos_a * dy_cam

        # Clamp to prevent runaway from bad detections
        mag = np.hypot(dx_robot, dy_robot)
        if mag > self.max_correction_mm:
            scale = self.max_correction_mm / mag
            dx_robot *= scale
            dy_robot *= scale
            log.debug(
                f"[SERVO] Correction clamped from {mag:.1f}mm "
                f"to {self.max_correction_mm:.1f}mm"
            )

        return dx_robot, dy_robot

    # ------------------------------------------------------------------
    # Main servo loop
    # ------------------------------------------------------------------

    def align(
        self,
        robot,
        ik_solver,
        detector_fn: Callable[[np.ndarray], Optional[Tuple[float, float]]],
        current_joints: np.ndarray,
        gripper_rz_deg: float = 0.0,
    ) -> Tuple[ServoResult, np.ndarray]:
        """Run the visual servo alignment loop.

        Repeatedly captures frames, detects the target, and applies small
        corrective robot moves until the target is centred in the gripper
        camera image (within *pixel_threshold* pixels) or *max_iterations*
        is reached.

        The camera must already be open (call open_camera() first).

        Args:
            robot: Connected and enabled DobotNova5 instance.
            ik_solver: IKSolver instance for the Nova5.
            detector_fn: Callable(frame: np.ndarray) -> Optional[(cx, cy)].
                Returns the target centroid in pixels, or None if not found.
            current_joints: Current joint angles in degrees (6-element array).
            gripper_rz_deg: Current gripper rz in degrees (used for the
                camera→robot-base coordinate transform).

        Returns:
            Tuple of (ServoResult, final_joint_angles).
        """
        from planner.trajectory import execute_trajectory

        if not self.camera_open:
            log.error("[SERVO] Camera not open — call open_camera() first")
            return ServoResult(converged=False, iterations=0,
                               final_error_px=float('inf')), current_joints

        img_cx = self.cam_width / 2.0
        img_cy = self.cam_height / 2.0

        corrections: list = []
        final_error = float('inf')
        miss_count = 0
        max_misses = 3  # consecutive detection failures before giving up
        actual_iterations = 0  # track real count for early-exit cases

        log.info(
            f"[SERVO] Start: max_iters={self.max_iterations}, "
            f"threshold={self.pixel_threshold}px, "
            f"scale={self.scale_mm_per_pixel}mm/px, "
            f"gain={self.gain}, gripper_rz={gripper_rz_deg:.1f}°"
        )

        for iteration in range(self.max_iterations):
            actual_iterations = iteration + 1
            # --- Capture ---
            frame = self.capture_frame()
            if frame is None:
                log.warning(f"[SERVO] iter {iteration}: no frame")
                miss_count += 1
                if miss_count >= max_misses:
                    log.error("[SERVO] Too many frame-capture failures; aborting")
                    break
                time.sleep(0.1)
                continue

            # --- Detect ---
            detection = detector_fn(frame)

            if self.save_debug:
                _save_debug_frame(
                    frame, detection, iteration, img_cx, img_cy
                )

            if detection is None:
                miss_count += 1
                log.warning(
                    f"[SERVO] iter {iteration}: target not detected "
                    f"(miss {miss_count}/{max_misses})"
                )
                if miss_count >= max_misses:
                    log.warning("[SERVO] Too many consecutive misses; aborting")
                    break
                time.sleep(0.1)
                continue
            miss_count = 0  # reset on successful detection

            target_cx, target_cy = detection
            ex = target_cx - img_cx
            ey = target_cy - img_cy
            error_px = float(np.hypot(ex, ey))
            final_error = error_px

            log.info(
                f"[SERVO] iter {iteration+1}/{self.max_iterations}: "
                f"target=({target_cx:.0f},{target_cy:.0f}) "
                f"err=({ex:+.1f},{ey:+.1f})px  |err|={error_px:.1f}px"
            )

            # --- Convergence check ---
            if error_px < self.pixel_threshold:
                log.info(
                    f"[SERVO] Converged after {iteration+1} iter(s)! "
                    f"error={error_px:.1f}px < {self.pixel_threshold}px"
                )
                return ServoResult(
                    converged=True,
                    iterations=iteration + 1,
                    final_error_px=error_px,
                    corrections=corrections,
                ), current_joints

            # --- Compute correction ---
            dx_mm, dy_mm = self._pixel_to_robot_delta(ex, ey, gripper_rz_deg)

            if abs(dx_mm) < 0.3 and abs(dy_mm) < 0.3:
                log.debug("[SERVO] Correction negligible (<0.3mm); skipping move")
                continue

            log.info(f"[SERVO] Correction: dx={dx_mm:+.2f}mm dy={dy_mm:+.2f}mm")
            corrections.append((dx_mm, dy_mm))

            # --- Apply correction ---
            try:
                pose = robot.get_pose()   # [x, y, z, rx, ry, rz] mm / deg
                new_x = pose[0] + dx_mm
                new_y = pose[1] + dy_mm
                new_z = pose[2]           # hold Z constant
                new_rpy = np.array([pose[3], pose[4], pose[5]])

                new_joints = ik_solver.solve_ik(
                    np.array([new_x, new_y, new_z]),
                    new_rpy,
                    seed_joints_deg=current_joints,
                )

                if new_joints is None:
                    log.warning("[SERVO] IK failed for corrective pose; skipping")
                    continue

                ok = execute_trajectory(
                    robot, current_joints, new_joints, max_step_deg=2.0
                )
                if ok:
                    current_joints = new_joints
                else:
                    log.warning("[SERVO] Corrective trajectory failed")

            except Exception as exc:
                log.error(f"[SERVO] Error during corrective move: {exc}")
                break

            time.sleep(self.settle_s)

        # Loop exhausted (or broken out of) without convergence
        log.warning(
            f"[SERVO] Finished without convergence: "
            f"{actual_iterations} iter(s), "
            f"final_error={final_error:.1f}px"
        )
        return ServoResult(
            converged=False,
            iterations=actual_iterations,
            final_error_px=final_error,
            corrections=corrections,
        ), current_joints


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def _save_debug_frame(
    frame: np.ndarray,
    detection: Optional[Tuple[float, float]],
    iteration: int,
    img_cx: float,
    img_cy: float,
):
    """Write an annotated servo frame to /tmp/servo_debug/servo_NNN.jpg."""
    vis = frame.copy()
    h, w = vis.shape[:2]
    cx, cy = int(img_cx), int(img_cy)

    # Image-centre crosshair
    cv2.line(vis, (cx - 25, cy), (cx + 25, cy), (0, 255, 255), 1)
    cv2.line(vis, (cx, cy - 25), (cx, cy + 25), (0, 255, 255), 1)
    cv2.putText(vis, "centre", (cx + 4, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    if detection is not None:
        tx, ty = int(detection[0]), int(detection[1])
        # Target circle
        cv2.circle(vis, (tx, ty), 10, (0, 0, 255), 2)
        cv2.circle(vis, (tx, ty), 2, (0, 0, 255), -1)
        # Line from centre to target
        cv2.line(vis, (cx, cy), (tx, ty), (0, 128, 255), 1)
        # Error text
        ex = tx - cx
        ey = ty - cy
        err_px = np.hypot(ex, ey)
        cv2.putText(
            vis,
            f"err=({ex:+d},{ey:+d}) {err_px:.0f}px",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
        )
    else:
        cv2.putText(vis, "NO DETECTION", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.putText(vis, f"Iter {iteration}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    out_dir = '/tmp/servo_debug'
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f'servo_{iteration:03d}.jpg'), vis)


# ---------------------------------------------------------------------------
# Convenience detector wrappers
# ---------------------------------------------------------------------------

def make_green_cube_detector(
    min_area: float = 200.0,
    hsv_low: Optional[np.ndarray] = None,
    hsv_high: Optional[np.ndarray] = None,
) -> Callable[[np.ndarray], Optional[Tuple[float, float]]]:
    """Return a detector_fn that finds the largest green cube in a frame.

    Wraps ``vision.green_cube_detector.detect_green_cubes`` for use as the
    ``detector_fn`` argument to ``VisualServo.align``.

    Args:
        min_area: Minimum contour area to accept (pixels^2).
        hsv_low:  Lower HSV bound (default: [35, 60, 60]).
        hsv_high: Upper HSV bound (default: [85, 255, 255]).

    Returns:
        Callable(frame) -> (cx, cy) or None.
    """
    # Import here so the module can be imported without cv2-heavy vision deps
    from vision.green_cube_detector import detect_green_cubes

    def _detect(frame: np.ndarray) -> Optional[Tuple[float, float]]:
        dets, _ = detect_green_cubes(
            frame, hsv_low=hsv_low, hsv_high=hsv_high, min_area=min_area
        )
        if not dets:
            return None
        # Return centroid of the largest detection
        best = dets[0]  # sorted by area descending
        return float(best.cx), float(best.cy)

    return _detect
