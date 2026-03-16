"""Fetch Game Controller — state machine for cube pick-and-place.

Orchestrates the full pipeline:
  IDLE → DETECT → APPROACH → REFINE → GRASP → LIFT → TRANSPORT → PLACE → DONE

Each state transition can be advanced manually (step-by-step) or automatically.
The controller is designed to run from a GUI thread with periodic update() calls.

All coordinates are in mm (base frame).
"""

import enum
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable

import cv2
import numpy as np

from logger import get_logger

log = get_logger('fetch_game')


class FetchState(enum.Enum):
    """States of the fetch game state machine."""
    IDLE = 'IDLE'
    DETECT = 'DETECT'           # Waiting for cube detection from overview cam
    APPROACH = 'APPROACH'       # Moving arm above target cube
    REFINE = 'REFINE'           # Using gripper cam to refine position
    OPEN_GRIPPER = 'OPEN_GRIP'  # Opening gripper before descent
    DESCEND = 'DESCEND'         # Lowering to grasp height
    GRASP = 'GRASP'             # Closing gripper
    LIFT = 'LIFT'               # Lifting cube
    TRANSPORT = 'TRANSPORT'     # Moving to place position
    PLACE = 'PLACE'             # Lowering to place height and opening gripper
    RETRACT = 'RETRACT'         # Moving back to safe position
    DONE = 'DONE'
    ERROR = 'ERROR'


@dataclass
class FetchGameState:
    """Observable state of the fetch game, for GUI display."""
    state: FetchState = FetchState.IDLE
    status_text: str = 'Ready'
    # Detection
    num_cubes_detected: int = 0
    target_cube_idx: int = -1
    target_cube_px: Optional[tuple] = None   # (cx, cy) in overview image
    target_cube_3d: Optional[np.ndarray] = None  # [x,y,z] mm in base frame
    # Refinement
    gripper_cam_detection: Optional[tuple] = None  # (cx, cy) in gripper cam
    gripper_cam_frame: Optional[np.ndarray] = None  # latest gripper cam frame (BGR)
    refined_pos_3d: Optional[np.ndarray] = None
    estimated_yaw_deg: float = 0.0
    refine_error_px: float = float('inf')
    # Target
    place_target_3d: Optional[np.ndarray] = None  # [x,y,z] mm in base frame
    # Progress
    auto_mode: bool = False
    step_count: int = 0


class FetchGameController:
    """State machine for the fetch game.

    Designed to be driven from a GUI — call update(frame) each camera tick
    to keep detection running, and step() or start_auto() to advance states.

    Args:
        app: The UnifiedPyQtApp instance (for robot, camera, config access).
        hover_height_mm: Height above cube for approach/refine (mm).
        grasp_height_mm: Height at which to close gripper (mm).
        lift_height_mm: Height to lift after grasping (mm).
        place_pos_mm: [x, y, z] target position for placing (mm).
        cube_selection_mode: 'largest', 'closest_to_center', 'closest_to_click'.
    """

    def __init__(self, app,
                 hover_height_mm: float = 40.0,
                 grasp_height_mm: float = 8.0,
                 lift_height_mm: float = 80.0,
                 place_pos_mm: np.ndarray = None,
                 cube_selection_mode: str = 'largest'):
        self.app = app
        self.hover_height_mm = hover_height_mm
        self.grasp_height_mm = grasp_height_mm
        self.lift_height_mm = lift_height_mm
        self.place_pos_mm = place_pos_mm if place_pos_mm is not None else np.array([100.0, 100.0, 12.0])
        self.cube_selection_mode = cube_selection_mode

        self.state = FetchGameState()
        self._click_xy = None  # For click-to-select mode
        self._transform = None
        self._solver = None
        self._worker_thread = None
        self._abort = False
        self._speed = 150

        # Lock protecting all FetchGameState field reads/writes.
        # RLock allows the same thread to re-acquire (e.g. step() -> _set_state()).
        self._lock = threading.RLock()

        # Callbacks for GUI updates
        self.on_state_changed: Optional[Callable] = None

    def ensure_ready(self):
        """Lazy-init calibration transform and IK solver."""
        if self._transform is None:
            from config_loader import config_path
            import os
            cal_path = config_path('calibration.yaml')
            if os.path.exists(cal_path):
                from calibration import CoordinateTransform
                self._transform = CoordinateTransform()
                self._transform.load(cal_path)

        if self._solver is None:
            try:
                from kinematics.arm101_ik_solver import Arm101IKSolver
                self._solver = Arm101IKSolver()
            except Exception as e:
                log.warning(f"Failed to init IK solver: {e}")

    def set_click_target(self, x: int, y: int):
        """Set click position for 'closest_to_click' selection mode."""
        self._click_xy = (x, y)
        self.cube_selection_mode = 'closest_to_click'
        log.info(f"Click target set: ({x}, {y})")

    def set_place_target(self, pos_mm: np.ndarray):
        """Set the place target position in base frame (mm)."""
        copy = pos_mm.copy()
        self.place_pos_mm = copy
        with self._lock:
            self.state.place_target_3d = copy.copy()
        log.info(f"Place target: ({pos_mm[0]:.0f}, {pos_mm[1]:.0f}, {pos_mm[2]:.0f})")

    def update_detection(self, frame: np.ndarray):
        """Run cube detection on an overview camera frame.

        Called from the GUI camera callback. Updates state with detection info
        but doesn't advance the state machine.

        Args:
            frame: BGR frame from overview camera.

        Returns:
            (detections, target_idx): List of CubeDetection and selected target index.
        """
        from vision.green_cube_detector import detect_green_cubes, select_target_cube
        self.ensure_ready()

        # Run detection outside the lock — this is the heavy computation.
        cubes, info = detect_green_cubes(frame)

        target_idx = select_target_cube(
            cubes,
            mode=self.cube_selection_mode,
            click_xy=self._click_xy,
        )

        # Pre-compute 3D position outside the lock (no state mutation yet).
        new_cube_3d = None
        cube_px = None
        if target_idx >= 0 and target_idx < len(cubes):
            cube = cubes[target_idx]
            cube_px = (cube.cx, cube.cy)
            if self._transform and self.app.camera:
                intr = self.app.camera.intrinsics
                ray_cam = np.array([
                    (cube.cx - intr.ppx) / intr.fx,
                    (cube.cy - intr.ppy) / intr.fy,
                    1.0
                ])
                T = self._transform.T_camera_to_base
                ray_base = T[:3, :3] @ ray_cam
                origin_base = T[:3, 3]
                cube_z = 10.0  # cube center height above table
                if abs(ray_base[2]) > 1e-6:
                    s = (cube_z - origin_base[2]) / ray_base[2]
                    if s > 0:
                        new_cube_3d = origin_base + s * ray_base
                    else:
                        log.debug(f"Negative ray scale {s:.2f}, ignoring")

        # Commit detection results and check for auto-advance under the lock.
        do_step = False
        with self._lock:
            self.state.num_cubes_detected = len(cubes)
            self.state.target_cube_idx = target_idx
            if cube_px is not None:
                self.state.target_cube_px = cube_px
                self.state.target_cube_3d = new_cube_3d
                # Auto-advance: if in DETECT state with auto_mode and we have a target
                if (self.state.auto_mode
                        and self.state.state == FetchState.DETECT
                        and self.state.target_cube_3d is not None
                        and not (self._worker_thread and self._worker_thread.is_alive())):
                    do_step = True
            else:
                self.state.target_cube_px = None
                self.state.target_cube_3d = None

        # Call step() outside the lock to avoid re-entrant locking issues.
        if do_step:
            self.step()

        return cubes, target_idx

    def _set_state(self, new_state: FetchState, status: str = ''):
        """Transition to a new state (thread-safe)."""
        with self._lock:
            old = self.state.state
            self.state.state = new_state
            self.state.status_text = status or new_state.value
            self.state.step_count += 1
            log.info(f"State: {old.value} -> {new_state.value}: {status}")
            callback = self.on_state_changed
        # Invoke callback outside the lock to prevent holding it during
        # arbitrary GUI operations (avoids deadlock with GUI thread).
        if callback:
            try:
                callback(self.state)
            except Exception:
                pass

    def reset(self):
        """Reset to IDLE state (thread-safe).

        _abort is set first (no lock needed — boolean write is atomic in CPython)
        so the worker thread can see it and stop early.  The join() is done
        outside the lock to prevent a deadlock where the worker waits to acquire
        _lock (to call step()) while we hold it waiting for the worker to finish.
        """
        self._abort = True
        worker = self._worker_thread
        if worker and worker.is_alive():
            worker.join(timeout=2.0)
        with self._lock:
            self._abort = False
            self.state = FetchGameState()
            self.state.place_target_3d = self.place_pos_mm.copy()
        self._set_state(FetchState.IDLE, 'Reset — ready')

    def step(self):
        """Advance one step in the state machine (non-blocking).

        Starts a worker thread for the current state's action.
        Thread-safe: acquiring _lock ensures the state read and the
        subsequent transition are atomic (no two threads can both observe
        the same state and both trigger the same action).

        Note: reset() is intentionally called *outside* the lock to avoid a
        deadlock where the worker thread (waiting to acquire _lock to call
        step()) is being joined while _lock is held by the calling thread.
        """
        do_reset = False
        with self._lock:
            if self._worker_thread and self._worker_thread.is_alive():
                log.warning("Worker still running, ignoring step")
                return

            self._abort = False
            current = self.state.state

            if current == FetchState.IDLE:
                self._set_state(FetchState.DETECT, 'Scanning for cubes...')

            elif current == FetchState.DETECT:
                if self.state.target_cube_3d is not None:
                    self._set_state(FetchState.APPROACH, 'Moving above target cube...')
                    self._run_async(self._do_approach)
                else:
                    self.state.status_text = 'No cube detected — waiting...'

            elif current == FetchState.APPROACH:
                # After approach, try gripper cam refinement
                self._set_state(FetchState.REFINE, 'Refining with gripper camera...')
                self._run_async(self._do_refine)

            elif current == FetchState.REFINE:
                self._set_state(FetchState.OPEN_GRIPPER, 'Opening gripper...')
                self._run_async(self._do_open_gripper)

            elif current == FetchState.OPEN_GRIPPER:
                self._set_state(FetchState.DESCEND, 'Descending to grasp...')
                self._run_async(self._do_descend)

            elif current == FetchState.DESCEND:
                self._set_state(FetchState.GRASP, 'Closing gripper...')
                self._run_async(self._do_grasp)

            elif current == FetchState.GRASP:
                self._set_state(FetchState.LIFT, 'Lifting cube...')
                self._run_async(self._do_lift)

            elif current == FetchState.LIFT:
                self._set_state(FetchState.TRANSPORT, 'Transporting to target...')
                self._run_async(self._do_transport)

            elif current == FetchState.TRANSPORT:
                self._set_state(FetchState.PLACE, 'Placing cube...')
                self._run_async(self._do_place)

            elif current == FetchState.PLACE:
                self._set_state(FetchState.RETRACT, 'Retracting...')
                self._run_async(self._do_retract)

            elif current == FetchState.RETRACT:
                self._set_state(FetchState.DONE, 'Fetch complete!')

            elif current in (FetchState.DONE, FetchState.ERROR):
                # reset() joins the worker thread — must be called outside the
                # lock to avoid deadlock (worker may be trying to acquire _lock
                # to call step() while we're waiting for it to finish here).
                do_reset = True

        if do_reset:
            self.reset()

    def start_auto(self):
        """Run the full fetch sequence automatically."""
        self.reset()
        with self._lock:
            self.state.auto_mode = True
        self._set_state(FetchState.DETECT, 'Auto: scanning...')

    def stop_auto(self):
        """Stop auto mode."""
        with self._lock:
            self.state.auto_mode = False
        self._abort = True

    def _run_async(self, fn):
        """Run a blocking function in a background thread."""
        self._worker_thread = threading.Thread(target=self._worker_wrapper,
                                                args=(fn,), daemon=True)
        self._worker_thread.start()

    def _worker_wrapper(self, fn):
        """Wrapper that catches exceptions and transitions to ERROR."""
        try:
            fn()
            # In auto mode, advance to next step.
            # Read auto_mode under the lock; step() also acquires the lock.
            with self._lock:
                should_advance = self.state.auto_mode and not self._abort
            if should_advance:
                time.sleep(0.2)
                self.step()
        except Exception as e:
            with self._lock:
                state_name = self.state.state.value
            log.error(f"Fetch game error in {state_name}: {e}")
            import traceback
            traceback.print_exc()
            self._set_state(FetchState.ERROR, f'Error: {e}')

    # ------------------------------------------------------------------
    # State actions (run in worker thread)
    # ------------------------------------------------------------------

    def _move_to_position(self, pos_mm: np.ndarray, label: str = 'target'):
        """Move arm to a 3D position using position-only IK."""
        if self._abort:
            return
        robot = self.app.robot
        if robot is None or self._solver is None:
            raise RuntimeError("No robot or IK solver available")

        angles = robot.get_angles()
        if not angles or len(angles) < 6:
            raise RuntimeError(f"Cannot get joint angles for {label}")
        seed = np.array(angles[:5])
        solution = self._solver.solve_ik_position(pos_mm, seed_motor_deg=seed)

        if solution is None:
            raise RuntimeError(f"IK failed for {label}: {pos_mm}")

        cmd = list(solution) + [angles[5]]
        robot.move_joints(cmd, speed=self._speed)
        time.sleep(0.5)  # Wait for motion to complete
        log.info(f"Moved to {label}: ({pos_mm[0]:.1f}, {pos_mm[1]:.1f}, {pos_mm[2]:.1f})")

    def _do_approach(self):
        """Move arm above the target cube."""
        with self._lock:
            target = self.state.target_cube_3d
        if target is None:
            raise RuntimeError("No target cube position")

        # Snapshot target position to avoid race with detection thread.
        hover_pos = target.copy()
        hover_pos[2] = self.hover_height_mm
        self._move_to_position(hover_pos, 'approach')
        refined = hover_pos.copy()
        refined[2] = target[2]  # keep original Z for later use
        with self._lock:
            self.state.status_text = f'Hovering above cube at z={self.hover_height_mm:.0f}mm'
            self.state.refined_pos_3d = refined

    def _do_refine(self):
        """Use gripper camera to refine position over the cube.

        Opens the gripper camera, detects the cube, computes pixel error
        from center, and iteratively corrects arm position.
        """
        from vision.green_cube_detector import detect_green_cubes
        import math

        config = self.app.config or {}
        gc = config.get('gripper_camera', {})
        cam_index = gc.get('device_index', 8)
        cam_w = gc.get('width', 640)
        cam_h = gc.get('height', 480)
        hfov_deg = gc.get('hfov_deg', 61.8)
        # Camera mount rotation relative to gripper (degrees)
        refine_cfg = config.get('visual_servoing', {})
        mount_angle_deg = refine_cfg.get('mount_angle_deg', 0.0)
        mount_angle_rad = math.radians(mount_angle_deg)

        # Compute scale from current height
        z_mm = self.hover_height_mm
        scale_mm_per_px = z_mm * math.tan(math.radians(hfov_deg / 2.0)) / (cam_w / 2.0)

        robot = self.app.robot
        if robot is None:
            with self._lock:
                self.state.status_text = 'Refine: no robot, skipping'
            return

        # Try to open gripper camera
        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            log.warning(f"Cannot open gripper cam /dev/video{cam_index}, skipping refine")
            with self._lock:
                self.state.status_text = 'Gripper cam unavailable, skipping refine'
            return

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

        # Warm up
        for _ in range(8):
            cap.read()

        img_cx = cam_w / 2.0
        img_cy = cam_h / 2.0
        max_iters = 8
        gain = 0.5
        pixel_threshold = 15.0

        try:
            for iteration in range(max_iters):
                if self._abort:
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                cubes, _ = detect_green_cubes(frame, min_area=100)

                # Annotate and publish gripper cam frame for GUI display
                gc_vis = frame.copy()
                # Draw crosshair at center
                h_gc, w_gc = gc_vis.shape[:2]
                cv2.drawMarker(gc_vis, (w_gc // 2, h_gc // 2), (255, 255, 0),
                               cv2.MARKER_CROSS, 30, 1)
                if cubes:
                    from vision.green_cube_detector import annotate_frame
                    gc_vis = annotate_frame(gc_vis, cubes, target_idx=0)
                with self._lock:
                    self.state.gripper_cam_frame = gc_vis

                if not cubes:
                    with self._lock:
                        self.state.gripper_cam_detection = None
                        self.state.status_text = (
                            f'Refine iter {iteration+1}: no cube in gripper cam')
                    time.sleep(0.2)
                    continue

                cube = cubes[0]
                ex = cube.cx - img_cx
                ey = cube.cy - img_cy
                error_px = np.hypot(ex, ey)
                with self._lock:
                    self.state.gripper_cam_detection = (cube.cx, cube.cy)
                    self.state.estimated_yaw_deg = cube.yaw_deg
                    self.state.refine_error_px = error_px
                    self.state.status_text = (
                        f'Refine iter {iteration+1}/{max_iters}: '
                        f'err={error_px:.0f}px yaw={cube.yaw_deg:.0f}deg'
                    )

                if error_px < pixel_threshold:
                    log.info(f"Refine converged: error={error_px:.1f}px")
                    with self._lock:
                        self.state.status_text = f'Refined! err={error_px:.1f}px'
                    break

                # Compute correction in robot frame (rotate by mount angle)
                ex_mm = ex * scale_mm_per_px
                ey_mm = ey * scale_mm_per_px
                cos_a = math.cos(mount_angle_rad)
                sin_a = math.sin(mount_angle_rad)
                dx_mm = (cos_a * ex_mm - sin_a * ey_mm) * gain
                dy_mm = (sin_a * ex_mm + cos_a * ey_mm) * gain

                # Clamp
                mag = np.hypot(dx_mm, dy_mm)
                if mag > 15.0:
                    dx_mm *= 15.0 / mag
                    dy_mm *= 15.0 / mag

                # Apply correction via small position move
                angles = robot.get_angles()
                if not angles or len(angles) < 6:
                    break

                seed = np.array(angles[:5])
                current_pos, _ = self._solver.forward_kin(seed)

                new_pos = current_pos.copy()
                new_pos[0] += dx_mm
                new_pos[1] += dy_mm

                solution = self._solver.solve_ik_position(new_pos, seed_motor_deg=seed)
                if solution is not None:
                    cmd = list(solution) + [angles[5]]
                    robot.move_joints(cmd, speed=100)
                    time.sleep(0.4)

                    # Update refined position
                    with self._lock:
                        self.state.refined_pos_3d = new_pos.copy()
            else:
                # Exhausted iterations without converging
                with self._lock:
                    last_error = self.state.refine_error_px
                    self.state.status_text = (
                        f'Refine: {last_error:.0f}px error '
                        f'(threshold {pixel_threshold:.0f}px)')
                log.warning(f"Refine did not converge after {max_iters} iterations "
                            f"(last error: {last_error:.0f}px)")

        finally:
            cap.release()

        # --- Align gripper yaw (J5 = wrist_roll) with detected cube orientation ---
        self._align_gripper_yaw()

    def _align_gripper_yaw(self):
        """Rotate wrist_roll (J5) to align gripper fingers with cube edges.

        Uses the estimated_yaw_deg captured during gripper-camera refinement.
        For the SO-ARM101, J5 (wrist_roll) rotates around the approach axis
        (roughly vertical during top-down grasp), so adjusting it doesn't
        change X/Y/Z position — it just reorients the gripper.

        The cube yaw is in image coordinates relative to the gripper camera.
        We rotate J5 by the detected yaw so the gripper fingers align with
        the nearest cube edge.
        """
        import math

        robot = self.app.robot
        if robot is None or self._solver is None:
            return

        yaw = self.state.estimated_yaw_deg
        if abs(yaw) < 2.0:
            log.info(f"Cube yaw {yaw:.1f}° is near-zero, skipping alignment")
            return

        config = self.app.config or {}
        refine_cfg = config.get('visual_servoing', {})
        mount_angle_deg = refine_cfg.get('mount_angle_deg', 0.0)

        # The detected yaw is in camera image coordinates.
        # Adjust for camera mount rotation relative to gripper.
        # Then negate: if the cube is rotated +20° in the image, we need
        # to rotate the gripper by +20° (same direction) to align.
        gripper_yaw_correction = yaw + mount_angle_deg

        angles = robot.get_angles()
        if not angles or len(angles) < 6:
            return

        # J5 is wrist_roll (index 4 in 0-based motor angles)
        current_j5 = angles[4]
        new_j5 = current_j5 + gripper_yaw_correction

        # Clamp to safe range (typical wrist_roll range is about ±150°)
        new_j5 = max(-150.0, min(150.0, new_j5))

        log.info(f"Aligning gripper: cube_yaw={yaw:.1f}° mount={mount_angle_deg:.1f}° "
                 f"J5: {current_j5:.1f}° -> {new_j5:.1f}°")

        cmd = list(angles)
        cmd[4] = new_j5
        robot.move_joints(cmd, speed=100)
        time.sleep(0.5)
        self.state.status_text = f'Gripper aligned: yaw correction {gripper_yaw_correction:.1f}°'

    def _do_open_gripper(self):
        """Open the gripper before descending."""
        robot = self.app.robot
        if robot is None:
            return
        if hasattr(robot, 'gripper_open'):
            robot.gripper_open()
            time.sleep(0.5)
        with self._lock:
            self.state.status_text = 'Gripper opened'

    def _do_descend(self):
        """Lower the arm to grasp height."""
        # Snapshot positions under the lock to avoid races with camera/GUI.
        with self._lock:
            pos = self.state.refined_pos_3d
            if pos is None:
                pos = self.state.target_cube_3d
        if pos is None:
            raise RuntimeError("No target position for descend")

        grasp_pos = pos.copy()
        grasp_pos[2] = self.grasp_height_mm
        self._move_to_position(grasp_pos, 'descend')
        with self._lock:
            self.state.status_text = f'At grasp height z={self.grasp_height_mm:.0f}mm'

    def _do_grasp(self):
        """Close the gripper to grasp the cube."""
        robot = self.app.robot
        if robot is None:
            return
        if hasattr(robot, 'gripper_close'):
            robot.gripper_close()
            time.sleep(0.8)
        with self._lock:
            self.state.status_text = 'Gripper closed — cube grasped'

    def _do_lift(self):
        """Lift the cube to a safe height."""
        with self._lock:
            pos = self.state.refined_pos_3d
            if pos is None:
                pos = self.state.target_cube_3d
        if pos is None:
            raise RuntimeError("No position for lift")

        lift_pos = pos.copy()
        lift_pos[2] = self.lift_height_mm
        self._move_to_position(lift_pos, 'lift')
        with self._lock:
            self.state.status_text = f'Lifted to z={self.lift_height_mm:.0f}mm'

    def _do_transport(self):
        """Move to the place target position (at lift height first, then above place)."""
        place = self.place_pos_mm.copy()
        # First move at lift height to avoid collisions
        transit_pos = place.copy()
        transit_pos[2] = self.lift_height_mm
        self._move_to_position(transit_pos, 'transport')
        with self._lock:
            self.state.status_text = f'Above place target at z={self.lift_height_mm:.0f}mm'

    def _do_place(self):
        """Lower to place height and open gripper."""
        robot = self.app.robot
        place = self.place_pos_mm.copy()
        self._move_to_position(place, 'place')
        time.sleep(0.3)

        if robot and hasattr(robot, 'gripper_open'):
            robot.gripper_open()
            time.sleep(0.5)
        with self._lock:
            self.state.status_text = 'Cube placed!'

    def _do_retract(self):
        """Move back to a safe position above the place location."""
        retract_pos = self.place_pos_mm.copy()
        retract_pos[2] = self.lift_height_mm
        self._move_to_position(retract_pos, 'retract')
        with self._lock:
            self.state.status_text = 'Retracted to safe height'
