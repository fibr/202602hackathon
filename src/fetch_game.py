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
                 grasp_height_mm: float = 12.0,
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
        self._gripper_cam = None
        self._speed = 150

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
        self.place_pos_mm = pos_mm.copy()
        self.state.place_target_3d = pos_mm.copy()
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

        cubes, info = detect_green_cubes(frame)
        self.state.num_cubes_detected = len(cubes)

        target_idx = select_target_cube(
            cubes,
            mode=self.cube_selection_mode,
            click_xy=self._click_xy,
        )
        self.state.target_cube_idx = target_idx

        if target_idx >= 0 and target_idx < len(cubes):
            cube = cubes[target_idx]
            self.state.target_cube_px = (cube.cx, cube.cy)

            # Ray-plane intersection for 3D position
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
                    self.state.target_cube_3d = origin_base + s * ray_base
        else:
            self.state.target_cube_px = None
            self.state.target_cube_3d = None

        return cubes, target_idx

    def _set_state(self, new_state: FetchState, status: str = ''):
        """Transition to a new state."""
        old = self.state.state
        self.state.state = new_state
        self.state.status_text = status or new_state.value
        self.state.step_count += 1
        log.info(f"State: {old.value} -> {new_state.value}: {status}")
        if self.on_state_changed:
            try:
                self.on_state_changed(self.state)
            except Exception:
                pass

    def reset(self):
        """Reset to IDLE state."""
        self._abort = True
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        self._abort = False
        self.state = FetchGameState()
        self.state.place_target_3d = self.place_pos_mm.copy()
        self._set_state(FetchState.IDLE, 'Reset — ready')

    def step(self):
        """Advance one step in the state machine (non-blocking).

        Starts a worker thread for the current state's action.
        """
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

        elif current == FetchState.DONE:
            self.reset()

        elif current == FetchState.ERROR:
            self.reset()

    def start_auto(self):
        """Run the full fetch sequence automatically."""
        self.state.auto_mode = True
        self.reset()
        self._set_state(FetchState.DETECT, 'Auto: scanning...')

    def stop_auto(self):
        """Stop auto mode."""
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
            # In auto mode, advance to next step
            if self.state.auto_mode and not self._abort:
                time.sleep(0.2)
                self.step()
        except Exception as e:
            log.error(f"Fetch game error in {self.state.state.value}: {e}")
            import traceback
            traceback.print_exc()
            self._set_state(FetchState.ERROR, f'Error: {e}')

    # ------------------------------------------------------------------
    # State actions (run in worker thread)
    # ------------------------------------------------------------------

    def _move_to_position(self, pos_mm: np.ndarray, label: str = 'target'):
        """Move arm to a 3D position using position-only IK."""
        robot = self.app.robot
        if robot is None or self._solver is None:
            raise RuntimeError("No robot or IK solver available")

        angles = robot.get_angles()
        seed = np.array(angles[:5]) if angles else None
        solution = self._solver.solve_ik_position(pos_mm, seed_motor_deg=seed)

        if solution is None:
            raise RuntimeError(f"IK failed for {label}: {pos_mm}")

        cmd = list(solution) + [angles[5] if angles else 0]
        robot.move_joints(cmd, speed=self._speed)
        time.sleep(0.5)  # Wait for motion to complete
        log.info(f"Moved to {label}: ({pos_mm[0]:.1f}, {pos_mm[1]:.1f}, {pos_mm[2]:.1f})")

    def _do_approach(self):
        """Move arm above the target cube."""
        target = self.state.target_cube_3d
        if target is None:
            raise RuntimeError("No target cube position")

        hover_pos = target.copy()
        hover_pos[2] = self.hover_height_mm
        self._move_to_position(hover_pos, 'approach')
        self.state.status_text = f'Hovering above cube at z={self.hover_height_mm:.0f}mm'

    def _do_refine(self):
        """Use gripper camera to refine position over the cube.

        Opens the gripper camera, detects the cube, computes pixel error
        from center, and iteratively corrects arm position.
        """
        from vision.green_cube_detector import detect_green_cubes
        import math

        config = self.app.config
        gc = config.get('gripper_camera', {})
        cam_index = gc.get('device_index', 8)
        cam_w = gc.get('width', 640)
        cam_h = gc.get('height', 480)
        hfov_deg = gc.get('hfov_deg', 61.8)

        # Compute scale from current height
        z_mm = self.hover_height_mm
        scale_mm_per_px = z_mm * math.tan(math.radians(hfov_deg / 2.0)) / (cam_w / 2.0)

        robot = self.app.robot
        if robot is None:
            self.state.status_text = 'Refine: no robot, skipping'
            return

        # Try to open gripper camera
        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            log.warning(f"Cannot open gripper cam /dev/video{cam_index}, skipping refine")
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
                if not cubes:
                    self.state.gripper_cam_detection = None
                    self.state.status_text = f'Refine iter {iteration+1}: no cube in gripper cam'
                    time.sleep(0.2)
                    continue

                cube = cubes[0]
                self.state.gripper_cam_detection = (cube.cx, cube.cy)
                self.state.estimated_yaw_deg = cube.yaw_deg

                ex = cube.cx - img_cx
                ey = cube.cy - img_cy
                error_px = np.hypot(ex, ey)
                self.state.refine_error_px = error_px

                self.state.status_text = (
                    f'Refine iter {iteration+1}/{max_iters}: '
                    f'err={error_px:.0f}px yaw={cube.yaw_deg:.0f}deg'
                )

                if error_px < pixel_threshold:
                    log.info(f"Refine converged: error={error_px:.1f}px")
                    self.state.status_text = f'Refined! err={error_px:.1f}px'
                    break

                # Compute correction in robot frame
                dx_mm = ex * scale_mm_per_px * gain
                dy_mm = ey * scale_mm_per_px * gain

                # Clamp
                mag = np.hypot(dx_mm, dy_mm)
                if mag > 15.0:
                    dx_mm *= 15.0 / mag
                    dy_mm *= 15.0 / mag

                # Apply correction via small position move
                angles = robot.get_angles()
                if angles is None:
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
                    self.state.refined_pos_3d = new_pos.copy()

        finally:
            cap.release()

    def _do_open_gripper(self):
        """Open the gripper before descending."""
        robot = self.app.robot
        if robot is None:
            return
        if hasattr(robot, 'gripper_open'):
            robot.gripper_open()
            time.sleep(0.5)
        self.state.status_text = 'Gripper opened'

    def _do_descend(self):
        """Lower the arm to grasp height."""
        # Use refined position if available, else use original detection
        pos = self.state.refined_pos_3d
        if pos is None:
            pos = self.state.target_cube_3d
        if pos is None:
            raise RuntimeError("No target position for descend")

        grasp_pos = pos.copy()
        grasp_pos[2] = self.grasp_height_mm
        self._move_to_position(grasp_pos, 'descend')
        self.state.status_text = f'At grasp height z={self.grasp_height_mm:.0f}mm'

    def _do_grasp(self):
        """Close the gripper to grasp the cube."""
        robot = self.app.robot
        if robot is None:
            return
        if hasattr(robot, 'gripper_close'):
            robot.gripper_close()
            time.sleep(0.8)
        self.state.status_text = 'Gripper closed — cube grasped'

    def _do_lift(self):
        """Lift the cube to a safe height."""
        pos = self.state.refined_pos_3d
        if pos is None:
            pos = self.state.target_cube_3d
        if pos is None:
            raise RuntimeError("No position for lift")

        lift_pos = pos.copy()
        lift_pos[2] = self.lift_height_mm
        self._move_to_position(lift_pos, 'lift')
        self.state.status_text = f'Lifted to z={self.lift_height_mm:.0f}mm'

    def _do_transport(self):
        """Move to the place target position (at lift height first, then above place)."""
        place = self.place_pos_mm.copy()
        # First move at lift height to avoid collisions
        transit_pos = place.copy()
        transit_pos[2] = self.lift_height_mm
        self._move_to_position(transit_pos, 'transport')
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
        self.state.status_text = 'Cube placed!'

    def _do_retract(self):
        """Move back to a safe position above the place location."""
        retract_pos = self.place_pos_mm.copy()
        retract_pos[2] = self.lift_height_mm
        self._move_to_position(retract_pos, 'retract')
        self.state.status_text = 'Retracted to safe height'
