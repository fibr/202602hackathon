"""Main orchestrator for the rod pick-and-stand system.

Ties together vision, calibration, planning, and robot control
in a state machine that detects a rod and picks it up.
"""

import time
import sys
import os
import numpy as np

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_loader import load_config, config_path
from logger import get_logger, get_log_file
from vision import RodDetector, create_camera, VisualServo, make_green_cube_detector
from calibration import CoordinateTransform
from robot import DobotNova5, Gripper
from planner import GraspPlanner
from planner.grasp_planner import MotionType, GripperAction
from planner.trajectory import execute_trajectory
from kinematics import IKSolver

log = get_logger('main')


def execute_waypoints(robot: DobotNova5, gripper: Gripper, waypoints: list,
                      ik_solver: IKSolver, linear_step_mm: float = 5.0,
                      max_step_deg: float = 5.0,
                      pause_after: str = None) -> np.ndarray:
    """Execute waypoints using local IK + smooth quintic trajectories.

    For JOINT waypoints: solve IK, then execute a smooth quintic trajectory
    from current joints to target joints (subdivided into small steps).
    For LINEAR waypoints: interpolate Cartesian path, IK each step, then
    execute each step as a small joint move.

    All moves use quintic smoothstep for zero velocity/acceleration at
    endpoints, plus retry logic for transient failures.

    Args:
        robot: Connected DobotNova5 instance.
        gripper: Gripper instance.
        waypoints: Ordered list of Waypoint objects to execute.
        ik_solver: IKSolver for pose→joints conversion.
        linear_step_mm: Cartesian interpolation step for LINEAR moves.
        max_step_deg: Maximum joint step per trajectory segment.
        pause_after: If given, stop after executing the waypoint with this
            label and return current joints. Remaining waypoints are skipped.
            The caller is responsible for executing the rest.

    Returns:
        Current joint angles after the last executed waypoint (degrees).
    """
    current_joints = robot.get_joint_angles()

    for wp in waypoints:
        target_pos = np.array([wp.x, wp.y, wp.z])
        target_rpy = np.array([wp.rx, wp.ry, wp.rz])

        print(f"  [{wp.label}] -> ({wp.x:.1f}, {wp.y:.1f}, {wp.z:.1f}) "
              f"rx={wp.rx:.1f} ry={wp.ry:.1f} rz={wp.rz:.1f} "
              f"motion={wp.motion.value} gripper={wp.gripper.value}")

        # Execute gripper action before motion if opening
        if wp.gripper == GripperAction.OPEN:
            gripper.open()

        if wp.motion == MotionType.JOINT:
            # Solve IK, then use smooth trajectory to get there
            joints = ik_solver.solve_ik(target_pos, target_rpy,
                                         seed_joints_deg=current_joints)
            if joints is None:
                log.error(f"IK failed for waypoint {wp.label}")
                raise RuntimeError(f"IK failed at {wp.label}: "
                                   f"pos={target_pos}, rpy={target_rpy}")
            ok = execute_trajectory(robot, current_joints, joints,
                                    max_step_deg=max_step_deg)
            if not ok:
                raise RuntimeError(f"Trajectory failed at {wp.label}")
            current_joints = joints

        else:  # LINEAR
            start_pos, start_rpy = ik_solver.forward_kin(current_joints)

            joint_path = ik_solver.interpolate_linear(
                start_pos, start_rpy, target_pos, target_rpy,
                seed_joints_deg=current_joints, step_mm=linear_step_mm)
            if joint_path is None:
                log.error(f"Linear interpolation failed for {wp.label}")
                raise RuntimeError(f"Linear IK failed at {wp.label}")

            # Execute each Cartesian step as a small trajectory
            for step_joints in joint_path:
                ok = execute_trajectory(robot, current_joints, step_joints,
                                        max_step_deg=max_step_deg)
                if not ok:
                    raise RuntimeError(f"Trajectory failed during {wp.label}")
                current_joints = step_joints

        # Execute gripper action after motion if closing
        if wp.gripper == GripperAction.CLOSE:
            gripper.close()

        # Pause after a specific waypoint (e.g. for visual servoing)
        if pause_after is not None and wp.label == pause_after:
            log.info(f"[EXECUTE] Pausing after waypoint '{pause_after}'")
            return current_joints

    return current_joints


def run_visual_servo_step(
    robot: DobotNova5,
    ik_solver: IKSolver,
    servo: VisualServo,
    current_joints: np.ndarray,
    gripper_rz_deg: float,
    config: dict,
) -> np.ndarray:
    """Run the visual servo correction loop and return updated joint angles.

    Opens the gripper camera, aligns the gripper over the detected target,
    then closes the camera.  Uses the green-cube detector by default; the
    detection parameters are taken from the ``visual_servo`` config section.

    Args:
        robot: Connected DobotNova5 instance.
        ik_solver: IKSolver instance.
        servo: VisualServo instance (camera not yet open).
        current_joints: Joint angles after PRE_GRASP (degrees).
        gripper_rz_deg: Gripper rz at the pre-grasp pose (degrees).
        config: Full project config dict.

    Returns:
        Updated joint angles after servo correction.
    """
    vs_cfg = config.get('visual_servo', {})

    # Build detector — reuse green-cube detector (tunable via config)
    detector_fn = make_green_cube_detector(
        min_area=vs_cfg.get('detector_min_area', 200.0),
    )

    log.info("[SERVO] Opening gripper camera...")
    try:
        servo.open_camera()
    except RuntimeError as exc:
        log.warning(f"[SERVO] Cannot open gripper camera: {exc}. Skipping servo.")
        return current_joints

    try:
        result, updated_joints = servo.align(
            robot=robot,
            ik_solver=ik_solver,
            detector_fn=detector_fn,
            current_joints=current_joints,
            gripper_rz_deg=gripper_rz_deg,
        )
        status = "CONVERGED" if result.converged else "NOT CONVERGED"
        log.info(
            f"[SERVO] {status}: {result.iterations} iter(s), "
            f"final_error={result.final_error_px:.1f}px, "
            f"corrections={len(result.corrections)}"
        )
        return updated_joints
    finally:
        servo.close_camera()
        log.info("[SERVO] Gripper camera closed.")


def _validate_ik(robot: DobotNova5, ik_solver: IKSolver):
    """Cross-check local FK/IK against the robot's built-in commands."""
    current_joints = robot.get_joint_angles()
    robot_pose = robot.get_pose()  # [x, y, z, rx, ry, rz]

    local_pos, local_rpy = ik_solver.forward_kin(current_joints)
    pos_err = np.linalg.norm(local_pos - robot_pose[:3])
    rpy_err = np.max(np.abs(local_rpy - robot_pose[3:6]))

    log.info(f"[IK VALIDATE] Robot joints: {np.round(current_joints, 2)}")
    log.info(f"[IK VALIDATE] Robot FK: pos={np.round(robot_pose[:3], 1)} rpy={np.round(robot_pose[3:6], 1)}")
    log.info(f"[IK VALIDATE] Local FK: pos={np.round(local_pos, 1)} rpy={np.round(local_rpy, 1)}")
    log.info(f"[IK VALIDATE] Position error: {pos_err:.2f} mm, RPY error: {rpy_err:.2f} deg")

    if pos_err > 5.0:
        log.warning(f"[IK VALIDATE] Position error {pos_err:.1f}mm exceeds 5mm — "
                    "check tool_length_mm in config/robot_config.yaml")
    else:
        log.info("[IK VALIDATE] OK — local IK matches robot")


def main():
    """Main entry point: detect rod, plan grasp, execute pick-and-stand."""
    log.info("=== Rod Pick-and-Stand System ===")
    log.info(f"Log file: {get_log_file()}")

    # Load config
    config = load_config()
    robot_cfg = config.get('robot', {})
    camera_cfg = config.get('camera', {})
    planner_cfg = config.get('planner', {})
    gripper_cfg = config.get('gripper', {})
    kin_cfg = config.get('kinematics', {})

    # Initialize components
    camera = create_camera(config)

    detector = RodDetector(
        min_aspect_ratio=camera_cfg.get('min_aspect_ratio', 2.2),
        min_area=camera_cfg.get('min_area', 500),
        depth_min_mm=camera_cfg.get('depth_min_mm', 6000),
        depth_max_mm=camera_cfg.get('depth_max_mm', 19000),
        max_brightness=camera_cfg.get('max_brightness', 120),
        max_brightness_std=camera_cfg.get('max_brightness_std', 40.0),
        min_convexity=camera_cfg.get('min_convexity', 0.85),
        workspace_roi=camera_cfg.get('workspace_roi'),
    )

    transform = CoordinateTransform()
    calibration_path = config_path('calibration.yaml')
    if os.path.exists(calibration_path):
        transform.load(calibration_path)
        log.info(f"Loaded calibration from {calibration_path}")
    else:
        log.warning("No calibration file found. Using identity transform.")

    planner = GraspPlanner(
        safe_z=planner_cfg.get('safe_z', 300.0),
        approach_offset_z=planner_cfg.get('approach_offset_z', 100.0),
        place_position=tuple(planner_cfg.get('place_position', [300.0, 0.0, 50.0])),
    )

    robot = DobotNova5(
        ip=robot_cfg.get('ip', '192.168.5.1'),
        dashboard_port=robot_cfg.get('dashboard_port', 29999),
    )

    gripper = Gripper(robot=robot)

    ik_solver = IKSolver(
        tool_length_mm=gripper_cfg.get('tool_length_mm', 120.0),
    )

    # === State Machine ===
    try:
        # INIT
        log.info("[INIT] Connecting to robot...")
        robot.connect()
        robot.clear_error()
        robot.enable()
        robot.set_speed(robot_cfg.get('speed_percent', 30))
        log.info("[INIT] Robot enabled.")

        # Validate local IK against robot's built-in IK
        if kin_cfg.get('validate_on_startup', True):
            _validate_ik(robot, ik_solver)

        log.info("[INIT] Starting camera...")
        camera.start()
        log.info("[INIT] Camera started.")

        # DETECT
        log.info("[DETECT] Looking for rod...")
        detection = None
        max_attempts = 10
        for attempt in range(max_attempts):
            color_image, depth_image, depth_frame = camera.get_frames()
            if color_image is None:
                continue

            detection = detector.detect(color_image, depth_image, depth_frame, camera)
            if detection and detection.confidence > 0.3:
                log.info(f"[DETECT] Rod found! center={detection.center_3d}, "
                         f"confidence={detection.confidence:.1%}")
                break
            time.sleep(0.2)
        else:
            log.error(f"[DETECT] Failed to detect rod after {max_attempts} attempts.")
            return

        # PLAN
        log.info("[PLAN] Computing grasp plan...")
        # Transform from camera frame (meters) to robot base frame (mm)
        center_base_m = transform.camera_to_base(detection.center_3d)
        axis_base = transform.camera_axis_to_base(detection.axis_3d)
        center_base_mm = center_base_m * 1000.0  # Convert to mm for robot

        waypoints = planner.plan(center_base_mm, axis_base)
        log.info(f"[PLAN] Generated {len(waypoints)} waypoints.")

        # EXECUTE
        log.info("[EXECUTE] Running pick-and-stand sequence (local IK + joint control)...")
        vs_cfg = config.get('visual_servo', {})
        servo_enabled = vs_cfg.get('enabled', False)
        linear_step = kin_cfg.get('linear_step_mm', 5.0)

        if servo_enabled:
            # --- Phase 1: move to PRE_GRASP (coarse positioning) ---
            log.info("[EXECUTE] Phase 1: coarse move to PRE_GRASP ...")
            current_joints = execute_waypoints(
                robot, gripper, waypoints, ik_solver,
                linear_step_mm=linear_step,
                pause_after='PRE_GRASP',
            )

            # --- Phase 2: gripper-camera visual servo (fine positioning) ---
            log.info("[EXECUTE] Phase 2: gripper-camera visual servo ...")
            pre_grasp_wp = next(
                (wp for wp in waypoints if wp.label == 'PRE_GRASP'), None
            )
            gripper_rz = pre_grasp_wp.rz if pre_grasp_wp is not None else 0.0

            servo = VisualServo.from_config(config)
            current_joints = run_visual_servo_step(
                robot, ik_solver, servo, current_joints,
                gripper_rz_deg=gripper_rz,
                config=config,
            )

            # After servo the robot TCP is centred over the target.
            # Update the rod-relative waypoints so the descent and lift
            # stay at the servo-corrected XY position rather than the
            # coarsely-detected position (which may have calibration error).
            corrected_pose = robot.get_pose()   # [x, y, z, rx, ry, rz] mm/deg
            corrected_x = corrected_pose[0]
            corrected_y = corrected_pose[1]
            log.info(
                f"[EXECUTE] Servo-corrected TCP: "
                f"x={corrected_x:.1f}mm, y={corrected_y:.1f}mm"
            )

            # Labels whose XY should track the servo-corrected position
            _rod_relative_labels = {'PRE_GRASP', 'GRASP_DESCEND', 'GRASP_CLOSE', 'LIFT'}
            for wp in waypoints:
                if wp.label in _rod_relative_labels:
                    wp.x = corrected_x
                    wp.y = corrected_y

            # --- Phase 3: execute remaining waypoints after PRE_GRASP ---
            log.info("[EXECUTE] Phase 3: continuing grasp sequence ...")
            remaining = [wp for wp in waypoints if wp.label != 'PRE_GRASP']
            execute_waypoints(
                robot, gripper, remaining, ik_solver,
                linear_step_mm=linear_step,
            )
        else:
            # No visual servo: execute all waypoints directly
            execute_waypoints(robot, gripper, waypoints, ik_solver,
                              linear_step_mm=linear_step)

        log.info("[DONE] Pick-and-stand complete!")

    except KeyboardInterrupt:
        log.warning("User interrupted.")
    except Exception as e:
        log.error(f"[ERROR] {e}", exc_info=True)
        raise
    finally:
        log.info("[CLEANUP] Shutting down...")
        try:
            camera.stop()
        except Exception:
            pass
        try:
            robot.disable()
            robot.disconnect()
        except Exception:
            pass
        log.info("[CLEANUP] Done.")


if __name__ == "__main__":
    main()
