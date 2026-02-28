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

from config_loader import load_config
from vision import RealSenseCamera, RodDetector
from calibration import CoordinateTransform
from robot import DobotNova5, Gripper
from planner import GraspPlanner
from planner.grasp_planner import MotionType, GripperAction


def execute_waypoints(robot: DobotNova5, gripper: Gripper, waypoints: list):
    """Execute a sequence of waypoints on the robot."""
    for wp in waypoints:
        print(f"  [{wp.label}] -> ({wp.x:.1f}, {wp.y:.1f}, {wp.z:.1f}) "
              f"rx={wp.rx:.1f} ry={wp.ry:.1f} rz={wp.rz:.1f} "
              f"motion={wp.motion.value} gripper={wp.gripper.value}")

        # Execute gripper action before motion if opening
        if wp.gripper == GripperAction.OPEN:
            gripper.open()

        # Execute motion (both use IK + jog, blocks until complete)
        if wp.motion == MotionType.JOINT:
            robot.move_joint(wp.x, wp.y, wp.z, wp.rx, wp.ry, wp.rz)
        else:
            robot.move_linear(wp.x, wp.y, wp.z, wp.rx, wp.ry, wp.rz)

        # Execute gripper action after motion if closing
        if wp.gripper == GripperAction.CLOSE:
            gripper.close()


def main():
    """Main entry point: detect rod, plan grasp, execute pick-and-stand."""
    print("=== Rod Pick-and-Stand System ===")
    print()

    # Load config
    config = load_config()
    robot_cfg = config.get('robot', {})
    camera_cfg = config.get('camera', {})
    planner_cfg = config.get('planner', {})

    # Initialize components
    camera = RealSenseCamera(
        width=camera_cfg.get('width', 640),
        height=camera_cfg.get('height', 480),
        fps=camera_cfg.get('fps', 15),
    )

    detector = RodDetector(
        min_aspect_ratio=camera_cfg.get('min_aspect_ratio', 3.0),
        depth_min_mm=camera_cfg.get('depth_min_mm', 100),
        depth_max_mm=camera_cfg.get('depth_max_mm', 1000),
    )

    transform = CoordinateTransform()
    calibration_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'calibration.yaml')
    if os.path.exists(calibration_path):
        transform.load(calibration_path)
        print("Loaded calibration from", calibration_path)
    else:
        print("WARNING: No calibration file found. Using identity transform.")
        print("  Run scripts/calibrate.py first, or create config/calibration.yaml")

    planner = GraspPlanner(
        safe_z=planner_cfg.get('safe_z', 300.0),
        approach_offset_z=planner_cfg.get('approach_offset_z', 100.0),
        place_position=tuple(planner_cfg.get('place_position', [300.0, 0.0, 50.0])),
    )

    robot = DobotNova5(
        ip=robot_cfg.get('ip', '192.168.5.1'),
        dashboard_port=robot_cfg.get('dashboard_port', 29999),
        motion_port=robot_cfg.get('motion_port', 30003),
    )

    gripper = Gripper(robot=robot)

    # === State Machine ===
    try:
        # INIT
        print("[INIT] Connecting to robot...")
        robot.connect()
        print(f"[INIT] Motion mode: {robot.motion_mode}")
        robot.clear_error()
        robot.enable()
        robot.set_speed(robot_cfg.get('speed_percent', 30))
        print("[INIT] Robot enabled.")

        print("[INIT] Starting camera...")
        camera.start()
        print("[INIT] Camera started.")

        # DETECT
        print("[DETECT] Looking for rod...")
        detection = None
        max_attempts = 10
        for attempt in range(max_attempts):
            color_image, depth_image, depth_frame = camera.get_frames()
            if color_image is None:
                continue

            detection = detector.detect(color_image, depth_image, depth_frame, camera)
            if detection and detection.confidence > 0.3:
                print(f"[DETECT] Rod found! center={detection.center_3d}, "
                      f"confidence={detection.confidence:.1%}")
                break
            time.sleep(0.2)
        else:
            print("[DETECT] Failed to detect rod after", max_attempts, "attempts.")
            return

        # PLAN
        print("[PLAN] Computing grasp plan...")
        # Transform from camera frame (meters) to robot base frame (mm)
        center_base_m = transform.camera_to_base(detection.center_3d)
        axis_base = transform.camera_axis_to_base(detection.axis_3d)
        center_base_mm = center_base_m * 1000.0  # Convert to mm for robot

        waypoints = planner.plan(center_base_mm, axis_base)
        print(f"[PLAN] Generated {len(waypoints)} waypoints.")

        # EXECUTE
        print("[EXECUTE] Running pick-and-stand sequence...")
        execute_waypoints(robot, gripper, waypoints)

        print("[DONE] Pick-and-stand complete!")

    except KeyboardInterrupt:
        print("\n[ABORT] User interrupted.")
    except Exception as e:
        print(f"[ERROR] {e}")
        raise
    finally:
        print("[CLEANUP] Shutting down...")
        try:
            camera.stop()
        except Exception:
            pass
        try:
            robot.disable()
            robot.disconnect()
        except Exception:
            pass
        print("[CLEANUP] Done.")


if __name__ == "__main__":
    main()
