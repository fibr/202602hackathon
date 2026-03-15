#!/usr/bin/env python3
"""Track a green cube and hover the arm 30mm above it.

Uses the overview camera to detect a green cube, transforms pixel coords
to robot base frame via hand-eye calibration, solves IK for a position
30mm above the cube, and continuously moves the arm to track it.

Usage:
    ./run.sh scripts/track_cube.py              # Run with default settings
    ./run.sh scripts/track_cube.py --height 50  # Hover 50mm above cube
    ./run.sh scripts/track_cube.py --speed 150  # Servo speed
    ./run.sh scripts/track_cube.py --no-move    # Detect only, don't move arm
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config_loader import load_config, connect_robot, config_path
from vision import create_camera
from vision.green_cube_detector import detect_green_cubes, annotate_frame
from calibration import CoordinateTransform
from kinematics.arm101_ik_solver import Arm101IKSolver


def main():
    parser = argparse.ArgumentParser(description='Track green cube, hover arm above')
    parser.add_argument('--height', type=float, default=30.0,
                        help='Hover height above cube in mm (default: 30)')
    parser.add_argument('--speed', type=int, default=200,
                        help='Servo move speed (default: 200)')
    parser.add_argument('--no-move', action='store_true',
                        help='Detection only, do not move the arm')
    parser.add_argument('--interval', type=float, default=0.1,
                        help='Control loop interval in seconds (default: 0.1)')
    args = parser.parse_args()

    config = load_config()

    # Camera
    print("Starting camera...")
    camera = create_camera(config)
    camera.start()
    print(f"  Camera: {camera.width}x{camera.height}")

    # Flush initial frames
    for _ in range(15):
        camera.get_frames()
        time.sleep(0.03)

    # Calibration transform (camera → robot base)
    calib_path = config_path('calibration.yaml')
    if not os.path.exists(calib_path):
        print(f"ERROR: No calibration at {calib_path}")
        print("  Run hand-eye calibration first.")
        camera.stop()
        return 1
    transform = CoordinateTransform()
    transform.load(calib_path)
    print(f"  Calibration loaded from {calib_path}")

    # Robot
    robot = None
    solver = None
    if not args.no_move:
        try:
            robot = connect_robot(config)
            print("  Robot connected")
            solver = Arm101IKSolver()
        except Exception as e:
            print(f"  WARNING: Robot not available: {e}")
            print("  Running in detection-only mode")

    # Tracking state
    last_target = None
    move_threshold_mm = 3.0  # Only move if target shifted by this much

    print(f"\n=== Cube Tracker ===")
    print(f"  Hover height: {args.height}mm above cube")
    print(f"  Move: {'enabled' if robot else 'detection only'}")
    print(f"  Press 'q' or ESC to quit\n")

    cv2.namedWindow('Cube Tracker', cv2.WINDOW_NORMAL)

    try:
        while True:
            color, depth, depth_frame = camera.get_frames()
            if color is None:
                continue

            # Detect green cubes
            cubes, info = detect_green_cubes(color)
            display = annotate_frame(color, cubes)

            # Status bar
            h, w = display.shape[:2]

            if cubes:
                cube = cubes[0]  # Track the largest one

                # Convert pixel to 3D in camera frame
                p_cam = camera.pixel_to_3d(cube.cx, cube.cy, depth_frame)

                # Transform to robot base frame (meters → mm)
                p_base_m = transform.camera_to_base(p_cam)
                p_base_mm = p_base_m * 1000.0

                # Target: hover above cube
                target_mm = p_base_mm.copy()
                target_mm[2] += args.height

                # Draw target info
                cv2.putText(display,
                            f'Cube: ({p_base_mm[0]:.0f}, {p_base_mm[1]:.0f}, '
                            f'{p_base_mm[2]:.0f})mm',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1)
                cv2.putText(display,
                            f'Target: ({target_mm[0]:.0f}, {target_mm[1]:.0f}, '
                            f'{target_mm[2]:.0f})mm  [+{args.height:.0f}mm]',
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 200, 255), 1)

                # Move arm if target changed enough
                if robot and solver:
                    should_move = True
                    if last_target is not None:
                        delta = np.linalg.norm(target_mm - last_target)
                        if delta < move_threshold_mm:
                            should_move = False

                    if should_move:
                        # Get current angles as IK seed
                        angles = robot.get_angles()
                        seed = np.array(angles[:5]) if angles else None

                        # Solve IK (position only, let orientation float)
                        solution = solver.solve_ik_position(
                            target_mm, seed_motor_deg=seed)

                        if solution is not None:
                            # Send as 6-joint command (gripper stays)
                            cmd = list(solution) + [angles[5] if angles else 0]
                            robot.move_joints(cmd, speed=args.speed)
                            last_target = target_mm.copy()
                            cv2.putText(display, 'TRACKING',
                                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 255, 0), 2)
                        else:
                            cv2.putText(display, 'IK FAILED (unreachable)',
                                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 0, 255), 2)
            else:
                cv2.putText(display, 'No cube detected',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1)

            # Show current arm pose
            if robot:
                angles = robot.get_angles()
                if angles:
                    pos, rpy = solver.forward_kin(np.array(angles[:5]))
                    cv2.putText(display,
                                f'TCP: ({pos[0]:.0f}, {pos[1]:.0f}, '
                                f'{pos[2]:.0f})mm',
                                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (200, 200, 200), 1)

            cv2.imshow('Cube Tracker', display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        if robot:
            try:
                robot.disconnect()
            except Exception:
                pass
        print("Done.")


if __name__ == '__main__':
    main()
