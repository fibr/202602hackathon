#!/usr/bin/env python3
"""Test calibration by moving the arm 5cm above each checkerboard corner.

Detects the checkerboard, transforms the 4 outer corners to robot base frame
using the saved calibration, then moves the arm to 5cm above each corner.
Uses the DobotNova5 driver (MovJ with ROS2 driver, jog fallback without).

Usage:
    ./run.sh scripts/test_calibration.py [--dry-run] [--hd]

    --dry-run  Detect and compute targets but don't move the robot
    --hd       Use 1280x720 resolution
"""

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from vision import RealSenseCamera
from calibration.transform import CoordinateTransform
from robot.dobot_api import DobotNova5

# Checkerboard parameters (must match detect_checkerboard.py)
BOARD_COLS = 7   # inner corners
BOARD_ROWS = 9   # inner corners
SQUARE_SIZE_M = 0.02  # 2cm

# Height above each corner to move to (meters)
HOVER_HEIGHT_M = 0.05  # 5cm

SPEED_PERCENT = 20  # slow for safety


def detect_checkerboard(camera):
    """Detect checkerboard and return corners + solvePnP pose."""
    print("Looking for checkerboard (press 'c' to capture, 'q' to abort)...")

    while True:
        color_image, depth_image, depth_frame = camera.get_frames()
        if color_image is None:
            continue

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Try SectorBased detector first
        found, corners = False, None
        try:
            found, corners = cv2.findChessboardCornersSB(
                gray, (BOARD_COLS, BOARD_ROWS),
                cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        except cv2.error:
            pass

        if not found:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            found, corners = cv2.findChessboardCorners(
                enhanced, (BOARD_COLS, BOARD_ROWS), flags)
            if found:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

        display = color_image.copy()
        if found:
            cv2.drawChessboardCorners(display, (BOARD_COLS, BOARD_ROWS), corners, found)
            cv2.putText(display, f"Found {len(corners)} corners - press 'c'",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No checkerboard",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Calibration Test', display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            return None, None, None, None
        if cv2.getWindowProperty('Calibration Test', cv2.WND_PROP_VISIBLE) < 1:
            return None, None, None, None

        if key == ord('c') and found:
            # solvePnP for board pose
            obj_points = np.zeros((BOARD_ROWS * BOARD_COLS, 3), dtype=np.float32)
            for r in range(BOARD_ROWS):
                for c in range(BOARD_COLS):
                    obj_points[r * BOARD_COLS + c] = [c * SQUARE_SIZE_M, r * SQUARE_SIZE_M, 0]

            camera_matrix = np.array([
                [camera.intrinsics.fx, 0, camera.intrinsics.ppx],
                [0, camera.intrinsics.fy, camera.intrinsics.ppy],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.array(camera.intrinsics.coeffs, dtype=np.float64)

            _, rvec, tvec = cv2.solvePnP(obj_points, corners, camera_matrix, dist_coeffs)
            R, _ = cv2.Rodrigues(rvec)
            T_board_in_cam = np.eye(4)
            T_board_in_cam[:3, :3] = R
            T_board_in_cam[:3, 3] = tvec.flatten()

            return corners, T_board_in_cam, color_image, depth_frame


def get_board_outer_corners_cam(T_board_in_cam):
    """Get the 4 outer corners of the checkerboard in camera frame.

    These are the actual board corners (not inner corners), i.e. offset
    by half a square from the first/last inner corners.
    """
    half = SQUARE_SIZE_M / 2.0
    max_x = (BOARD_COLS - 1) * SQUARE_SIZE_M
    max_y = (BOARD_ROWS - 1) * SQUARE_SIZE_M

    # 4 outer corners in board frame (on the board plane, Z=0)
    board_corners = [
        np.array([-half, -half, 0, 1]),                    # top-left
        np.array([max_x + half, -half, 0, 1]),             # top-right
        np.array([max_x + half, max_y + half, 0, 1]),      # bottom-right
        np.array([-half, max_y + half, 0, 1]),              # bottom-left
    ]
    labels = ["top-left", "top-right", "bottom-right", "bottom-left"]

    corners_cam = []
    for pt in board_corners:
        p_cam = (T_board_in_cam @ pt)[:3]
        corners_cam.append(p_cam)

    return corners_cam, labels


def main():
    dry_run = '--dry-run' in sys.argv
    hd = '--hd' in sys.argv
    width, height = (1280, 720) if hd else (640, 480)

    # Load calibration
    calib_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'calibration.yaml')
    if not os.path.exists(calib_path):
        print(f"ERROR: No calibration file at {calib_path}")
        print("Run detect_checkerboard.py first to create it.")
        sys.exit(1)

    transform = CoordinateTransform()
    transform.load(calib_path)
    print("Loaded calibration from", calib_path)

    # Detect checkerboard
    camera = RealSenseCamera(width=width, height=height, fps=15)
    camera.start()
    print("Camera started.\n")

    result = detect_checkerboard(camera)
    camera.stop()
    cv2.destroyAllWindows()

    if result[0] is None:
        print("Aborted.")
        return

    corners_2d, T_board_in_cam, color_image, depth_frame = result

    # Get 4 outer corners in camera frame
    corners_cam, labels = get_board_outer_corners_cam(T_board_in_cam)

    # Transform to robot base frame
    print("\n=== Checkerboard corners in robot base frame ===")
    targets_base = []
    for i, (p_cam, label) in enumerate(zip(corners_cam, labels)):
        p_base = transform.camera_to_base(p_cam)
        # Convert to mm for robot commands
        p_base_mm = p_base * 1000.0
        hover_mm = p_base_mm.copy()
        hover_mm[2] += HOVER_HEIGHT_M * 1000.0  # add 5cm in Z

        print(f"\n  Corner {i} ({label}):")
        print(f"    Camera frame:  [{p_cam[0]:.4f}, {p_cam[1]:.4f}, {p_cam[2]:.4f}] m")
        print(f"    Robot base:    [{p_base_mm[0]:.1f}, {p_base_mm[1]:.1f}, {p_base_mm[2]:.1f}] mm")
        print(f"    Hover target:  [{hover_mm[0]:.1f}, {hover_mm[1]:.1f}, {hover_mm[2]:.1f}] mm")
        targets_base.append(hover_mm)

    if dry_run:
        print("\n[DRY RUN] Would move to above targets. Exiting.")
        return

    # Connect to robot using the driver
    print("\n=== Connecting to robot ===")
    robot = DobotNova5()
    robot.connect()
    print(f"  Motion mode: {robot.motion_mode}")

    robot.enable()
    robot.set_speed(SPEED_PERCENT)

    # Get current pose for orientation reference
    pose = robot.get_pose()
    print(f"  Current pose: {', '.join(f'{v:.1f}' for v in pose)}")

    # Use current orientation (rx, ry, rz) for all targets
    rx, ry, rz = pose[3], pose[4], pose[5]
    print(f"  Using orientation: rx={rx:.1f}, ry={ry:.1f}, rz={rz:.1f}")

    input("\nPress Enter to start moving to corners (Ctrl+C to abort)...")

    try:
        for i, (target_mm, label) in enumerate(zip(targets_base, labels)):
            print(f"\n--- Moving to corner {i} ({label}) ---")
            print(f"  Target: [{target_mm[0]:.1f}, {target_mm[1]:.1f}, {target_mm[2]:.1f}] mm")

            ok = robot.move_joint(
                target_mm[0], target_mm[1], target_mm[2], rx, ry, rz,
                speed_percent=SPEED_PERCENT)

            if not ok:
                print(f"  ERROR: move failed for corner {i}, skipping")
                continue

            # Report actual pose
            actual = robot.get_pose()
            print(f"  Actual pose:  {', '.join(f'{v:.1f}' for v in actual)}")
            err = np.linalg.norm(actual[:3] - target_mm)
            print(f"  Position error: {err:.1f} mm")

            if i < len(targets_base) - 1:
                input("  Press Enter for next corner...")

        print("\n=== Done! ===")

    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        robot.jog_stop()

    finally:
        robot.disconnect()
        print("Robot disconnected.")


if __name__ == "__main__":
    main()
