#!/usr/bin/env python3
"""Test calibration by moving the arm 5cm above each checkerboard corner.

Detects the checkerboard, transforms the 4 outer corners to robot base frame
using the saved calibration, then jogs the arm to 5cm above each corner.

Usage:
    ./run.sh scripts/test_calibration.py [--dry-run] [--hd]

    --dry-run  Detect and compute targets but don't move the robot
    --hd       Use 1280x720 resolution
"""

import sys
import os
import time
import socket
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from vision import RealSenseCamera
from calibration.transform import CoordinateTransform

# Checkerboard parameters (must match detect_checkerboard.py)
BOARD_COLS = 7   # inner corners
BOARD_ROWS = 9   # inner corners
SQUARE_SIZE_M = 0.02  # 2cm

# Height above each corner to move to (meters)
HOVER_HEIGHT_M = 0.05  # 5cm

# Robot jog parameters
SPEED_PERCENT = 20       # slow for safety
ANGLE_TOLERANCE = 0.8    # degrees - close enough to stop jogging
MAX_JOG_ITERS = 30       # max jog iterations per target
JOG_SETTLE_TIME = 0.15   # seconds to wait after stopping a jog


class RobotConnection:
    """Minimal dashboard connection for jog-based motion."""

    def __init__(self, ip, port=29999):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(3)
        self.sock.connect((ip, port))
        try:
            self.sock.recv(1024)
        except socket.timeout:
            pass

    def send(self, cmd):
        self.sock.send(f"{cmd}\n".encode())
        time.sleep(0.05)
        try:
            return self.sock.recv(4096).decode().strip()
        except socket.timeout:
            return ""

    def parse_vals(self, resp):
        try:
            inner = resp.split('{')[1].split('}')[0]
            return [float(x) for x in inner.split(',')]
        except (IndexError, ValueError):
            return None

    def get_pose(self):
        return self.parse_vals(self.send('GetPose()'))

    def get_angles(self):
        return self.parse_vals(self.send('GetAngle()'))

    def inverse_kin(self, x, y, z, rx, ry, rz):
        resp = self.send(f'InverseKin({x},{y},{z},{rx},{ry},{rz})')
        return self.parse_vals(resp)

    def enable(self):
        print("  Enabling robot...")
        self.send('DisableRobot()')
        time.sleep(1)
        self.send('ClearError()')
        self.send('EnableRobot()')
        time.sleep(1)
        mode = self.send('RobotMode()')
        print(f"  RobotMode: {mode}")

    def close(self):
        self.send('MoveJog()')
        self.sock.close()


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
    # Inner corner grid spans (BOARD_COLS-1)*SQUARE_SIZE in X, (BOARD_ROWS-1)*SQUARE_SIZE in Y
    # Outer board corners are 1 square beyond the inner corners in each direction
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


def jog_to_angles(robot, target_angles, label=""):
    """Jog each joint toward target angles iteratively."""
    for iteration in range(MAX_JOG_ITERS):
        current = robot.get_angles()
        if not current:
            print(f"    ERROR: can't read joint angles")
            return False

        errors = [(abs(target_angles[i] - current[i]), i) for i in range(6)]
        max_err = max(e for e, _ in errors)

        if max_err < ANGLE_TOLERANCE:
            print(f"    Reached target ({iteration} iterations, max err {max_err:.2f} deg)")
            return True

        # Jog the joint with largest error
        errors.sort(reverse=True)
        err, ji = errors[0]
        direction = '+' if target_angles[ji] > current[ji] else '-'
        jog_time = min(0.8, err / 20.0)
        jog_time = max(0.05, jog_time)

        robot.send(f'MoveJog(J{ji+1}{direction})')
        time.sleep(jog_time)
        robot.send('MoveJog()')
        time.sleep(JOG_SETTLE_TIME)

    print(f"    WARNING: did not converge after {MAX_JOG_ITERS} iterations")
    return False


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

    # Connect to robot
    print("\n=== Connecting to robot ===")
    robot = RobotConnection('192.168.5.1')
    robot.enable()
    robot.send(f'SpeedFactor({SPEED_PERCENT})')

    # Get current pose for orientation reference
    current_pose = robot.get_pose()
    if not current_pose:
        print("ERROR: can't read robot pose")
        robot.close()
        return

    print(f"  Current pose: {', '.join(f'{v:.1f}' for v in current_pose)}")

    # Use current orientation (rx, ry, rz) for all targets
    rx, ry, rz = current_pose[3], current_pose[4], current_pose[5]
    print(f"  Using orientation: rx={rx:.1f}, ry={ry:.1f}, rz={rz:.1f}")

    input("\nPress Enter to start moving to corners (Ctrl+C to abort)...")

    try:
        for i, (target_mm, label) in enumerate(zip(targets_base, labels)):
            print(f"\n--- Moving to corner {i} ({label}) ---")
            print(f"  Target: [{target_mm[0]:.1f}, {target_mm[1]:.1f}, {target_mm[2]:.1f}] mm")

            # Solve IK for hover position
            target_joints = robot.inverse_kin(
                target_mm[0], target_mm[1], target_mm[2], rx, ry, rz)

            if not target_joints:
                print(f"  ERROR: IK failed for corner {i}, skipping")
                continue

            print(f"  IK solution: {', '.join(f'{v:.1f}' for v in target_joints)}")

            # Jog to target
            success = jog_to_angles(robot, target_joints, label)

            # Report actual pose
            actual = robot.get_pose()
            if actual:
                print(f"  Actual pose:  {', '.join(f'{v:.1f}' for v in actual)}")
                err = np.sqrt(sum((actual[j] - target_mm[j])**2 for j in range(3)))
                print(f"  Position error: {err:.1f} mm")

            if i < len(targets_base) - 1:
                input("  Press Enter for next corner...")

        print("\n=== Done! ===")

    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        robot.send('MoveJog()')

    finally:
        robot.close()
        print("Robot disconnected.")


if __name__ == "__main__":
    main()
