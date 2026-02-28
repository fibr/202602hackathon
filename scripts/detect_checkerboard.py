#!/usr/bin/env python3
"""Detect a checkerboard and calibrate camera-to-robot transform via click.

Checkerboard spec: 8x10 squares, 2cm square size -> 7x9 inner corners.

Workflow:
  1. Position robot TCP at a checkerboard corner
  2. Click that corner in the camera image
  3. Script snaps to nearest detected corner, reads robot TCP, records pair
  4. Repeat for 3+ corners
  5. Press 's' to solve and save calibration

Usage:
    ./run.sh scripts/detect_checkerboard.py [--hd]

    --hd   Use 1280x720 resolution (default: 640x480)

Keys:
    s  Solve & save calibration (needs 3+ points)
    u  Undo last point
    r  Reset all points
    q  Quit
"""

import sys
import os
import socket
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from vision import RealSenseCamera
from config_loader import load_config

# Checkerboard parameters
BOARD_COLS = 7   # inner corners (8 squares - 1)
BOARD_ROWS = 9   # inner corners (10 squares - 1)
SQUARE_SIZE_M = 0.02  # 2cm squares

SNAP_RADIUS_PX = 30  # max pixel distance to snap click to a corner


class RobotConnection:
    """Lightweight dashboard connection for reading pose only."""

    def __init__(self, ip, port=29999):
        self.ip = ip
        self.port = port
        self.sock = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(2)
        self.sock.connect((self.ip, self.port))
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

    def get_pose(self):
        """Get TCP pose [x,y,z,rx,ry,rz] in mm/deg, or None."""
        resp = self.send('GetPose()')
        try:
            inner = resp.split('{')[1].split('}')[0]
            return [float(x) for x in inner.split(',')]
        except (IndexError, ValueError):
            return None

    def close(self):
        if self.sock:
            self.sock.close()


def detect_corners(gray):
    """Find checkerboard corners using multiple strategies."""
    try:
        found, corners = cv2.findChessboardCornersSB(
            gray, (BOARD_COLS, BOARD_ROWS),
            cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if found:
            return found, corners
    except cv2.error:
        pass

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(enhanced, (BOARD_COLS, BOARD_ROWS), flags)
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        return found, corners

    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
             cv2.CALIB_CB_NORMALIZE_IMAGE |
             cv2.CALIB_CB_FILTER_QUADS)
    found, corners = cv2.findChessboardCorners(sharpened, (BOARD_COLS, BOARD_ROWS), flags)
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        return found, corners

    for cols, rows in [(BOARD_COLS, BOARD_ROWS - 2), (BOARD_COLS - 2, BOARD_ROWS),
                       (BOARD_COLS - 2, BOARD_ROWS - 2)]:
        try:
            found, corners = cv2.findChessboardCornersSB(
                gray, (cols, rows), cv2.CALIB_CB_EXHAUSTIVE)
            if found:
                print(f"  (detected {cols}x{rows} subset of board)")
                return found, corners
        except cv2.error:
            pass

    return False, None


def compute_board_pose(corners_2d, intrinsics):
    """solvePnP -> (T_board_in_cam 4x4, obj_points)."""
    n = len(corners_2d)
    if n == BOARD_ROWS * BOARD_COLS:
        cols, rows = BOARD_COLS, BOARD_ROWS
    else:
        for c, r in [(BOARD_COLS, BOARD_ROWS - 2), (BOARD_COLS - 2, BOARD_ROWS),
                     (BOARD_COLS - 2, BOARD_ROWS - 2)]:
            if c * r == n:
                cols, rows = c, r
                break
        else:
            cols, rows = BOARD_COLS, BOARD_ROWS

    obj_points = np.zeros((rows * cols, 3), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            obj_points[r * cols + c] = [c * SQUARE_SIZE_M, r * SQUARE_SIZE_M, 0]

    camera_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float64)

    _, rvec, tvec = cv2.solvePnP(obj_points, corners_2d, camera_matrix, dist_coeffs)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T, obj_points


def corner_3d_in_cam(corner_idx, T_board_in_cam, n_cols=BOARD_COLS):
    """Get the 3D position of a detected corner in camera frame (meters)."""
    row = corner_idx // n_cols
    col = corner_idx % n_cols
    p_board = np.array([col * SQUARE_SIZE_M, row * SQUARE_SIZE_M, 0, 1])
    p_cam = (T_board_in_cam @ p_board)[:3]
    return p_cam


def solve_rigid_transform(pts_cam, pts_robot):
    """SVD-based rigid transform: T such that pts_robot = T @ pts_cam.

    Args:
        pts_cam: Nx3 points in camera frame (meters)
        pts_robot: Nx3 points in robot frame (meters)

    Returns:
        4x4 homogeneous transform T_cam_to_base
    """
    assert len(pts_cam) >= 3
    A = np.array(pts_cam)
    B = np.array(pts_robot)

    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    A_c = A - centroid_A
    B_c = B - centroid_B

    H = A_c.T @ B_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Correct reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def main():
    hd = '--hd' in sys.argv
    width, height = (1280, 720) if hd else (640, 480)

    config = load_config()
    rc = config.get('robot', {})
    ip = rc.get('ip', '192.168.5.1')

    print("=== Click-to-Calibrate ===")
    print(f"Board: {BOARD_COLS+1}x{BOARD_ROWS+1} squares, {SQUARE_SIZE_M*100:.0f}cm")
    print(f"Resolution: {width}x{height}")
    print()
    print("Workflow:")
    print("  1. Position robot TCP at a checkerboard corner")
    print("  2. Click that corner in the image")
    print("  3. Repeat for 3+ corners")
    print("  4. Press 's' to solve & save")
    print()
    print("Keys: [s] solve & save  [u] undo  [r] reset  [q] quit")
    print()

    # Connect to robot
    print(f"Connecting to robot at {ip}...")
    robot = RobotConnection(ip)
    robot.connect()
    pose = robot.get_pose()
    if pose:
        print(f"  Robot pose: {', '.join(f'{v:.1f}' for v in pose)}")
    else:
        print("  WARNING: couldn't read robot pose")

    # Start camera
    camera = RealSenseCamera(width=width, height=height, fps=15)
    camera.start()
    print("Camera started.\n")

    # State
    pairs = []  # list of (p_cam_3d_meters, p_robot_3d_meters, corner_2d_px)
    current_corners = None  # latest detected 2D corners
    current_T_board = None  # latest board pose
    click_point = None      # set by mouse callback
    status_msg = "Click a corner where the robot TCP is"

    def on_mouse(event, x, y, flags, param):
        nonlocal click_point
        if event == cv2.EVENT_LBUTTONDOWN:
            click_point = (x, y)

    cv2.namedWindow('Calibration')
    cv2.setMouseCallback('Calibration', on_mouse)

    try:
        while True:
            color_image, depth_image, depth_frame = camera.get_frames()
            if color_image is None:
                continue

            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            found, corners = detect_corners(gray)

            display = color_image.copy()

            if found:
                current_corners = corners
                rvec_tvec = compute_board_pose(corners, camera.intrinsics)
                current_T_board = rvec_tvec[0]

                cv2.drawChessboardCorners(display, (BOARD_COLS, BOARD_ROWS), corners, found)

                n_cols = BOARD_COLS
                n = len(corners)
                if n != BOARD_ROWS * BOARD_COLS:
                    for c, r in [(BOARD_COLS, BOARD_ROWS - 2), (BOARD_COLS - 2, BOARD_ROWS),
                                 (BOARD_COLS - 2, BOARD_ROWS - 2)]:
                        if c * r == n:
                            n_cols = c
                            break
            else:
                current_corners = None
                current_T_board = None

            # Handle click
            if click_point is not None and current_corners is not None and current_T_board is not None:
                cx, cy = click_point
                click_point = None

                # Find nearest corner
                pts_2d = current_corners.reshape(-1, 2)
                dists = np.sqrt((pts_2d[:, 0] - cx)**2 + (pts_2d[:, 1] - cy)**2)
                best_idx = np.argmin(dists)
                best_dist = dists[best_idx]

                if best_dist > SNAP_RADIUS_PX:
                    status_msg = f"Click too far from any corner ({best_dist:.0f}px > {SNAP_RADIUS_PX}px)"
                    print(f"  {status_msg}")
                else:
                    # Get corner 3D in camera frame
                    p_cam = corner_3d_in_cam(best_idx, current_T_board, n_cols)

                    # Read robot pose
                    pose = robot.get_pose()
                    if pose is None:
                        status_msg = "ERROR: can't read robot pose"
                        print(f"  {status_msg}")
                    else:
                        # Robot pose XYZ in mm -> convert to meters
                        p_robot_m = np.array(pose[:3]) / 1000.0
                        corner_px = tuple(pts_2d[best_idx].astype(int))

                        pairs.append((p_cam, p_robot_m, corner_px))

                        row = best_idx // n_cols
                        col = best_idx % n_cols
                        status_msg = (f"Point {len(pairs)}: corner({col},{row}) "
                                      f"cam=[{p_cam[0]:.3f},{p_cam[1]:.3f},{p_cam[2]:.3f}]m "
                                      f"robot=[{pose[0]:.1f},{pose[1]:.1f},{pose[2]:.1f}]mm")
                        print(f"  {status_msg}")

            # Draw recorded points
            for i, (_, _, px) in enumerate(pairs):
                cv2.circle(display, px, 8, (0, 255, 255), 2)
                cv2.putText(display, str(i + 1), (px[0] + 10, px[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Status bar
            bar_text = f"{len(pairs)} pts | {status_msg}"
            if not found:
                bar_text = f"{len(pairs)} pts | No board detected"
            cv2.putText(display, bar_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(display, "[s]solve [u]undo [r]reset [q]quit", (10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow('Calibration', display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break
            if cv2.getWindowProperty('Calibration', cv2.WND_PROP_VISIBLE) < 1:
                break

            if key == ord('u') and pairs:
                removed = pairs.pop()
                status_msg = f"Undid point {len(pairs) + 1}"
                print(f"  {status_msg}")

            if key == ord('r'):
                pairs.clear()
                status_msg = "Reset - click corners to start"
                print(f"  {status_msg}")

            if key == ord('s'):
                if len(pairs) < 3:
                    status_msg = f"Need 3+ points (have {len(pairs)})"
                    print(f"  {status_msg}")
                    continue

                pts_cam = [p[0] for p in pairs]
                pts_robot = [p[1] for p in pairs]
                T_cam2base = solve_rigid_transform(pts_cam, pts_robot)

                print(f"\n=== Calibration Result ({len(pairs)} points) ===")
                print(f"T_camera_to_base:")
                print(T_cam2base)

                # Camera position in robot frame
                cam_pos = T_cam2base[:3, 3] * 1000
                print(f"\nCamera position in robot frame: [{cam_pos[0]:.1f}, {cam_pos[1]:.1f}, {cam_pos[2]:.1f}] mm")

                # Reprojection error
                errors = []
                for p_cam, p_robot, _ in pairs:
                    p_hom = np.append(p_cam, 1.0)
                    p_est = (T_cam2base @ p_hom)[:3]
                    err_mm = np.linalg.norm(p_est - p_robot) * 1000
                    errors.append(err_mm)
                print(f"Reprojection errors: {', '.join(f'{e:.1f}' for e in errors)} mm")
                print(f"Mean error: {np.mean(errors):.1f} mm, Max: {np.max(errors):.1f} mm")

                # Save
                from calibration.transform import CoordinateTransform
                ct = CoordinateTransform()
                ct.T_camera_to_base = T_cam2base
                out_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'calibration.yaml')
                ct.save(out_path)
                print(f"Saved to {out_path}")
                status_msg = f"Saved! Mean error: {np.mean(errors):.1f}mm"

    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        robot.close()
        cv2.destroyAllWindows()
        print("\nDone.")


if __name__ == "__main__":
    main()
