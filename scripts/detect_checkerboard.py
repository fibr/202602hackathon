#!/usr/bin/env python3
"""Interactive calibration: jog the arm + click on the TCP tip in the image.

Place the TCP tip touching the checkerboard surface and click on it in the
camera image. The click ray is intersected with the checkerboard plane (known
from PnP — no depth sensor needed) to get the 3D point in camera frame.
Robot TCP position comes from GetPose().

Workflow:
  1. Jog the robot so the TCP tip touches a spot on the checkerboard
  2. Click on the TCP tip in the camera image
  3. Script intersects click ray with board plane for camera-frame 3D
  4. Repeat for 4+ positions spread across the board
  5. Press Enter to solve with RANSAC + least-squares refinement and save

Usage:
    ./run.sh scripts/detect_checkerboard.py [--sd]
    ./run.sh scripts/detect_checkerboard.py --verify [--sd] [--dry-run]

    --sd       Use 640x480 resolution (default: 1280x720 HD)
    --verify   Verify calibration: detect board, move arm 5cm above each
               outer corner. Requires saved calibration + robot connection.
    --dry-run  With --verify: compute targets but don't move the robot

Arm control: GUI panel on the right side of the window (XY pad, Z, gripper,
    speed, enable/home). Keyboard shortcuts: 1-6/!@#$%^ jog, space stop,
    c/o gripper, [/] speed, v enable, p print pose.

Calibration:
    click      Record correspondence (ray-plane intersection with board)
    Enter      Solve & save (RANSAC + least-squares, needs 4+ points)
    u          Undo last point
    n          Clear all points
    Esc        Quit
"""

import sys
import os
import socket
import time
import random
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from vision import RealSenseCamera, CameraIntrinsics
from config_loader import load_config
from gui.robot_controls import RobotControlPanel, PANEL_WIDTH
from calibration import CoordinateTransform
from visualization import RobotOverlay

# Checkerboard parameters
BOARD_COLS = 7   # inner corners (8 squares - 1)
BOARD_ROWS = 9   # inner corners (10 squares - 1)
SQUARE_SIZE_M = 0.02  # 2cm squares

SNAP_RADIUS_PX = 30  # max pixel distance to snap click to a corner

# RANSAC parameters
RANSAC_ITERATIONS = 500
RANSAC_INLIER_THRESHOLD_MM = 15.0


class RobotConnection:
    """Dashboard connection for pose queries and arm control."""

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

    def _parse_vals(self, resp):
        """Extract comma-separated floats from '{...}' in response."""
        try:
            inner = resp.split('{')[1].split('}')[0]
            return [float(x) for x in inner.split(',')]
        except (IndexError, ValueError):
            return None

    def get_pose(self):
        """Get TCP pose [x,y,z,rx,ry,rz] in mm/deg, or None."""
        return self._parse_vals(self.send('GetPose()'))

    def get_angles(self):
        """Get joint angles [j1..j6] in deg, or None."""
        return self._parse_vals(self.send('GetAngle()'))

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
    """solvePnP -> (T_board_in_cam 4x4, obj_points, reproj_error_px).

    reproj_error_px is the RMS reprojection error in pixels — measures how
    well the current intrinsics explain the detected corners.
    """
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

    # Compute reprojection error
    projected, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
    reproj_err = np.sqrt(np.mean(
        (corners_2d.reshape(-1, 2) - projected.reshape(-1, 2)) ** 2))

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T, obj_points, reproj_err


def corner_3d_in_cam(corner_idx, T_board_in_cam, n_cols=BOARD_COLS):
    """Get the 3D position of a detected corner in camera frame (meters)."""
    row = corner_idx // n_cols
    col = corner_idx % n_cols
    p_board = np.array([col * SQUARE_SIZE_M, row * SQUARE_SIZE_M, 0, 1])
    p_cam = (T_board_in_cam @ p_board)[:3]
    return p_cam


def pixel_to_ray(pixel, intrinsics):
    """Convert a pixel to a normalized ray direction, accounting for distortion.

    Uses cv2.undistortPoints to remove lens distortion, giving a more accurate
    ray direction than raw pinhole backprojection.

    Args:
        pixel: (x, y) pixel coordinates
        intrinsics: CameraIntrinsics or pyrealsense2 intrinsics

    Returns:
        np.ndarray [x, y, 1] unnormalized ray direction in camera frame
    """
    px, py = pixel
    camera_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float64)

    # undistortPoints with no newCameraMatrix returns normalized coords
    pts = cv2.undistortPoints(
        np.array([[[px, py]]], dtype=np.float64),
        camera_matrix, dist_coeffs)
    return np.array([pts[0][0][0], pts[0][0][1], 1.0])


def ray_plane_intersect(pixel, intrinsics, T_board_in_cam):
    """Intersect a camera ray through a pixel with the checkerboard plane.

    The board plane is z=0 in board frame, transformed to camera frame via
    T_board_in_cam. Returns the 3D intersection point in camera frame (meters).

    Uses cv2.undistortPoints for accurate ray computation with distortion.

    Args:
        pixel: (x, y) pixel coordinates
        intrinsics: CameraIntrinsics or pyrealsense2 intrinsics
        T_board_in_cam: 4x4 board-to-camera transform from solvePnP

    Returns:
        np.ndarray [x, y, z] in camera frame (meters), or None if ray is
        parallel to the plane.
    """
    ray_dir = pixel_to_ray(pixel, intrinsics)

    # Board plane: the board's z=0 plane in camera frame
    # Normal = R @ [0,0,1] (board z-axis in camera frame)
    # Point on plane = translation column of T_board_in_cam
    R = T_board_in_cam[:3, :3]
    t = T_board_in_cam[:3, 3]
    plane_normal = R[:, 2]  # third column of R
    plane_point = t

    # Ray-plane intersection: ray_origin=0, ray_dir
    # t_param = dot(plane_normal, plane_point) / dot(plane_normal, ray_dir)
    denom = np.dot(plane_normal, ray_dir)
    if abs(denom) < 1e-8:
        return None  # ray parallel to plane

    t_param = np.dot(plane_normal, plane_point) / denom
    if t_param < 0:
        return None  # intersection behind camera

    return ray_dir * t_param


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


def _refine_transform(T_init, pts_cam, pts_robot):
    """Refine rigid transform via least-squares optimization on SE(3).

    Parameterizes the transform as 6 DOF (3 rotation via Rodrigues + 3 translation)
    and minimizes the sum of squared residuals.

    Args:
        T_init: 4x4 initial transform estimate
        pts_cam: Nx3 points in camera frame (meters)
        pts_robot: Nx3 points in robot frame (meters)

    Returns:
        4x4 refined transform
    """
    A = np.array(pts_cam)
    B = np.array(pts_robot)

    # Extract initial params: rotation as Rodrigues vector + translation
    R_init = T_init[:3, :3]
    t_init = T_init[:3, 3]
    rotvec_init = Rotation.from_matrix(R_init).as_rotvec()
    x0 = np.concatenate([rotvec_init, t_init])

    def residuals(x):
        R = Rotation.from_rotvec(x[:3]).as_matrix()
        t = x[3:6]
        transformed = (R @ A.T).T + t
        return (transformed - B).ravel()

    result = least_squares(residuals, x0, method='lm')

    R_opt = Rotation.from_rotvec(result.x[:3]).as_matrix()
    t_opt = result.x[3:6]
    T = np.eye(4)
    T[:3, :3] = R_opt
    T[:3, 3] = t_opt
    return T


def solve_robust_transform(pts_cam, pts_robot):
    """RANSAC + SVD + least-squares refinement for rigid transform.

    Args:
        pts_cam: Nx3 points in camera frame (meters)
        pts_robot: Nx3 points in robot frame (meters)

    Returns:
        (T_cam_to_base 4x4, inlier_mask bool array)
    """
    N = len(pts_cam)
    A = np.array(pts_cam)
    B = np.array(pts_robot)

    if N < 4:
        # Not enough for RANSAC, plain SVD + refine
        T = solve_rigid_transform(pts_cam, pts_robot)
        T = _refine_transform(T, A, B)
        return T, np.ones(N, dtype=bool)

    threshold_m = RANSAC_INLIER_THRESHOLD_MM / 1000.0

    best_inliers = None
    best_count = 0

    for _ in range(RANSAC_ITERATIONS):
        idx = random.sample(range(N), 3)
        T_cand = solve_rigid_transform(A[idx], B[idx])

        errors = np.array([
            np.linalg.norm((T_cand @ np.append(A[i], 1.0))[:3] - B[i])
            for i in range(N)
        ])
        inliers = errors < threshold_m
        count = inliers.sum()

        if count > best_count:
            best_count = count
            best_inliers = inliers

    # Refit SVD on inliers, then refine with least-squares
    inlier_idx = np.where(best_inliers)[0]
    T_svd = solve_rigid_transform(A[inlier_idx], B[inlier_idx])
    T_final = _refine_transform(T_svd, A[inlier_idx], B[inlier_idx])
    return T_final, best_inliers


VERIFY_HOVER_HEIGHT_M = 0.05  # 5cm above each corner
VERIFY_SPEED_PERCENT = 20


def _get_board_outer_corners_cam(T_board_in_cam):
    """Get 4 outer corners of the checkerboard in camera frame (meters)."""
    half = SQUARE_SIZE_M / 2.0
    max_x = (BOARD_COLS - 1) * SQUARE_SIZE_M
    max_y = (BOARD_ROWS - 1) * SQUARE_SIZE_M
    board_corners = [
        np.array([-half, -half, 0, 1]),
        np.array([max_x + half, -half, 0, 1]),
        np.array([max_x + half, max_y + half, 0, 1]),
        np.array([-half, max_y + half, 0, 1]),
    ]
    labels = ["top-left", "top-right", "bottom-right", "bottom-left"]
    corners_cam = [(T_board_in_cam @ pt)[:3] for pt in board_corners]
    return corners_cam, labels


def run_verify(width, height, dry_run):
    """Verify calibration by moving the arm above checkerboard corners."""
    calib_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'calibration.yaml')
    if not os.path.exists(calib_path):
        print(f"ERROR: No calibration file at {calib_path}")
        print("Run detect_checkerboard.py first to create it.")
        sys.exit(1)

    transform = CoordinateTransform()
    transform.load(calib_path)
    print(f"Loaded calibration from {calib_path}")

    camera = RealSenseCamera(width=width, height=height, fps=15)
    camera.start()
    print("Camera started. Looking for checkerboard (press 'c' to capture, 'q' to abort)...")

    corners_2d = None
    T_board_in_cam = None
    try:
        while True:
            color_image, _, _ = camera.get_frames()
            if color_image is None:
                continue
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            found, corners = detect_corners(gray)

            display = color_image.copy()
            if found:
                cv2.drawChessboardCorners(display, (BOARD_COLS, BOARD_ROWS), corners, found)
                cv2.putText(display, f"Found {len(corners)} corners - press 'c'",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "No checkerboard",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Calibration Verify', display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("Aborted.")
                return
            try:
                if cv2.getWindowProperty('Calibration Verify', cv2.WND_PROP_VISIBLE) < 1:
                    print("Aborted.")
                    return
            except cv2.error:
                print("Aborted.")
                return

            if key == ord('c') and found:
                T_board_in_cam, _, _ = compute_board_pose(corners, camera.intrinsics)
                corners_2d = corners
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()

    corners_cam, labels = _get_board_outer_corners_cam(T_board_in_cam)

    print("\n=== Checkerboard corners in robot base frame ===")
    targets_base = []
    for i, (p_cam, label) in enumerate(zip(corners_cam, labels)):
        p_base = transform.camera_to_base(p_cam)
        p_base_mm = p_base * 1000.0
        hover_mm = p_base_mm.copy()
        hover_mm[2] += VERIFY_HOVER_HEIGHT_M * 1000.0
        print(f"\n  Corner {i} ({label}):")
        print(f"    Camera frame:  [{p_cam[0]:.4f}, {p_cam[1]:.4f}, {p_cam[2]:.4f}] m")
        print(f"    Robot base:    [{p_base_mm[0]:.1f}, {p_base_mm[1]:.1f}, {p_base_mm[2]:.1f}] mm")
        print(f"    Hover target:  [{hover_mm[0]:.1f}, {hover_mm[1]:.1f}, {hover_mm[2]:.1f}] mm")
        targets_base.append(hover_mm)

    if dry_run:
        print("\n[DRY RUN] Would move to above targets. Exiting.")
        return

    from robot.dobot_api import DobotNova5
    print("\n=== Connecting to robot ===")
    robot = DobotNova5()
    robot.connect()
    print("  Connected.")
    robot.enable()
    robot.set_speed(VERIFY_SPEED_PERCENT)

    pose = robot.get_pose()
    print(f"  Current pose: {', '.join(f'{v:.1f}' for v in pose)}")
    rx, ry, rz = pose[3], pose[4], pose[5]
    print(f"  Using orientation: rx={rx:.1f}, ry={ry:.1f}, rz={rz:.1f}")

    input("\nPress Enter to start moving to corners (Ctrl+C to abort)...")

    try:
        for i, (target_mm, label) in enumerate(zip(targets_base, labels)):
            print(f"\n--- Moving to corner {i} ({label}) ---")
            print(f"  Target: [{target_mm[0]:.1f}, {target_mm[1]:.1f}, {target_mm[2]:.1f}] mm")
            ok = robot.movj(target_mm[0], target_mm[1], target_mm[2], rx, ry, rz)
            if not ok:
                print(f"  ERROR: move failed for corner {i}, skipping")
                continue
            actual = robot.get_pose()
            print(f"  Actual pose:  {', '.join(f'{v:.1f}' for v in actual)}")
            err = np.linalg.norm(np.array(actual[:3]) - target_mm)
            print(f"  Position error: {err:.1f} mm")
            if i < len(targets_base) - 1:
                input("  Press Enter for next corner...")
        print("\n=== Verification complete! ===")
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
    finally:
        robot.disconnect()
        print("Robot disconnected.")


def main():
    sd = '--sd' in sys.argv
    verify = '--verify' in sys.argv
    dry_run = '--dry-run' in sys.argv
    width, height = (640, 480) if sd else (1280, 720)

    if verify:
        run_verify(width, height, dry_run)
        return

    config = load_config()
    rc = config.get('robot', {})
    ip = rc.get('ip', '192.168.5.1')
    port = rc.get('dashboard_port', 29999)

    print("=== Interactive Calibration ===")
    print(f"Board: {BOARD_COLS+1}x{BOARD_ROWS+1} squares, {SQUARE_SIZE_M*100:.0f}cm")
    print(f"Resolution: {width}x{height}")
    print()
    print("Arm control: GUI panel on right (XY pad, Z, gripper, speed, enable/home)")
    print("             Keyboard: 1-6/!@#$%^ jog, space stop, c/o gripper, [/] speed, v enable")
    print("Intrinsics:  i capture frame | I calibrate & save (need 10+ frames)")
    print("Plane:       g save ground plane from current board detection")
    print("Hand-eye:    click record point | Enter solve | u undo | n clear")
    print("             Esc quit | p print pose")
    print()

    # Connect to robot (optional — camera-only mode if unavailable)
    robot = None
    print(f"Connecting to robot at {ip}:{port}...")
    try:
        robot = RobotConnection(ip, port)
        robot.connect()
        print(f"  Connected.")

        robot.send('DisableRobot()')
        time.sleep(1)
        robot.send('ClearError()')
        robot.send('EnableRobot()')
        time.sleep(1)

        speed = 30
        robot.send(f'SpeedFactor({speed})')

        pose = robot.get_pose()
        if pose:
            print(f"  Robot pose: {', '.join(f'{v:.1f}' for v in pose)}")
        else:
            print("  WARNING: couldn't read robot pose")
    except (ConnectionRefusedError, socket.timeout, OSError) as e:
        print(f"  WARNING: Cannot connect to robot: {e}")
        print(f"  Continuing in camera-only mode (no arm control)")
        robot = None

    # Start camera
    camera = RealSenseCamera(width=width, height=height, fps=15)
    camera.start()
    print("Camera started.\n")

    # GUI panel (only if robot connected)
    panel = None
    if robot:
        panel = RobotControlPanel(robot, panel_x=width, panel_height=height)
        panel.speed = speed
        panel.status_msg = "Touch TCP to board, then click on it"

    # Robot overlay (load calibration if available)
    robot_overlay = None
    calibration_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'calibration.yaml')
    if os.path.exists(calibration_path):
        transform = CoordinateTransform()
        transform.load(calibration_path)
        gripper_cfg = config.get('gripper', {})
        robot_overlay = RobotOverlay(
            T_camera_to_base=transform.T_camera_to_base,
            tool_length_mm=gripper_cfg.get('tool_length_mm', 120.0),
            base_offset_mm=transform.base_offset_mm,
            base_rpy_deg=transform.base_rpy_deg,
        )
        print(f"Loaded calibration for robot overlay")

    # State
    pairs = []  # list of (p_cam_3d_meters, p_robot_3d_meters, corner_2d_px)
    current_corners = None
    current_T_board = None
    click_point = None

    # Intrinsics calibration state
    intr_frames = []  # list of (obj_points, img_points) for cv2.calibrateCamera
    intr_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'camera_intrinsics.yaml')

    def on_mouse(event, x, y, flags, param):
        nonlocal click_point
        # Route to panel if in panel area
        if panel and x >= width:
            panel.handle_mouse(event, x, y, flags)
            return
        # Camera area: record calibration click
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

            # Create canvas (expanded with panel area if robot connected)
            canvas_w = width + PANEL_WIDTH if panel else width
            canvas = np.zeros((height, canvas_w, 3), dtype=np.uint8)
            display = color_image.copy()

            reproj_err = None
            if found:
                current_corners = corners
                current_T_board, _, reproj_err = compute_board_pose(
                    corners, camera.intrinsics)

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

            # Handle click — ray-plane intersection with checkerboard plane
            if click_point is not None:
                cx, cy = click_point
                click_point = None

                if current_T_board is None:
                    msg = "No board detected - need board for ray-plane intersection"
                    if panel:
                        panel.status_msg = msg
                    print(f"  {msg}")
                elif not robot:
                    msg = "No robot connected - can't record calibration point"
                    print(f"  {msg}")
                else:
                    p_cam = ray_plane_intersect((cx, cy), camera.intrinsics, current_T_board)
                    if p_cam is None:
                        msg = "Ray parallel to board plane - try different angle"
                        if panel:
                            panel.status_msg = msg
                        print(f"  {msg}")
                    else:
                        pose = robot.get_pose()
                        if pose is None:
                            msg = "ERROR: can't read robot pose"
                            if panel:
                                panel.status_msg = msg
                            print(f"  {msg}")
                        else:
                            p_robot_m = np.array(pose[:3]) / 1000.0
                            pairs.append((p_cam, p_robot_m, (cx, cy)))

                            msg = (f"Pt {len(pairs)}: "
                                   f"cam=[{p_cam[0]:.3f},{p_cam[1]:.3f},{p_cam[2]:.3f}] "
                                   f"robot=[{pose[0]:.1f},{pose[1]:.1f},{pose[2]:.1f}]")
                            if panel:
                                panel.status_msg = msg
                            print(f"  {msg}")

            # Draw recorded points
            for i, (_, _, px) in enumerate(pairs):
                cv2.circle(display, px, 8, (0, 255, 255), 2)
                cv2.putText(display, str(i + 1), (px[0] + 10, px[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Robot joint overlay
            if robot_overlay and camera.intrinsics is not None:
                if robot:
                    angles = robot.get_angles()
                    if angles:
                        display = robot_overlay.draw_joints(
                            display, np.array(angles), camera.intrinsics)
                    else:
                        display = robot_overlay.draw_base_marker(
                            display, camera.intrinsics)
                else:
                    display = robot_overlay.draw_base_marker(
                        display, camera.intrinsics)

            # Status bar on camera image
            board_status = f"Board OK reproj:{reproj_err:.2f}px" if found and reproj_err is not None else ("Board OK" if found else "No board")
            intr_str = f"Intr:{len(intr_frames)}" if intr_frames else ""
            if panel:
                jog_str = " JOG" if panel.jogging else ""
                bar_text = f"{len(pairs)} pts | {intr_str} | Spd:{panel.speed}%{jog_str} | {board_status}"
            else:
                bar_text = f"{len(pairs)} pts | {intr_str} | {board_status} | No robot"
            cv2.putText(display, bar_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(display, "i intr capture | I calibrate | g plane | Enter solve | u undo | Esc quit",
                        (10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

            # Compose canvas: camera on left, panel on right (if robot)
            canvas[0:height, 0:width] = display
            if panel:
                panel.draw(canvas)

            cv2.imshow('Calibration', canvas)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # Esc
                break
            if cv2.getWindowProperty('Calibration', cv2.WND_PROP_VISIBLE) < 1:
                break

            # --- Arm control via shared panel keyboard handler ---
            if key != 255 and panel and panel.handle_key(key):
                pass  # consumed by panel

            # Print pose (keep as keyboard shortcut)
            elif key == ord('p') and robot:
                pose = robot.get_pose()
                angles = robot.get_angles()
                if pose and angles:
                    print(f"  Pose:   {', '.join(f'{v:.2f}' for v in pose)}")
                    print(f"  Joints: {', '.join(f'{v:.2f}' for v in angles)}")
                    if panel:
                        panel.status_msg = "Pose printed to console"

            # --- Calibration controls ---

            elif key == ord('u') and pairs:
                pairs.pop()
                msg = f"Undid -> {len(pairs)} pts remain"
                if panel:
                    panel.status_msg = msg
                print(f"  {msg}")

            elif key == ord('n'):
                pairs.clear()
                msg = "Cleared all points"
                if panel:
                    panel.status_msg = msg
                print(f"  {msg}")

            # --- Intrinsics calibration ---

            elif key == ord('i'):
                # Capture frame for intrinsics calibration
                if not found or current_corners is None:
                    msg = "No board detected — can't capture for intrinsics"
                    if panel:
                        panel.status_msg = msg
                    print(f"  {msg}")
                else:
                    n = len(current_corners)
                    # Determine grid size for this detection
                    i_cols, i_rows = BOARD_COLS, BOARD_ROWS
                    if n != BOARD_ROWS * BOARD_COLS:
                        for cc, rr in [(BOARD_COLS, BOARD_ROWS - 2),
                                       (BOARD_COLS - 2, BOARD_ROWS),
                                       (BOARD_COLS - 2, BOARD_ROWS - 2)]:
                            if cc * rr == n:
                                i_cols, i_rows = cc, rr
                                break
                        else:
                            msg = f"Unexpected corner count {n}, skipping"
                            if panel:
                                panel.status_msg = msg
                            print(f"  {msg}")
                            continue
                    obj_pts = np.zeros((i_rows * i_cols, 3), dtype=np.float32)
                    for rr in range(i_rows):
                        for cc in range(i_cols):
                            obj_pts[rr * i_cols + cc] = [
                                cc * SQUARE_SIZE_M, rr * SQUARE_SIZE_M, 0]
                    intr_frames.append((obj_pts, current_corners.copy()))
                    msg = f"Intrinsics frame {len(intr_frames)} captured"
                    if panel:
                        panel.status_msg = msg
                    print(f"  {msg}")

            elif key == ord('I'):
                # Run intrinsics calibration
                if len(intr_frames) < 5:
                    msg = f"Need 5+ frames for intrinsics (have {len(intr_frames)})"
                    if panel:
                        panel.status_msg = msg
                    print(f"  {msg}")
                else:
                    obj_points = [f[0] for f in intr_frames]
                    img_points = [f[1] for f in intr_frames]
                    print(f"\n=== Calibrating intrinsics from {len(intr_frames)} frames ===")
                    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                        obj_points, img_points, (width, height), None, None)
                    print(f"  Reprojection error: {ret:.4f} px")
                    print(f"  Camera matrix:\n{mtx}")
                    print(f"  Distortion: {dist.ravel()}")

                    # Save
                    calib_intr = CameraIntrinsics(
                        fx=mtx[0, 0], fy=mtx[1, 1],
                        ppx=mtx[0, 2], ppy=mtx[1, 2],
                        coeffs=dist.ravel().tolist())
                    calib_intr.width = width
                    calib_intr.height = height
                    calib_intr.save(intr_path)
                    print(f"  Saved to {intr_path}")

                    # Apply immediately
                    camera.intrinsics = calib_intr
                    msg = f"Intrinsics saved! reproj={ret:.3f}px"
                    if panel:
                        panel.status_msg = msg
                    print(f"  {msg}")

            # --- Ground plane calibration ---

            elif key == ord('g'):
                # Save ground plane normal + offset from current board detection
                if current_T_board is None:
                    msg = "No board detected — can't save ground plane"
                    if panel:
                        panel.status_msg = msg
                    print(f"  {msg}")
                else:
                    # Board z=0 plane in camera frame
                    R_board = current_T_board[:3, :3]
                    t_board = current_T_board[:3, 3]
                    plane_normal = R_board[:, 2]  # board Z axis in camera frame
                    plane_d = np.dot(plane_normal, t_board)  # signed distance from origin

                    plane_path = os.path.join(
                        os.path.dirname(__file__), '..', 'config', 'ground_plane.yaml')
                    import yaml
                    os.makedirs(os.path.dirname(plane_path), exist_ok=True)
                    data = {
                        'plane_normal': plane_normal.tolist(),
                        'plane_d': float(plane_d),
                        'T_board_in_cam': current_T_board.tolist(),
                    }
                    with open(plane_path, 'w') as f:
                        yaml.dump(data, f, default_flow_style=False)

                    print(f"\n=== Ground Plane Saved ===")
                    print(f"  Normal (cam): [{plane_normal[0]:.4f}, {plane_normal[1]:.4f}, {plane_normal[2]:.4f}]")
                    print(f"  Distance: {plane_d:.4f} m ({plane_d*1000:.1f} mm)")
                    print(f"  Saved to {plane_path}")
                    msg = f"Plane saved: d={plane_d*1000:.1f}mm"
                    if panel:
                        panel.status_msg = msg

            elif key == 13:  # Enter
                if len(pairs) < 3:
                    msg = f"Need 3+ points (have {len(pairs)})"
                    if panel:
                        panel.status_msg = msg
                    print(f"  {msg}")
                    continue

                pts_cam = [p[0] for p in pairs]
                pts_robot = [p[1] for p in pairs]

                T_cam2base, inlier_mask = solve_robust_transform(pts_cam, pts_robot)

                n_inliers = inlier_mask.sum()
                n_outliers = len(pairs) - n_inliers

                print(f"\n=== Calibration Result ({len(pairs)} pts, {n_inliers} inliers, {n_outliers} outliers) ===")
                print("T_camera_to_base:")
                print(T_cam2base)

                cam_pos = T_cam2base[:3, 3] * 1000
                print(f"\nCamera in robot frame: [{cam_pos[0]:.1f}, {cam_pos[1]:.1f}, {cam_pos[2]:.1f}] mm")

                # Per-point errors
                print("\nPer-point errors:")
                errors = []
                for i, (p_cam, p_robot, _) in enumerate(pairs):
                    p_hom = np.append(p_cam, 1.0)
                    p_est = (T_cam2base @ p_hom)[:3]
                    err_mm = np.linalg.norm(p_est - p_robot) * 1000
                    errors.append(err_mm)
                    tag = "  " if inlier_mask[i] else "* "
                    print(f"  {tag}Point {i+1}: {err_mm:.1f} mm{'  <-- OUTLIER' if not inlier_mask[i] else ''}")

                inlier_errors = [e for e, m in zip(errors, inlier_mask) if m]
                print(f"\nInlier mean: {np.mean(inlier_errors):.1f} mm, max: {np.max(inlier_errors):.1f} mm")

                # Save
                ct = CoordinateTransform()
                ct.T_camera_to_base = T_cam2base
                out_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'calibration.yaml')
                ct.save(out_path)
                print(f"Saved to {out_path}")
                msg = f"Saved! {n_inliers}/{len(pairs)} inliers, mean {np.mean(inlier_errors):.1f}mm"
                if panel:
                    panel.status_msg = msg

    except KeyboardInterrupt:
        pass
    finally:
        if robot:
            robot.send('MoveJog()')  # stop any jog
        camera.stop()
        if robot:
            robot.close()
        cv2.destroyAllWindows()
        print("\nDone.")


if __name__ == "__main__":
    main()
