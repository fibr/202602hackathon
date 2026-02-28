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
  5. Press 's' to solve with RANSAC + least-squares refinement and save

Usage:
    ./run.sh scripts/detect_checkerboard.py [--hd]

    --sd   Use 640x480 resolution (default: 1280x720 HD)

Arm control (OpenCV window must be focused):
    1-6        Jog J1+..J6+ (hold key, press space to stop)
    !@#$%^     Jog J1-..J6- (shift + 1-6)
    space      Stop jog
    W/S        Y+ / Y-  forward/back (20mm step)
    A/D        X- / X+  left/right   (20mm step)
    Q/E        Z+ / Z-  up/down      (10mm step)
    r/R t/T    Rx/Ry +/- (5 deg step)
    y/Y        Rz+ / Rz-
    [/]        Speed -/+ 10%
    c/o        Gripper close/open
    v          Enable robot
    p          Print pose

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
from vision import RealSenseCamera
from config_loader import load_config

# Checkerboard parameters
BOARD_COLS = 7   # inner corners (8 squares - 1)
BOARD_ROWS = 9   # inner corners (10 squares - 1)
SQUARE_SIZE_M = 0.02  # 2cm squares

SNAP_RADIUS_PX = 30  # max pixel distance to snap click to a corner

# Arm control parameters
CART_STEP_MM = 10.0      # vertical (Q/E) step
CART_STEP_MM_XY = 20.0   # horizontal (WASD) step
CART_STEP_DEG = 5.0
CART_KEYS = {
    ord('w'): (1, +1), ord('s'): (1, -1),   # Y+ / Y- (forward/back)
    ord('a'): (0, -1), ord('d'): (0, +1),   # X- / X+ (left/right)
    ord('q'): (2, +1), ord('e'): (2, -1),   # Z+ / Z- (up/down)
    ord('r'): (3, +1), ord('R'): (3, -1),   # Rx+ / Rx-
    ord('t'): (4, +1), ord('T'): (4, -1),   # Ry+ / Ry-
    ord('y'): (5, +1), ord('Y'): (5, -1),   # Rz+ / Rz-
}
CART_LABELS = {0: 'X', 1: 'Y', 2: 'Z', 3: 'Rx', 4: 'Ry', 5: 'Rz'}

JOG_POS = {ord('1'): 'J1+', ord('2'): 'J2+', ord('3'): 'J3+',
            ord('4'): 'J4+', ord('5'): 'J5+', ord('6'): 'J6+'}
JOG_NEG = {ord('!'): 'J1-', ord('@'): 'J2-', ord('#'): 'J3-',
            ord('$'): 'J4-', ord('%'): 'J5-', ord('^'): 'J6-'}
ALL_JOG = {**JOG_POS, **JOG_NEG}

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


def ray_plane_intersect(pixel, intrinsics, T_board_in_cam):
    """Intersect a camera ray through a pixel with the checkerboard plane.

    The board plane is z=0 in board frame, transformed to camera frame via
    T_board_in_cam. Returns the 3D intersection point in camera frame (meters).

    Args:
        pixel: (x, y) pixel coordinates
        intrinsics: RealSense intrinsics (fx, fy, ppx, ppy)
        T_board_in_cam: 4x4 board-to-camera transform from solvePnP

    Returns:
        np.ndarray [x, y, z] in camera frame (meters), or None if ray is
        parallel to the plane.
    """
    # Ray direction in camera frame (unnormalized)
    px, py = pixel
    ray_dir = np.array([
        (px - intrinsics.ppx) / intrinsics.fx,
        (py - intrinsics.ppy) / intrinsics.fy,
        1.0
    ])

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


def do_cart_step(robot, axis_idx, sign):
    """Step in Cartesian space: read pose, offset, MovL to target.

    Returns:
        status message string
    """
    pose = robot.get_pose()
    if not pose or len(pose) < 6:
        return "ERROR: can't read pose"

    if axis_idx < 2:       # X, Y (horizontal)
        step = CART_STEP_MM_XY
    elif axis_idx == 2:    # Z (vertical)
        step = CART_STEP_MM
    else:                  # rotations
        step = CART_STEP_DEG
    target = list(pose)
    target[axis_idx] += sign * step

    axis_name = CART_LABELS[axis_idx]
    dir_ch = '+' if sign > 0 else '-'

    tp = target
    cmd = f'MovL(pose={{{tp[0]:.2f},{tp[1]:.2f},{tp[2]:.2f},{tp[3]:.2f},{tp[4]:.2f},{tp[5]:.2f}}})'
    resp = robot.send(cmd)
    code = resp.split(',')[0] if resp else '-1'
    if code != '0':
        return f"ERROR: {cmd} -> {resp}"

    # Wait for motion to complete
    time.sleep(0.3)
    prev = robot.get_angles()
    for _ in range(50):
        time.sleep(0.1)
        cur = robot.get_angles()
        if prev and cur and len(prev) >= 6 and len(cur) >= 6:
            if max(abs(cur[i] - prev[i]) for i in range(6)) < 0.05:
                break
        prev = cur

    new_pose = robot.get_pose()
    if new_pose:
        val = ','.join(f'{v:.1f}' for v in new_pose)
        return f"{axis_name}{dir_ch} done  Pose: [{val}]"
    return f"{axis_name}{dir_ch} done"


def main():
    sd = '--sd' in sys.argv
    width, height = (640, 480) if sd else (1280, 720)

    config = load_config()
    rc = config.get('robot', {})
    ip = rc.get('ip', '192.168.5.1')
    port = rc.get('dashboard_port', 29999)

    print("=== Interactive Calibration ===")
    print(f"Board: {BOARD_COLS+1}x{BOARD_ROWS+1} squares, {SQUARE_SIZE_M*100:.0f}cm")
    print(f"Resolution: {width}x{height}")
    print()
    print("Arm control: WASD move, Q/E up/down, 1-6/!@#$%^ jog, space stop")
    print("             c/o gripper, [/] speed, v enable, p pose")
    print("Calibration: touch TCP to board, click on it, Enter solve, u undo, n clear, Esc quit")
    print()

    # Connect to robot
    print(f"Connecting to robot at {ip}:{port}...")
    robot = RobotConnection(ip, port)
    try:
        robot.connect()
    except ConnectionRefusedError:
        print(f"  ERROR: Connection refused at {ip}:{port}")
        print(f"  - Is the robot powered on and booted? (takes ~60s after power cycle)")
        print(f"  - Is another dashboard session already connected? (only one allowed)")
        sys.exit(1)
    except socket.timeout:
        print(f"  ERROR: Connection timed out to {ip}:{port}")
        print(f"  - Is the robot on the network? Try: ping {ip}")
        print(f"  - Check cable connection and IP configuration")
        sys.exit(1)
    except OSError as e:
        print(f"  ERROR: Cannot connect to robot: {e}")
        print(f"  - Verify robot IP in config/robot_config.yaml")
        sys.exit(1)
    print(f"  Connected.")

    # Enable robot
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

    # Start camera
    camera = RealSenseCamera(width=width, height=height, fps=15)
    camera.start()
    print("Camera started.\n")

    # State
    pairs = []  # list of (p_cam_3d_meters, p_robot_3d_meters, corner_2d_px)
    current_corners = None
    current_T_board = None
    click_point = None
    status_msg = "Touch TCP to board, then click on it"
    jogging = False
    last_pose_time = 0.0
    cached_pose = None

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

            # Handle click — ray-plane intersection with checkerboard plane
            if click_point is not None:
                cx, cy = click_point
                click_point = None

                if current_T_board is None:
                    status_msg = "No board detected - need board for ray-plane intersection"
                    print(f"  {status_msg}")
                else:
                    p_cam = ray_plane_intersect((cx, cy), camera.intrinsics, current_T_board)
                    if p_cam is None:
                        status_msg = "Ray parallel to board plane - try different angle"
                        print(f"  {status_msg}")
                    else:
                        pose = robot.get_pose()
                        if pose is None:
                            status_msg = "ERROR: can't read robot pose"
                            print(f"  {status_msg}")
                        else:
                            p_robot_m = np.array(pose[:3]) / 1000.0
                            pairs.append((p_cam, p_robot_m, (cx, cy)))

                            status_msg = (f"Pt {len(pairs)}: "
                                          f"cam=[{p_cam[0]:.3f},{p_cam[1]:.3f},{p_cam[2]:.3f}] "
                                          f"robot=[{pose[0]:.1f},{pose[1]:.1f},{pose[2]:.1f}]")
                            print(f"  {status_msg}")

            # Draw recorded points
            for i, (_, _, px) in enumerate(pairs):
                cv2.circle(display, px, 8, (0, 255, 255), 2)
                cv2.putText(display, str(i + 1), (px[0] + 10, px[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Robot pose overlay (top-right, throttled to avoid lag)
            now = time.time()
            if now - last_pose_time > 0.5:
                last_pose_time = now
                cached_pose = robot.get_pose()
            if cached_pose:
                pose_str = f"TCP: {cached_pose[0]:.1f},{cached_pose[1]:.1f},{cached_pose[2]:.1f}"
                cv2.putText(display, pose_str, (width - 300, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

            # Status bar
            board_status = "Board OK" if found else "No board"
            jog_str = " JOG" if jogging else ""
            bar_text = f"{len(pairs)} pts | Spd:{speed}%{jog_str} | {board_status} | {status_msg}"
            cv2.putText(display, bar_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(display, "WASD move | QE up/dn | 1-6 jog | co grip | Enter solve | u undo | n clear | Esc quit",
                        (10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

            cv2.imshow('Calibration', display)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # Esc only (q is Z+ now)
                break
            if cv2.getWindowProperty('Calibration', cv2.WND_PROP_VISIBLE) < 1:
                break

            # --- Arm control ---

            # Joint jog
            if key in ALL_JOG:
                axis = ALL_JOG[key]
                robot.send(f'MoveJog({axis})')
                jogging = True
                status_msg = f"JOG {axis}"

            # Stop jog
            elif key == ord(' '):
                robot.send('MoveJog()')
                jogging = False
                status_msg = "Stopped"

            # Cartesian step
            elif key in CART_KEYS:
                if jogging:
                    robot.send('MoveJog()')
                    jogging = False
                axis_idx, sign = CART_KEYS[key]
                status_msg = do_cart_step(robot, axis_idx, sign)

            # Gripper
            elif key == ord('c'):
                robot.send('ToolDOInstant(2,0)')
                robot.send('ToolDOInstant(1,1)')
                status_msg = "Gripper CLOSED"
            elif key == ord('o'):
                robot.send('ToolDOInstant(1,0)')
                robot.send('ToolDOInstant(2,1)')
                status_msg = "Gripper OPEN"

            # Speed
            elif key == ord('['):
                speed = max(1, speed - 10)
                robot.send(f'SpeedFactor({speed})')
                status_msg = f"Speed: {speed}%"
            elif key == ord(']'):
                speed = min(100, speed + 10)
                robot.send(f'SpeedFactor({speed})')
                status_msg = f"Speed: {speed}%"

            # Enable
            elif key == ord('v'):
                robot.send('DisableRobot()')
                time.sleep(1)
                robot.send('ClearError()')
                robot.send('EnableRobot()')
                time.sleep(1)
                status_msg = "Robot enabled"

            # Print pose
            elif key == ord('p'):
                pose = robot.get_pose()
                angles = robot.get_angles()
                if pose and angles:
                    print(f"  Pose:   {', '.join(f'{v:.2f}' for v in pose)}")
                    print(f"  Joints: {', '.join(f'{v:.2f}' for v in angles)}")
                    status_msg = f"Pose printed to console"

            # --- Calibration controls ---

            elif key == ord('u') and pairs:
                pairs.pop()
                status_msg = f"Undid -> {len(pairs)} pts remain"
                print(f"  {status_msg}")

            elif key == ord('n'):
                pairs.clear()
                status_msg = "Cleared all points"
                print(f"  {status_msg}")

            elif key == 13:  # Enter
                if len(pairs) < 3:
                    status_msg = f"Need 3+ points (have {len(pairs)})"
                    print(f"  {status_msg}")
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
                from calibration.transform import CoordinateTransform
                ct = CoordinateTransform()
                ct.T_camera_to_base = T_cam2base
                out_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'calibration.yaml')
                ct.save(out_path)
                print(f"Saved to {out_path}")
                status_msg = f"Saved! {n_inliers}/{len(pairs)} inliers, mean {np.mean(inlier_errors):.1f}mm"

    except KeyboardInterrupt:
        pass
    finally:
        robot.send('MoveJog()')  # stop any jog
        camera.stop()
        robot.close()
        cv2.destroyAllWindows()
        print("\nDone.")


if __name__ == "__main__":
    main()
