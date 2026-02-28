#!/usr/bin/env python3
"""Detect a checkerboard with RealSense and compute camera-to-robot transform.

Checkerboard spec: 8x10 squares, 2cm square size -> 7x9 inner corners.
The user places the checkerboard at a known position relative to the robot base.

Usage:
    ./run.sh scripts/detect_checkerboard.py [--hd]

    --hd   Use 1280x720 resolution (default: 640x480)

Keys:
    c  Capture and compute (freezes on detected board)
    s  Save calibration to config/calibration.yaml
    r  Reset / re-detect
    q  Quit
"""

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from vision import RealSenseCamera

# Note: solvePnP uses 2D corners + known square size to compute pose,
# so it does NOT need depth. This works even if the board is at the edge
# of the depth sensor range (~0.3m for D435i).

# Checkerboard parameters
BOARD_COLS = 7   # inner corners (8 squares - 1)
BOARD_ROWS = 9   # inner corners (10 squares - 1)
SQUARE_SIZE_M = 0.02  # 2cm squares


def detect_corners(gray):
    """Find checkerboard corners using multiple strategies.

    Tries (in order):
    1. SectorBased detector (OpenCV 4+, most robust)
    2. Classic detector with CLAHE preprocessing
    3. Classic detector with aggressive adaptive threshold
    """
    # Strategy 1: SectorBased detector (best for small/distant boards)
    try:
        found, corners = cv2.findChessboardCornersSB(
            gray, (BOARD_COLS, BOARD_ROWS),
            cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if found:
            return found, corners
    except cv2.error:
        pass  # older OpenCV without SB support

    # Strategy 2: CLAHE contrast enhancement + classic detector
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
             cv2.CALIB_CB_NORMALIZE_IMAGE)
    found, corners = cv2.findChessboardCorners(enhanced, (BOARD_COLS, BOARD_ROWS), flags)
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        return found, corners

    # Strategy 3: sharpen + classic detector
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

    # Also try smaller board sizes in case edges are cut off
    for cols, rows in [(BOARD_COLS, BOARD_ROWS - 2), (BOARD_COLS - 2, BOARD_ROWS),
                       (BOARD_COLS - 2, BOARD_ROWS - 2)]:
        found, corners = cv2.findChessboardCornersSB(
            gray, (cols, rows), cv2.CALIB_CB_EXHAUSTIVE)
        if found:
            print(f"  (detected {cols}x{rows} subset of board)")
            return found, corners

    return False, None


def get_corners_3d(corners, depth_frame, camera):
    """Get 3D positions of detected corners using depth."""
    points_3d = []
    for pt in corners.reshape(-1, 2):
        x, y = int(round(pt[0])), int(round(pt[1]))
        x = max(0, min(x, camera.width - 1))
        y = max(0, min(y, camera.height - 1))
        p3d = camera.pixel_to_3d(x, y, depth_frame)
        points_3d.append(p3d)
    return np.array(points_3d)


def compute_board_pose(corners_2d, intrinsics, n_corners=None):
    """Use solvePnP to get checkerboard pose in camera frame.

    Returns (rvec, tvec, obj_points) where tvec is the board origin
    in camera coordinates (meters).
    """
    # Figure out detected grid size from corner count
    n = len(corners_2d) if n_corners is None else n_corners
    if n == BOARD_ROWS * BOARD_COLS:
        cols, rows = BOARD_COLS, BOARD_ROWS
    else:
        # Find matching smaller grid
        for c, r in [(BOARD_COLS, BOARD_ROWS - 2), (BOARD_COLS - 2, BOARD_ROWS),
                     (BOARD_COLS - 2, BOARD_ROWS - 2)]:
            if c * r == n:
                cols, rows = c, r
                break
        else:
            cols, rows = BOARD_COLS, BOARD_ROWS  # fallback

    # Object points: 3D coordinates of inner corners in board frame
    obj_points = np.zeros((rows * cols, 3), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            obj_points[r * cols + c] = [c * SQUARE_SIZE_M, r * SQUARE_SIZE_M, 0]

    # Camera matrix from RealSense intrinsics
    camera_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float64)

    _, rvec, tvec = cv2.solvePnP(obj_points, corners_2d, camera_matrix, dist_coeffs)
    return rvec, tvec, obj_points


def board_pose_to_transform(rvec, tvec):
    """Convert solvePnP result to 4x4 transform (board frame in camera frame)."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def main():
    hd = '--hd' in sys.argv
    width, height = (1280, 720) if hd else (640, 480)

    print("=== Checkerboard Detection & Calibration ===")
    print(f"Board: {BOARD_COLS+1}x{BOARD_ROWS+1} squares, {SQUARE_SIZE_M*100:.0f}cm square size")
    print(f"Looking for {BOARD_COLS}x{BOARD_ROWS} inner corners")
    print(f"Resolution: {width}x{height}" + (" (use --hd for 1280x720)" if not hd else ""))
    print()
    print("Keys: [c] capture  [s] save calibration  [r] reset  [q] quit")
    print()

    camera = RealSenseCamera(width=width, height=height, fps=15)
    camera.start()
    print("Camera started.")

    captured = False
    captured_image = None
    board_T_cam = None  # 4x4: board frame expressed in camera frame
    corners_3d_cam = None

    try:
        while True:
            if not captured:
                color_image, depth_image, depth_frame = camera.get_frames()
                if color_image is None:
                    continue

                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                found, corners = detect_corners(gray)

                display = color_image.copy()

                # Build debug view: CLAHE enhanced grayscale
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                debug_gray = clahe.apply(gray)
                debug_view = cv2.cvtColor(debug_gray, cv2.COLOR_GRAY2BGR)

                if found:
                    n_corners = len(corners)
                    cv2.drawChessboardCorners(display, (BOARD_COLS, BOARD_ROWS), corners, found)
                    cv2.drawChessboardCorners(debug_view, (BOARD_COLS, BOARD_ROWS), corners, found)
                    cv2.putText(display, f"Corners: {n_corners} - press 'c' to capture",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display, "No checkerboard found",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(debug_view, "CLAHE enhanced",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                display = np.hstack([display, debug_view])
            else:
                display = captured_image

            cv2.imshow('Checkerboard Calibration', display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break
            if cv2.getWindowProperty('Checkerboard Calibration', cv2.WND_PROP_VISIBLE) < 1:
                break

            if key == ord('c') and not captured:
                if not found:
                    print("No checkerboard detected - can't capture.")
                    continue

                print("\nCapturing...")

                # solvePnP for board pose
                rvec, tvec, obj_points = compute_board_pose(corners, camera.intrinsics)
                board_T_cam = board_pose_to_transform(rvec, tvec)

                # Also get depth-based 3D corners
                corners_3d_cam = get_corners_3d(corners, depth_frame, camera)

                # Board origin (first inner corner) in camera frame
                origin_cam = tvec.flatten()
                # Board corner opposite
                far_corner = board_T_cam @ np.array([
                    (BOARD_COLS - 1) * SQUARE_SIZE_M,
                    (BOARD_ROWS - 1) * SQUARE_SIZE_M,
                    0, 1])

                print(f"\n--- Board Pose in Camera Frame ---")
                print(f"  Origin (corner 0,0): [{origin_cam[0]:.4f}, {origin_cam[1]:.4f}, {origin_cam[2]:.4f}] m")
                print(f"  Far corner:          [{far_corner[0]:.4f}, {far_corner[1]:.4f}, {far_corner[2]:.4f}] m")
                print(f"  Distance to board:   {np.linalg.norm(origin_cam):.3f} m")

                # Show depth-based 3D for comparison
                c0_depth = corners_3d_cam[0]
                print(f"\n  Depth-based corner 0: [{c0_depth[0]:.4f}, {c0_depth[1]:.4f}, {c0_depth[2]:.4f}] m")

                # Board axes in camera frame
                R_board = board_T_cam[:3, :3]
                x_axis = R_board[:, 0]  # along cols (short edge for 7 corners)
                y_axis = R_board[:, 1]  # along rows (long edge for 9 corners)
                z_axis = R_board[:, 2]  # normal to board
                print(f"\n  Board X axis (cols) in cam: [{x_axis[0]:.3f}, {x_axis[1]:.3f}, {x_axis[2]:.3f}]")
                print(f"  Board Y axis (rows) in cam: [{y_axis[0]:.3f}, {y_axis[1]:.3f}, {y_axis[2]:.3f}]")
                print(f"  Board Z axis (normal) in cam: [{z_axis[0]:.3f}, {z_axis[1]:.3f}, {z_axis[2]:.3f}]")

                print(f"\n  Board dimensions: X = {(BOARD_COLS-1)*SQUARE_SIZE_M*100:.0f}cm, Y = {(BOARD_ROWS-1)*SQUARE_SIZE_M*100:.0f}cm")
                print(f"\n--- Next Steps ---")
                print(f"  You said the board corner is 36cm and 21cm from robot base.")
                print(f"  Use the robot's GetPose() or known position to determine")
                print(f"  the board origin in robot base frame, then we can compute")
                print(f"  the camera-to-base transform.")
                print(f"\n  Press 's' to save (after entering robot-frame coords),")
                print(f"  or 'r' to re-detect.")

                # Freeze display with detection overlay
                captured_image = color_image.copy()
                cv2.drawChessboardCorners(captured_image, (BOARD_COLS, BOARD_ROWS), corners, True)

                # Draw origin and axes on image
                img_pts, _ = cv2.projectPoints(
                    np.float32([[0, 0, 0], [0.06, 0, 0], [0, 0.06, 0], [0, 0, 0.06]]),
                    rvec, tvec,
                    np.array([[camera.intrinsics.fx, 0, camera.intrinsics.ppx],
                              [0, camera.intrinsics.fy, camera.intrinsics.ppy],
                              [0, 0, 1]], dtype=np.float64),
                    np.array(camera.intrinsics.coeffs, dtype=np.float64))
                o = tuple(img_pts[0].ravel().astype(int))
                px = tuple(img_pts[1].ravel().astype(int))
                py = tuple(img_pts[2].ravel().astype(int))
                pz = tuple(img_pts[3].ravel().astype(int))
                cv2.line(captured_image, o, px, (0, 0, 255), 2)   # X = red
                cv2.line(captured_image, o, py, (0, 255, 0), 2)   # Y = green
                cv2.line(captured_image, o, pz, (255, 0, 0), 2)   # Z = blue
                cv2.putText(captured_image, "CAPTURED - 'r' to redo, 's' to save, 'q' to quit",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                captured_image = np.hstack([captured_image, depth_colormap])
                captured = True

            elif key == ord('r'):
                captured = False
                board_T_cam = None
                print("\nReset - looking for checkerboard...")

            elif key == ord('s') and captured and board_T_cam is not None:
                print("\n--- Save Calibration ---")
                print("Enter the board origin (corner 0,0) position in ROBOT BASE frame.")
                print("(The corner where detection starts - see red/green axes on image)")
                print("Units: meters. Robot base is origin.")
                try:
                    bx = float(input("  Board origin X in robot frame (m): "))
                    by = float(input("  Board origin Y in robot frame (m): "))
                    bz = float(input("  Board origin Z in robot frame (m): "))
                except (ValueError, EOFError):
                    print("  Invalid input, skipping save.")
                    continue

                # T_board_in_base: where the board origin is in robot frame
                # For now, assume board lies flat (Z up in robot frame = -Z in board normal)
                # We'll refine orientation later; the key data point is translation.
                board_origin_base = np.array([bx, by, bz])

                # board_T_cam gives us: P_cam = board_T_cam @ P_board
                # We want: P_base = T_cam2base @ P_cam
                # board_origin in cam frame = tvec
                # board_origin in base frame = user input
                # So T_cam2base @ tvec_hom = board_origin_base_hom

                # For a proper solution we need rotation too.
                # Use the board axes: board X/Y should map to known robot axes.
                # Ask user which direction the board X (cols) points in robot frame.
                print("\n  Which robot-frame direction does the board's RED axis (X/cols) point?")
                print("  Options: +x, -x, +y, -y, +z, -z")
                board_x_dir = input("  Board X -> robot axis: ").strip().lower()
                print("  Which robot-frame direction does the board's GREEN axis (Y/rows) point?")
                board_y_dir = input("  Board Y -> robot axis: ").strip().lower()

                axis_map = {
                    '+x': np.array([1, 0, 0]), '-x': np.array([-1, 0, 0]),
                    '+y': np.array([0, 1, 0]), '-y': np.array([0, -1, 0]),
                    '+z': np.array([0, 0, 1]), '-z': np.array([0, 0, -1]),
                }
                if board_x_dir not in axis_map or board_y_dir not in axis_map:
                    print("  Invalid axis, skipping save.")
                    continue

                # Build rotation: R_board_in_base
                R_board_base = np.column_stack([
                    axis_map[board_x_dir],
                    axis_map[board_y_dir],
                    np.cross(axis_map[board_x_dir], axis_map[board_y_dir])
                ])

                # T_board_in_base
                T_board_base = np.eye(4)
                T_board_base[:3, :3] = R_board_base
                T_board_base[:3, 3] = board_origin_base

                # T_board_in_cam = board_T_cam (from solvePnP)
                # T_cam2base = T_board_base @ inv(T_board_cam)
                T_cam2base = T_board_base @ np.linalg.inv(board_T_cam)

                print(f"\n  Camera-to-base transform:")
                print(T_cam2base)

                # Validate: camera origin in base frame
                cam_origin_base = T_cam2base[:3, 3]
                print(f"\n  Camera position in robot frame: [{cam_origin_base[0]:.3f}, {cam_origin_base[1]:.3f}, {cam_origin_base[2]:.3f}] m")

                # Validate: board origin round-trip
                board_origin_cam_hom = np.append(board_T_cam[:3, 3], 1.0)
                rt = (T_cam2base @ board_origin_cam_hom)[:3]
                print(f"  Board origin round-trip: [{rt[0]:.4f}, {rt[1]:.4f}, {rt[2]:.4f}] (should be [{bx:.4f}, {by:.4f}, {bz:.4f}])")

                save = input("\n  Save to config/calibration.yaml? [y/N]: ").strip().lower()
                if save == 'y':
                    from calibration.transform import CoordinateTransform
                    ct = CoordinateTransform()
                    ct.T_camera_to_base = T_cam2base
                    out_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'calibration.yaml')
                    ct.save(out_path)
                    print(f"  Saved to {out_path}")
                else:
                    print("  Not saved.")

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Camera stopped.")


if __name__ == "__main__":
    main()
