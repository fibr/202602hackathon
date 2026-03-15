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
    ./run.sh scripts/detect_checkerboard.py --arm101 [--safe] [--sd]
    ./run.sh scripts/detect_checkerboard.py --verify [--sd] [--dry-run]

    --sd       Use 640x480 resolution (default: 1280x720 HD)
    --arm101   Use LeRobot arm101 instead of Dobot Nova5
    --safe     Start in safe mode (reduced torque/speed, arm101 only)
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

Pure algorithm helpers have been extracted to src/calibration/calib_helpers.py.
This script re-exports them and keeps the global _board_detector and workflow
functions (run_verify, connect_arm101, main).
"""

import sys
import os
import socket
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from vision import RealSenseCamera, CameraIntrinsics, create_camera
from vision.board_detector import BoardDetector, BoardDetection
from config_loader import load_config
from gui.robot_controls import RobotControlPanel, PANEL_WIDTH
from calibration import CoordinateTransform
from visualization import RobotOverlay

# Re-export all algorithm helpers from the canonical module so that any code
# still importing from this script continues to work unchanged.
from calibration.calib_helpers import (  # noqa: F401
    BOARD_COLS,
    BOARD_ROWS,
    SQUARE_SIZE_M,
    SNAP_RADIUS_PX,
    RANSAC_ITERATIONS,
    RANSAC_INLIER_THRESHOLD_MM,
    detect_corners as _detect_corners_impl,
    compute_board_pose as _compute_board_pose_impl,
    corner_3d_in_cam,
    pixel_to_ray,
    ray_plane_intersect,
    solve_rigid_transform,
    _refine_transform,
    solve_robust_transform,
    _get_board_outer_corners_cam as _get_board_outer_corners_cam_impl,
)

# Module-level board detector, set by main() from config.
# GUI views that import this module can set this directly to configure
# the board detector before calling detect_corners / compute_board_pose.
_board_detector: BoardDetector = None

VERIFY_HOVER_HEIGHT_M = 0.05  # 5cm above each corner
VERIFY_SPEED_PERCENT = 20


# ---------------------------------------------------------------------------
# Backward-compatible wrappers that pass the module global to calib_helpers
# ---------------------------------------------------------------------------

def detect_corners(gray):
    """Find board corners — uses the module-level _board_detector if set.

    See calibration.calib_helpers.detect_corners for full documentation.
    Pass board_detector explicitly when using calib_helpers directly.
    """
    return _detect_corners_impl(gray, board_detector=_board_detector)


def compute_board_pose(corners_2d, intrinsics, detection=None):
    """solvePnP → (T_board_in_cam 4x4, obj_points, reproj_error_px).

    See calibration.calib_helpers.compute_board_pose for full documentation.
    Pass board_detector explicitly when using calib_helpers directly.
    """
    return _compute_board_pose_impl(corners_2d, intrinsics,
                                    detection=detection,
                                    board_detector=_board_detector)


def _get_board_outer_corners_cam(T_board_in_cam):
    """Get 4 outer corners of the calibration board in camera frame (metres).

    See calibration.calib_helpers._get_board_outer_corners_cam for docs.
    Pass board_detector explicitly when using calib_helpers directly.
    """
    return _get_board_outer_corners_cam_impl(T_board_in_cam,
                                             board_detector=_board_detector)


# ---------------------------------------------------------------------------
# Robot connection helper (Nova5 only — arm101 uses connect_arm101 below)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Workflow functions
# ---------------------------------------------------------------------------

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

    verify_config = load_config()
    verify_config['camera'] = dict(verify_config.get('camera', {}), width=width, height=height)
    cam_type = verify_config.get('camera', {}).get('type', 'realsense')
    print(f"Starting {cam_type} camera ({width}x{height})...")
    camera = create_camera(verify_config)
    camera.start()
    width, height = camera.width, camera.height
    print(f"Camera started ({width}x{height}). Looking for checkerboard (press 'c' to capture, 'q' to abort)...")

    corners_2d = None
    T_board_in_cam = None
    try:
        while True:
            color_image, _, _ = camera.get_frames()
            if color_image is None:
                continue
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            found, corners, detection = detect_corners(gray)

            display = color_image.copy()
            if found:
                if _board_detector is not None and detection is not None:
                    _board_detector.draw_corners(display, detection)
                else:
                    cv2.drawChessboardCorners(display, (BOARD_COLS, BOARD_ROWS), corners, found)
                cv2.putText(display, f"Found {len(corners)} corners - press 'c'",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "No board detected",
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
                T_board_in_cam, _, _ = compute_board_pose(
                    corners, camera.intrinsics, detection)
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


def connect_arm101(config, safe_mode=False):
    """Connect to LeRobot arm101 follower. Returns (robot, speed)."""
    from robot.lerobot_arm101 import LeRobotArm101

    ac = config.get('arm101', {})
    port = ac.get('port', '')
    baudrate = ac.get('baudrate', 1_000_000)
    motor_ids = ac.get('motor_ids', [1, 2, 3, 4, 5, 6])

    print(f"=== LeRobot arm101 Calibration ===")
    if safe_mode:
        print("  ** SAFE MODE: reduced torque and speed **")

    arm = LeRobotArm101(
        port=port, baudrate=baudrate,
        motor_ids=motor_ids, safe_mode=safe_mode)
    arm.connect()
    arm.enable_torque()
    speed = arm.speed
    return arm, speed


def main():
    global _board_detector

    sd = '--sd' in sys.argv
    verify = '--verify' in sys.argv
    dry_run = '--dry-run' in sys.argv
    use_arm101 = '--arm101' in sys.argv
    safe_mode = '--safe' in sys.argv
    width, height = (640, 480) if sd else (1280, 720)

    config = load_config()

    # Initialize board detector from config
    _board_detector = BoardDetector.from_config(config)

    if verify:
        run_verify(width, height, dry_run)
        return

    rc = config.get('robot', {})
    ip = rc.get('ip', '192.168.5.1')
    port = rc.get('dashboard_port', 29999)

    print("=== Interactive Calibration ===")
    print(f"Board: {_board_detector.describe()}")
    print(f"Resolution: {width}x{height}")
    print()
    print("Arm control: GUI panel on right (XY pad, Z, gripper, speed, enable/home)")
    print("             Keyboard: 1-6/!@#$%^ jog, space stop, c/o gripper, [/] speed, v enable")
    print("Intrinsics:  'i' or button to capture | button to calibrate & save (5+ frames)")
    print("             button to visualize reprojection errors")
    print("Plane:       'g' or button to capture | button to save")
    print("Hand-eye:    click record point | Enter or button to solve | u undo | n clear")
    print("             Esc quit | p print pose")
    print()

    # Connect to robot (optional — camera-only mode if unavailable)
    robot = None
    speed = 30
    if use_arm101:
        if safe_mode:
            pass  # handled by connect_arm101
        try:
            robot, speed = connect_arm101(config, safe_mode=safe_mode)
            angles = robot.get_angles()
            if angles:
                print(f"  Joints: {', '.join(f'{v:.1f}' for v in angles)}")
            pose = robot.get_pose()
            if pose:
                print(f"  TCP:    {', '.join(f'{v:.1f}' for v in pose)}")
        except Exception as e:
            print(f"  WARNING: Cannot connect to arm101: {e}")
            print(f"  Continuing in camera-only mode (no arm control)")
            robot = None
    else:
        if safe_mode:
            print("  Note: --safe is only supported for arm101, ignoring.")
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
    cam_type = config.get('camera', {}).get('type', 'realsense')
    main_config = dict(config)
    main_config['camera'] = dict(config.get('camera', {}), width=width, height=height)
    print(f"Starting {cam_type} camera ({width}x{height})...")
    camera = create_camera(main_config)
    camera.start()
    # Use actual resolution (camera may not support requested size)
    width, height = camera.width, camera.height
    print(f"Camera started ({width}x{height}).\n")

    # GUI panel (always shown; robot=None disables arm controls)
    panel = RobotControlPanel(robot, panel_x=width, panel_height=height, config=config)
    if robot:
        panel.speed = speed
        if use_arm101:
            panel.status_msg = "arm101: jog arm, camera features available"
        else:
            panel.status_msg = "Touch TCP to board, then click on it"
    else:
        panel.status_msg = "Camera-only mode"

    # Plane save callback (GUI button)
    def save_plane():
        import yaml as _yaml
        if len(plane_samples) < 1:
            panel.status_msg = "No plane samples — press 'g' to capture"
            print(f"  {panel.status_msg}")
            return
        normals = np.array([s[0] for s in plane_samples])
        distances = np.array([s[1] for s in plane_samples])
        # Flip any normals pointing opposite to the majority
        ref = normals[0]
        for idx in range(1, len(normals)):
            if np.dot(normals[idx], ref) < 0:
                normals[idx] = -normals[idx]
                distances[idx] = -distances[idx]
        avg_normal = normals.mean(axis=0)
        avg_normal /= np.linalg.norm(avg_normal)
        avg_d = distances.mean()
        std_d = distances.std() if len(distances) > 1 else 0.0
        angles_deg = [np.degrees(np.arccos(np.clip(
            np.dot(n, avg_normal), -1, 1))) for n in normals]
        max_angle = max(angles_deg)
        plane_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', 'ground_plane.yaml')
        os.makedirs(os.path.dirname(plane_path), exist_ok=True)
        data = {
            'plane_normal': avg_normal.tolist(),
            'plane_d': float(avg_d),
            'num_samples': len(plane_samples),
            'std_d_mm': float(std_d * 1000),
            'max_angle_deg': float(max_angle),
        }
        with open(plane_path, 'w') as f:
            _yaml.dump(data, f, default_flow_style=False)
        print(f"\n=== Ground Plane ({len(plane_samples)} samples) ===")
        print(f"  Normal: [{avg_normal[0]:.5f}, {avg_normal[1]:.5f}, {avg_normal[2]:.5f}]")
        print(f"  Distance: {avg_d*1000:.1f} mm (std: {std_d*1000:.1f} mm)")
        print(f"  Max angular deviation: {max_angle:.2f} deg")
        print(f"  Saved to {plane_path}")
        panel.status_msg = f"Plane saved: {len(plane_samples)}x, d={avg_d*1000:.1f}mm std={std_d*1000:.1f}mm"
        plane_samples.clear()

    panel.add_button(
        lambda: f"Save Plane ({len(plane_samples)})",
        save_plane,
        color=(0, 100, 100))

    # Hand-eye solve callback (GUI button)
    def solve_handeye():
        if len(pairs) < 3:
            panel.status_msg = f"Need 3+ points (have {len(pairs)})"
            print(f"  {panel.status_msg}")
            return
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
        ct = CoordinateTransform()
        ct.T_camera_to_base = T_cam2base
        out_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'calibration.yaml')
        ct.save(out_path)
        print(f"Saved to {out_path}")
        panel.status_msg = f"Saved! {n_inliers}/{len(pairs)} inliers, mean {np.mean(inlier_errors):.1f}mm"

    panel.add_button(
        lambda: f"Solve HandEye ({len(pairs)})",
        solve_handeye,
        color=(0, 100, 0))

    # Intrinsics capture callback (GUI button + 'i' key)
    def capture_intrinsics_frame():
        if current_corners is None:
            msg = "No board detected — can't capture"
            panel.status_msg = msg
            print(f"  {msg}")
            return
        if _board_detector is not None and current_detection is not None:
            intr_detections.append(current_detection)
            n_corners = len(current_detection.corners)
            partial = " (partial)" if current_detection.is_partial else ""
            msg = f"Intrinsics frame {len(intr_detections)}: {n_corners} corners{partial}"
            panel.status_msg = msg
            print(f"  {msg}")
        else:
            n = len(current_corners)
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
                    panel.status_msg = msg
                    print(f"  {msg}")
                    return
            obj_pts = np.zeros((i_rows * i_cols, 3), dtype=np.float32)
            for rr in range(i_rows):
                for cc in range(i_cols):
                    obj_pts[rr * i_cols + cc] = [
                        cc * SQUARE_SIZE_M, rr * SQUARE_SIZE_M, 0]
            intr_frames.append((obj_pts, current_corners.copy()))
            msg = f"Intrinsics frame {len(intr_frames)} captured"
            panel.status_msg = msg
            print(f"  {msg}")

    panel.add_button(
        lambda: f"Capture Intr ({len(intr_detections) or len(intr_frames)})",
        capture_intrinsics_frame,
        color=(100, 80, 0))

    # Intrinsics calibrate callback (GUI button + 'I' key)
    def calibrate_intrinsics():
        n_frames = len(intr_detections) if intr_detections else len(intr_frames)
        if n_frames < 5:
            msg = f"Need 5+ frames (have {n_frames})"
            panel.status_msg = msg
            print(f"  {msg}")
            return
        if intr_detections and _board_detector is not None:
            print(f"\n=== Calibrating intrinsics from {len(intr_detections)} frames ({_board_detector.describe()}) ===")
            try:
                ret, calib_intr = _board_detector.calibrate_intrinsics(
                    intr_detections, (width, height))
                print(f"  Reprojection error: {ret:.4f} px")
                print(f"  Camera matrix:\n{calib_intr.camera_matrix}")
                print(f"  Distortion: {calib_intr.dist_coeffs}")
                calib_intr.save(intr_path)
                print(f"  Saved to {intr_path}")
                camera.intrinsics = calib_intr
                msg = f"Intrinsics saved! reproj={ret:.3f}px ({len(intr_detections)} frames)"
                panel.status_msg = msg
                print(f"  {msg}")
            except Exception as e:
                msg = f"Calibration failed: {e}"
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

            calib_intr = CameraIntrinsics(
                fx=mtx[0, 0], fy=mtx[1, 1],
                ppx=mtx[0, 2], ppy=mtx[1, 2],
                coeffs=dist.ravel().tolist())
            calib_intr.width = width
            calib_intr.height = height
            calib_intr.save(intr_path)
            print(f"  Saved to {intr_path}")

            camera.intrinsics = calib_intr
            msg = f"Intrinsics saved! reproj={ret:.3f}px"
            panel.status_msg = msg
            print(f"  {msg}")

    panel.add_button(
        lambda: f"Calibrate Intr ({len(intr_detections) or len(intr_frames)})",
        calibrate_intrinsics,
        color=(0, 100, 100))

    # Ground plane capture callback (GUI button + 'g' key)
    def capture_plane_sample():
        if current_T_board is None:
            panel.status_msg = "No board — can't capture plane sample"
            return
        R_board = current_T_board[:3, :3]
        t_board = current_T_board[:3, 3]
        normal = R_board[:, 2]
        d = np.dot(normal, t_board)
        plane_samples.append((normal, d))
        msg = f"Plane sample {len(plane_samples)}: d={d*1000:.1f}mm"
        panel.status_msg = msg
        print(f"  {msg}  normal=[{normal[0]:.4f},{normal[1]:.4f},{normal[2]:.4f}]")

    panel.add_button(
        lambda: f"Capture Plane ({len(plane_samples)})",
        capture_plane_sample,
        color=(80, 80, 0))

    # Intrinsics visualization callback
    def visualize_intrinsics():
        """Show all captured intrinsics frames with reprojection errors."""
        n_det = len(intr_detections)
        n_leg = len(intr_frames)
        if n_det == 0 and n_leg == 0:
            panel.status_msg = "No intrinsics frames captured yet"
            return

        # Get current intrinsics for reprojection
        intr = camera.intrinsics
        cam_mtx = intr.camera_matrix
        dist_c = intr.dist_coeffs

        # Build per-frame data: list of (frame_idx, corners_2d, obj_pts, per_pt_errors)
        frame_data = []
        if n_det and _board_detector is not None:
            for i, det in enumerate(intr_detections):
                obj_pts = _board_detector.get_object_points(det)
                corners_2d = det.corners.reshape(-1, 2)
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, corners_2d.astype(np.float64), cam_mtx, dist_c)
                if not ok:
                    continue
                projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, cam_mtx, dist_c)
                projected = projected.reshape(-1, 2)
                errors = np.linalg.norm(corners_2d - projected, axis=1)
                frame_data.append((i, corners_2d, projected, errors))
        else:
            for i, (obj_pts, img_pts) in enumerate(intr_frames):
                corners_2d = img_pts.reshape(-1, 2)
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, corners_2d.astype(np.float64), cam_mtx, dist_c)
                if not ok:
                    continue
                projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, cam_mtx, dist_c)
                projected = projected.reshape(-1, 2)
                errors = np.linalg.norm(corners_2d - projected, axis=1)
                frame_data.append((i, corners_2d, projected, errors))

        if not frame_data:
            panel.status_msg = "No valid frames for visualization"
            return

        # Collect all errors for color scale
        all_errors = np.concatenate([fd[3] for fd in frame_data])
        max_err = max(all_errors.max(), 1.0)
        mean_err = all_errors.mean()
        print(f"\n=== Intrinsics Visualization: {len(frame_data)} frames ===")
        print(f"  Mean reproj error: {mean_err:.3f}px, Max: {max_err:.3f}px")
        for idx, _, _, errs in frame_data:
            print(f"  Frame {idx+1}: mean={errs.mean():.3f}px max={errs.max():.3f}px ({len(errs)} pts)")

        # Draw all frames overlaid on a single canvas
        vis = np.zeros((height, width, 3), dtype=np.uint8)
        frame_colors = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (255, 100, 255), (100, 255, 255),
            (200, 150, 100), (100, 200, 150), (150, 100, 200),
            (200, 200, 200),
        ]

        for fi, (idx, corners, projected, errors) in enumerate(frame_data):
            base_color = frame_colors[fi % len(frame_colors)]
            for j in range(len(corners)):
                err_ratio = min(errors[j] / max(max_err, 0.5), 1.0)
                r = int(255 * err_ratio)
                g = int(255 * (1.0 - err_ratio))
                color = (0, g, r)  # BGR

                cx, cy = int(corners[j][0]), int(corners[j][1])
                px, py = int(projected[j][0]), int(projected[j][1])

                cv2.circle(vis, (cx, cy), 4, color, -1)
                cv2.circle(vis, (px, py), 4, color, 1)
                cv2.line(vis, (cx, cy), (px, py), color, 1)

            cx_mean = int(corners[:, 0].mean())
            cy_mean = int(corners[:, 1].mean())
            cv2.putText(vis, f"F{idx+1}", (cx_mean - 10, cy_mean),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, base_color, 1)

        cv2.rectangle(vis, (0, 0), (width, 50), (0, 0, 0), -1)
        cv2.putText(vis, f"Intrinsics: {len(frame_data)} frames, "
                    f"mean={mean_err:.2f}px, max={max_err:.2f}px  "
                    f"[filled=detected, hollow=reprojected, green=low err, red=high]",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        cv2.putText(vis, "Scroll=zoom  Drag=pan  Esc/q=close",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        win_name = "Intrinsics Visualization"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        zoom = 1.0
        pan_x, pan_y = 0, 0
        dragging = False
        drag_start = (0, 0)
        pan_start = (0, 0)

        def _vis_mouse(event, x, y, flags, param):
            nonlocal zoom, pan_x, pan_y, dragging, drag_start, pan_start
            if event == cv2.EVENT_MOUSEWHEEL:
                old_zoom = zoom
                if flags > 0:
                    zoom = min(zoom * 1.2, 10.0)
                else:
                    zoom = max(zoom / 1.2, 0.5)
                pan_x = int(x - (x - pan_x) * zoom / old_zoom)
                pan_y = int(y - (y - pan_y) * zoom / old_zoom)
            elif event == cv2.EVENT_LBUTTONDOWN:
                dragging = True
                drag_start = (x, y)
                pan_start = (pan_x, pan_y)
            elif event == cv2.EVENT_MOUSEMOVE and dragging:
                pan_x = pan_start[0] + (x - drag_start[0])
                pan_y = pan_start[1] + (y - drag_start[1])
            elif event == cv2.EVENT_LBUTTONUP:
                dragging = False

        cv2.setMouseCallback(win_name, _vis_mouse)

        while True:
            zh, zw = int(height * zoom), int(width * zoom)
            zoomed = cv2.resize(vis, (zw, zh), interpolation=cv2.INTER_NEAREST)
            vx = max(0, -pan_x)
            vy = max(0, -pan_y)
            vx2 = min(zw, width - pan_x)
            vy2 = min(zh, height - pan_y)
            viewport = np.zeros((height, width, 3), dtype=np.uint8)
            dx = max(0, pan_x)
            dy = max(0, pan_y)
            cw = min(vx2 - vx, width - dx)
            ch = min(vy2 - vy, height - dy)
            if cw > 0 and ch > 0:
                viewport[dy:dy+ch, dx:dx+cw] = zoomed[vy:vy+ch, vx:vx+cw]
            cv2.imshow(win_name, viewport)
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or key == ord('q'):
                break
            try:
                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
        cv2.destroyWindow(win_name)
        panel.status_msg = f"Vis closed (mean={mean_err:.2f}px)"

    panel.add_button(
        lambda: f"Visualize Intr ({len(intr_detections) or len(intr_frames)})",
        visualize_intrinsics,
        color=(100, 0, 100))

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
    current_detection = None  # BoardDetection (for charuco ID-based ops)
    current_T_board = None
    click_point = None

    # Intrinsics calibration state
    intr_frames = []      # list of (obj_points, img_points) for legacy calibration
    intr_detections = []  # list of BoardDetection for BoardDetector calibration
    intr_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'camera_intrinsics.yaml')

    # Ground plane calibration state
    plane_samples = []  # list of (normal_vec, distance) from multiple board detections

    def on_mouse(event, x, y, flags, param):
        nonlocal click_point
        # Route to panel if in panel area
        if x >= width:
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
            found, corners, detection = detect_corners(gray)

            # Create canvas (camera + panel)
            canvas_w = width + PANEL_WIDTH
            canvas = np.zeros((height, canvas_w, 3), dtype=np.uint8)
            display = color_image.copy()

            reproj_err = None
            if found:
                current_corners = corners
                current_detection = detection
                current_T_board, _, reproj_err = compute_board_pose(
                    corners, camera.intrinsics, detection)

                if _board_detector is not None and detection is not None:
                    _board_detector.draw_corners(display, detection)
                else:
                    cv2.drawChessboardCorners(
                        display, (BOARD_COLS, BOARD_ROWS), corners, found)
            else:
                current_corners = None
                current_detection = None
                current_T_board = None

            # Handle click — ray-plane intersection with checkerboard plane
            if click_point is not None:
                cx, cy = click_point
                click_point = None

                if current_T_board is None:
                    msg = "No board detected - need board for ray-plane intersection"
                    panel.status_msg = msg
                    print(f"  {msg}")
                elif not robot:
                    msg = "No robot connected - can't record calibration point"
                    print(f"  {msg}")
                else:
                    p_cam = ray_plane_intersect((cx, cy), camera.intrinsics, current_T_board)
                    if p_cam is None:
                        msg = "Ray parallel to board plane - try different angle"
                        panel.status_msg = msg
                        print(f"  {msg}")
                    else:
                        pose = robot.get_pose()
                        if pose is None:
                            msg = "ERROR: can't read robot pose"
                            panel.status_msg = msg
                            print(f"  {msg}")
                        else:
                            p_robot_m = np.array(pose[:3]) / 1000.0
                            pairs.append((p_cam, p_robot_m, (cx, cy)))

                            msg = (f"Pt {len(pairs)}: "
                                   f"cam=[{p_cam[0]:.3f},{p_cam[1]:.3f},{p_cam[2]:.3f}] "
                                   f"robot=[{pose[0]:.1f},{pose[1]:.1f},{pose[2]:.1f}]")
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

            # --- HUD overlay ---
            board_status = (f"Board: reproj {reproj_err:.2f}px"
                            if found and reproj_err is not None
                            else ("Board OK" if found else "No board"))
            n_intr = len(intr_detections) if intr_detections else len(intr_frames)
            intr_str = f"  Intr:{n_intr}" if n_intr else ""
            plane_str = f"  Plane:{len(plane_samples)}" if plane_samples else ""
            jog_str = " JOG" if panel.jogging else ""
            bar_text = f"{len(pairs)} pts |{intr_str}{plane_str} | Spd:{panel.speed}%{jog_str} | {board_status}"
            cv2.rectangle(display, (0, 0), (width, 32), (0, 0, 0), -1)
            cv2.putText(display, bar_text, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0) if found else (0, 0, 255), 1)

            help_lines = [
                "[i] capture intrinsics  [g] capture plane  [click] hand-eye pt  [p] pose",
                "[Enter] solve hand-eye  [u] undo  [n] clear  [Esc] quit",
            ]
            line_h = 22
            bar_h = line_h * len(help_lines) + 8
            cv2.rectangle(display, (0, height - bar_h), (width, height),
                          (0, 0, 0), -1)
            for i, line in enumerate(help_lines):
                y = height - bar_h + 4 + line_h * (i + 1)
                cv2.putText(display, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (220, 220, 220), 1)

            canvas[0:height, 0:width] = display
            panel.draw(canvas)

            cv2.imshow('Calibration', canvas)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # Esc
                break
            try:
                if cv2.getWindowProperty('Calibration', cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            if key != 255 and panel and panel.handle_key(key):
                pass  # consumed by panel

            elif key == ord('p') and robot:
                pose = robot.get_pose()
                angles = robot.get_angles()
                if pose:
                    print(f"  Pose:   {', '.join(f'{v:.2f}' for v in pose)}")
                if angles:
                    print(f"  Joints: {', '.join(f'{v:.2f}' for v in angles)}")
                if pose or angles:
                    panel.status_msg = "Pose printed to console"

            elif key == ord('u') and pairs:
                pairs.pop()
                msg = f"Undid -> {len(pairs)} pts remain"
                panel.status_msg = msg
                print(f"  {msg}")

            elif key == ord('n'):
                pairs.clear()
                msg = "Cleared all points"
                panel.status_msg = msg
                print(f"  {msg}")

            elif key == ord('i'):
                capture_intrinsics_frame()

            elif key == ord('g'):
                capture_plane_sample()

            elif key == 13:  # Enter — solve hand-eye (same as button)
                solve_handeye()

    except KeyboardInterrupt:
        pass
    finally:
        if robot:
            if use_arm101:
                robot.close()
            else:
                robot.send('MoveJog()')  # stop any jog
                robot.close()
        camera.stop()
        cv2.destroyAllWindows()
        print("\nDone.")


if __name__ == "__main__":
    main()
