#!/usr/bin/env python3
"""Hand-eye calibration for arm101 with a fixed external camera.

Moves the arm through diverse poses in safe mode, collects correspondences
between robot TCP (from FK) and camera observations, then solves for the
camera-to-robot-base transform.

Supports multiple tracking methods:
  --method depth   : Best — uses RealSense D435i RGB+depth (requires hardware)
  --method click   : Interactive — user clicks gripper tip in each frame
  --method auto    : Automated — frame differencing (unreliable if cube occluded)

Usage:
    ./run.sh scripts/handeye_calibrate.py --method click --camera 4
    ./run.sh scripts/handeye_calibrate.py --method depth
    ./run.sh scripts/handeye_calibrate.py --method auto --camera 4
"""

import sys
import os
import time
import argparse
import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from robot.lerobot_arm101 import LeRobotArm101
from kinematics.arm101_ik_solver import Arm101IKSolver
from calibration.transform import CoordinateTransform

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')


# ──────────────────── Helpers ────────────────────

def get_fk_pose(solver, arm, joint_count=5):
    """Get TCP position from FK. Returns (pos_mm, rpy_deg) or (None, None)."""
    angles = arm.get_angles()
    if angles is None:
        return None, None
    try:
        return solver.forward_kin(np.array(angles[:joint_count]))
    except Exception as e:
        print(f"  FK error: {e}")
        return None, None


def wait_for_motion(arm, timeout=3.0, threshold=0.5):
    """Wait until arm stops moving."""
    prev = arm.read_all_angles()
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(0.15)
        curr = arm.read_all_angles()
        if all(abs(c - p) < threshold for c, p in zip(curr, prev)):
            return True
        prev = curr
    return False


def generate_calibration_poses(arm, n_poses=16):
    """Generate calibration poses by perturbing the current joint angles."""
    current = arm.read_all_angles()
    if current is None or any(a < -180 for a in current):
        raise RuntimeError("Cannot read current arm angles")
    print(f"  Current angles: {[f'{a:.1f}' for a in current]}")

    base = np.array(current[:5], dtype=float)
    gripper = current[5]
    perturbations = [
        [0,0,0,0,0], [30,0,0,0,0], [-30,0,0,0,0],
        [0,25,0,0,0], [0,-25,0,0,0], [0,0,30,0,0], [0,0,-30,0,0],
        [25,20,0,0,0], [-25,-20,0,0,0], [25,0,25,0,0], [-25,0,-25,0,0],
        [0,25,25,0,0], [0,-25,-25,0,0], [30,20,20,0,0], [-30,-20,-20,0,0],
        [20,-20,25,0,0], [-20,25,-20,0,0], [0,0,0,25,0], [0,0,0,-25,0],
        [25,15,15,10,0],
    ]
    return [list(base + np.array(d, dtype=float)) + [gripper]
            for d in perturbations[:n_poses]]


# ──────────────────── Click method ────────────────────

_click_point = None

def _click_handler(event, x, y, flags, param):
    global _click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        _click_point = (x, y)


def collect_click(frame, pose_idx, total):
    """Show frame, wait for user to click gripper tip. Returns (cx, cy) or None."""
    global _click_point
    _click_point = None
    win = "Click gripper tip"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, _click_handler)

    while True:
        vis = frame.copy()
        cv2.putText(vis, f"Pose {pose_idx+1}/{total} — click gripper tip (s=skip, q=quit)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if _click_point:
            cv2.circle(vis, _click_point, 6, (0, 0, 255), -1)
        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            return 'quit'
        if key == ord('s'):
            return None
        if _click_point and (key == ord(' ') or key == 13):  # space or enter confirms
            return _click_point


# ──────────────────── Auto method (frame differencing) ────────────────────

def find_moving_green(frame_home, frame_current, min_area=80):
    """Find green blobs in the motion region between home and current frame."""
    gray_h = cv2.cvtColor(frame_home, cv2.COLOR_BGR2GRAY)
    gray_c = cv2.cvtColor(frame_current, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_h, gray_c)
    _, motion = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    motion = cv2.dilate(motion, k, iterations=3)

    hsv = cv2.cvtColor(frame_current, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([35, 60, 60]), np.array([85, 255, 255]))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    moving_green = cv2.bitwise_and(green, motion)

    cnts, _ = cv2.findContours(moving_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    best = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(best) < min_area:
        return None, None
    M = cv2.moments(best)
    if M['m00'] == 0:
        return None, None
    return int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])


# ──────────────────── Depth method (RealSense) ────────────────────

def find_closest_in_depth(depth_image, camera, depth_frame,
                           home_depth=None, min_d=100, max_d=700):
    """Find closest object in depth within valid range, optionally diffed vs home."""
    curr = depth_image.astype(float)
    curr[curr == 0] = 9999

    if home_depth is not None:
        home = home_depth.astype(float)
        home[home == 0] = 9999
        closer = (home - curr) > 30
        valid = (depth_image > min_d) & (depth_image < max_d) & closer
        if not np.any(valid):
            valid = (depth_image > min_d) & (depth_image < max_d)
    else:
        valid = (depth_image > min_d) & (depth_image < max_d)

    if not np.any(valid):
        return None, None, None

    masked = depth_image.astype(float)
    masked[~valid] = 9999
    py, px = np.unravel_index(np.argmin(masked), masked.shape)
    d = int(depth_image[py, px])
    pt = camera.pixel_to_3d(px, py, depth_frame)
    return pt, (px, py), d


# ──────────────────── Solvers ────────────────────

def solve_rigid_transform(pts_cam, pts_robot):
    """SVD rigid transform: T_camera_to_base from 3D-3D correspondences.

    pts_cam in meters, pts_robot in mm.
    """
    A = np.array(pts_cam, dtype=np.float64)
    B = np.array(pts_robot, dtype=np.float64) / 1000.0

    ca, cb = A.mean(0), B.mean(0)
    Ac, Bc = A - ca, B - cb
    U, _, Vt = np.linalg.svd(Ac.T @ Bc)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = cb - R @ ca

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    errs = [np.linalg.norm((T @ np.append(A[i], 1))[:3] - B[i]) * 1000
            for i in range(len(A))]
    print(f"\n  SVD residuals (mm): mean={np.mean(errs):.1f}, max={np.max(errs):.1f}")
    for i, e in enumerate(errs):
        print(f"    Pt {i}: {e:.1f}{' ***' if e > 20 else ''}")

    pos = T[:3, 3] * 1000
    print(f"  Camera in robot frame: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}] mm")
    return T


def solve_pnp(pts_3d_mm, pts_2d_px, K, dist):
    """PnP: T_camera_to_base from 3D (robot mm) and 2D (pixel) correspondences."""
    obj = np.array(pts_3d_mm, dtype=np.float64).reshape(-1, 1, 3)
    img = np.array(pts_2d_px, dtype=np.float64).reshape(-1, 1, 2)

    if len(obj) < 4:
        return None

    best, best_e = None, 1e9
    if len(obj) >= 6:
        ok, rv, tv, inl = cv2.solvePnPRansac(
            obj, img, K, dist, iterationsCount=2000, reprojectionError=8.0)
        if ok and inl is not None:
            p, _ = cv2.projectPoints(obj, rv, tv, K, dist)
            e = np.mean(np.linalg.norm(p.reshape(-1, 2) - img.reshape(-1, 2), axis=1))
            print(f"  RANSAC: {len(inl)}/{len(obj)} inliers, err={e:.2f}px")
            if e < best_e:
                best_e, best = e, (rv, tv)

    for nm, fl in [("ITER", cv2.SOLVEPNP_ITERATIVE), ("EPNP", cv2.SOLVEPNP_EPNP)]:
        try:
            ok, rv, tv = cv2.solvePnP(obj, img, K, dist, flags=fl)
            if ok:
                p, _ = cv2.projectPoints(obj, rv, tv, K, dist)
                e = np.mean(np.linalg.norm(p.reshape(-1, 2) - img.reshape(-1, 2), axis=1))
                print(f"  {nm}: err={e:.2f}px")
                if e < best_e:
                    best_e, best = e, (rv, tv)
        except Exception:
            pass

    if best is None:
        return None

    rv, tv = best
    Rb2c, _ = cv2.Rodrigues(rv)
    T = np.eye(4)
    T[:3, :3] = Rb2c
    T[:3, 3] = tv.flatten()
    T_c2b = np.linalg.inv(T)
    pos = T_c2b[:3, 3]
    print(f"  Camera in robot frame: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}] mm")
    return T_c2b


# ──────────────────── Save ────────────────────

def save_calibration(T_cam2base, filepath, is_meters=True):
    """Save calibration transform."""
    ct = CoordinateTransform()
    ct.T_camera_to_base = T_cam2base
    from scipy.spatial.transform import Rotation
    R = T_cam2base[:3, :3]
    rpy = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
    scale = 1000.0 if is_meters else 1.0
    ct.base_offset_mm = (T_cam2base[:3, 3] * scale).copy()
    ct.base_rpy_deg = rpy
    ct.save(filepath)
    print(f"  Saved to {filepath}")


# ──────────────────── Main ────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hand-eye calibration for arm101")
    parser.add_argument('--method', choices=['depth', 'click', 'auto'], default='click',
                        help='Tracking method (default: click)')
    parser.add_argument('--camera', type=int, default=4,
                        help='Webcam device index for click/auto methods')
    parser.add_argument('--n-poses', type=int, default=16)
    parser.add_argument('--port', type=str, default='')
    parser.add_argument('--output', type=str,
                        default=os.path.join(PROJECT_ROOT, 'config', 'calibration_arm101.yaml'))
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Hand-Eye Calibration: arm101 (method={args.method})")
    print("=" * 60)

    # Camera setup
    camera_rs = None
    cap = None
    K, dist = None, None

    if args.method == 'depth':
        print("\n[1] Starting RealSense D435i...")
        from vision.camera import RealSenseCamera
        camera_rs = RealSenseCamera(width=640, height=480, fps=15)
        camera_rs.start()
        for _ in range(30):
            camera_rs.get_frames()
            time.sleep(0.05)
        print(f"  Ready: {camera_rs.width}x{camera_rs.height}")
    else:
        print(f"\n[1] Opening webcam /dev/video{args.camera}...")
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"  ERROR: Cannot open camera {args.camera}")
            return 1
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for _ in range(15):
            cap.read()
            time.sleep(0.05)
        print("  Camera ready.")

        # Load intrinsics for PnP
        cam_yaml = os.path.join(PROJECT_ROOT, 'config', 'cameras.yaml')
        with open(cam_yaml) as f:
            cdata = yaml.safe_load(f)
        for cname, cinfo in cdata['cameras'].items():
            if cinfo.get('device_index') == args.camera:
                intr = cinfo['intrinsics']
                K = np.array(intr['camera_matrix'], dtype=np.float64)
                dist = np.array(intr['dist_coeffs'], dtype=np.float64)
                print(f"  Intrinsics from {cname}: fx={K[0,0]:.1f}")
                break
        if K is None:
            K = np.array([[554.3, 0, 320], [0, 554.3, 240], [0, 0, 1]], dtype=np.float64)
            dist = np.zeros(5, dtype=np.float64)
            print("  Using default intrinsics (estimated)")

    # FK solver
    print("\n[2] FK solver...")
    solver = Arm101IKSolver()

    # Connect arm
    print("\n[3] Arm101 (safe mode)...")
    port = args.port or LeRobotArm101.find_port()
    arm = LeRobotArm101(port=port, safe_mode=True)
    arm.connect()
    arm.enable_torque()
    time.sleep(0.5)

    tcp, _ = get_fk_pose(solver, arm)
    if tcp is not None:
        print(f"  TCP: [{tcp[0]:.1f}, {tcp[1]:.1f}, {tcp[2]:.1f}] mm")

    # Generate poses
    print(f"\n[4] Collecting data ({args.n_poses} poses)...")
    poses = generate_calibration_poses(arm, n_poses=args.n_poses)

    pts_3d_cam = []   # For depth method: 3D in camera frame (m)
    pts_3d_robot = []  # 3D in robot frame (mm)
    pts_2d = []        # For click/auto: 2D pixels

    # For auto method: take home reference
    home_frame = None
    home_depth = None
    if args.method == 'auto':
        _, home_frame = cap.read()
    if args.method == 'depth':
        _, home_depth, _ = camera_rs.get_frames()

    for i, pose in enumerate(poses):
        print(f"\n  --- Pose {i+1}/{len(poses)} ---")

        arm.move_joints(pose)
        time.sleep(0.8)
        wait_for_motion(arm, timeout=4.0)
        time.sleep(0.8)

        tcp_pos, _ = get_fk_pose(solver, arm)
        if tcp_pos is None:
            print("  FK failed, skipping.")
            continue

        if args.method == 'depth':
            # Get depth frames
            ds = []
            df_last = None
            c_last = None
            for _ in range(5):
                c, d, df = camera_rs.get_frames()
                if d is not None:
                    ds.append(d)
                    df_last = df
                    c_last = c
                time.sleep(0.05)
            if not ds:
                print("  No depth, skipping.")
                continue
            d_avg = np.median(np.stack(ds), axis=0).astype(np.uint16)
            pt3d, px, dv = find_closest_in_depth(
                d_avg, camera_rs, df_last,
                home_depth if i > 0 else None)
            if pt3d is None or np.linalg.norm(pt3d) < 0.01:
                print("  No valid depth point, skipping.")
                continue
            print(f"  Depth: 3D=[{pt3d[0]*1000:.1f},{pt3d[1]*1000:.1f},{pt3d[2]*1000:.1f}]mm @ px{px}")
            print(f"  FK:   TCP=[{tcp_pos[0]:.1f},{tcp_pos[1]:.1f},{tcp_pos[2]:.1f}]mm")
            pts_3d_cam.append(pt3d.copy())
            pts_3d_robot.append(tcp_pos.copy())
            if i == 0:
                home_depth = d_avg.copy()

        elif args.method == 'click':
            _, frame = cap.read()
            if frame is None:
                continue
            result = collect_click(frame, i, len(poses))
            if result == 'quit':
                break
            if result is None:
                continue
            cx, cy = result
            print(f"  Click: ({cx}, {cy}), TCP=[{tcp_pos[0]:.1f},{tcp_pos[1]:.1f},{tcp_pos[2]:.1f}]mm")
            pts_2d.append([float(cx), float(cy)])
            pts_3d_robot.append(tcp_pos.copy())

        elif args.method == 'auto':
            _, frame = cap.read()
            if frame is None or home_frame is None:
                continue
            cx, cy = find_moving_green(home_frame, frame)
            if cx is None:
                print("  No moving green detected, skipping.")
                continue
            print(f"  Auto: ({cx},{cy}), TCP=[{tcp_pos[0]:.1f},{tcp_pos[1]:.1f},{tcp_pos[2]:.1f}]mm")
            pts_2d.append([float(cx), float(cy)])
            pts_3d_robot.append(tcp_pos.copy())

    n = len(pts_3d_robot)
    print(f"\n  Collected {n} correspondences.")

    # Return home
    if poses:
        arm.move_joints(poses[0])
        wait_for_motion(arm)

    # Solve
    T = None
    is_meters = False
    if args.method == 'depth' and len(pts_3d_cam) >= 3:
        T = solve_rigid_transform(pts_3d_cam, pts_3d_robot)
        is_meters = True
    elif args.method in ('click', 'auto') and len(pts_2d) >= 4:
        T = solve_pnp(pts_3d_robot, pts_2d, K, dist)
        is_meters = False  # PnP output is in mm (same units as input)

    if T is not None:
        save_calibration(T, args.output, is_meters=is_meters)
        print(f"\n  SUCCESS: {args.output}")
        # Debug data
        dfile = os.path.join(PROJECT_ROOT, 'config', 'calibration_arm101_points.yaml')
        dd = {'method': args.method, 'n_points': n,
              'points_robot_mm': [p.tolist() for p in pts_3d_robot]}
        if pts_2d:
            dd['points_2d_pixel'] = pts_2d
        if pts_3d_cam:
            dd['points_camera_m'] = [p.tolist() for p in pts_3d_cam]
        with open(dfile, 'w') as f:
            yaml.dump(dd, f, default_flow_style=False)
    else:
        print("\n  FAILED or insufficient data.")

    arm.disable_torque()
    arm.disconnect()
    if cap:
        cap.release()
    if camera_rs:
        camera_rs.stop()
    cv2.destroyAllWindows()
    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
