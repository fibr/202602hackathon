#!/usr/bin/env python3
"""GUI tool for servo calibration and manual hand-eye calibration.

Shows a live camera feed while you move the arm by hand (torque off).
Two modes:

  SERVO CALIBRATION (default):
    Move the arm to its URDF zero pose and press SPACE to save offsets.
    Live readout of raw servo positions and angles overlaid on camera.

  HAND-EYE CALIBRATION (--handeye):
    Move arm to diverse poses, press SPACE to capture each correspondence
    (joint angles via FK + yellow tape pixel detection). Press S to solve
    when enough points are collected.

Usage:
    ./run.sh scripts/calibration_gui.py                 # Servo calibration
    ./run.sh scripts/calibration_gui.py --handeye       # Hand-eye calibration
"""

import sys
import os
import time
import argparse
import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config_loader import load_config
from robot.lerobot_arm101 import LeRobotArm101
from kinematics.arm101_ik_solver import Arm101IKSolver

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
OFFSET_FILE = os.path.join(PROJECT_ROOT, 'config', 'servo_offsets.yaml')
HANDEYE_FILE = os.path.join(PROJECT_ROOT, 'config', 'calibration_arm101.yaml')

MOTOR_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
               'wrist_flex', 'wrist_roll', 'gripper']

# STS3215 register for raw position
ADDR_PRESENT_POSITION = 56

# Yellow tape HSV defaults
YELLOW_HSV_LOW = np.array([18, 80, 120])
YELLOW_HSV_HIGH = np.array([35, 255, 255])


def read_all_raw(arm):
    """Read raw servo positions for all motors."""
    positions = {}
    for mid in arm.motor_ids:
        pos, result, error = arm.packet_handler.read2ByteTxRx(
            arm.port_handler, mid, ADDR_PRESENT_POSITION)
        positions[mid] = pos
    return positions


def find_yellow_tape(frame, min_area=50):
    """Detect yellow tape centroid. Returns (cx, cy) or (None, None)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_HSV_LOW, YELLOW_HSV_HIGH)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, mask
    best = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(best) < min_area:
        return None, None, mask
    M = cv2.moments(best)
    if M['m00'] == 0:
        return None, None, mask
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy, mask


def draw_servo_overlay(frame, raw_positions, offsets, angles_deg):
    """Draw servo info overlay on frame."""
    y0 = 30
    cv2.putText(frame, "SERVO CALIBRATION — torque OFF, move arm by hand",
                (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    y0 += 30

    for i, mid in enumerate(sorted(raw_positions.keys())):
        name = MOTOR_NAMES[mid - 1] if mid <= len(MOTOR_NAMES) else f"motor_{mid}"
        raw = raw_positions[mid]
        offset = offsets.get(name, {}).get('zero_raw', 2048)
        cal_deg = (raw - offset) * 360.0 / 4096.0
        uncal_deg = (raw - 2048) * 360.0 / 4096.0

        color = (200, 200, 200)
        text = f"J{mid} {name[:10]:<10}  raw={raw:4d}  cal={cal_deg:+6.1f}deg  (uncal={uncal_deg:+.0f})"
        cv2.putText(frame, text, (10, y0 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    # FK pose if available
    if angles_deg is not None:
        n_joints = len(raw_positions)
        y_fk = y0 + n_joints * 22 + 10
        ang_str = ", ".join(f"{a:.1f}" for a in angles_deg)
        cv2.putText(frame, f"Angles: [{ang_str}]",
                    (10, y_fk), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 200), 1)

    # Instructions at bottom
    h = frame.shape[0]
    cv2.putText(frame, "SPACE=save zero offsets | R=read | ESC=quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


def draw_handeye_overlay(frame, tcp_pos, yellow_pt, n_collected):
    """Draw hand-eye calibration overlay."""
    h = frame.shape[0]
    cv2.putText(frame, "HAND-EYE CALIBRATION — torque OFF, move arm by hand",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    if tcp_pos is not None:
        cv2.putText(frame, f"TCP: [{tcp_pos[0]:.1f}, {tcp_pos[1]:.1f}, {tcp_pos[2]:.1f}] mm",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

    if yellow_pt[0] is not None:
        cx, cy = yellow_pt
        cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
        cv2.circle(frame, (cx, cy), 14, (0, 255, 255), 2)
        cv2.putText(frame, f"Yellow: ({cx},{cy})",
                    (cx + 16, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    else:
        cv2.putText(frame, "No yellow tape detected",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.putText(frame, f"Collected: {n_collected} points (need >= 6)",
                (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, "SPACE=capture | S=joint solve (offsets+extrinsics) | U=undo | ESC=quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)


def load_offsets():
    """Load saved servo zero offsets."""
    if os.path.exists(OFFSET_FILE):
        with open(OFFSET_FILE, 'r') as f:
            data = yaml.safe_load(f)
        return data.get('zero_offsets', {})
    return {}


def save_offsets(raw_positions):
    """Save current raw positions as zero offsets."""
    offsets = {}
    for mid, pos in sorted(raw_positions.items()):
        name = MOTOR_NAMES[mid - 1]
        offsets[name] = {
            'motor_id': mid,
            'zero_raw': pos,
        }
    save_offsets_dict(offsets)
    return offsets


def save_offsets_dict(offsets):
    """Save an offsets dict to servo_offsets.yaml."""
    data = {
        'description': 'Servo zero offsets for SO-ARM101',
        'zero_offsets': offsets,
        'notes': {
            'usage': 'angle_deg = (raw_position - zero_raw) * 360/4096',
            'default': '2048 (servo center) if no offset defined',
        }
    }
    os.makedirs(os.path.dirname(OFFSET_FILE), exist_ok=True)
    with open(OFFSET_FILE, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"  Saved offsets to {OFFSET_FILE}")


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


def save_handeye_calibration(T_cam2base, filepath):
    """Save hand-eye calibration transform."""
    from calibration.transform import CoordinateTransform
    from scipy.spatial.transform import Rotation

    ct = CoordinateTransform()
    ct.T_camera_to_base = T_cam2base
    R = T_cam2base[:3, :3]
    rpy = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
    ct.base_offset_mm = T_cam2base[:3, 3].copy()
    ct.base_rpy_deg = rpy
    ct.save(filepath)
    print(f"  Saved to {filepath}")


def solve_and_save_handeye(pts_3d_robot, pts_2d, K, dist):
    """Solve PnP and save calibration."""
    T = solve_pnp(pts_3d_robot, pts_2d, K, dist)
    if T is not None:
        save_handeye_calibration(T, HANDEYE_FILE)
        return True
    return False


DEG_PER_POS = 360.0 / 4096.0


def joint_solve(raw_positions_list, pts_2d, K, dist_coeffs, solver):
    """Jointly optimize servo zero offsets + camera extrinsics.

    Instead of trusting FK (which uses potentially wrong offsets), we
    parameterize the offsets and extrinsic together and minimize
    reprojection error.

    Parameters to optimize (11 total):
      - 5 servo zero offsets (raw units, one per joint excl. gripper)
      - 3 rotation (Rodrigues rvec)
      - 3 translation (tvec)

    For each observation:
      raw_pos -> angles(offsets) -> FK -> TCP_robot -> project(T, K) -> pixel
    Minimize: sum of ||pixel_predicted - pixel_observed||^2

    Args:
        raw_positions_list: List of dicts {motor_id: raw_pos} per capture.
        pts_2d: List of [cx, cy] pixel observations.
        K: 3x3 camera intrinsic matrix.
        dist_coeffs: Distortion coefficients.
        solver: Arm101IKSolver instance.

    Returns:
        (offsets_dict, T_cam2base) or (None, None) on failure.
    """
    from scipy.optimize import least_squares

    n = len(pts_2d)
    if n < 6:
        print(f"  Need >= 6 points for joint solve (have {n})")
        return None, None

    # Current offsets as initial guess
    current_offsets = load_offsets()
    offset_init = np.array([
        current_offsets.get(name, {}).get('zero_raw', 2048)
        for name in MOTOR_NAMES[:5]
    ], dtype=float)

    # Initial extrinsic guess from standard PnP with current offsets
    pts_3d_init = []
    for raw_pos in raw_positions_list:
        angles = np.array([
            (raw_pos[mid] - offset_init[mid - 1]) * DEG_PER_POS
            for mid in range(1, 6)
        ])
        tcp, _ = solver.forward_kin(angles)
        pts_3d_init.append(tcp)

    T_init = solve_pnp(pts_3d_init, pts_2d, K, dist_coeffs)
    if T_init is None:
        print("  Initial PnP failed, using identity as seed")
        rvec_init = np.zeros(3)
        tvec_init = np.array([0.0, 0.0, 500.0])
    else:
        # T_init is cam2base, PnP needs base2cam
        T_b2c = np.linalg.inv(T_init)
        rvec_init, _ = cv2.Rodrigues(T_b2c[:3, :3])
        rvec_init = rvec_init.flatten()
        tvec_init = T_b2c[:3, 3]

    # Pack: [5 offsets, 3 rvec, 3 tvec]
    x0 = np.concatenate([offset_init, rvec_init, tvec_init])

    pts_2d_arr = np.array(pts_2d, dtype=np.float64)

    def residuals(x):
        offsets_raw = x[:5]
        rvec = x[5:8]
        tvec = x[8:11]

        R_b2c, _ = cv2.Rodrigues(rvec)

        errs = []
        for i in range(n):
            # raw -> angles using candidate offsets
            angles = np.array([
                (raw_positions_list[i][mid] - offsets_raw[mid - 1]) * DEG_PER_POS
                for mid in range(1, 6)
            ])
            # FK -> TCP in robot frame (mm)
            tcp, _ = solver.forward_kin(angles)

            # Project to pixel: p = K @ (R @ tcp + t)
            p_cam = R_b2c @ tcp + tvec
            if p_cam[2] <= 0:
                errs.extend([1000.0, 1000.0])
                continue
            px = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
            py = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]

            errs.append(px - pts_2d_arr[i, 0])
            errs.append(py - pts_2d_arr[i, 1])

        return np.array(errs)

    print(f"\n  Joint optimization: {n} observations, 11 parameters "
          f"({2*n} residuals)")
    print(f"  Initial offsets: {offset_init.astype(int).tolist()}")

    # Initial error
    r0 = residuals(x0).reshape(-1, 2)
    e0 = np.linalg.norm(r0, axis=1)
    print(f"  Initial error: mean={np.mean(e0):.1f}px, max={np.max(e0):.1f}px")

    result = least_squares(residuals, x0, method='lm', max_nfev=5000)

    offsets_opt = result.x[:5]
    rvec_opt = result.x[5:8]
    tvec_opt = result.x[8:11]

    # Per-point residuals
    res = result.fun.reshape(-1, 2)
    errs_px = np.linalg.norm(res, axis=1)
    print(f"\n  Final error: mean={np.mean(errs_px):.2f}px, "
          f"median={np.median(errs_px):.2f}px, max={np.max(errs_px):.2f}px")

    # Per-point breakdown
    print(f"  Per-point residuals:")
    for i in range(n):
        flag = " ***" if errs_px[i] > 2 * np.median(errs_px) else ""
        print(f"    #{i+1:2d}: {errs_px[i]:6.2f}px  "
              f"(dx={res[i,0]:+6.1f}, dy={res[i,1]:+6.1f}){flag}")

    # Parameter uncertainty from Jacobian
    # cov ≈ inv(J^T J) * s^2,  where s^2 = cost / (n_residuals - n_params)
    param_names = (
        [f"{name}_offset" for name in MOTOR_NAMES[:5]]
        + ['rvec_x', 'rvec_y', 'rvec_z', 'tvec_x', 'tvec_y', 'tvec_z']
    )
    n_residuals = 2 * n
    n_params = 11
    dof = max(1, n_residuals - n_params)
    s2 = result.cost / dof  # variance estimate
    try:
        J = result.jac
        JtJ = J.T @ J
        cov = np.linalg.inv(JtJ) * s2
        std_devs = np.sqrt(np.abs(np.diag(cov)))

        print(f"\n  Parameter estimates (value ± 1σ uncertainty):")
        for i, name in enumerate(MOTOR_NAMES[:5]):
            old = int(offset_init[i])
            new = offsets_opt[i]
            sigma_raw = std_devs[i]
            sigma_deg = sigma_raw * DEG_PER_POS
            delta_deg = (new - old) * DEG_PER_POS
            confidence = "GOOD" if sigma_deg < 3.0 else "WEAK" if sigma_deg < 10.0 else "BAD"
            print(f"    {name:<16}: {old} -> {int(round(new)):5d}  "
                  f"(Δ={delta_deg:+6.1f}°)  ±{sigma_deg:.1f}°  [{confidence}]")

        # Extrinsic uncertainties
        print(f"    {'tvec_x':<16}: {tvec_opt[0]:+8.1f}mm  ±{std_devs[8]:.1f}mm")
        print(f"    {'tvec_y':<16}: {tvec_opt[1]:+8.1f}mm  ±{std_devs[9]:.1f}mm")
        print(f"    {'tvec_z':<16}: {tvec_opt[2]:+8.1f}mm  ±{std_devs[10]:.1f}mm")
        rvec_deg_sigma = [np.degrees(std_devs[5+j]) for j in range(3)]
        print(f"    {'rotation':<16}: ±({rvec_deg_sigma[0]:.2f}°, "
              f"{rvec_deg_sigma[1]:.2f}°, {rvec_deg_sigma[2]:.2f}°)")
    except np.linalg.LinAlgError:
        print("  WARNING: Could not compute parameter uncertainties (singular Jacobian)")
        print(f"  Optimized offsets:")
        for i, name in enumerate(MOTOR_NAMES[:5]):
            old = int(offset_init[i])
            new = int(round(offsets_opt[i]))
            delta_deg = (new - old) * DEG_PER_POS
            print(f"    {name:<16}: {old} -> {new}  (Δ={delta_deg:+.1f}°)")

    # Quality assessment
    quality = ("EXCELLENT" if np.mean(errs_px) < 3 else
               "GOOD" if np.mean(errs_px) < 8 else
               "OK" if np.mean(errs_px) < 20 else
               "POOR" if np.mean(errs_px) < 50 else "BAD")
    print(f"\n  Overall quality: {quality} (mean {np.mean(errs_px):.1f}px)")
    if quality in ("POOR", "BAD"):
        n_outliers = np.sum(errs_px > 2 * np.median(errs_px))
        print(f"  Suggestions:")
        if n_outliers > 0:
            print(f"    - {n_outliers} outlier(s) marked with *** — "
                  f"undo them (U) and re-solve")
        print(f"    - Check joint signs in arm101_ik_solver.py JOINT_SIGNS")
        print(f"    - Check camera intrinsics (currently estimated, not calibrated)")
        print(f"    - Try more diverse poses (vary all joints, not just 1-2)")

    # Build T_cam2base
    R_b2c, _ = cv2.Rodrigues(rvec_opt)
    T_b2c = np.eye(4)
    T_b2c[:3, :3] = R_b2c
    T_b2c[:3, 3] = tvec_opt
    T_c2b = np.linalg.inv(T_b2c)

    pos = T_c2b[:3, 3]
    print(f"  Camera in robot frame: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}] mm")

    # Build offsets dict
    offsets_dict = {}
    for i, name in enumerate(MOTOR_NAMES[:5]):
        offsets_dict[name] = {
            'motor_id': i + 1,
            'zero_raw': int(round(offsets_opt[i])),
        }

    return offsets_dict, T_c2b


def main():
    parser = argparse.ArgumentParser(description="GUI calibration tool for arm101")
    parser.add_argument('--handeye', action='store_true',
                        help='Hand-eye calibration mode (default: servo calibration)')
    parser.add_argument('--camera', type=int, default=None,
                        help='Camera device index (default: from config)')
    args = parser.parse_args()

    config = load_config()

    # Camera setup
    cam_cfg = config.get('camera', {})
    cam_idx = args.camera if args.camera is not None else cam_cfg.get('device_index', 4)
    print(f"Opening camera /dev/video{cam_idx}...")
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {cam_idx}")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Flush initial frames
    for _ in range(10):
        cap.read()
        time.sleep(0.03)
    print("  Camera ready.")

    # Connect arm (no torque)
    ac = config.get('arm101', {})
    port = ac.get('port', '') or LeRobotArm101.find_port()
    print(f"Connecting to arm on {port}...")
    arm = LeRobotArm101(port=port, baudrate=ac.get('baudrate', 1_000_000),
                         motor_ids=ac.get('motor_ids', [1, 2, 3, 4, 5, 6]),
                         speed=ac.get('speed', 200))
    arm.connect()
    print("  Disabling torque — you can move the arm by hand.")
    arm.disable_torque()

    # FK solver
    solver = Arm101IKSolver()

    # Load camera intrinsics for handeye mode
    K, dist_coeffs = None, None
    if args.handeye:
        cam_yaml = os.path.join(PROJECT_ROOT, 'config', 'cameras.yaml')
        if os.path.exists(cam_yaml):
            with open(cam_yaml) as f:
                cdata = yaml.safe_load(f)
            for cname, cinfo in cdata.get('cameras', {}).items():
                if cinfo.get('device_index') == cam_idx:
                    intr = cinfo['intrinsics']
                    K = np.array(intr['camera_matrix'], dtype=np.float64)
                    dist_coeffs = np.array(intr['dist_coeffs'], dtype=np.float64)
                    print(f"  Intrinsics from {cname}: fx={K[0,0]:.1f}")
                    break
        if K is None:
            K = np.array([[554.3, 0, 320], [0, 554.3, 240], [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.zeros(5, dtype=np.float64)
            print("  Using default intrinsics (estimated)")

    # State
    offsets = load_offsets()
    status_msg = ""
    status_time = 0
    # Hand-eye state
    pts_3d_robot = []
    pts_2d = []
    raw_positions_list = []  # raw servo positions per capture (for joint solve)

    mode_name = "Hand-Eye Calibration" if args.handeye else "Servo Calibration"
    win = f"arm101 {mode_name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print(f"\n=== {mode_name} GUI ===")
    print("Torque is OFF. Move the arm by hand.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Read servo state
            raw_positions = read_all_raw(arm)
            angles = arm.get_angles()

            # FK
            tcp_pos = None
            if angles is not None:
                try:
                    tcp_pos, _ = solver.forward_kin(np.array(angles[:5]))
                except Exception:
                    pass

            if args.handeye:
                # Yellow tape detection
                cx, cy, mask = find_yellow_tape(frame)
                draw_handeye_overlay(frame, tcp_pos, (cx, cy), len(pts_2d))

                # Show small mask in corner
                mask_small = cv2.resize(mask, (160, 120))
                mask_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
                frame[0:120, frame.shape[1]-160:frame.shape[1]] = mask_bgr
            else:
                draw_servo_overlay(frame, raw_positions, offsets, angles)

            # Status message (fades after 3s)
            if status_msg and time.time() - status_time < 3.0:
                cv2.putText(frame, status_msg, (10, frame.shape[0] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow(win, frame)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # ESC
                break

            try:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            if key == ord(' '):
                if args.handeye:
                    # Capture hand-eye point (raw positions + pixel)
                    if tcp_pos is not None and cx is not None:
                        pts_3d_robot.append(tcp_pos.copy())
                        pts_2d.append([float(cx), float(cy)])
                        raw_positions_list.append(dict(raw_positions))
                        status_msg = f"Captured #{len(pts_2d)}: TCP=[{tcp_pos[0]:.0f},{tcp_pos[1]:.0f},{tcp_pos[2]:.0f}] px=({cx},{cy})"
                        status_time = time.time()
                        print(f"  {status_msg}")
                    else:
                        status_msg = "SKIP: no TCP or no yellow tape"
                        status_time = time.time()
                else:
                    # Save servo offsets
                    offsets = save_offsets(raw_positions)
                    status_msg = f"Saved offsets to {os.path.basename(OFFSET_FILE)}"
                    status_time = time.time()
                    print(f"  {status_msg}")
                    for mid, pos in sorted(raw_positions.items()):
                        name = MOTOR_NAMES[mid - 1]
                        print(f"    {name}: raw={pos}")

            elif key == ord('s') and args.handeye:
                # Joint solve: optimize servo offsets + camera extrinsics together
                if len(pts_2d) < 6:
                    status_msg = f"Need >= 6 points for joint solve (have {len(pts_2d)})"
                    status_time = time.time()
                else:
                    print(f"\nJoint solve: {len(pts_2d)} points, optimizing offsets + extrinsics...")
                    opt_offsets, T_c2b = joint_solve(
                        raw_positions_list, pts_2d, K, dist_coeffs, solver)
                    if opt_offsets is not None:
                        # Save both servo offsets and hand-eye calibration
                        save_offsets_dict(opt_offsets)
                        save_handeye_calibration(T_c2b, HANDEYE_FILE)
                        offsets = load_offsets()  # reload for display
                        status_msg = "Joint solve OK — saved offsets + extrinsics"
                    else:
                        status_msg = "Joint solve FAILED"
                    status_time = time.time()
                    print(f"  {status_msg}")

            elif key == ord('u') and args.handeye:
                # Undo last capture
                if pts_2d:
                    pts_2d.pop()
                    pts_3d_robot.pop()
                    raw_positions_list.pop()
                    status_msg = f"Undone — {len(pts_2d)} points remaining"
                    status_time = time.time()

            elif key == ord('r') and not args.handeye:
                # Reload offsets
                offsets = load_offsets()
                status_msg = "Reloaded offsets from file"
                status_time = time.time()

    except KeyboardInterrupt:
        pass
    finally:
        arm.disconnect()
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
