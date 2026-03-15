"""Pure algorithm helpers for calibration workflows.

This module centralises the reusable, side-effect-free (or minimally I/O-bound)
calibration functions that were previously duplicated between
``scripts/calibration_gui.py`` and ``scripts/detect_checkerboard.py``.

Both scripts and all GUI views in ``src/gui/views/`` now import from here.

Sections
--------
1.  Shared constants (motor names, file paths, board geometry, RANSAC params)
2.  Servo / arm101 helpers  (find_yellow_tape, load/save offsets, solve_pnp,
    joint_solve, save_handeye_calibration, read_all_raw, draw overlays)
3.  Checkerboard / geometry helpers  (detect_corners, compute_board_pose,
    pixel_to_ray, ray_plane_intersect, solve_rigid_transform,
    solve_robust_transform, _get_board_outer_corners_cam)
"""

import os
import random

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# 1.  Shared constants
# ---------------------------------------------------------------------------

# --- Arm101 / servo ---

MOTOR_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
               'wrist_flex', 'wrist_roll', 'gripper']

# STS3215 register for raw position
ADDR_PRESENT_POSITION = 56

# Yellow tape HSV detection range
YELLOW_HSV_LOW = np.array([18, 80, 120])
YELLOW_HSV_HIGH = np.array([35, 255, 255])

# Raw encoder ticks → degrees
DEG_PER_POS = 360.0 / 4096.0

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
from config_loader import config_path as _config_path
OFFSET_FILE = _config_path('servo_offsets.yaml')
HANDEYE_FILE = _config_path('calibration_arm101.yaml')

# --- Checkerboard / geometry ---

# Legacy checkerboard parameters (fallback when no BoardDetector config)
BOARD_COLS = 7    # inner corners (8 squares − 1)
BOARD_ROWS = 9    # inner corners (10 squares − 1)
SQUARE_SIZE_M = 0.02  # 2 cm squares

SNAP_RADIUS_PX = 30  # max pixel distance to snap a click to a detected corner

# RANSAC parameters for solve_robust_transform
RANSAC_ITERATIONS = 500
RANSAC_INLIER_THRESHOLD_MM = 15.0


# ---------------------------------------------------------------------------
# 2.  Servo / arm101 helpers
# ---------------------------------------------------------------------------

def read_all_raw(arm):
    """Read raw servo positions for all motors.

    Args:
        arm: LeRobotArm101 instance (already connected).

    Returns:
        dict mapping motor_id (int) to raw position (int).
    """
    positions = {}
    for mid in arm.motor_ids:
        pos, result, error = arm.packet_handler.read2ByteTxRx(
            arm.port_handler, mid, ADDR_PRESENT_POSITION)
        positions[mid] = pos
    return positions


def find_yellow_tape(frame, min_area=50):
    """Detect yellow tape centroid in an BGR frame.

    Uses HSV colour thresholding with morphological open/close to remove
    noise.

    Args:
        frame: BGR image (numpy ndarray).
        min_area: Minimum contour area in pixels to be considered a detection.

    Returns:
        (cx, cy, mask) where (cx, cy) are int pixel coordinates of the
        largest yellow blob centroid, or (None, None, mask) if not found.
    """
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
    """Draw servo calibration info overlay onto *frame* in-place.

    Args:
        frame: BGR image to annotate.
        raw_positions: dict {motor_id: raw_pos}.
        offsets: dict as returned by load_offsets().
        angles_deg: sequence of joint angles in degrees, or None.
    """
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
        text = (f"J{mid} {name[:10]:<10}  raw={raw:4d}  cal={cal_deg:+6.1f}deg"
                f"  (uncal={uncal_deg:+.0f})")
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
    """Draw hand-eye calibration info overlay onto *frame* in-place.

    Args:
        frame: BGR image to annotate.
        tcp_pos: [x, y, z] TCP position in mm, or None.
        yellow_pt: (cx, cy) yellow tape centroid pixels, or (None, None).
        n_collected: number of correspondences collected so far.
    """
    h = frame.shape[0]
    cv2.putText(frame, "HAND-EYE CALIBRATION — torque OFF, move arm by hand",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    if tcp_pos is not None:
        cv2.putText(frame,
                    f"TCP: [{tcp_pos[0]:.1f}, {tcp_pos[1]:.1f}, {tcp_pos[2]:.1f}] mm",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

    cx, cy = yellow_pt
    if cx is not None:
        cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
        cv2.circle(frame, (cx, cy), 14, (0, 255, 255), 2)
        cv2.putText(frame, f"Yellow: ({cx},{cy})",
                    (cx + 16, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    else:
        cv2.putText(frame, "No yellow tape detected",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.putText(frame, f"Collected: {n_collected} points (need >= 6)",
                (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame,
                "SPACE=capture | S=joint solve (offsets+extrinsics) | U=undo | ESC=quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)


def load_offsets():
    """Load saved servo zero offsets from servo_offsets.yaml.

    Returns:
        dict  {motor_name: {'motor_id': int, 'zero_raw': int, ...}}
        or empty dict if no file exists.
    """
    if os.path.exists(OFFSET_FILE):
        with open(OFFSET_FILE, 'r') as f:
            data = yaml.safe_load(f)
        return data.get('zero_offsets', {})
    return {}


def save_offsets(raw_positions):
    """Save current raw servo positions as zero offsets.

    Args:
        raw_positions: dict {motor_id: raw_pos} as returned by read_all_raw().

    Returns:
        offsets dict that was saved.
    """
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
    """Persist an offsets dict to servo_offsets.yaml.

    Args:
        offsets: dict  {motor_name: {'motor_id': int, 'zero_raw': int, ...}}
    """
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
    """Compute T_camera_to_base from 3-D (robot mm) / 2-D (pixel) correspondences.

    Tries RANSAC (when ≥ 6 points), ITERATIVE, and EPNP solvers and returns
    the best result by reprojection error.

    Args:
        pts_3d_mm: Nx3 array of 3-D points in robot base frame (mm).
        pts_2d_px: Nx2 array of 2-D pixel coordinates.
        K: 3x3 camera intrinsic matrix.
        dist: distortion coefficients.

    Returns:
        4x4 T_cam_to_base homogeneous transform, or None if PnP failed.
    """
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
            e = np.mean(np.linalg.norm(
                p.reshape(-1, 2) - img.reshape(-1, 2), axis=1))
            print(f"  RANSAC: {len(inl)}/{len(obj)} inliers, err={e:.2f}px")
            if e < best_e:
                best_e, best = e, (rv, tv)

    for nm, fl in [("ITER", cv2.SOLVEPNP_ITERATIVE), ("EPNP", cv2.SOLVEPNP_EPNP)]:
        try:
            ok, rv, tv = cv2.solvePnP(obj, img, K, dist, flags=fl)
            if ok:
                p, _ = cv2.projectPoints(obj, rv, tv, K, dist)
                e = np.mean(np.linalg.norm(
                    p.reshape(-1, 2) - img.reshape(-1, 2), axis=1))
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
    """Save a hand-eye calibration transform to a YAML file.

    Uses CoordinateTransform to write a calibration.yaml compatible file.

    Args:
        T_cam2base: 4x4 camera-to-base homogeneous transform.
        filepath: Destination path (created if needed).
    """
    from .transform import CoordinateTransform

    ct = CoordinateTransform()
    ct.T_camera_to_base = T_cam2base
    R = T_cam2base[:3, :3]
    rpy = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
    ct.base_offset_mm = T_cam2base[:3, 3].copy()
    ct.base_rpy_deg = rpy
    ct.save(filepath)
    print(f"  Saved to {filepath}")


def solve_and_save_handeye(pts_3d_robot, pts_2d, K, dist):
    """PnP solve and immediately save to the default HANDEYE_FILE path.

    Args:
        pts_3d_robot: Nx3 TCP positions in robot frame (mm).
        pts_2d: Nx2 pixel observations.
        K: 3x3 intrinsic matrix.
        dist: distortion coefficients.

    Returns:
        True on success, False if PnP failed.
    """
    T = solve_pnp(pts_3d_robot, pts_2d, K, dist)
    if T is not None:
        save_handeye_calibration(T, HANDEYE_FILE)
        return True
    return False


def joint_solve(raw_positions_list, pts_2d, K, dist_coeffs, solver, progress_callback=None):
    """Jointly optimise servo zero offsets + camera extrinsics.

    Instead of trusting FK (which uses potentially wrong offsets), we
    parameterise the offsets and extrinsic together and minimise
    reprojection error.

    Parameters to optimise (11 total):
      - 5 servo zero offsets (raw units, one per joint excl. gripper)
      - 3 rotation (Rodrigues rvec)
      - 3 translation (tvec)

    For each observation:
      raw_pos → angles(offsets) → FK → TCP_robot → project(T, K) → pixel
    Minimise: sum of ||pixel_predicted − pixel_observed||²

    Args:
        raw_positions_list: List of dicts {motor_id: raw_pos} per capture.
        pts_2d: List of [cx, cy] pixel observations.
        K: 3x3 camera intrinsic matrix.
        dist_coeffs: Distortion coefficients.
        solver: Arm101IKSolver instance.
        progress_callback: Optional callable(iteration: int, max_iterations: int)
            called during optimization to report progress.

    Returns:
        (offsets_dict, T_cam2base) or (None, None) on failure.
    """
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
            # raw → angles using candidate offsets
            angles = np.array([
                (raw_positions_list[i][mid] - offsets_raw[mid - 1]) * DEG_PER_POS
                for mid in range(1, 6)
            ])
            # FK → TCP in robot frame (mm)
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

    print(f"\n  Joint optimisation: {n} observations, 11 parameters "
          f"({2*n} residuals)")
    print(f"  Initial offsets: {offset_init.astype(int).tolist()}")

    # Initial error
    r0 = residuals(x0).reshape(-1, 2)
    e0 = np.linalg.norm(r0, axis=1)
    print(f"  Initial error: mean={np.mean(e0):.1f}px, max={np.max(e0):.1f}px")

    # Setup progress tracking
    iteration_count = [0]  # Use list to allow modification in nested function
    max_iterations = 5000

    def progress_wrapper(xk, *args, **kwargs):
        """Wrapper to track optimization progress."""
        iteration_count[0] += 1
        if progress_callback is not None:
            progress_callback(iteration_count[0], max_iterations)

    result = least_squares(residuals, x0, method='lm', max_nfev=max_iterations,
                          callback=progress_wrapper)

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
    # cov ≈ inv(J^T J) * s²,  where s² = cost / (n_residuals − n_params)
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
            confidence = ("GOOD" if sigma_deg < 3.0
                          else "WEAK" if sigma_deg < 10.0 else "BAD")
            print(f"    {name:<16}: {old} -> {int(round(new)):5d}  "
                  f"(Δ={delta_deg:+6.1f}°)  ±{sigma_deg:.1f}°  [{confidence}]")

        # Extrinsic uncertainties
        print(f"    {'tvec_x':<16}: {tvec_opt[0]:+8.1f}mm  ±{std_devs[8]:.1f}mm")
        print(f"    {'tvec_y':<16}: {tvec_opt[1]:+8.1f}mm  ±{std_devs[9]:.1f}mm")
        print(f"    {'tvec_z':<16}: {tvec_opt[2]:+8.1f}mm  ±{std_devs[10]:.1f}mm")
        rvec_deg_sigma = [np.degrees(std_devs[5 + j]) for j in range(3)]
        print(f"    {'rotation':<16}: ±({rvec_deg_sigma[0]:.2f}°, "
              f"{rvec_deg_sigma[1]:.2f}°, {rvec_deg_sigma[2]:.2f}°)")
    except np.linalg.LinAlgError:
        print("  WARNING: Could not compute parameter uncertainties "
              "(singular Jacobian)")
        print(f"  Optimised offsets:")
        for i, name in enumerate(MOTOR_NAMES[:5]):
            old = int(offset_init[i])
            new = int(round(offsets_opt[i]))
            delta_deg = (new - old) * DEG_PER_POS
            print(f"    {name:<16}: {old} -> {new}  (Δ={delta_deg:+.1f}°)")

    # Quality assessment
    quality = ("EXCELLENT" if np.mean(errs_px) < 3
               else "GOOD" if np.mean(errs_px) < 8
               else "OK" if np.mean(errs_px) < 20
               else "POOR" if np.mean(errs_px) < 50 else "BAD")
    print(f"\n  Overall quality: {quality} (mean {np.mean(errs_px):.1f}px)")
    if quality in ("POOR", "BAD"):
        n_outliers = np.sum(errs_px > 2 * np.median(errs_px))
        print(f"  Suggestions:")
        if n_outliers > 0:
            print(f"    - {n_outliers} outlier(s) marked with *** — "
                  f"undo them (U) and re-solve")
        print(f"    - Check joint signs in arm101_ik_solver.py JOINT_SIGNS")
        print(f"    - Check camera intrinsics (currently estimated, "
              f"not calibrated)")
        print(f"    - Try more diverse poses (vary all joints, not just 1-2)")

    # Build T_cam2base
    R_b2c, _ = cv2.Rodrigues(rvec_opt)
    T_b2c = np.eye(4)
    T_b2c[:3, :3] = R_b2c
    T_b2c[:3, 3] = tvec_opt
    T_c2b = np.linalg.inv(T_b2c)

    pos = T_c2b[:3, 3]
    print(f"  Camera in robot frame: "
          f"[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}] mm")

    # Build offsets dict
    offsets_dict = {}
    for i, name in enumerate(MOTOR_NAMES[:5]):
        offsets_dict[name] = {
            'motor_id': i + 1,
            'zero_raw': int(round(offsets_opt[i])),
        }

    return offsets_dict, T_c2b


# ---------------------------------------------------------------------------
# 3.  Checkerboard / geometry helpers
# ---------------------------------------------------------------------------

def detect_corners(gray, board_detector=None):
    """Find board corners in a grayscale image.

    Uses *board_detector* when provided (supports CharuCo / ArUco boards).
    Falls back to legacy OpenCV checkerboard detection with multiple
    enhancement strategies.

    Args:
        gray: Grayscale image (numpy ndarray).
        board_detector: Optional BoardDetector instance.

    Returns:
        (found: bool, corners: ndarray or None, detection: BoardDetection or None)
    """
    if board_detector is not None:
        det = board_detector.detect(gray)
        if det is not None:
            return True, det.corners, det
        return False, None, None

    # Legacy fallback (no BoardDetector configured)
    try:
        found, corners = cv2.findChessboardCornersSB(
            gray, (BOARD_COLS, BOARD_ROWS),
            cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if found:
            return found, corners, None
    except cv2.error:
        pass

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(
        enhanced, (BOARD_COLS, BOARD_ROWS), flags)
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        return found, corners, None

    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
             cv2.CALIB_CB_NORMALIZE_IMAGE |
             cv2.CALIB_CB_FILTER_QUADS)
    found, corners = cv2.findChessboardCorners(
        sharpened, (BOARD_COLS, BOARD_ROWS), flags)
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        return found, corners, None

    for cols, rows in [(BOARD_COLS, BOARD_ROWS - 2),
                       (BOARD_COLS - 2, BOARD_ROWS),
                       (BOARD_COLS - 2, BOARD_ROWS - 2)]:
        try:
            found, corners = cv2.findChessboardCornersSB(
                gray, (cols, rows), cv2.CALIB_CB_EXHAUSTIVE)
            if found:
                print(f"  (detected {cols}x{rows} subset of board)")
                return found, corners, None
        except cv2.error:
            pass

    return False, None, None


def compute_board_pose(corners_2d, intrinsics, detection=None,
                       board_detector=None):
    """solvePnP → (T_board_in_cam 4x4, obj_points, reproj_error_px).

    reproj_error_px is the RMS reprojection error in pixels — measures how
    well the current intrinsics explain the detected corners.

    Args:
        corners_2d: Detected corners (Nx1x2 or Nx2).
        intrinsics: CameraIntrinsics.
        detection: Optional BoardDetection for charuco ID-based obj points.
        board_detector: Optional BoardDetector instance.

    Returns:
        (T 4x4, obj_points ndarray, reproj_err_px float)
    """
    if board_detector is not None and detection is not None:
        return board_detector.compute_pose(detection, intrinsics)

    # Legacy fallback (no detector or no detection metadata)
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
    projected, _ = cv2.projectPoints(
        obj_points, rvec, tvec, camera_matrix, dist_coeffs)
    reproj_err = np.sqrt(np.mean(
        (corners_2d.reshape(-1, 2) - projected.reshape(-1, 2)) ** 2))

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T, obj_points, reproj_err


def corner_3d_in_cam(corner_idx, T_board_in_cam, n_cols=BOARD_COLS):
    """Get the 3-D position of a detected corner in camera frame (meters).

    Args:
        corner_idx: Linear index into the corners array.
        T_board_in_cam: 4x4 board-to-camera transform.
        n_cols: Number of corner columns (default BOARD_COLS).

    Returns:
        np.ndarray [x, y, z] in camera frame (metres).
    """
    row = corner_idx // n_cols
    col = corner_idx % n_cols
    p_board = np.array([col * SQUARE_SIZE_M, row * SQUARE_SIZE_M, 0, 1])
    p_cam = (T_board_in_cam @ p_board)[:3]
    return p_cam


def pixel_to_ray(pixel, intrinsics):
    """Convert a pixel to a normalised ray direction, accounting for distortion.

    Uses cv2.undistortPoints to remove lens distortion, giving a more accurate
    ray direction than raw pinhole backprojection.

    Args:
        pixel: (x, y) pixel coordinates.
        intrinsics: CameraIntrinsics or pyrealsense2 intrinsics.

    Returns:
        np.ndarray [x, y, 1] unnormalised ray direction in camera frame.
    """
    px, py = pixel
    camera_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float64)

    # undistortPoints with no newCameraMatrix returns normalised coords
    pts = cv2.undistortPoints(
        np.array([[[px, py]]], dtype=np.float64),
        camera_matrix, dist_coeffs)
    return np.array([pts[0][0][0], pts[0][0][1], 1.0])


def ray_plane_intersect(pixel, intrinsics, T_board_in_cam):
    """Intersect a camera ray with the checkerboard plane.

    The board plane is z=0 in board frame, transformed to camera frame via
    T_board_in_cam.  Returns the 3-D intersection point in camera frame
    (metres).

    Uses cv2.undistortPoints for accurate ray computation with distortion.

    Args:
        pixel: (x, y) pixel coordinates.
        intrinsics: CameraIntrinsics or pyrealsense2 intrinsics.
        T_board_in_cam: 4x4 board-to-camera transform from solvePnP.

    Returns:
        np.ndarray [x, y, z] in camera frame (metres), or None if the ray
        is parallel to the plane.
    """
    ray_dir = pixel_to_ray(pixel, intrinsics)

    # Board plane: the board's z=0 plane in camera frame
    # Normal = R @ [0,0,1]  (board z-axis in camera frame)
    # Point on plane = translation column of T_board_in_cam
    R = T_board_in_cam[:3, :3]
    t = T_board_in_cam[:3, 3]
    plane_normal = R[:, 2]   # third column of R
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
    """SVD-based rigid transform: T such that pts_robot ≈ T @ pts_cam.

    Args:
        pts_cam: Nx3 points in camera frame (metres).
        pts_robot: Nx3 points in robot frame (metres).

    Returns:
        4x4 homogeneous transform T_cam_to_base.
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
    """Refine a rigid transform via least-squares optimisation on SE(3).

    Parameterises the transform as 6 DOF (3 rotation via Rodrigues + 3
    translation) and minimises the sum of squared residuals.

    Args:
        T_init: 4x4 initial transform estimate.
        pts_cam: Nx3 points in camera frame (metres).
        pts_robot: Nx3 points in robot frame (metres).

    Returns:
        4x4 refined transform.
    """
    A = np.array(pts_cam)
    B = np.array(pts_robot)

    # Extract initial params: rotation as Rodrigues vector + translation
    R_init = T_init[:3, :3]
    t_init = T_init[:3, 3]
    rotvec_init = Rotation.from_matrix(R_init).as_rotvec()
    x0 = np.concatenate([rotvec_init, t_init])

    def residuals(x):
        R_loc = Rotation.from_rotvec(x[:3]).as_matrix()
        t_loc = x[3:6]
        transformed = (R_loc @ A.T).T + t_loc
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
        pts_cam: Nx3 points in camera frame (metres).
        pts_robot: Nx3 points in robot frame (metres).

    Returns:
        (T_cam_to_base 4x4, inlier_mask bool ndarray)
    """
    N = len(pts_cam)
    A = np.array(pts_cam)
    B = np.array(pts_robot)

    if N < 4:
        # Not enough for RANSAC — plain SVD + refine
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


def _get_board_outer_corners_cam(T_board_in_cam, board_detector=None):
    """Get the 4 outer corners of the calibration board in camera frame (metres).

    Args:
        T_board_in_cam: 4x4 board-to-camera transform.
        board_detector: Optional BoardDetector instance (for non-default board
            dimensions).

    Returns:
        (corners_cam, labels) where corners_cam is a list of 4 np.ndarray
        [x,y,z] positions in camera frame (metres) and labels is a list of
        4 strings ('top-left', 'top-right', 'bottom-right', 'bottom-left').
    """
    if board_detector is not None:
        sq = board_detector.square_size_m
        inner_cols = board_detector.inner_cols
        inner_rows = board_detector.inner_rows
    else:
        sq = SQUARE_SIZE_M
        inner_cols = BOARD_COLS
        inner_rows = BOARD_ROWS
    half = sq / 2.0
    max_x = (inner_cols - 1) * sq
    max_y = (inner_rows - 1) * sq
    board_corners = [
        np.array([-half, -half, 0, 1]),
        np.array([max_x + half, -half, 0, 1]),
        np.array([max_x + half, max_y + half, 0, 1]),
        np.array([-half, max_y + half, 0, 1]),
    ]
    labels = ["top-left", "top-right", "bottom-right", "bottom-left"]
    corners_cam = [(T_board_in_cam @ pt)[:3] for pt in board_corners]
    return corners_cam, labels
