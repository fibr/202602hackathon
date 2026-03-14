#!/usr/bin/env python3
"""Detect green cubes with overview camera and move SO-ARM101 above each one.

Pipeline:
  1. Open overview camera (Logitech BRIO, /dev/video4)
  2. Detect green cubes using HSV color segmentation
  3. Map pixel coordinates to ARM101 workspace (pixel-to-arm calibration)
  4. Use position-only IK to move the arm above each cube
  5. Capture screenshots at each position

The pixel-to-arm calibration can be tuned via config/cube_calib.yaml.

Usage:
  ./run.sh scripts/visit_cubes.py                    # Detect cubes, move arm
  ./run.sh scripts/visit_cubes.py --detect-only       # Camera detection only
  ./run.sh scripts/visit_cubes.py --manual             # Use manual positions
  ./run.sh scripts/visit_cubes.py --hover-z 120        # Hover height (mm)
  ./run.sh scripts/visit_cubes.py --speed 100          # Servo speed (0-4095)
  ./run.sh scripts/visit_cubes.py --safe               # Safe mode (reduced torque)
"""

import sys
import os
import time
import argparse
import cv2
import numpy as np
import yaml
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision.green_cube_detector import detect_green_cubes, annotate_frame, CubeDetection
from config_loader import load_config

# ---- Configuration ----
OVERVIEW_CAM_INDEX = 4          # Logitech BRIO (overview of workspace)

# Output directory for screenshots
SCREENSHOT_DIR = '/tmp/cube_visit_screenshots'

# Calibration file
CALIB_FILE = os.path.join(os.path.dirname(__file__), '..', 'config', 'cube_calib.yaml')

# Default pixel-to-arm calibration for BRIO → ARM101
# The BRIO is mounted looking down at the table. The ARM101 base is visible.
# Arm base is approximately at pixel (530, 350) in the overview image.
DEFAULT_CALIB = {
    'description': 'Approximate pixel-to-arm mapping for overview camera -> ARM101',
    'arm_base_pixel': [530, 350],   # Where ARM101 base appears in 640x480 BRIO image
    'mm_per_pixel_x': -0.8,        # Negative: camera X→ arm X (left-right mapping)
    'mm_per_pixel_y': -0.6,        # Negative: camera Y (down) → arm Y (forward)
    'table_z_mm': 30.0,            # Height of table surface in arm frame (mm)
    'arm_base_offset_mm': [0.0, 0.0],
}

# Manual cube positions (fallback when camera calibration is poor)
# These are approximate XY positions in ARM101 base frame (mm)
MANUAL_CUBE_POSITIONS = [
    {'name': 'Cube-Front',       'x': 100, 'y':   0},
    {'name': 'Cube-Front-Left',  'x':  80, 'y':  60},
    {'name': 'Cube-Front-Right', 'x':  80, 'y': -60},
    {'name': 'Cube-Left',        'x': 120, 'y':  80},
    {'name': 'Cube-Right',       'x': 120, 'y': -80},
]


def load_calibration() -> dict:
    """Load pixel-to-arm calibration from file, or use defaults."""
    path = os.path.normpath(CALIB_FILE)
    if os.path.exists(path):
        with open(path, 'r') as f:
            calib = yaml.safe_load(f)
        print(f"  Loaded calibration from {path}")
        return calib
    print(f"  Using default calibration (no {os.path.basename(path)} found)")
    return dict(DEFAULT_CALIB)


def pixel_to_arm_xy(px: int, py: int, calib: dict) -> tuple:
    """Convert overview camera pixel to ARM101 arm-frame XY coordinates (mm).

    Uses a simple linear mapping:
      arm_x = (px - base_px) * mm_per_pixel_x + offset_x
      arm_y = (py - base_py) * mm_per_pixel_y + offset_y
    """
    base_px, base_py = calib['arm_base_pixel']
    scale_x = calib['mm_per_pixel_x']
    scale_y = calib['mm_per_pixel_y']
    off_x, off_y = calib.get('arm_base_offset_mm', [0.0, 0.0])
    arm_x = (px - base_px) * scale_x + off_x
    arm_y = (py - base_py) * scale_y + off_y
    return arm_x, arm_y


def open_camera(index: int, width: int = 640, height: int = 480):
    """Open a USB camera with MJPEG codec."""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera /dev/video{index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    for _ in range(10):
        cap.read()
    return cap


def capture_frame(cap) -> np.ndarray:
    """Capture a single frame, retrying a few times."""
    for _ in range(5):
        ret, frame = cap.read()
        if ret and frame is not None:
            return frame
    raise RuntimeError("Failed to capture frame")


def save_screenshot(frame: np.ndarray, name: str, cube_info: str = "") -> str:
    """Save a screenshot with metadata overlay."""
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%H%M%S')
    filename = f"{name}_{timestamp}.jpg"
    filepath = os.path.join(SCREENSHOT_DIR, filename)

    vis = frame.copy()
    cv2.putText(vis, f"{name} | {cube_info}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imwrite(filepath, vis)
    return filepath


def connect_arm(safe_mode: bool = False):
    """Connect to the SO-ARM101 and enable torque.

    Returns:
        LeRobotArm101 instance.
    """
    from robot.lerobot_arm101 import LeRobotArm101

    port = LeRobotArm101.find_port()
    print(f"  Connecting to arm on {port}...")
    arm = LeRobotArm101(port=port, safe_mode=safe_mode)
    arm.connect()
    arm.enable_torque()
    print(f"  Arm enabled (safe_mode={safe_mode})")
    return arm


def point_arm_at(arm, target_mm: np.ndarray, seed_angles: list = None,
                 speed: int = 100):
    """Move the arm to point at a target position using position-only IK.

    Args:
        arm: LeRobotArm101 instance.
        target_mm: [x, y, z] target in arm frame (mm).
        seed_angles: Starting joint angles (5, degrees). None = use current.
        speed: Servo speed.

    Returns:
        Joint angles used (5 floats in degrees), or None if IK failed.
    """
    from kinematics.arm101_ik_solver import Arm101IKSolver

    solver = Arm101IKSolver()

    if seed_angles is None:
        current = arm.read_all_angles()
        seed_angles = np.array(current[:5])

    # Try with provided seed first, then neutral, then various seeds
    seeds_to_try = [
        seed_angles,
        None,  # neutral
        np.array([0.0, 60.0, -30.0, -30.0, 0.0]),
        np.array([0.0, 45.0, -45.0, -30.0, 0.0]),
        np.array([30.0, 60.0, -30.0, -30.0, 0.0]),
        np.array([-30.0, 60.0, -30.0, -30.0, 0.0]),
    ]

    result = None
    for seed in seeds_to_try:
        result = solver.solve_ik_position(target_mm, seed_motor_deg=seed)
        if result is not None:
            break

    if result is None:
        print(f"  IK failed for target {target_mm} (tried {len(seeds_to_try)} seeds)")
        return None

    # Verify FK
    fk_pos, fk_rpy = solver.forward_kin(result)
    pos_err = np.linalg.norm(fk_pos - target_mm)
    print(f"  IK OK: err={pos_err:.1f}mm FK=({fk_pos[0]:.0f},{fk_pos[1]:.0f},{fk_pos[2]:.0f})")

    # Build 6-joint command (5 IK joints + keep gripper where it is)
    current = arm.read_all_angles()
    angles_6 = list(result) + [current[5]]
    print(f"  Moving to angles: [{', '.join(f'{a:.1f}' for a in angles_6)}]")
    arm.write_all_angles(angles_6, speed=speed)
    return result


def wait_for_motion(arm, timeout: float = 5.0, threshold: float = 1.0):
    """Wait for arm to finish moving by polling joint angles."""
    prev = np.array(arm.read_all_angles())
    stable_count = 0
    start = time.time()

    while time.time() - start < timeout:
        time.sleep(0.2)
        current = np.array(arm.read_all_angles())
        if np.max(np.abs(current - prev)) < threshold:
            stable_count += 1
            if stable_count >= 3:
                return True
        else:
            stable_count = 0
        prev = current
    return False


def run_detect_only(args):
    """Detection-only mode: show detections without moving arm."""
    print("=== Green Cube Detection (ARM101 workspace) ===")
    calib = load_calibration()

    cap = open_camera(args.camera)
    try:
        frame = capture_frame(cap)
        detections, _ = detect_green_cubes(frame)

        print(f"\nDetected {len(detections)} green cubes:")
        for i, det in enumerate(detections):
            arm_x, arm_y = pixel_to_arm_xy(det.cx, det.cy, calib)
            print(f"  Cube {i+1}: pixel ({det.cx},{det.cy}) "
                  f"-> arm ({arm_x:.1f}, {arm_y:.1f}) mm, "
                  f"area={det.area:.0f}px^2")

        vis = annotate_frame(frame, detections)
        for i, det in enumerate(detections):
            arm_x, arm_y = pixel_to_arm_xy(det.cx, det.cy, calib)
            txt = f"ARM101: ({arm_x:.0f},{arm_y:.0f})mm"
            cv2.putText(vis, txt, (det.bbox[0], det.bbox[1] + det.bbox[3] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        path = save_screenshot(vis, "detection_overview", f"{len(detections)} cubes")
        print(f"\nSaved detection image: {path}")

    finally:
        cap.release()


def run_manual_mode(args):
    """Move arm to pre-defined cube positions."""
    print("=== Manual Cube Visit Mode (ARM101) ===")
    print(f"  Hover height: {args.hover_z}mm")
    print(f"  Speed: {args.speed}")
    print(f"  Safe mode: {args.safe}")
    print(f"  Positions: {len(MANUAL_CUBE_POSITIONS)}")

    from kinematics.arm101_ik_solver import Arm101IKSolver
    solver = Arm101IKSolver()

    # Pre-check IK for all positions
    print("\nPre-checking IK feasibility...")
    feasible = []
    seed = None
    for pos_info in MANUAL_CUBE_POSITIONS:
        target = np.array([pos_info['x'], pos_info['y'], args.hover_z])
        joints = solver.solve_ik_position(target, seed_motor_deg=seed)
        if joints is not None:
            fk_pos, _ = solver.forward_kin(joints)
            err = np.linalg.norm(fk_pos - target)
            feasible.append((pos_info, joints))
            seed = joints
            print(f"  {pos_info['name']}: OK (err={err:.1f}mm)")
        else:
            print(f"  {pos_info['name']}: SKIP (IK fail)")

    if not feasible:
        print("\nNo feasible positions! Try adjusting --hover-z or manual positions.")
        return

    print(f"\n{len(feasible)}/{len(MANUAL_CUBE_POSITIONS)} positions reachable.")

    # Open camera for screenshots
    cap = None
    try:
        cap = open_camera(args.camera)
        print(f"Camera /dev/video{args.camera} opened for screenshots.")
    except Exception as e:
        print(f"Camera not available ({e}), will skip camera screenshots.")

    # Connect to arm
    print("\nConnecting to ARM101...")
    arm = connect_arm(safe_mode=args.safe)

    screenshots = []

    try:
        initial_angles = arm.read_all_angles()
        print(f"Current angles: [{', '.join(f'{a:.1f}' for a in initial_angles)}]")

        seed = np.array(initial_angles[:5])

        for i, (pos_info, pre_joints) in enumerate(feasible):
            name = pos_info['name']
            target = np.array([pos_info['x'], pos_info['y'], args.hover_z])

            print(f"\n{'='*50}")
            print(f"Cube {i+1}/{len(feasible)}: {name}")
            print(f"Target: ({target[0]:.0f}, {target[1]:.0f}, {target[2]:.0f})mm")

            result = point_arm_at(arm, target, seed_angles=seed, speed=args.speed)

            if result is not None:
                seed = result
                print(f"  Waiting for motion to complete...")
                wait_for_motion(arm)

                # Get actual position via FK
                pose = arm.get_pose()
                if pose:
                    info = f"Pose: ({pose[0]:.0f},{pose[1]:.0f},{pose[2]:.0f})"
                else:
                    info = f"Target: ({target[0]:.0f},{target[1]:.0f},{target[2]:.0f})"
                print(f"  Arrived! {info}")

                # Capture screenshot
                if cap is not None:
                    time.sleep(0.5)
                    frame = capture_frame(cap)
                    vis = frame.copy()
                    cv2.putText(vis, f"Above {name}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(vis, info, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    path = save_screenshot(vis, f"cube_{i+1}_{name}", info)
                    screenshots.append(path)
                    print(f"  Screenshot: {path}")

                time.sleep(args.pause)
            else:
                print(f"  FAILED to reach {name}")

        # Return to home
        print(f"\n{'='*50}")
        print("Returning to home position...")
        arm.write_all_angles(initial_angles, speed=args.speed)
        wait_for_motion(arm)

    except KeyboardInterrupt:
        print("\nInterrupted!")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            arm.disable_torque()
            arm.disconnect()
        except Exception:
            pass
        if cap is not None:
            cap.release()

    print_summary(screenshots)


def run_camera_guided(args):
    """Detect cubes with camera and move arm to each one."""
    print("=== Camera-Guided Cube Visit (ARM101) ===")

    calib = load_calibration()

    from kinematics.arm101_ik_solver import Arm101IKSolver
    solver = Arm101IKSolver()

    # Step 1: Detect cubes
    print("\nStep 1: Detecting green cubes...")
    cap = open_camera(args.camera)
    frame = capture_frame(cap)
    detections, _ = detect_green_cubes(frame)

    if not detections:
        print("No green cubes detected! Try --manual mode or check camera/lighting.")
        cap.release()
        return

    # Step 2: Map to arm coordinates
    print(f"\nStep 2: Mapping {len(detections)} cubes to ARM101 workspace...")
    targets = []
    table_z = calib.get('table_z_mm', 30.0)

    for i, det in enumerate(detections):
        arm_x, arm_y = pixel_to_arm_xy(det.cx, det.cy, calib)
        target = {'name': f'Cube-{i+1}', 'x': arm_x, 'y': arm_y, 'z': table_z,
                  'px': det.cx, 'py': det.cy, 'area': det.area}
        targets.append(target)
        print(f"  Cube {i+1}: pixel ({det.cx},{det.cy}) "
              f"-> arm ({arm_x:.1f}, {arm_y:.1f}, {table_z:.1f}) mm")

    # Save detection overview
    vis = annotate_frame(frame, detections)
    for i, (det, tgt) in enumerate(zip(detections, targets)):
        txt = f"({tgt['x']:.0f},{tgt['y']:.0f})mm"
        cv2.putText(vis, txt, (det.bbox[0], det.bbox[1] + det.bbox[3] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    det_path = save_screenshot(vis, "detection_overview", f"{len(detections)} cubes")
    print(f"  Saved: {det_path}")

    # Step 3: Pre-check IK
    print(f"\nStep 3: Pre-checking IK...")
    feasible = []
    seed = None
    for tgt in targets:
        hover_pos = np.array([tgt['x'], tgt['y'], args.hover_z])
        joints = solver.solve_ik_position(hover_pos, seed_motor_deg=seed)
        if joints is not None:
            fk_pos, _ = solver.forward_kin(joints)
            err = np.linalg.norm(fk_pos - hover_pos)
            feasible.append(tgt)
            seed = joints
            print(f"  {tgt['name']}: OK (err={err:.1f}mm)")
        else:
            print(f"  {tgt['name']}: SKIP (IK fail at "
                  f"{hover_pos[0]:.0f},{hover_pos[1]:.0f},{hover_pos[2]:.0f})")

    if not feasible:
        print("\nNo feasible targets! Calibration may be off. Try --manual mode.")
        cap.release()
        return

    # Step 4: Connect and execute
    print(f"\nStep 4: Connecting to ARM101 and visiting {len(feasible)} cubes...")
    arm = connect_arm(safe_mode=args.safe)

    screenshots = [det_path]

    try:
        initial_angles = arm.read_all_angles()
        print(f"Current angles: [{', '.join(f'{a:.1f}' for a in initial_angles)}]")

        seed = np.array(initial_angles[:5])

        for i, tgt in enumerate(feasible):
            target = np.array([tgt['x'], tgt['y'], args.hover_z])

            print(f"\n{'='*50}")
            print(f"Cube {i+1}/{len(feasible)}: {tgt['name']}")
            print(f"Target: ({tgt['x']:.0f}, {tgt['y']:.0f}) mm, hover_z={args.hover_z}mm")

            result = point_arm_at(arm, target, seed_angles=seed, speed=args.speed)

            if result is not None:
                seed = result
                print(f"  Waiting for motion...")
                wait_for_motion(arm)

                pose = arm.get_pose()
                if pose:
                    info = f"Pose: ({pose[0]:.0f},{pose[1]:.0f},{pose[2]:.0f})"
                else:
                    info = f"Target: ({target[0]:.0f},{target[1]:.0f},{target[2]:.0f})"
                print(f"  Arrived! {info}")

                # Screenshot with detection overlay
                time.sleep(0.5)
                frame = capture_frame(cap)
                vis = annotate_frame(frame, detections)
                # Highlight current target
                cv2.circle(vis, (tgt['px'], tgt['py']), 15, (0, 0, 255), 3)
                cv2.putText(vis, f"VISITING {tgt['name']}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(vis, info, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                path = save_screenshot(vis, f"visit_{tgt['name']}", info)
                screenshots.append(path)
                print(f"  Screenshot: {path}")

                time.sleep(args.pause)
            else:
                print(f"  FAILED to reach {tgt['name']}")

        # Return home
        print(f"\n{'='*50}")
        print("Returning to home position...")
        arm.write_all_angles(initial_angles, speed=args.speed)
        wait_for_motion(arm)

    except KeyboardInterrupt:
        print("\nInterrupted!")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            arm.disable_torque()
            arm.disconnect()
        except Exception:
            pass
        cap.release()

    print_summary(screenshots)


def print_summary(screenshots: list):
    """Print summary and create composite."""
    print(f"\n{'='*50}")
    print(f"Done! {len(screenshots)} screenshots in {SCREENSHOT_DIR}/")
    for p in screenshots:
        print(f"  {p}")

    if len(screenshots) >= 2:
        create_composite(screenshots)


def create_composite(screenshot_paths: list):
    """Create a composite image from all screenshots."""
    images = []
    for path in screenshot_paths:
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (480, 360))
            images.append(img)

    if len(images) < 2:
        return

    cols = min(3, len(images))
    rows = (len(images) + cols - 1) // cols

    blank = np.zeros_like(images[0])
    while len(images) < rows * cols:
        images.append(blank)

    grid_rows = []
    for r in range(rows):
        row_imgs = images[r * cols:(r + 1) * cols]
        grid_rows.append(np.hstack(row_imgs))
    composite = np.vstack(grid_rows)

    path = os.path.join(SCREENSHOT_DIR, 'composite_all_cubes.jpg')
    cv2.imwrite(path, composite)
    print(f"\nComposite image: {path}")


def main():
    parser = argparse.ArgumentParser(description="Visit green cubes with SO-ARM101")
    parser.add_argument('--detect-only', action='store_true',
                        help='Detection only - no arm motion')
    parser.add_argument('--manual', action='store_true',
                        help='Use manual pre-defined positions instead of camera detection')
    parser.add_argument('--hover-z', type=float, default=120.0,
                        help='Hover height above table in mm (default: 120)')
    parser.add_argument('--speed', type=int, default=150,
                        help='Servo speed 0-4095 (default: 150)')
    parser.add_argument('--camera', type=int, default=OVERVIEW_CAM_INDEX,
                        help=f'Camera device index (default: {OVERVIEW_CAM_INDEX})')
    parser.add_argument('--pause', type=float, default=3.0,
                        help='Pause at each cube in seconds (default: 3.0)')
    parser.add_argument('--safe', action='store_true',
                        help='Safe mode (reduced torque/speed)')
    args = parser.parse_args()

    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    if args.detect_only:
        run_detect_only(args)
    elif args.manual:
        run_manual_mode(args)
    else:
        run_camera_guided(args)


if __name__ == '__main__':
    main()
