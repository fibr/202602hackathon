#!/usr/bin/env python3
"""Visit green cubes with calibrated ARM101.

Key improvements over visit_cubes.py:
  - Uses per-motor servo offsets (config/servo_offsets.yaml) via explicit _deg_to_pos_motor()
  - Prints actual vs commanded angles for debugging
  - Constrains targets to reachable workspace
  - Takes screenshots with arm position overlay

Usage:
  ./run.sh scripts/visit_cubes_calibrated.py
  ./run.sh scripts/visit_cubes_calibrated.py --hover-z 150
"""

import sys
import os
import time
import argparse
import cv2
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

SCREENSHOT_DIR = '/tmp/cube_visit_calibrated'
OVERVIEW_CAM_INDEX = 4

# Pixel-to-arm calibration (approximate, needs tuning)
# arm_base_pixel: where the ARM101 base is in the BRIO 640x480 image
# mm_per_pixel: scale factors (camera pixel to arm-frame mm)
CALIB = {
    'arm_base_pixel': [530, 350],
    'mm_per_pixel_x': -0.8,    # Camera X → arm X (negative: cam-right = arm-left)
    'mm_per_pixel_y': -0.6,    # Camera Y → arm Y
}

# Workspace limits (arm frame mm) — positions outside these are unreachable
WORKSPACE = {
    'x_min': 150,    # Minimum forward reach
    'x_max': 350,    # Maximum forward reach
    'y_min': -150,   # Right limit
    'y_max': 150,    # Left limit
    'z_min': 80,     # Minimum safe height (above table)
    'z_max': 300,    # Maximum comfortable height
}


def open_camera(index, width=640, height=480):
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


def capture_frame(cap):
    for _ in range(5):
        ret, frame = cap.read()
        if ret and frame is not None:
            return frame
    raise RuntimeError("Failed to capture frame")


def wait_for_motion(arm, timeout=5.0, threshold=1.0):
    prev = np.array(arm.read_all_angles())
    stable = 0
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(0.2)
        cur = np.array(arm.read_all_angles())
        if np.max(np.abs(cur - prev)) < threshold:
            stable += 1
            if stable >= 3:
                return True
        else:
            stable = 0
        prev = cur
    return False


def pixel_to_arm(px, py):
    """Convert pixel coordinates to arm-frame XY (mm)."""
    base_px, base_py = CALIB['arm_base_pixel']
    arm_x = (px - base_px) * CALIB['mm_per_pixel_x']
    arm_y = (py - base_py) * CALIB['mm_per_pixel_y']
    return arm_x, arm_y


def clamp_to_workspace(x, y, z):
    """Clamp target to reachable workspace."""
    x = np.clip(x, WORKSPACE['x_min'], WORKSPACE['x_max'])
    y = np.clip(y, WORKSPACE['y_min'], WORKSPACE['y_max'])
    z = np.clip(z, WORKSPACE['z_min'], WORKSPACE['z_max'])
    return x, y, z


# Safe intermediate position — arm up and centered, guarantees high Z
SAFE_UP_ANGLES = [0, 10, -10, -10, 0]  # FK ≈ (370, 0, 230)mm — well above table


def angles_to_calibrated_positions(arm, angles):
    """Convert joint angles in degrees to raw positions using per-motor calibration.

    Uses arm._deg_to_pos_motor() for motors 1-5 (IK joints) to apply per-motor
    zero offsets from servo_offsets.yaml. Motor 6 (gripper) uses standard conversion.

    Args:
        arm: LeRobotArm101 instance
        angles: List of 6 target angles in degrees

    Returns:
        List of 6 raw positions (0-4095)
    """
    positions = []
    for i, angle in enumerate(angles):
        motor_id = arm.motor_ids[i]
        if motor_id <= 5:  # IK joints use per-motor calibration
            pos = arm._deg_to_pos_motor(angle, motor_id)
        else:  # Gripper (motor 6) uses standard conversion
            pos = arm._deg_to_pos(angle)
        positions.append(pos)
    return positions


def move_to_target(arm, solver, target_mm, initial_angles, speed=150,
                   use_waypoint=True):
    """Move arm to target using IK with waypoint safety.

    To prevent table collisions during motion, the arm first moves to a
    safe "up" position before moving to the target. This avoids the arm
    swinging through low-Z configurations when servos move independently.

    Returns: (success, actual_angles, actual_fk_pos)
    """
    seeds = [
        np.array(initial_angles[:5]),
        None,  # neutral
        np.array([0, 15, -15, -30, 0]),
        np.array([0, 20, -20, -20, 0]),
        np.array([0, 10, -30, -30, 0]),
        np.array([20, 15, -15, -15, 0]),
        np.array([-20, 15, -15, -15, 0]),
    ]

    result = None
    for seed in seeds:
        result = solver.solve_ik_position(target_mm, seed_motor_deg=seed)
        if result is not None:
            # Sanity check: reject solutions with extreme angles
            if np.any(np.abs(result) > 80):
                print(f"    Rejecting extreme solution: [{','.join(f'{j:.0f}' for j in result)}]")
                result = None
                continue
            # FK safety check: reject if Z too low
            fk_check, _ = solver.forward_kin(result)
            if fk_check[2] < 50:
                print(f"    Rejecting low-Z solution: Z={fk_check[2]:.0f}mm")
                result = None
                continue
            break

    if result is None:
        return False, None, None

    fk_pos, _ = solver.forward_kin(result)
    pos_err = np.linalg.norm(fk_pos - target_mm)
    print(f"    IK: joints=[{','.join(f'{j:.0f}' for j in result)}] "
          f"FK=({fk_pos[0]:.0f},{fk_pos[1]:.0f},{fk_pos[2]:.0f}) err={pos_err:.1f}mm")

    grip = initial_angles[5]

    # Step 1: Move to safe "up" position first (prevents table collision during transit)
    if use_waypoint:
        safe_cmd = list(SAFE_UP_ANGLES) + [grip]
        print(f"    -> Safe waypoint first...")
        # Convert angles to calibrated positions using per-motor offsets
        safe_positions = angles_to_calibrated_positions(arm, safe_cmd)
        arm.write_all_positions(safe_positions, speed=speed)
        time.sleep(1.5)
        wait_for_motion(arm, timeout=4)

    # Step 2: Move to target
    cmd = list(result) + [grip]
    # Convert angles to calibrated positions using per-motor offsets
    cmd_positions = angles_to_calibrated_positions(arm, cmd)
    arm.write_all_positions(cmd_positions, speed=speed)
    time.sleep(2.0)
    wait_for_motion(arm, timeout=5)
    time.sleep(0.5)

    # Read actual
    actual = arm.read_all_angles()
    actual_pos, _ = solver.forward_kin(np.array(actual[:5], dtype=float))

    angle_errs = [actual[i] - result[i] for i in range(5)]
    print(f"    Actual: joints=[{','.join(f'{a:.0f}' for a in actual[:5])}] "
          f"FK=({actual_pos[0]:.0f},{actual_pos[1]:.0f},{actual_pos[2]:.0f})")
    print(f"    Angle tracking error: [{','.join(f'{e:+.1f}' for e in angle_errs)}]")

    return True, actual[:5], actual_pos


def main():
    parser = argparse.ArgumentParser(description="Visit cubes with calibrated ARM101")
    parser.add_argument('--camera', type=int, default=OVERVIEW_CAM_INDEX)
    parser.add_argument('--speed', type=int, default=150)
    parser.add_argument('--hover-z', type=float, default=150.0,
                        help='Hover height in mm (default: 150)')
    args = parser.parse_args()

    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    from robot.lerobot_arm101 import LeRobotArm101
    from kinematics.arm101_ik_solver import Arm101IKSolver
    from vision.green_cube_detector import detect_green_cubes, annotate_frame

    solver = Arm101IKSolver()

    # Open camera
    cap = open_camera(args.camera)
    print(f"Camera /dev/video{args.camera} opened.")

    # Detect cubes before connecting arm (so arm doesn't occlude)
    print("\n=== Step 1: Detecting Green Cubes ===")
    frame = capture_frame(cap)
    detections, _ = detect_green_cubes(frame)
    print(f"  Found {len(detections)} green cubes")

    # Map to arm coordinates and filter reachable
    targets = []
    for i, det in enumerate(detections):
        arm_x, arm_y = pixel_to_arm(det.cx, det.cy)
        arm_x, arm_y, arm_z = clamp_to_workspace(arm_x, arm_y, args.hover_z)

        # Check if original (unclamped) position was in workspace
        orig_x, orig_y = pixel_to_arm(det.cx, det.cy)
        in_workspace = (WORKSPACE['x_min'] <= orig_x <= WORKSPACE['x_max'] and
                        WORKSPACE['y_min'] <= orig_y <= WORKSPACE['y_max'])

        targets.append({
            'name': f'Cube-{i+1}',
            'px': det.cx, 'py': det.cy,
            'arm_x': arm_x, 'arm_y': arm_y, 'arm_z': arm_z,
            'in_workspace': in_workspace,
            'area': det.area,
        })
        ws_str = "IN WORKSPACE" if in_workspace else "CLAMPED"
        print(f"  Cube {i+1}: pixel ({det.cx},{det.cy}) -> "
              f"arm ({arm_x:.0f},{arm_y:.0f},{arm_z:.0f}) mm [{ws_str}]")

    # Save detection overview
    vis = annotate_frame(frame, detections)
    for i, (det, tgt) in enumerate(zip(detections, targets)):
        color = (0, 255, 0) if tgt['in_workspace'] else (0, 0, 255)
        txt = f"({tgt['arm_x']:.0f},{tgt['arm_y']:.0f})mm"
        cv2.putText(vis, txt, (det.cx - 30, det.cy + det.bbox[3]//2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    det_path = os.path.join(SCREENSHOT_DIR, 'detection.jpg')
    cv2.imwrite(det_path, vis)
    screenshots = [det_path]

    reachable = [t for t in targets if t['in_workspace']]
    if not reachable:
        print("\nNo cubes in reachable workspace! Calibration may need adjustment.")
        print("Trying clamped positions anyway...")
        reachable = targets

    # Connect arm
    print(f"\n=== Step 2: Connecting ARM101 ===")
    port = LeRobotArm101.find_port()
    arm = LeRobotArm101(port=port, safe_mode=False)
    arm.connect()
    arm.enable_torque()

    initial = arm.read_all_angles()
    init_pos, _ = solver.forward_kin(np.array(initial[:5], dtype=float))
    print(f"  Initial: angles=[{','.join(f'{a:.0f}' for a in initial[:5])}] "
          f"FK=({init_pos[0]:.0f},{init_pos[1]:.0f},{init_pos[2]:.0f})")

    # Visit each cube
    print(f"\n=== Step 3: Visiting {len(reachable)} cubes ===")
    try:
        for i, tgt in enumerate(reachable):
            print(f"\n{'='*50}")
            print(f"  {tgt['name']}: target ({tgt['arm_x']:.0f}, {tgt['arm_y']:.0f}, "
                  f"{tgt['arm_z']:.0f}) mm")

            target_mm = np.array([tgt['arm_x'], tgt['arm_y'], tgt['arm_z']])
            success, actual_angles, actual_pos = move_to_target(
                arm, solver, target_mm, initial, speed=args.speed)

            if success:
                time.sleep(0.5)
                frame = capture_frame(cap)
                vis = frame.copy()

                h, w = vis.shape[:2]
                overlay = vis.copy()
                cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)

                # Highlight target cube
                cv2.circle(vis, (tgt['px'], tgt['py']), 20, (0, 0, 255), 3)
                cv2.putText(vis, f"Target: {tgt['name']}", (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                if actual_pos is not None:
                    cv2.putText(vis, f"Arm FK: ({actual_pos[0]:.0f},{actual_pos[1]:.0f},{actual_pos[2]:.0f})mm",
                                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
                cv2.putText(vis, f"Target: ({tgt['arm_x']:.0f},{tgt['arm_y']:.0f},{tgt['arm_z']:.0f})mm",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

                path = os.path.join(SCREENSHOT_DIR, f'{tgt["name"]}.jpg')
                cv2.imwrite(path, vis)
                screenshots.append(path)
                print(f"    Screenshot: {path}")
            else:
                print(f"    SKIP: IK failed")

            time.sleep(1.0)

        # Return home
        print(f"\n{'='*50}")
        print("  Returning to initial (via safe waypoint)...")
        safe_cmd = list(SAFE_UP_ANGLES) + [initial[5]]
        # Convert angles to calibrated positions using per-motor offsets
        safe_positions = angles_to_calibrated_positions(arm, safe_cmd)
        arm.write_all_positions(safe_positions, speed=args.speed)
        time.sleep(1.5)
        wait_for_motion(arm, timeout=4)
        # Convert initial angles to calibrated positions using per-motor offsets
        initial_positions = angles_to_calibrated_positions(arm, initial)
        arm.write_all_positions(initial_positions, speed=args.speed)
        wait_for_motion(arm)

    except KeyboardInterrupt:
        print("\nInterrupted!")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        arm.disable_torque()
        arm.disconnect()
        cap.release()

    # Create composite
    if len(screenshots) >= 2:
        images = []
        for p in screenshots:
            img = cv2.imread(p)
            if img is not None:
                img = cv2.resize(img, (480, 360))
                images.append(img)
        cols = min(3, len(images))
        rows = (len(images) + cols - 1) // cols
        blank = np.zeros_like(images[0])
        while len(images) < rows * cols:
            images.append(blank)
        grid_rows = []
        for r in range(rows):
            grid_rows.append(np.hstack(images[r*cols:(r+1)*cols]))
        composite = np.vstack(grid_rows)
        comp_path = os.path.join(SCREENSHOT_DIR, 'composite.jpg')
        cv2.imwrite(comp_path, composite)
        print(f"\nComposite: {comp_path}")
        screenshots.append(comp_path)

    print(f"\nDone! {len(screenshots)} images in {SCREENSHOT_DIR}/")


if __name__ == '__main__':
    main()
