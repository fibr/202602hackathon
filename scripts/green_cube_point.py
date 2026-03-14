#!/usr/bin/env python3
"""Detect green cubes via USB cameras and point SO101 arm at each one.

Pipeline:
  1. Open overview camera (Logitech BRIO, /dev/video4) and gripper camera (/dev/video8)
  2. Detect green cubes using HSV color segmentation
  3. Map pixel coordinates to arm workspace via a simple calibration transform
  4. Use position-only IK to move the arm to point at each cube

Calibration approach:
  The overview camera is fixed. We define a pixel-to-workspace homography by
  specifying known correspondences between pixel coordinates and arm-frame XY
  coordinates.  Without depth, we assume all cubes are on the table plane (Z=0
  in arm frame, or a fixed table_z).

  Initial calibration is approximate; run with --calibrate to interactively
  pick reference points.

Usage:
  ./run.sh scripts/green_cube_point.py                    # Full pipeline
  ./run.sh scripts/green_cube_point.py --detect-only      # Detection only (no arm)
  ./run.sh scripts/green_cube_point.py --calibrate        # Interactive calibration
  ./run.sh scripts/green_cube_point.py --safe              # Safe mode (reduced torque)
"""

import sys
import os
import time
import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision.green_cube_detector import detect_green_cubes, annotate_frame, CubeDetection
from config_loader import load_config


# ---- Camera configuration ----
OVERVIEW_CAM_INDEX = 4    # Logitech BRIO (overview of workspace)
GRIPPER_CAM_INDEX = 8     # USB Camera (mounted on gripper)

# ---- Calibration file ----
CALIB_FILE = os.path.join(os.path.dirname(__file__), '..', 'config', 'cube_calib.yaml')

# ---- Default approximate pixel-to-arm calibration ----
# These are rough defaults for the BRIO at 640x480 viewing the SO101 workspace.
# The arm base is approximately at pixel (530, 350) in the overview image.
# The workspace is roughly 300mm wide in the camera's field of view.
# This gives a rough mm/pixel scale.
DEFAULT_CALIB = {
    'description': 'Approximate pixel-to-arm mapping for overview camera',
    'arm_base_pixel': [530, 350],   # Where arm base appears in overview image
    'mm_per_pixel_x': -0.8,         # Negative: camera X increases left, arm X increases right
    'mm_per_pixel_y': -0.6,         # Negative: camera Y increases down, arm Y increases forward
    'table_z_mm': 30.0,             # Height of table surface in arm frame (mm)
    'arm_base_offset_mm': [0.0, 0.0],  # XY offset from arm base to workspace center
}


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


def save_calibration(calib: dict):
    """Save calibration to YAML file."""
    path = os.path.normpath(CALIB_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(calib, f, default_flow_style=False)
    print(f"  Saved calibration to {path}")


def pixel_to_arm_xy(px: int, py: int, calib: dict) -> tuple[float, float]:
    """Convert overview camera pixel to arm-frame XY coordinates (mm).

    Uses a simple linear mapping:
      arm_x = (px - base_px) * mm_per_pixel_x + offset_x
      arm_y = (py - base_py) * mm_per_pixel_y + offset_y

    Args:
        px, py: Pixel coordinates in overview camera image.
        calib: Calibration dict.

    Returns:
        (arm_x_mm, arm_y_mm): Position in arm base frame.
    """
    base_px, base_py = calib['arm_base_pixel']
    scale_x = calib['mm_per_pixel_x']
    scale_y = calib['mm_per_pixel_y']
    off_x, off_y = calib.get('arm_base_offset_mm', [0.0, 0.0])

    arm_x = (px - base_px) * scale_x + off_x
    arm_y = (py - base_py) * scale_y + off_y
    return arm_x, arm_y


def open_camera(index: int, width: int = 640, height: int = 480) -> cv2.VideoCapture:
    """Open a USB camera with MJPEG codec."""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera /dev/video{index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Let auto-exposure settle
    for _ in range(5):
        cap.read()
    return cap


def capture_frame(cap: cv2.VideoCapture) -> np.ndarray:
    """Capture a single frame, retrying a few times."""
    for _ in range(3):
        ret, frame = cap.read()
        if ret and frame is not None:
            return frame
    raise RuntimeError("Failed to capture frame")


def connect_arm(safe_mode: bool = False):
    """Connect to the SO101 arm and enable torque.

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


def point_arm_at(arm, target_mm: np.ndarray, seed_angles: list = None):
    """Move the arm to point at a target position using position-only IK.

    Args:
        arm: LeRobotArm101 instance.
        target_mm: [x, y, z] target in arm frame (mm).
        seed_angles: Starting joint angles (5, degrees). None = use current.

    Returns:
        Joint angles used (5 floats in degrees), or None if IK failed.
    """
    from kinematics.arm101_ik_solver import Arm101IKSolver

    solver = Arm101IKSolver()

    if seed_angles is None:
        current = arm.read_all_angles()
        seed_angles = np.array(current[:5])

    result = solver.solve_ik_position(target_mm, seed_motor_deg=seed_angles)
    if result is None:
        print(f"  IK failed for target {target_mm}")
        return None

    # Build 6-joint command (5 IK joints + keep gripper where it is)
    current = arm.read_all_angles()
    angles_6 = list(result) + [current[5]]
    print(f"  Moving to angles: [{', '.join(f'{a:.1f}' for a in angles_6)}]")
    arm.write_all_angles(angles_6, speed=100)
    return result


def run_detect_only():
    """Detection-only mode: show camera feeds with green cube detections."""
    print("=== Green Cube Detection (no arm) ===")

    calib = load_calibration()

    # Open cameras
    print("Opening cameras...")
    overview = open_camera(OVERVIEW_CAM_INDEX)
    gripper = open_camera(GRIPPER_CAM_INDEX)
    print(f"  Overview: /dev/video{OVERVIEW_CAM_INDEX}")
    print(f"  Gripper:  /dev/video{GRIPPER_CAM_INDEX}")

    print("\nPress 'q' or Esc to quit, 's' to save snapshot")
    snapshot_count = 0

    try:
        while True:
            # Capture frames
            frame_ov = capture_frame(overview)
            frame_gr = capture_frame(gripper)

            # Detect cubes
            dets_ov, _ = detect_green_cubes(frame_ov)
            dets_gr, _ = detect_green_cubes(frame_gr)

            # Annotate
            vis_ov = annotate_frame(frame_ov, dets_ov)
            vis_gr = annotate_frame(frame_gr, dets_gr, label_prefix="GCube")

            # Add arm coordinate info for overview detections
            for i, det in enumerate(dets_ov):
                ax, ay = pixel_to_arm_xy(det.cx, det.cy, calib)
                txt = f"Arm: ({ax:.0f}, {ay:.0f})mm"
                cv2.putText(vis_ov, txt, (det.bbox[0], det.bbox[1] + det.bbox[3] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # Status bar
            cv2.putText(vis_ov, f"Overview: {len(dets_ov)} cubes | q=quit s=save",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(vis_gr, f"Gripper: {len(dets_gr)} cubes",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("Overview Camera", vis_ov)
            cv2.imshow("Gripper Camera", vis_gr)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                snapshot_count += 1
                os.makedirs('/tmp/cube_snapshots', exist_ok=True)
                cv2.imwrite(f'/tmp/cube_snapshots/overview_{snapshot_count}.jpg', vis_ov)
                cv2.imwrite(f'/tmp/cube_snapshots/gripper_{snapshot_count}.jpg', vis_gr)
                print(f"  Snapshot #{snapshot_count} saved")

    finally:
        overview.release()
        gripper.release()
        cv2.destroyAllWindows()
        print(f"\nDone. {snapshot_count} snapshots saved.")


def run_calibrate():
    """Interactive calibration mode.

    Click on the arm base and known positions in the overview camera to
    establish pixel-to-arm coordinate mapping.
    """
    print("=== Interactive Calibration ===")
    print("Step 1: Click on the arm base pivot in the overview camera.")
    print("Step 2: Click on known positions and enter their arm-frame coordinates.")
    print("Press 'q' when done, 'r' to reset.")

    overview = open_camera(OVERVIEW_CAM_INDEX)
    calib = load_calibration()
    clicks = []
    arm_base_set = False

    def on_mouse(event, x, y, flags, param):
        nonlocal arm_base_set
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
            if not arm_base_set:
                calib['arm_base_pixel'] = [x, y]
                arm_base_set = True
                print(f"  Arm base set to pixel ({x}, {y})")
                print("  Now click on cubes/known points. Enter arm coords in terminal.")
            else:
                print(f"  Clicked pixel ({x}, {y})")
                print("  Enter arm X,Y in mm (e.g., '100,50') or 'skip': ", end='', flush=True)

    cv2.namedWindow("Calibrate")
    cv2.setMouseCallback("Calibrate", on_mouse)

    calib_points = []  # list of (px, py, arm_x, arm_y)

    try:
        while True:
            frame = capture_frame(overview)
            vis = frame.copy()

            # Draw arm base
            if arm_base_set:
                bx, by = calib['arm_base_pixel']
                cv2.circle(vis, (bx, by), 8, (0, 0, 255), 2)
                cv2.putText(vis, "ARM BASE", (bx + 10, by - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Draw calibration points
            for px, py, ax, ay in calib_points:
                cv2.circle(vis, (px, py), 5, (255, 0, 0), -1)
                cv2.putText(vis, f"({ax:.0f},{ay:.0f})", (px + 5, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # Detect cubes for reference
            dets, _ = detect_green_cubes(frame)
            for det in dets:
                cv2.circle(vis, (det.cx, det.cy), 3, (0, 255, 0), -1)

            status = "Click ARM BASE first" if not arm_base_set else f"Calib points: {len(calib_points)} | q=save&quit r=reset"
            cv2.putText(vis, status, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow("Calibrate", vis)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                calib_points.clear()
                clicks.clear()
                arm_base_set = False
                print("  Reset calibration.")

    finally:
        overview.release()
        cv2.destroyAllWindows()

    # If we have calibration points, compute best-fit linear mapping
    if len(calib_points) >= 2:
        print(f"\nComputing calibration from {len(calib_points)} points...")
        bx, by = calib['arm_base_pixel']
        dpx = np.array([(p[0] - bx) for p in calib_points])
        dpy = np.array([(p[1] - by) for p in calib_points])
        ax = np.array([p[2] for p in calib_points])
        ay = np.array([p[3] for p in calib_points])

        # Fit: arm_x = dpx * scale_x, arm_y = dpy * scale_y
        if np.sum(dpx**2) > 0:
            calib['mm_per_pixel_x'] = float(np.sum(ax * dpx) / np.sum(dpx**2))
        if np.sum(dpy**2) > 0:
            calib['mm_per_pixel_y'] = float(np.sum(ay * dpy) / np.sum(dpy**2))

        print(f"  mm_per_pixel_x = {calib['mm_per_pixel_x']:.4f}")
        print(f"  mm_per_pixel_y = {calib['mm_per_pixel_y']:.4f}")

    save_calibration(calib)
    print("Calibration complete.")


def run_full_pipeline(safe_mode: bool = False):
    """Full pipeline: detect cubes and point arm at each one."""
    print("=== Green Cube Point Pipeline ===")
    print(f"  Safe mode: {safe_mode}")

    calib = load_calibration()

    # Open overview camera
    print("Opening overview camera...")
    overview = open_camera(OVERVIEW_CAM_INDEX)

    # Connect arm
    print("Connecting to arm...")
    arm = connect_arm(safe_mode=safe_mode)

    # Read initial arm position
    initial_angles = arm.read_all_angles()
    print(f"  Initial angles: [{', '.join(f'{a:.1f}' for a in initial_angles)}]")

    try:
        # Step 1: Detect cubes
        print("\nStep 1: Detecting green cubes...")
        frame = capture_frame(overview)
        detections, _ = detect_green_cubes(frame)
        print(f"  Found {len(detections)} green cubes")

        if not detections:
            print("  No cubes detected! Check lighting and camera position.")
            return

        # Step 2: Map to arm coordinates
        print("\nStep 2: Computing arm-frame coordinates...")
        targets = []
        table_z = calib.get('table_z_mm', 30.0)

        for i, det in enumerate(detections):
            ax, ay = pixel_to_arm_xy(det.cx, det.cy, calib)
            target = np.array([ax, ay, table_z])
            targets.append(target)
            print(f"  Cube {i+1}: pixel ({det.cx},{det.cy}) -> arm ({ax:.1f}, {ay:.1f}, {table_z:.1f}) mm")

        # Save annotated detection image
        vis = annotate_frame(frame, detections)
        for i, (det, target) in enumerate(zip(detections, targets)):
            txt = f"({target[0]:.0f},{target[1]:.0f})mm"
            cv2.putText(vis, txt, (det.bbox[0], det.bbox[1] + det.bbox[3] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        os.makedirs('/tmp/cube_snapshots', exist_ok=True)
        cv2.imwrite('/tmp/cube_snapshots/pipeline_detections.jpg', vis)
        print("  Saved detection image to /tmp/cube_snapshots/pipeline_detections.jpg")

        # Step 3: Point at each cube
        print(f"\nStep 3: Pointing at {len(targets)} cubes...")
        seed = np.array(initial_angles[:5])

        for i, target in enumerate(targets):
            print(f"\n--- Cube {i+1}/{len(targets)} ---")
            print(f"  Target: [{target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}] mm")

            # Move arm above the cube (elevated approach)
            approach_target = target.copy()
            approach_target[2] += 80  # 80mm above table
            print(f"  Approach: [{approach_target[0]:.1f}, {approach_target[1]:.1f}, {approach_target[2]:.1f}] mm")

            result = point_arm_at(arm, approach_target, seed_angles=seed)
            if result is not None:
                seed = result
                print(f"  Pointing at cube {i+1}. Waiting 3s...")
                time.sleep(3)
            else:
                print(f"  SKIP cube {i+1}: IK failed")

        # Return to initial position
        print(f"\nReturning to initial position...")
        arm.write_all_angles(initial_angles, speed=100)
        time.sleep(2)

        print("\nDone! Pointed at all reachable cubes.")

    finally:
        try:
            arm.disable_torque()
            arm.disconnect()
        except Exception:
            pass
        overview.release()
        cv2.destroyAllWindows()


def main():
    detect_only = '--detect-only' in sys.argv
    calibrate = '--calibrate' in sys.argv
    safe_mode = '--safe' in sys.argv

    if calibrate:
        run_calibrate()
    elif detect_only:
        run_detect_only()
    else:
        run_full_pipeline(safe_mode=safe_mode)


if __name__ == '__main__':
    main()
