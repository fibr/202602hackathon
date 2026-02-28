#!/usr/bin/env python3
"""Collect rod images and optionally command the robot to hover over detections.

Shows a live camera feed. Press SPACE to capture a frame pair (color + depth).
Press 'g' to move the robot 200mm above the detected rod centroid, with the
gripper perpendicular to the rod axis.

Robot connection is optional — if the robot is unreachable, all camera/detection
features still work.

Each capture saves to data/rod_dataset/:
  {NNN}_color.png      -- BGR color image
  {NNN}_depth.png      -- raw uint16 depth in mm (lossless)
  {NNN}_depth_vis.png  -- colorized depth for quick browsing

Usage:
    ./run.sh scripts/collect_dataset.py [--sd] [--no-robot]

Controls:
    SPACE       Capture current frame
    d           Toggle detection overlay
    g           Move robot to hover 200mm above detected rod
    q / Esc     Quit
"""

import sys
import os
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from vision import RealSenseCamera
from vision.rod_detector import RodDetector
from calibration import CoordinateTransform
from config_loader import load_config

DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'rod_dataset')
HOVER_HEIGHT_MM = 200.0   # Height above rod centroid
GRIPPER_DOWN_RX = 180.0   # rx when gripper points straight down


def find_next_index(dataset_dir):
    """Find the next available index by scanning existing files."""
    if not os.path.exists(dataset_dir):
        return 0
    existing = [f for f in os.listdir(dataset_dir) if f.endswith('_color.png')]
    if not existing:
        return 0
    indices = []
    for f in existing:
        try:
            indices.append(int(f.split('_')[0]))
        except ValueError:
            pass
    return max(indices) + 1 if indices else 0


def try_connect_robot(config):
    """Try to connect to the robot. Returns (robot, gripper) or (None, None)."""
    try:
        from robot import DobotNova5
        robot_cfg = config.get('robot', {})
        robot = DobotNova5(
            ip=robot_cfg.get('ip', '192.168.5.1'),
            dashboard_port=robot_cfg.get('dashboard_port', 29999),
        )
        robot.connect()
        robot.clear_error()
        robot.enable()
        robot.set_speed(robot_cfg.get('speed_percent', 30))
        mode = robot.get_mode()
        print(f"  Robot connected (mode={mode})")
        return robot
    except Exception as e:
        print(f"  Robot not available: {e}")
        return None


def compute_hover_pose(detection, transform):
    """Compute robot pose to hover above the rod centroid.

    Returns:
        (x, y, z, rx, ry, rz) in mm/degrees for the robot, or None on failure.
        Gripper points down (rx=180), rz perpendicular to rod axis.
    """
    # Transform camera-frame 3D point to robot base frame (meters)
    center_base_m = transform.camera_to_base(detection.center_3d)
    axis_base = transform.camera_axis_to_base(detection.axis_3d)

    # Convert to mm
    rod_x = center_base_m[0] * 1000.0
    rod_y = center_base_m[1] * 1000.0
    rod_z = center_base_m[2] * 1000.0

    # Hover above the rod
    hover_z = rod_z + HOVER_HEIGHT_MM

    # Gripper rotation: rz perpendicular to rod axis (projected onto XY plane)
    grasp_rz = np.degrees(np.arctan2(axis_base[1], axis_base[0]))
    # Add 90 degrees to be perpendicular to the rod, not parallel
    grasp_rz += 90.0

    return (rod_x, rod_y, hover_z, GRIPPER_DOWN_RX, 0.0, grasp_rz)


def main():
    sd = '--sd' in sys.argv
    no_robot = '--no-robot' in sys.argv
    width, height = (640, 480) if sd else (1280, 720)

    os.makedirs(DATASET_DIR, exist_ok=True)
    next_idx = find_next_index(DATASET_DIR)

    config = load_config()
    cam_cfg = config.get('camera', {})

    detector = RodDetector(
        min_aspect_ratio=cam_cfg.get('min_aspect_ratio', 2.2),
        min_area=cam_cfg.get('min_area', 500),
        depth_min_mm=cam_cfg.get('depth_min_mm', 6000),
        depth_max_mm=cam_cfg.get('depth_max_mm', 19000),
        max_brightness=cam_cfg.get('max_brightness', 120),
        max_brightness_std=cam_cfg.get('max_brightness_std', 40.0),
        min_convexity=cam_cfg.get('min_convexity', 0.85),
        workspace_roi=cam_cfg.get('workspace_roi'),
    )

    # Load calibration
    transform = CoordinateTransform()
    calibration_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'calibration.yaml')
    if os.path.exists(calibration_path):
        transform.load(calibration_path)
        print(f"Loaded calibration from {calibration_path}")
    else:
        print("WARNING: No calibration file — robot goto will use identity transform")

    # Try robot connection (optional)
    robot = None
    if not no_robot:
        print("Connecting to robot...")
        robot = try_connect_robot(config)

    print()
    print("=== Rod Dataset Collection ===")
    print(f"Resolution: {width}x{height}")
    print(f"Save directory: {os.path.abspath(DATASET_DIR)}")
    print(f"Next index: {next_idx}")
    print(f"Robot: {'connected' if robot else 'not connected'}")
    print()
    print("SPACE=capture  d=detect  g=goto rod  q=quit")
    print()

    camera = RealSenseCamera(width=width, height=height, fps=15)
    camera.start()

    cv2.namedWindow('Collect Dataset')

    # Let auto-exposure settle
    for _ in range(30):
        camera.get_frames()

    show_detection = False
    count = 0
    last_detection = None

    try:
        while True:
            color, depth_img, depth_frame = camera.get_frames()
            if color is None:
                continue

            vis = color.copy()

            # Run detection overlay if toggled
            if show_detection:
                detection = detector.detect(color, depth_img, depth_frame, camera)
                if detection:
                    last_detection = detection
                    vis = detector.draw_detection(vis, detection)

                    # Show hover target info
                    pose = compute_hover_pose(detection, transform)
                    if pose:
                        x, y, z, rx, ry, rz = pose
                        cv2.putText(vis,
                                    f"Hover: ({x:.0f},{y:.0f},{z:.0f}) rz={rz:.0f}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 255), 2)
                else:
                    last_detection = None

                # Draw ROI
                roi_mask = getattr(detector, '_last_roi_mask', None)
                if roi_mask is not None:
                    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis, contours, -1, (0, 255, 255), 1)

            # Status bar
            h_img = vis.shape[0]
            robot_str = "robot:ON" if robot else "robot:OFF"
            det_str = "det:ON" if show_detection else "det:OFF"
            status = f"#{next_idx}  |  {count} captured  |  {det_str}  |  {robot_str}  |  SPACE=capture  d=detect  g=goto  q=quit"
            cv2.rectangle(vis, (0, h_img - 30), (vis.shape[1], h_img), (0, 0, 0), -1)
            cv2.putText(vis, status, (10, h_img - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            cv2.imshow('Collect Dataset', vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break
            try:
                if cv2.getWindowProperty('Collect Dataset', cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            if key == ord('d'):
                show_detection = not show_detection
                print(f"  Detection overlay: {'ON' if show_detection else 'OFF'}")

            if key == ord('g'):
                if not show_detection:
                    print("  Turn on detection first (press 'd')")
                elif last_detection is None:
                    print("  No rod detected — nothing to go to")
                elif robot is None:
                    print("  Robot not connected")
                else:
                    pose = compute_hover_pose(last_detection, transform)
                    if pose:
                        x, y, z, rx, ry, rz = pose
                        print(f"  Moving to hover: ({x:.1f}, {y:.1f}, {z:.1f}) rx={rx:.1f} ry={ry:.1f} rz={rz:.1f}")
                        ok = robot.movj(x, y, z, rx, ry, rz)
                        if ok:
                            print(f"  Arrived at hover position")
                        else:
                            print(f"  Move failed or timed out")

            if key == ord(' '):
                idx_str = f"{next_idx:03d}"
                color_path = os.path.join(DATASET_DIR, f"{idx_str}_color.png")
                depth_path = os.path.join(DATASET_DIR, f"{idx_str}_depth.png")
                depth_vis_path = os.path.join(DATASET_DIR, f"{idx_str}_depth_vis.png")

                cv2.imwrite(color_path, color)
                cv2.imwrite(depth_path, depth_img)
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imwrite(depth_vis_path, depth_colormap)

                print(f"  [{idx_str}] Saved to {DATASET_DIR}/")
                next_idx += 1
                count += 1

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        if robot:
            try:
                robot.disconnect()
            except Exception:
                pass
        print(f"\nDone. {count} frames captured (indices up to {next_idx - 1}).")
        print(f"Dataset: {os.path.abspath(DATASET_DIR)}")


if __name__ == "__main__":
    main()
