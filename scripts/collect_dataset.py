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
    r           Toggle robot skeleton overlay (base always shown)
    g           Move robot to hover 200mm above detected rod
    Arrows      Nudge base overlay position XY (10mm steps)
    +/-         Nudge base overlay position Z (10mm steps)
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
from visualization import RobotOverlay
from gui.robot_controls import RobotControlPanel, PANEL_WIDTH

DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'rod_dataset')


class DobotPanelAdapter:
    """Adapter to make DobotNova5 compatible with RobotControlPanel.

    The panel expects send(), get_pose() -> list, get_angles() -> list.
    DobotNova5 has _send(), get_pose() -> np.ndarray, get_joint_angles().
    """

    def __init__(self, robot):
        self._robot = robot

    def send(self, cmd):
        return self._robot._send(cmd)

    def get_pose(self):
        p = self._robot.get_pose()
        return list(p) if p is not None else None

    def get_angles(self):
        a = self._robot.get_joint_angles()
        return list(a) if a is not None else None


HOVER_HEIGHT_MM = 200.0    # Height above rod centroid
GRIPPER_DOWN_RX = 180.0    # rx when gripper points straight down


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


def pixel_to_ray_camera(u, v, intrinsics):
    """Convert pixel (u, v) to a unit ray direction in camera frame.

    Uses the RealSense intrinsics (fx, fy, ppx, ppy).
    """
    ray = np.array([
        (u - intrinsics.ppx) / intrinsics.fx,
        (v - intrinsics.ppy) / intrinsics.fy,
        1.0,
    ])
    return ray / np.linalg.norm(ray)


def ray_plane_intersect(origin, direction, plane_z):
    """Intersect a ray with a horizontal plane at z=plane_z in base frame.

    Args:
        origin: ray origin [x, y, z] in base frame (mm)
        direction: ray direction [dx, dy, dz] in base frame
        plane_z: z-height of the plane in base frame (mm)

    Returns:
        [x, y, z] intersection point, or None if ray is parallel to plane.
    """
    if abs(direction[2]) < 1e-9:
        return None
    t = (plane_z - origin[2]) / direction[2]
    if t < 0:
        return None  # Intersection behind camera
    return origin + t * direction


def compute_hover_pose(detection, transform, intrinsics, rod_diameter_mm):
    """Compute robot pose to hover above the rod centroid.

    Instead of using RGBD depth (unreliable at this range), cast a ray from
    the camera through the 2D centroid pixel and intersect with the table
    plane (z=0) + rod radius. The rod center is at z = rod_diameter/2.

    The rod axis is computed by casting rays through the two endpoints of the
    bounding rect and intersecting with the same plane.

    Returns:
        (x, y, z, rx, ry, rz) in mm/degrees for the robot, or None on failure.
        Gripper points down (rx=180), rz perpendicular to rod axis.
    """
    T = transform.T_camera_to_base
    R = T[:3, :3]
    cam_origin_m = T[:3, 3]  # Camera origin in base frame (meters)
    cam_origin_mm = cam_origin_m * 1000.0

    rod_z = rod_diameter_mm / 2.0  # Rod centroid height above table

    # Cast ray through centroid pixel
    cx, cy = detection.center_2d
    ray_cam = pixel_to_ray_camera(cx, cy, intrinsics)
    ray_base = R @ ray_cam  # Rotate ray to base frame

    rod_center = ray_plane_intersect(cam_origin_mm, ray_base, rod_z)
    if rod_center is None:
        return None

    # Compute rod axis in base frame from bounding rect endpoints
    if detection.contour is not None:
        import cv2
        rect = cv2.minAreaRect(detection.contour)
        (rcx, rcy), (rw, rh), angle = rect
        if rw < rh:
            rw, rh = rh, rw
            angle += 90
        angle_rad = np.radians(angle)
        dx = int(rw / 2 * np.cos(angle_rad))
        dy = int(rw / 2 * np.sin(angle_rad))

        ray_p1 = pixel_to_ray_camera(cx - dx, cy - dy, intrinsics)
        ray_p2 = pixel_to_ray_camera(cx + dx, cy + dy, intrinsics)
        p1 = ray_plane_intersect(cam_origin_mm, R @ ray_p1, rod_z)
        p2 = ray_plane_intersect(cam_origin_mm, R @ ray_p2, rod_z)

        if p1 is not None and p2 is not None:
            axis_base = p2 - p1
            axis_norm = np.linalg.norm(axis_base[:2])  # XY only
            if axis_norm > 0:
                axis_base = axis_base / np.linalg.norm(axis_base)
            else:
                axis_base = np.array([1, 0, 0])
        else:
            axis_base = np.array([1, 0, 0])
    else:
        axis_base = np.array([1, 0, 0])

    # Hover above the rod
    hover_z = rod_z + HOVER_HEIGHT_MM

    # Gripper rotation: rz perpendicular to rod axis (projected onto XY plane)
    grasp_rz = np.degrees(np.arctan2(axis_base[1], axis_base[0]))
    grasp_rz += 90.0  # Perpendicular to rod, not parallel

    return (rod_center[0], rod_center[1], hover_z, GRIPPER_DOWN_RX, 0.0, grasp_rz)


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

    rod_cfg = config.get('rod', {})
    rod_diameter_mm = rod_cfg.get('diameter_mm', 27.0)

    # Load calibration
    transform = CoordinateTransform()
    calibration_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'calibration.yaml')
    if os.path.exists(calibration_path):
        transform.load(calibration_path)
        print(f"Loaded calibration from {calibration_path}")
    else:
        print("WARNING: No calibration file — robot goto will use identity transform")

    # Robot overlay for projecting joint positions into camera image
    gripper_cfg = config.get('gripper', {})
    robot_cfg = config.get('robot', {})
    base_offset = robot_cfg.get('base_offset_mm')
    robot_overlay = RobotOverlay(
        T_camera_to_base=transform.T_camera_to_base,
        tool_length_mm=gripper_cfg.get('tool_length_mm', 200.0),
        base_offset_mm=np.array(base_offset) if base_offset else None,
    )

    # Try robot connection (optional)
    robot = None
    if not no_robot:
        print("Connecting to robot...")
        robot = try_connect_robot(config)

    # GUI panel (only if robot connected)
    panel = None
    if robot:
        adapter = DobotPanelAdapter(robot)
        panel = RobotControlPanel(adapter, panel_x=width, panel_height=height)
        robot_speed = config.get('robot', {}).get('speed_percent', 30)
        panel.speed = robot_speed

    print()
    print("=== Rod Dataset Collection ===")
    print(f"Resolution: {width}x{height}")
    print(f"Save directory: {os.path.abspath(DATASET_DIR)}")
    print(f"Next index: {next_idx}")
    print(f"Robot: {'connected' if robot else 'not connected'}")
    print()
    print("SPACE=capture  d=detect  r=robot overlay  g=goto rod  q=quit")
    print("Arrow keys: nudge base overlay (10mm steps)  +/-: nudge base Z")
    print()

    camera = RealSenseCamera(width=width, height=height, fps=15)
    camera.start()

    def on_mouse(event, x, y, flags, param):
        if panel and x >= width:
            panel.handle_mouse(event, x, y, flags)

    cv2.namedWindow('Collect Dataset')
    cv2.setMouseCallback('Collect Dataset', on_mouse)

    # Let auto-exposure settle
    for _ in range(30):
        camera.get_frames()

    show_detection = False
    show_robot_overlay = True  # Always show by default
    count = 0
    last_detection = None

    try:
        while True:
            color, depth_img, depth_frame = camera.get_frames()
            if color is None:
                continue

            canvas_w = width + PANEL_WIDTH if panel else width
            canvas = np.zeros((height, canvas_w, 3), dtype=np.uint8)
            vis = color.copy()

            # Run detection overlay if toggled
            if show_detection:
                detection = detector.detect(color, depth_img, depth_frame, camera)
                if detection:
                    last_detection = detection
                    vis = detector.draw_detection(vis, detection)

                    # Show hover target info
                    pose = compute_hover_pose(detection, transform,
                                              camera.intrinsics, rod_diameter_mm)
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

            # Robot overlay: always show base marker, show joints if connected
            if show_robot_overlay and camera.intrinsics is not None:
                if robot:
                    try:
                        joint_angles = robot.get_joint_angles()
                        vis = robot_overlay.draw_joints(vis, joint_angles,
                                                        camera.intrinsics)
                    except Exception:
                        vis = robot_overlay.draw_base_marker(vis, camera.intrinsics)
                else:
                    vis = robot_overlay.draw_base_marker(vis, camera.intrinsics)

            # Status bar
            h_img = vis.shape[0]
            robot_str = "robot:ON" if robot else "robot:OFF"
            det_str = "det:ON" if show_detection else "det:OFF"
            overlay_str = "skel:ON" if show_robot_overlay else "skel:OFF"
            status = f"#{next_idx}  |  {count} captured  |  {det_str}  |  {overlay_str}  |  {robot_str}  |  SPACE=capture  d=detect  r=skel  g=goto  q=quit"
            cv2.rectangle(vis, (0, h_img - 30), (vis.shape[1], h_img), (0, 0, 0), -1)
            cv2.putText(vis, status, (10, h_img - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            canvas[0:height, 0:width] = vis
            if panel:
                panel.draw(canvas)

            cv2.imshow('Collect Dataset', canvas)
            key_raw = cv2.waitKeyEx(1)
            key = key_raw & 0xFF

            if key == ord('q') or key == 27:
                break
            try:
                if cv2.getWindowProperty('Collect Dataset', cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            # Panel keyboard handling
            if key != 255 and panel and panel.handle_key(key):
                pass

            elif key == ord('d'):
                show_detection = not show_detection
                print(f"  Detection overlay: {'ON' if show_detection else 'OFF'}")

            elif key == ord('r'):
                show_robot_overlay = not show_robot_overlay
                print(f"  Robot skeleton overlay: {'ON' if show_robot_overlay else 'OFF'}")

            # Arrow keys to nudge base overlay position (10mm steps)
            NUDGE_MM = 10.0
            if key_raw == 65361:    # Left arrow -> base X-
                robot_overlay.nudge_base(dx_mm=-NUDGE_MM)
                print(f"  Base offset: {robot_overlay.base_offset_m * 1000} mm")
            elif key_raw == 65363:  # Right arrow -> base X+
                robot_overlay.nudge_base(dx_mm=NUDGE_MM)
                print(f"  Base offset: {robot_overlay.base_offset_m * 1000} mm")
            elif key_raw == 65362:  # Up arrow -> base Y-
                robot_overlay.nudge_base(dy_mm=-NUDGE_MM)
                print(f"  Base offset: {robot_overlay.base_offset_m * 1000} mm")
            elif key_raw == 65364:  # Down arrow -> base Y+
                robot_overlay.nudge_base(dy_mm=NUDGE_MM)
                print(f"  Base offset: {robot_overlay.base_offset_m * 1000} mm")
            elif key == ord('+') or key == ord('='):  # Z+
                robot_overlay.nudge_base(dz_mm=NUDGE_MM)
                print(f"  Base offset: {robot_overlay.base_offset_m * 1000} mm")
            elif key == ord('-'):                      # Z-
                robot_overlay.nudge_base(dz_mm=-NUDGE_MM)
                print(f"  Base offset: {robot_overlay.base_offset_m * 1000} mm")

            elif key == ord('g'):
                if not show_detection:
                    print("  Turn on detection first (press 'd')")
                elif last_detection is None:
                    print("  No rod detected — nothing to go to")
                elif robot is None:
                    print("  Robot not connected")
                else:
                    pose = compute_hover_pose(last_detection, transform,
                                              camera.intrinsics, rod_diameter_mm)
                    if pose:
                        x, y, z, rx, ry, rz = pose
                        print(f"  Moving to hover: ({x:.1f}, {y:.1f}, {z:.1f}) rx={rx:.1f} ry={ry:.1f} rz={rz:.1f}")
                        ok = robot.movj(x, y, z, rx, ry, rz)
                        if ok:
                            print(f"  Arrived at hover position")
                        else:
                            print(f"  Move failed or timed out")

            elif key == ord(' '):
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
        if panel and panel.jogging:
            panel._stop_jog()
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
