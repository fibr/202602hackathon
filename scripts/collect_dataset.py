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
    ./run.sh scripts/collect_dataset.py --snapshot [--sd]

Modes:
    (default)   Live interactive feed with capture, detection, robot goto
    --snapshot  Single-frame capture: run detection, save 6-stage debug images
                to /tmp/rod_debug/, print detection log, and exit.

Controls (live mode):
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
from datetime import datetime
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


SNAPSHOT_DIR = "/tmp/rod_debug"

# --- Snapshot debug drawing helpers ---

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _put_title(image, title, subtitle=None):
    """Draw a title bar at the top of the image."""
    vis = image.copy()
    bar_h = 40 if subtitle is None else 60
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (vis.shape[1], bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
    cv2.putText(vis, title, (10, 28), _FONT, 0.8, (255, 255, 255), 2)
    if subtitle:
        cv2.putText(vis, subtitle, (10, 52), _FONT, 0.5, (0, 255, 255), 1)
    return vis


def _draw_roi_overlay(image, roi_mask):
    """Draw the workspace ROI as a semi-transparent green zone with border."""
    vis = image.copy()
    if roi_mask is None:
        return _put_title(vis, "3. ROI", "No workspace ROI configured")
    overlay = vis.copy()
    overlay[roi_mask > 0] = (0, 180, 0)
    cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)
    roi_pct = 100.0 * np.count_nonzero(roi_mask) / roi_mask.size
    return _put_title(vis, "3. Workspace ROI", f"ROI covers {roi_pct:.1f}% of frame")


def _draw_candidates(image, detector, roi_mask):
    """Draw all candidate segments with per-candidate detail labels."""
    vis = image.copy()
    GREEN, RED = (0, 255, 0), (0, 0, 255)
    if roi_mask is not None:
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 255), 1)

    candidates = getattr(detector, '_last_candidates', [])
    masks = getattr(detector, '_last_masks', None)
    h, w = image.shape[:2]
    n_pass = sum(1 for c in candidates if c['passed'])
    n_fail = len(candidates) - n_pass

    for c in candidates:
        passed = c['passed']
        color = GREEN if passed else RED
        cx, cy = c['center']

        if masks is not None and c['index'] < len(masks):
            seg_mask = masks[c['index']]
            if seg_mask.shape != (h, w):
                seg_mask = cv2.resize(seg_mask, (w, h),
                                      interpolation=cv2.INTER_NEAREST)
            binary = (seg_mask > 0.5).astype(np.uint8)
            seg_contours, _ = cv2.findContours(binary * 255,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, seg_contours, -1, color, 2 if passed else 1)
            if passed:
                overlay = vis.copy()
                overlay[binary > 0] = GREEN
                cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)

        line1 = f"#{c['index']} ar={c['aspect_ratio']:.1f} cx={c['convexity']:.2f} V={c['mean_v']:.0f}+/-{c['std_v']:.0f}"
        line2 = c['reject_reason'] if c['reject_reason'] else f"score={c.get('score', 0):.1f}"
        for dy_off, text in [(-22, line1), (-8, line2)]:
            (tw, th), _ = cv2.getTextSize(text, _FONT, 0.35, 1)
            cv2.rectangle(vis, (cx - 2, cy + dy_off - th - 1),
                          (cx + tw + 2, cy + dy_off + 2), (0, 0, 0), -1)
            cv2.putText(vis, text, (cx, cy + dy_off), _FONT, 0.35, color, 1)
        cv2.circle(vis, (cx, cy), 4, color, -1)

    subtitle = f"{n_pass} passed, {n_fail} rejected (ar>={detector.min_aspect_ratio} cx>={detector.min_convexity} V<={detector.max_brightness})"
    return _put_title(vis, "5. Candidates: shape + color filter", subtitle)


def run_snapshot(config, width, height):
    """Single-frame capture with 6-stage debug pipeline. Saves to /tmp/rod_debug/."""
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = f"{SNAPSHOT_DIR}/{ts}"

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

    print(f"Starting camera ({width}x{height})...")
    camera = RealSenseCamera(width=width, height=height, fps=15)
    camera.start()

    for _ in range(30):
        camera.get_frames()
    time.sleep(0.5)

    print("Capturing frame...")
    color_image, depth_image, depth_frame = camera.get_frames()
    camera.stop()

    if color_image is None:
        print("ERROR: No frame captured")
        return

    print(f"Frame: {color_image.shape[1]}x{color_image.shape[0]}")

    info_lines = [
        f"Timestamp: {ts}",
        f"Resolution: {width}x{height}",
        f"",
        f"--- Detector config ---",
        f"min_aspect_ratio: {detector.min_aspect_ratio}",
        f"min_area: {detector.min_area}",
        f"depth_range: {detector.depth_min_mm}-{detector.depth_max_mm} mm",
        f"max_brightness (V): {detector.max_brightness}",
        f"max_brightness_std: {detector.max_brightness_std}",
        f"min_convexity: {detector.min_convexity}",
        f"workspace_roi: {detector.workspace_roi is not None}",
        f"fastsam: conf={detector.fastsam_conf} iou={detector.fastsam_iou} imgsz={detector.fastsam_imgsz}",
        f"",
    ]

    # 1. Color
    cv2.imwrite(f"{prefix}_1_color.png",
                _put_title(color_image, "1. Color input", f"{width}x{height}"))

    # 2. Depth
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    valid = depth_image[depth_image > 0]
    depth_sub = ""
    if len(valid) > 0:
        depth_sub = f"range: {int(valid.min())}-{int(valid.max())} mm, median: {int(np.median(valid))} mm"
    cv2.imwrite(f"{prefix}_2_depth.png",
                _put_title(depth_colormap, "2. Depth", depth_sub))

    # Run detection
    print("Running FastSAM detection...")
    t0 = time.time()
    detection = detector.detect(color_image, depth_image, depth_frame, camera)
    t_detect = time.time() - t0
    print(f"Inference time: {t_detect:.2f}s")
    info_lines.append(f"--- Pipeline results ---")
    info_lines.append(f"Inference time: {t_detect:.2f}s")

    # 3. ROI
    roi_mask = getattr(detector, '_last_roi_mask', None)
    cv2.imwrite(f"{prefix}_3_roi.png",
                _draw_roi_overlay(color_image, roi_mask))

    # 4. Segments
    masks = getattr(detector, '_last_masks', None)
    n_masks = len(masks) if masks is not None else 0
    segments_vis = detector.draw_all_segments(color_image)
    segments_vis = _put_title(segments_vis, "4. FastSAM segments",
                              f"{n_masks} segments found in {t_detect:.2f}s")
    cv2.imwrite(f"{prefix}_4_segments.png", segments_vis)
    info_lines.append(f"Total segments: {n_masks}")

    # 5. Candidates
    cv2.imwrite(f"{prefix}_5_candidates.png",
                _draw_candidates(color_image, detector, roi_mask))

    candidates = getattr(detector, '_last_candidates', [])
    n_candidates = len(candidates)
    n_pass = sum(1 for c in candidates if c['passed'])
    info_lines.append(f"Candidates in ROI (area >= {detector.min_area}): {n_candidates}")
    info_lines.append(f"Passed shape+color filter: {n_pass}")
    info_lines.append(f"")

    for c in candidates:
        status = "PASS" if c['passed'] else "FAIL"
        reason = f"  ({c['reject_reason']})" if c['reject_reason'] else ""
        info_lines.append(
            f"  [{status}] seg#{c['index']}: area={c['area']:.0f} "
            f"ar={c['aspect_ratio']:.1f} cx={c['convexity']:.2f} "
            f"size=({c['size'][0]:.0f}x{c['size'][1]:.0f}) "
            f"V={c['mean_v']:.0f}+/-{c['std_v']:.0f} S={c['mean_s']:.0f} "
            f"dark={c['darkness_score']:.2f} unif={c['uniformity_score']:.2f} "
            f"overlap={c['overlap_ratio']:.0%} "
            f"score={c.get('score', '-')}{reason}")

    # 6. Detection
    if detection:
        info_lines += [
            f"", f"=== DETECTION ===",
            f"Center 2D: {detection.center_2d}",
            f"Center 3D: [{detection.center_3d[0]:.4f}, {detection.center_3d[1]:.4f}, {detection.center_3d[2]:.4f}] m",
            f"Axis 3D:   [{detection.axis_3d[0]:.4f}, {detection.axis_3d[1]:.4f}, {detection.axis_3d[2]:.4f}]",
            f"Confidence: {detection.confidence:.0%}",
        ]
        det_vis = detector.draw_detection(color_image, detection)
        det_vis = _put_title(det_vis, "6. Detection",
                             f"conf={detection.confidence:.0%} z={detection.center_3d[2]:.3f}m @ {detection.center_2d}")
        cv2.imwrite(f"{prefix}_6_detection.png", det_vis)
        print(f"\nROD DETECTED!")
        print(f"  Center 2D: {detection.center_2d}")
        print(f"  Center 3D: [{detection.center_3d[0]:.4f}, {detection.center_3d[1]:.4f}, {detection.center_3d[2]:.4f}] m")
        print(f"  Axis:      [{detection.axis_3d[0]:.4f}, {detection.axis_3d[1]:.4f}, {detection.axis_3d[2]:.4f}]")
        print(f"  Confidence: {detection.confidence:.0%}")
    else:
        cv2.imwrite(f"{prefix}_6_detection.png",
                    _put_title(color_image, "6. Detection", "NO ROD DETECTED"))
        info_lines += [f"", f"=== NO DETECTION ==="]
        print("\nNo rod detected.")

    # Info file
    info_text = '\n'.join(info_lines)
    with open(f"{prefix}_info.txt", 'w') as f:
        f.write(info_text)
    print(f"\n{info_text}")
    print(f"\nDebug images: {prefix}_*.png")


def main():
    sd = '--sd' in sys.argv
    no_robot = '--no-robot' in sys.argv
    snapshot = '--snapshot' in sys.argv
    width, height = (640, 480) if sd else (1280, 720)

    config = load_config()

    if snapshot:
        run_snapshot(config, width, height)
        return

    os.makedirs(DATASET_DIR, exist_ok=True)
    next_idx = find_next_index(DATASET_DIR)
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
