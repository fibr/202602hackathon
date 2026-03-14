#!/usr/bin/env python3
"""Visual servo standalone test — live detection + pixel error display on cam_8.

Opens /dev/video8 (the gripper-mounted downward camera), runs green-cube detection
on every frame, and overlays the pixel error from the image centre.  No robot
motion is performed — this script is purely for tuning two key parameters:

  scale_mm_per_pixel   — mm of robot XY movement per pixel of error
  mount_angle_deg      — rotation of the camera relative to the gripper (degrees)

The live overlay shows:
  • Target detection circle (green) and line from image centre to target
  • Pixel error components  (ex, ey) and Euclidean magnitude |err|
  • Estimated robot-base correction in mm (at gripper_rz = 0°)
  • Current values of the two tuning parameters
  • A coloured convergence indicator (green = within threshold, yellow = outside)

Usage:
    ./run.sh scripts/test_visual_servo.py
    ./run.sh scripts/test_visual_servo.py --cam 8
    ./run.sh scripts/test_visual_servo.py --scale 0.18 --angle 45.0
    ./run.sh scripts/test_visual_servo.py --flip-x   # mirror X axis
    ./run.sh scripts/test_visual_servo.py --flip-y   # mirror Y axis

Keyboard controls (in the OpenCV window):
    +  / =       Increase scale_mm_per_pixel by 0.01
    -            Decrease scale_mm_per_pixel by 0.01
    [            Decrease mount_angle_deg by 1°
    ]            Increase mount_angle_deg by 1°
    x            Toggle cam_flip_x
    y            Toggle cam_flip_y
    p            Print current params to terminal
    s            Save current params as visual_servo overrides in config/settings.yaml
    q / Esc      Quit
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import yaml

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config_loader import load_config
from vision.green_cube_detector import detect_green_cubes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def open_camera(index: int, width: int = 640, height: int = 480) -> cv2.VideoCapture:
    """Open a V4L2 camera with MJPEG codec and warm up auto-exposure."""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera /dev/video{index}. "
            "Is the device connected and not in use by another process?"
        )
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Warm-up: let auto-exposure settle
    for _ in range(8):
        cap.read()
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera /dev/video{index} opened at {actual_w}x{actual_h}")
    return cap


def pixel_to_robot_delta(
    ex: float,
    ey: float,
    scale_mm_per_pixel: float,
    mount_angle_deg: float,
    gripper_rz_deg: float,
    gain: float,
    flip_x: bool,
    flip_y: bool,
    max_correction_mm: float,
) -> tuple:
    """Convert a pixel error to a robot-base-frame XY correction (mm).

    Mirrors the logic in VisualServo._pixel_to_robot_delta so the display
    shows exactly what the servo would send.

    Returns:
        (dx_mm, dy_mm, clamped) where clamped is True if the output was clipped.
    """
    if flip_x:
        ex = -ex
    if flip_y:
        ey = -ey

    dx_cam = ex * scale_mm_per_pixel * gain
    dy_cam = ey * scale_mm_per_pixel * gain

    angle_rad = np.radians(gripper_rz_deg + mount_angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    dx_robot = cos_a * dx_cam - sin_a * dy_cam
    dy_robot = sin_a * dx_cam + cos_a * dy_cam

    mag = np.hypot(dx_robot, dy_robot)
    clamped = mag > max_correction_mm
    if clamped:
        s = max_correction_mm / mag
        dx_robot *= s
        dy_robot *= s

    return dx_robot, dy_robot, clamped


def draw_crosshair(img: np.ndarray, cx: int, cy: int,
                   color=(0, 220, 220), size: int = 20, thickness: int = 1):
    """Draw a small crosshair at (cx, cy)."""
    cv2.line(img, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(img, (cx, cy - size), (cx, cy + size), color, thickness)


def put_text_bg(img, text, pos, scale=0.5, color=(255, 255, 255),
                bg=(0, 0, 0), thickness=1):
    """Draw text with a dark background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - 2, y - th - 2), (x + tw + 2, y + baseline + 2),
                  bg, -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness,
                cv2.LINE_AA)


def save_settings_override(
    scale: float,
    angle: float,
    flip_x: bool,
    flip_y: bool,
    settings_path: str,
):
    """Persist visual_servo tuning values into config/settings.yaml."""
    existing = {}
    if os.path.exists(settings_path):
        with open(settings_path) as f:
            existing = yaml.safe_load(f) or {}

    existing.setdefault('visual_servo', {})
    existing['visual_servo']['scale_mm_per_pixel'] = round(float(scale), 4)
    existing['visual_servo']['mount_angle_deg'] = round(float(angle), 2)
    existing['visual_servo']['cam_flip_x'] = bool(flip_x)
    existing['visual_servo']['cam_flip_y'] = bool(flip_y)

    os.makedirs(os.path.dirname(settings_path), exist_ok=True)
    with open(settings_path, 'w') as f:
        yaml.dump(existing, f, default_flow_style=False)
    print(f"Saved visual_servo overrides to {settings_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visual servo tuning display — cam_8 + green-cube detection, no robot motion."
    )
    parser.add_argument('--cam', type=int, default=None,
                        help='Camera device index (default: from config, usually 8)')
    parser.add_argument('--scale', type=float, default=None,
                        help='Initial scale_mm_per_pixel override')
    parser.add_argument('--angle', type=float, default=None,
                        help='Initial mount_angle_deg override')
    parser.add_argument('--flip-x', action='store_true',
                        help='Enable cam_flip_x (mirror X component of pixel error)')
    parser.add_argument('--flip-y', action='store_true',
                        help='Enable cam_flip_y (mirror Y component of pixel error)')
    parser.add_argument('--gripper-rz', type=float, default=0.0,
                        help='Simulated gripper rz angle in degrees (default 0)')
    parser.add_argument('--width', type=int, default=None,
                        help='Capture width (default: from config, usually 640)')
    parser.add_argument('--height', type=int, default=None,
                        help='Capture height (default: from config, usually 480)')
    args = parser.parse_args()

    # --- Load config ---
    config = load_config()
    gc_cfg = config.get('gripper_camera', {})
    vs_cfg = config.get('visual_servo', {})

    cam_index = args.cam if args.cam is not None else gc_cfg.get('device_index', 8)
    cam_w = args.width if args.width is not None else gc_cfg.get('width', 640)
    cam_h = args.height if args.height is not None else gc_cfg.get('height', 480)

    scale = args.scale if args.scale is not None else vs_cfg.get('scale_mm_per_pixel', 0.25)
    angle = args.angle if args.angle is not None else vs_cfg.get('mount_angle_deg', 0.0)
    flip_x = args.flip_x or vs_cfg.get('cam_flip_x', False)
    flip_y = args.flip_y or vs_cfg.get('cam_flip_y', False)
    gain = vs_cfg.get('gain', 0.6)
    threshold_px = vs_cfg.get('pixel_threshold', 20.0)
    max_corr_mm = vs_cfg.get('max_correction_mm', 30.0)
    gripper_rz = args.gripper_rz

    settings_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.yaml')
    )

    print("=" * 60)
    print("  Visual Servo Tuning — cam_8 live feed (NO robot motion)")
    print("=" * 60)
    print(f"  Camera:            /dev/video{cam_index}  ({cam_w}x{cam_h})")
    print(f"  scale_mm_per_pixel: {scale:.4f}")
    print(f"  mount_angle_deg:    {angle:.2f}°")
    print(f"  cam_flip_x:         {flip_x}")
    print(f"  cam_flip_y:         {flip_y}")
    print(f"  gain:               {gain}")
    print(f"  pixel_threshold:    {threshold_px} px")
    print(f"  gripper_rz (sim):   {gripper_rz:.1f}°")
    print()
    print("Keyboard: +/- scale  [/] angle  x flip-x  y flip-y")
    print("          p print    s save     q/Esc quit")
    print()

    # --- Open camera ---
    try:
        cap = open_camera(cam_index, cam_w, cam_h)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    img_cx = cam_w / 2.0
    img_cy = cam_h / 2.0

    last_detection = None
    frame_count = 0
    fps_t0 = time.time()
    fps_display = 0.0

    try:
        while True:
            # --- Capture frame ---
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Warning: failed to capture frame; retrying...")
                time.sleep(0.05)
                continue

            frame_count += 1
            elapsed = time.time() - fps_t0
            if elapsed >= 1.0:
                fps_display = frame_count / elapsed
                frame_count = 0
                fps_t0 = time.time()

            # --- Detect ---
            dets, _ = detect_green_cubes(frame)

            vis = frame.copy()

            # --- Draw image-centre crosshair ---
            draw_crosshair(vis, int(img_cx), int(img_cy), color=(0, 220, 220), size=25)

            # --- Overlay detection + error ---
            if dets:
                det = dets[0]  # largest detection
                last_detection = (det.cx, det.cy)

                tx, ty = det.cx, det.cy
                ex = tx - img_cx
                ey = ty - img_cy
                err_px = float(np.hypot(ex, ey))

                dx_mm, dy_mm, clamped = pixel_to_robot_delta(
                    ex, ey, scale, angle, gripper_rz,
                    gain, flip_x, flip_y, max_corr_mm,
                )

                # Convergence colour
                if err_px < threshold_px:
                    err_color = (0, 220, 0)    # green — within threshold
                    status_str = "CONVERGED"
                else:
                    err_color = (0, 180, 255)  # orange — outside threshold
                    status_str = f"|err|={err_px:.1f}px"

                # Error line: centre → target
                cv2.arrowedLine(vis, (int(img_cx), int(img_cy)),
                                (tx, ty), err_color, 2, tipLength=0.15)

                # Target circle
                cv2.circle(vis, (tx, ty), 10, (0, 255, 0), 2)
                cv2.circle(vis, (tx, ty), 2, (0, 255, 0), -1)
                # Bounding box
                bx, by, bw, bh = det.bbox
                cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 200, 0), 1)

                # Pixel error label near the target
                put_text_bg(vis, f"({ex:+.0f},{ey:+.0f})px",
                            (tx + 12, ty - 6), scale=0.45, color=err_color)
                if clamped:
                    put_text_bg(vis, "CLAMPED", (tx + 12, ty + 14),
                                scale=0.4, color=(0, 0, 255))

                # Main status (top-left)
                y_off = 28
                put_text_bg(vis, status_str, (10, y_off),
                            scale=0.6, color=err_color)
                y_off += 22
                put_text_bg(vis, f"err: ({ex:+.1f}, {ey:+.1f})  |{err_px:.1f}| px",
                            (10, y_off), scale=0.48)
                y_off += 18
                put_text_bg(vis, f"mm corr (rz=0): dx={dx_mm:+.2f}  dy={dy_mm:+.2f} mm",
                            (10, y_off), scale=0.48)

            else:
                last_detection = None
                put_text_bg(vis, "NO DETECTION", (10, 28),
                            scale=0.65, color=(0, 0, 220))

            # --- Parameter panel (bottom-left) ---
            h = vis.shape[0]
            lines = [
                f"scale={scale:.4f} mm/px  (+/- to adjust)",
                f"angle={angle:.1f} deg     ([/] to adjust)",
                f"flip_x={flip_x}  flip_y={flip_y}  (x/y toggle)",
                f"threshold={threshold_px:.0f}px  gain={gain:.2f}  rz={gripper_rz:.1f}deg",
                f"FPS={fps_display:.1f}   s=save  p=print  q=quit",
            ]
            for i, ln in enumerate(reversed(lines)):
                put_text_bg(vis, ln, (8, h - 10 - i * 18),
                            scale=0.42, color=(200, 220, 200))

            cv2.imshow("Visual Servo Tuning — cam_8 (no robot)", vis)

            # --- Keyboard ---
            key = cv2.waitKey(30) & 0xFF

            if key in (ord('q'), 27):   # q or Esc
                break

            elif key in (ord('+'), ord('=')):
                scale = round(scale + 0.01, 4)
                print(f"scale_mm_per_pixel → {scale:.4f}")

            elif key == ord('-'):
                scale = round(max(0.001, scale - 0.01), 4)
                print(f"scale_mm_per_pixel → {scale:.4f}")

            elif key == ord('['):
                angle = round(angle - 1.0, 2)
                print(f"mount_angle_deg → {angle:.2f}")

            elif key == ord(']'):
                angle = round(angle + 1.0, 2)
                print(f"mount_angle_deg → {angle:.2f}")

            elif key == ord('x'):
                flip_x = not flip_x
                print(f"cam_flip_x → {flip_x}")

            elif key == ord('y'):
                flip_y = not flip_y
                print(f"cam_flip_y → {flip_y}")

            elif key == ord('p'):
                print()
                print("--- Current visual_servo parameters ---")
                print(f"  scale_mm_per_pixel: {scale:.4f}")
                print(f"  mount_angle_deg:    {angle:.2f}")
                print(f"  cam_flip_x:         {flip_x}")
                print(f"  cam_flip_y:         {flip_y}")
                if last_detection:
                    lx, ly = last_detection
                    ex = lx - img_cx
                    ey = ly - img_cy
                    err_px = float(np.hypot(ex, ey))
                    print(f"  Last detection:     pixel ({lx},{ly})  "
                          f"err=({ex:+.1f},{ey:+.1f})  |{err_px:.1f}|px")
                print()

            elif key == ord('s'):
                save_settings_override(scale, angle, flip_x, flip_y, settings_path)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nCamera released. Final parameters:")
        print(f"  scale_mm_per_pixel: {scale:.4f}")
        print(f"  mount_angle_deg:    {angle:.2f}")
        print(f"  cam_flip_x:         {flip_x}")
        print(f"  cam_flip_y:         {flip_y}")
        print()
        print("To persist these values, run again and press 's', or add to")
        print("config/settings.yaml under 'visual_servo:'")


if __name__ == '__main__':
    main()
