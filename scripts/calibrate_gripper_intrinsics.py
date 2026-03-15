#!/usr/bin/env python3
"""Automated intrinsics calibration for the gripper-mounted camera.

Opens the gripper-mounted wrist camera, jogs the arm around the safe HOME position while
detecting the CharUco board, then runs OpenCV camera calibration to produce
accurate fx, fy, ppx, ppy, and distortion coefficients.

The arm first moves to the configured home_angles, then makes small jogs from
there — it NEVER moves more than --max-jog-deg away from home on any joint.
This keeps the arm in a safe, predictable region throughout calibration.

Calibrated intrinsics are saved to:
  - config/camera_intrinsics.yaml  (primary, used by other scripts)
  - config/cameras.yaml            (camera registry, source: calibrated)

Usage:
    ./run.sh scripts/calibrate_gripper_intrinsics.py
    ./run.sh scripts/calibrate_gripper_intrinsics.py --no-robot     # manual capture
    ./run.sh scripts/calibrate_gripper_intrinsics.py --max-jog-deg 20  # tighter range
    ./run.sh scripts/calibrate_gripper_intrinsics.py --show-window     # live feed

Workflow:
  1. Place the CharUco board flat on the table in view of the gripper camera.
  2. Run this script — it moves to home, then jogs around it auto-capturing frames.
  3. Calibration runs automatically once enough good frames are collected.
  4. Results saved to config/camera_intrinsics.yaml and cameras.yaml.

Key controls (when --show-window is set):
  c  - Force-capture current frame
  r  - Run calibration now (even if < min-frames)
  q  - Quit without saving
"""

import argparse
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import yaml

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config_loader import load_config, config_path
from vision.board_detector import BoardDetector
from rig_lock import RigLock

# ──────────────────────────────────────────────────────────────────────────────
# Camera helpers
# ──────────────────────────────────────────────────────────────────────────────

def open_gripper_camera(config):
    """Open the gripper-mounted camera and return a cv2.VideoCapture."""
    gc = config.get('gripper_camera', {})
    dev_idx = gc.get('device_index', 0)
    w = gc.get('width', 640)
    h = gc.get('height', 480)
    print(f"  Opening gripper camera /dev/video{dev_idx} ({w}x{h})...")
    cap = cv2.VideoCapture(dev_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open gripper camera at /dev/video{dev_idx}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    # Flush stale frames from buffer
    for _ in range(10):
        cap.read()
    print(f"  Gripper camera ready.")
    return cap


def capture_frame(cap, n_warmup=3):
    """Capture a fresh frame, discarding stale buffer frames."""
    for _ in range(n_warmup):
        cap.read()
    ok, frame = cap.read()
    return frame if ok else None


# ──────────────────────────────────────────────────────────────────────────────
# Board detection helpers
# ──────────────────────────────────────────────────────────────────────────────

def detect_board(frame, detector):
    """Detect CharUco board in frame. Returns (detection, annotated_frame)."""
    if frame is None:
        return None, frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detection = detector.detect(gray)
    annotated = frame.copy()
    if detection is not None:
        detector.draw_corners(annotated, detection)
        n = len(detection.corners) if detection.corners is not None else 0
        cv2.putText(annotated, f"Corners: {n}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(annotated, "No board", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return detection, annotated


# ──────────────────────────────────────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────────────────────────────────────

def run_calibration(detections, image_size, detector):
    """Run cv2.calibrateCamera from collected detections.

    Returns:
        (rms, K, dist) or (None, None, None) on failure.
    """
    obj_points_list = []
    img_points_list = []
    for det in detections:
        obj_pts = detector.get_object_points(det)
        img_pts = det.corners.reshape(-1, 1, 2).astype(np.float32)
        obj_points_list.append(obj_pts)
        img_points_list.append(img_pts)
    try:
        rms, K, dist, _, _ = cv2.calibrateCamera(
            obj_points_list, img_points_list, image_size, None, None)
        return rms, K, dist
    except cv2.error as e:
        print(f"  Calibration failed: {e}")
        return None, None, None


# ──────────────────────────────────────────────────────────────────────────────
# Saving results
# ──────────────────────────────────────────────────────────────────────────────

def save_intrinsics(K, dist, image_size, rms, camera_name='camera_camera_0'):
    """Save calibrated intrinsics to camera_intrinsics.yaml and cameras.yaml."""
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    ppx = float(K[0, 2])
    ppy = float(K[1, 2])
    dist_list = [float(d) for d in dist.ravel()]
    w, h = int(image_size[0]), int(image_size[1])

    # ── camera_intrinsics.yaml ──
    intr_path = config_path('camera_intrinsics.yaml')
    intr_data = {
        'camera_matrix': [
            [fx, 0.0, ppx],
            [0.0, fy, ppy],
            [0.0, 0.0, 1.0],
        ],
        'dist_coeffs': dist_list,
        'image_size': [w, h],
        # Legacy flat keys (some scripts read these directly)
        'fx': fx, 'fy': fy, 'ppx': ppx, 'ppy': ppy,
        'dist': dist_list,
        'width': w, 'height': h,
        'rms_error': float(rms),
        'calibrated_at': datetime.now().isoformat(),
    }
    with open(intr_path, 'w') as f:
        yaml.dump(intr_data, f, default_flow_style=False)
    print(f"  Saved intrinsics → {intr_path}")

    # ── cameras.yaml ──
    cameras_path = config_path('cameras.yaml')
    if os.path.exists(cameras_path):
        with open(cameras_path, 'r') as f:
            cameras_data = yaml.safe_load(f) or {}
        if 'cameras' in cameras_data and camera_name in cameras_data['cameras']:
            cam = cameras_data['cameras'][camera_name]
            cam['intrinsics'] = {
                'camera_matrix': [
                    [fx, 0, ppx],
                    [0, fy, ppy],
                    [0, 0, 1],
                ],
                'dist_coeffs': dist_list,
                'image_size': [w, h],
                'source': 'calibrated',
                'rms_error': float(rms),
                'calibrated_at': datetime.now().isoformat(),
            }
            with open(cameras_path, 'w') as f:
                yaml.dump(cameras_data, f, default_flow_style=False)
            print(f"  Updated cameras.yaml ({camera_name}) → source: calibrated")
        else:
            # List available cameras to help user pick the right name
            avail = list((cameras_data.get('cameras') or {}).keys())
            print(f"  Warning: '{camera_name}' not found in cameras.yaml.")
            print(f"  Available cameras: {avail}")
            print(f"  Re-run with --camera-name <name> to update the registry.")

    return intr_path


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def show_undistort_comparison(frame, K, dist):
    """Show side-by-side original vs undistorted frame."""
    h, w = frame.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    undistorted = cv2.undistort(frame, K, dist, None, new_K)
    cv2.putText(frame, "Original", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(undistorted, "Undistorted", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    combined = np.hstack([frame, undistorted])
    cv2.imshow("Before / After Undistortion", combined)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()


def draw_status(frame, n_frames, min_frames, rms=None):
    """Draw capture progress HUD on frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    bar_w = int(w * min(n_frames / max(min_frames, 1), 1.0))
    cv2.rectangle(frame, (0, h - 6), (bar_w, h), (0, 200, 100), -1)
    txt = f"Frames: {n_frames}/{min_frames}"
    if rms is not None:
        txt += f"  RMS: {rms:.3f}px"
    cv2.putText(frame, txt, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame


# ──────────────────────────────────────────────────────────────────────────────
# Arm helpers
# ──────────────────────────────────────────────────────────────────────────────

def move_to_home(arm, home_angles, speed=80, settle_s=2.0):
    """Move all joints to home_angles at low speed and wait to settle.

    Args:
        arm: LeRobotArm101 instance.
        home_angles: List of 6 joint angles in degrees.
        speed: Servo speed (0-4095, lower = slower).
        settle_s: Seconds to wait for arm to reach home.
    """
    print(f"  Moving to home position: {home_angles[:5]} (J1-J5)...")
    try:
        arm.write_all_angles(home_angles[:6], speed=speed)
    except ValueError as e:
        print(f"  WARNING: Home move safety rejected: {e}")
        return False
    time.sleep(settle_s)
    print(f"  At home position.")
    return True


def jog_sequence(home_angles, jog_deg, n_steps, max_jog_deg):
    """Generate (joint_idx, target_angle_deg) for the sweep pattern.

    For each joint 0-4:
        sweep from home +jog, +2*jog, ..., up to max_jog_deg
        return to home
        sweep from home -jog, -2*jog, ..., up to max_jog_deg
        return to home

    Yields (joint_idx, target_angle_deg) for each capture position.
    Note: target_angle_deg is ABSOLUTE, not relative.
    """
    for joint in range(5):
        home = home_angles[joint]
        # Positive sweep
        for step in range(1, n_steps + 1):
            delta = step * jog_deg
            if delta > max_jog_deg:
                break
            yield (joint, home + delta)
        yield (joint, home)  # return to home
        # Negative sweep
        for step in range(1, n_steps + 1):
            delta = step * jog_deg
            if delta > max_jog_deg:
                break
            yield (joint, home - delta)
        yield (joint, home)  # return to home


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Automated gripper camera intrinsics calibration')
    parser.add_argument('--no-robot', action='store_true',
                        help='Camera-only mode: manual capture (press c or space)')
    parser.add_argument('--jog-deg', type=float, default=10.0,
                        help='Jog step size in degrees per step (default: 10.0)')
    parser.add_argument('--n-steps', type=int, default=2,
                        help='Steps per direction per joint (default: 2)')
    parser.add_argument('--max-jog-deg', type=float, default=25.0,
                        help='Max absolute jog from home in degrees (default: 25.0)')
    parser.add_argument('--settle-s', type=float, default=0.8,
                        help='Settle time after each jog in seconds (default: 0.8s)')
    parser.add_argument('--min-frames', type=int, default=15,
                        help='Minimum good frames before calibration (default: 15)')
    parser.add_argument('--min-corners', type=int, default=6,
                        help='Minimum CharUco corners to accept a frame (default: 6)')
    parser.add_argument('--show-window', action='store_true',
                        help='Display live camera feed window')
    parser.add_argument('--camera-name', default='camera_camera_0',
                        help='Camera key in cameras.yaml to update (default: camera_camera_0)')
    parser.add_argument('--no-move-home', action='store_true',
                        help='Skip moving to home at startup (arm already positioned)')
    args = parser.parse_args()

    # Acquire exclusive rig lock before touching hardware
    rig_lock = RigLock(holder='calibrate_gripper_intrinsics')
    rig_lock.acquire()
    print("[INFO] Rig lock acquired.")
    
    try:

        print("=" * 60)
        print("  GRIPPER CAMERA INTRINSICS CALIBRATION")
        print("=" * 60)

        config = load_config()
        bd_cfg = config.get('calibration_board', {})
        home_angles = config.get('arm101', {}).get('home_angles',
                                                    [0.0, 0.0, 90.0, 90.0, 0.0, 0.0])
        print(f"  Board: {bd_cfg.get('type', 'charuco')} {bd_cfg.get('cols')}x"
              f"{bd_cfg.get('rows')} ({bd_cfg.get('square_size_mm')}mm squares)")
        print(f"  Home angles (J1-J6): {home_angles}")
        print(f"  Jog range: home ±{args.max_jog_deg}° (step={args.jog_deg}°)")

        detector = BoardDetector.from_config(config)
        print(f"  Detector: {detector.describe()}")

        # ── Open camera ──
        cap = open_gripper_camera(config)

        # ── Connect to robot ──
        arm = None
        if not args.no_robot:
            try:
                from config_loader import connect_robot
                print("\n--- Connecting to arm ---")
                arm = connect_robot(config, safe_mode=True)
                arm.enable_torque()
                print(f"  Arm connected, torque enabled (safe mode)")

                # Move to home position first
                if not args.no_move_home:
                    move_to_home(arm, home_angles, speed=80, settle_s=2.0)
            except Exception as e:
                print(f"  WARNING: Could not connect to arm: {e}")
                print(f"  Falling back to manual capture mode")
                arm = None

        # ── Display window ──
        win = None
        if args.show_window:
            win = "Gripper Intrinsics Calibration  [c=capture  r=calibrate  q=quit]"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, 800, 600)

        # ── Verify board is visible ──
        print(f"\n--- Verifying board visibility ---")
        board_found = False
        for attempt in range(8):
            frame = capture_frame(cap)
            detection, annotated = detect_board(frame, detector)
            if win:
                cv2.imshow(win, annotated)
                cv2.waitKey(100)
            if detection is not None and len(detection.corners) >= args.min_corners:
                board_found = True
                print(f"  Board detected: {len(detection.corners)} corners ✓")
                break
            time.sleep(0.3)

        if not board_found:
            print(f"  WARNING: Board not visible at home position.")
            if arm is not None:
                print(f"  Will try to find it during jogging...")
            else:
                print(f"  Please position the board in view, then press 'c' to capture.")

        # ── Collect frames ──
        good_frames = []    # list of (BGR frame, BoardDetection)
        image_size = None   # set on first capture
        rms_current = None  # latest preliminary RMS

        MOTOR_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
                       'wrist_flex', 'wrist_roll']

        def try_capture(frame):
            """Detect board and append to good_frames. Returns (captured, annotated)."""
            nonlocal image_size
            detection, annotated = detect_board(frame, detector)
            if detection is not None and len(detection.corners) >= args.min_corners:
                h, w = frame.shape[:2]
                if image_size is None:
                    image_size = (w, h)
                good_frames.append((frame.copy(), detection))
                return True, annotated
            return False, annotated

        print(f"\n--- Collecting calibration frames ---")
        print(f"  Target: {args.min_frames} frames  "
              f"(jog={args.jog_deg}° x {args.n_steps} steps, "
              f"max_jog=±{args.max_jog_deg}°)")

        # Capture at starting (home) position
        frame = capture_frame(cap)
        ok, _ = try_capture(frame)
        if ok:
            print(f"  [home] Captured: {len(good_frames[-1][1].corners)} corners ✓ "
                  f"({len(good_frames)})")
        else:
            print(f"  [home] No board at home — continuing with jog sequence")

        # ── Automated jogging ──
        if arm is not None:
            pose_idx = 0
            # Track current absolute angles per joint (start at home)
            current_angles = list(home_angles[:6])

            for joint, target_abs in jog_sequence(home_angles, args.jog_deg,
                                                   args.n_steps, args.max_jog_deg):
                pose_idx += 1
                delta = target_abs - home_angles[joint]
                sign_str = f"+{delta:.0f}" if delta >= 0 else f"{delta:.0f}"

                # Build full angle command: only change the target joint
                new_angles = list(current_angles)
                new_angles[joint] = target_abs
                # Keep gripper at home
                new_angles[5] = home_angles[5] if len(home_angles) > 5 else 0.0

                try:
                    arm.write_all_angles(new_angles, speed=80)
                    current_angles[joint] = target_abs
                except ValueError as e:
                    print(f"  [{pose_idx}] J{joint+1} {MOTOR_NAMES[joint]} "
                          f"{sign_str}°: SAFETY {e}")
                    # Reset joint to home to be safe
                    try:
                        reset = list(current_angles)
                        reset[joint] = home_angles[joint]
                        arm.write_all_angles(reset, speed=80)
                        current_angles[joint] = home_angles[joint]
                        time.sleep(0.5)
                    except Exception:
                        pass
                    continue
                except Exception as e:
                    print(f"  [{pose_idx}] J{joint+1} {MOTOR_NAMES[joint]} "
                          f"{sign_str}°: ERROR {e}")
                    continue

                time.sleep(args.settle_s)

                # Capture frame
                frame = capture_frame(cap)
                ok, annotated = try_capture(frame)
                n_caps = len(good_frames)

                if ok:
                    nc = len(good_frames[-1][1].corners)
                    label = f"home{sign_str}°" if delta != 0 else "home"
                    print(f"  [{pose_idx}] J{joint+1} {MOTOR_NAMES[joint]} "
                          f"{label}: {nc} corners ✓ ({n_caps})")
                else:
                    label = f"home{sign_str}°" if delta != 0 else "home"
                    print(f"  [{pose_idx}] J{joint+1} {MOTOR_NAMES[joint]} "
                          f"{label}: no board")

                # Update window
                if win and annotated is not None:
                    img = draw_status(annotated.copy(), n_caps,
                                      args.min_frames, rms_current)
                    cv2.imshow(win, img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("  Quit requested.")
                        break
                    elif key == ord('c') or key == ord(' '):
                        f2 = capture_frame(cap)
                        ok2, _ = try_capture(f2)
                        if ok2:
                            print(f"  [manual] Captured: "
                                  f"{len(good_frames[-1][1].corners)} corners")
                    elif key == ord('r'):
                        print("  Early calibration requested.")
                        break

                # Run preliminary calibration when we first hit target
                if n_caps >= args.min_frames and rms_current is None:
                    print(f"\n  Reached {args.min_frames} frames — "
                          f"running preliminary calibration...")
                    sz = image_size or (640, 480)
                    dets = [d for _, d in good_frames]
                    rms, K, dist = run_calibration(dets, sz, detector)
                    if K is not None:
                        rms_current = rms
                        print(f"  Preliminary RMS: {rms:.3f}px  "
                              f"(continuing for better coverage)")

            # Always return arm to home when done
            print(f"\n  Returning arm to home position...")
            try:
                arm.write_all_angles(list(home_angles[:6]), speed=80)
                time.sleep(1.5)
            except Exception as e:
                print(f"  WARNING: Could not return to home: {e}")
            try:
                arm.disable_torque()
                print(f"  Torque disabled.")
            except Exception:
                pass

        else:
            # Manual capture mode
            print(f"  Manual mode — press 'c' or SPACE to capture, 'r' to calibrate, 'q' to quit.")
            auto_interval = 2.0  # seconds between auto-captures in headless mode
            last_capture = time.time()
            while len(good_frames) < args.min_frames:
                frame = capture_frame(cap, n_warmup=1)
                detection, annotated = detect_board(frame, detector)
                if win:
                    img = draw_status(annotated.copy() if annotated is not None else np.zeros((480, 640, 3), np.uint8),
                                      len(good_frames), args.min_frames, rms_current)
                    cv2.imshow(win, img)
                    key = cv2.waitKey(50) & 0xFF
                    if key == ord('c') or key == ord(' '):
                        ok, _ = try_capture(frame)
                        if ok:
                            print(f"  Captured: {len(good_frames[-1][1].corners)} corners "
                                  f"({len(good_frames)})")
                        else:
                            print(f"  No board detected")
                    elif key == ord('r'):
                        break
                    elif key == ord('q'):
                        cap.release()
                        if win:
                            cv2.destroyAllWindows()
                        return 0
                else:
                    # Headless: auto-capture periodically if board visible
                    now = time.time()
                    if now - last_capture >= auto_interval:
                        ok, _ = try_capture(frame)
                        if ok:
                            print(f"  Auto-captured: "
                                  f"{len(good_frames[-1][1].corners)} corners "
                                  f"({len(good_frames)})")
                            last_capture = now
                        else:
                            time.sleep(0.5)

        cap.release()

        # ── Final calibration ──
        print(f"\n--- Final calibration ({len(good_frames)} frames) ---")

        if len(good_frames) < 3:
            print(f"  ERROR: Need at least 3 frames, got {len(good_frames)}.")
            print(f"  Tips:")
            print(f"    - Make sure CharUco board is flat on the table and visible")
            print(f"    - Try --max-jog-deg 30 for wider coverage")
            print(f"    - Reduce --min-corners to accept partial detections")
            if win:
                cv2.destroyAllWindows()
            return 1

        h, w = good_frames[0][0].shape[:2]
        sz = image_size or (w, h)
        detections = [det for _, det in good_frames]

        rms, K, dist = run_calibration(detections, sz, detector)
        if K is None:
            print(f"  Calibration failed.")
            if win:
                cv2.destroyAllWindows()
            return 1

        # ── Print results ──
        print(f"\n{'='*60}")
        print(f"  CALIBRATION RESULTS")
        print(f"{'='*60}")
        print(f"  Frames used:  {len(detections)}")
        print(f"  Image size:   {sz[0]}x{sz[1]}")
        quality = '[EXCELLENT]' if rms < 0.5 else '[GOOD]' if rms < 1.0 else '[ACCEPTABLE]' if rms < 2.0 else '[HIGH — collect more frames]'
        print(f"  RMS error:    {rms:.4f} px  {quality}")
        print(f"  Focal length: fx={K[0,0]:.2f}  fy={K[1,1]:.2f}")
        print(f"  Principal pt: cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
        k = dist.ravel()
        print(f"  Distortion:   k1={k[0]:.4f}  k2={k[1]:.4f}  "
              f"p1={k[2]:.4f}  p2={k[3]:.4f}  k3={k[4]:.4f}")

        if rms < 0.5:
            print(f"\n  Excellent calibration! RMS < 0.5px.")
        elif rms < 1.0:
            print(f"\n  Good calibration. RMS < 1.0px.")
        elif rms < 2.0:
            print(f"\n  Acceptable. Consider more diverse frames.")
        else:
            print(f"\n  High error. Collect more frames from varied angles.")

        # Show undistortion comparison
        if win and good_frames:
            try:
                show_undistort_comparison(good_frames[-1][0].copy(), K, dist)
            except Exception:
                pass

        # ── Save ──
        save_intrinsics(K, dist, sz, rms, args.camera_name)

        print(f"\n  Done! Next step: run servo direction calibration.")
        print(f"  ./run.sh scripts/auto_servo_calib.py --save")

        if win:
            cv2.destroyAllWindows()

        return 0


    finally:
        rig_lock.release()

if __name__ == '__main__':
    sys.exit(main())
