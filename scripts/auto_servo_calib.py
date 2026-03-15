#!/usr/bin/env python3
"""Automated servo direction & offset calibration using gripper camera + CharUco board.

Connects to the ARM101, opens the gripper-mounted camera, and systematically
jogs each joint by small increments while capturing ChArUco board poses.
The collected data is fed into the brute-force sign solver to determine
optimal joint signs and zero offsets.

The solver tries all 32 sign combinations and picks the one that minimises
the inconsistency of the board position in robot base frame across captures.

Usage:
    ./run.sh scripts/auto_servo_calib.py
    ./run.sh scripts/auto_servo_calib.py --jog-deg 5.0    # smaller jog steps
    ./run.sh scripts/auto_servo_calib.py --jog-deg 10.0   # larger jog steps
    ./run.sh scripts/auto_servo_calib.py --save            # save results
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import yaml

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config_loader import load_config, connect_robot, config_path
from vision.camera import CameraIntrinsics
from vision.board_detector import BoardDetector
from calibration.calib_helpers import read_all_raw, load_offsets
from calibration.sign_solver import (
    _brute_force_signs, save_calibration_results, MOTOR_NAMES)
from kinematics.arm101_ik_solver import Arm101IKSolver


def load_gripper_camera_intrinsics(config):
    """Load intrinsics for the gripper camera."""
    intr_path = config_path('camera_intrinsics.yaml')
    if os.path.exists(intr_path):
        with open(intr_path) as f:
            data = yaml.safe_load(f)
        K = np.array(data['camera_matrix'])
        dist = np.array(data['dist_coeffs'])
        return CameraIntrinsics(
            fx=K[0, 0], fy=K[1, 1], ppx=K[0, 2], ppy=K[1, 2],
            coeffs=dist.tolist(),
        )
    gc = config.get('gripper_camera', {})
    w = gc.get('width', 640)
    hfov = gc.get('hfov_deg', 60.0)
    fx = (w / 2) / np.tan(np.radians(hfov / 2))
    return CameraIntrinsics(
        fx=fx, fy=fx, ppx=w / 2, ppy=gc.get('height', 480) / 2,
        coeffs=[0.0] * 5,
    )


def open_gripper_camera(config):
    """Open the gripper-mounted camera and return a cv2.VideoCapture."""
    gc = config.get('gripper_camera', {})
    dev_idx = gc.get('device_index', 8)
    w = gc.get('width', 640)
    h = gc.get('height', 480)
    print(f"  Opening gripper camera /dev/video{dev_idx} ({w}x{h})...")
    cap = cv2.VideoCapture(dev_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open gripper camera at /dev/video{dev_idx}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    for _ in range(10):
        cap.read()
    print(f"  Gripper camera ready.")
    return cap


def capture_frame(cap, n_warmup=3):
    """Capture a frame, discarding stale buffers."""
    for _ in range(n_warmup):
        cap.read()
    ok, frame = cap.read()
    return frame if ok else None


def detect_board_pose(frame, config, intrinsics):
    """Detect board and return (T_board_in_cam, n_corners, reproj_err) or (None, 0, None).

    Creates a fresh BoardDetector each time to avoid stateful legacy pattern issues.
    """
    if frame is None:
        return None, 0, None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = BoardDetector.from_config(config)
    detection = detector.detect(gray)
    if detection is None:
        return None, 0, None
    T, obj_pts, reproj_err = detector.compute_pose(detection, intrinsics)
    if T is None:
        return None, 0, None
    return T, len(detection.corners), reproj_err


def main():
    parser = argparse.ArgumentParser(description='Automated servo calibration')
    parser.add_argument('--jog-deg', type=float, default=8.0,
                        help='Jog step size in degrees (default: 8.0)')
    parser.add_argument('--n-steps', type=int, default=2,
                        help='Number of jog steps per direction per joint (default: 2)')
    parser.add_argument('--settle-s', type=float, default=0.5,
                        help='Settle time after each jog in seconds (default: 0.5)')
    parser.add_argument('--min-corners', type=int, default=8,
                        help='Minimum CharUco corners to accept a capture (default: 8)')
    parser.add_argument('--max-reproj', type=float, default=2.0,
                        help='Maximum reprojection error in px to accept (default: 2.0)')
    parser.add_argument('--save', action='store_true',
                        help='Save results to servo_offsets.yaml')
    args = parser.parse_args()

    print("=" * 60)
    print("  AUTO SERVO CALIBRATION")
    print("=" * 60)

    config = load_config()
    intrinsics = load_gripper_camera_intrinsics(config)
    bd = config.get('calibration_board', {})
    print(f"  Board: {bd.get('type', 'charuco')} {bd.get('cols')}x{bd.get('rows')} "
          f"({bd.get('square_size_mm')}mm squares)")
    print(f"  Camera intrinsics: fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f}")

    # Load current offsets
    offsets_dict = load_offsets()
    offsets_raw = np.array([
        offsets_dict.get(name, {}).get('zero_raw', 2048)
        for name in MOTOR_NAMES[:5]
    ], dtype=float)
    print(f"  Current offsets: {offsets_raw.astype(int).tolist()}")

    # Connect to arm
    print("\n--- Connecting to arm ---")
    arm = connect_robot(config, safe_mode=True)
    cap = open_gripper_camera(config)
    solver = Arm101IKSolver()
    print(f"  IK solver: signs={solver.signs.tolist()}")
    arm.enable_torque()
    print(f"  Torque enabled (safe mode)")

    # ---------------------------------------------------------------
    # Phase 1: Verify board visibility (try a few captures)
    # ---------------------------------------------------------------
    print("\n--- Phase 1: Verify board visibility ---")
    T0, n0, err0 = None, 0, None
    for attempt in range(5):
        frame = capture_frame(cap)
        T0, n0, err0 = detect_board_pose(frame, config, intrinsics)
        if T0 is not None:
            break
        time.sleep(0.3)
    if T0 is None:
        print(f"  Board NOT detected at starting position.")
        print(f"  Will try to find it by jogging joints...")
    else:
        board_dist = np.linalg.norm(T0[:3, 3])
        print(f"  Board detected: {n0} corners, reproj={err0:.2f}px, "
              f"dist={board_dist*1000:.0f}mm")

    # ---------------------------------------------------------------
    # Phase 2: Collect captures by jogging each joint
    # ---------------------------------------------------------------
    jog = args.jog_deg
    n_steps = args.n_steps
    print(f"\n--- Phase 2: Collecting captures (jog={jog}°, steps={n_steps}) ---")

    captures = []

    # Capture at starting position (if board was detected)
    raw = read_all_raw(arm)
    if (T0 is not None and n0 >= args.min_corners
            and err0 is not None and err0 < args.max_reproj):
        captures.append({'raw': raw, 'T_board_in_cam': T0})
        print(f"  [start] Captured: {n0} corners ✓")
    else:
        print(f"  [start] No board at start, will collect during jog")

    n_joints = 5
    total_moves = n_joints * n_steps * 4  # +steps, -steps(return), -steps, +steps(return)
    move_idx = 0

    for joint in range(n_joints):
        joint_name = MOTOR_NAMES[joint]

        # Sweep positive: +jog, +2*jog, +3*jog...
        for step in range(n_steps):
            move_idx += 1
            try:
                arm.jog_joint(joint, +1, step_deg=jog, speed=80)
            except ValueError as e:
                print(f"  [{move_idx}/{total_moves}] J{joint+1} {joint_name} "
                      f"+: SAFETY {e}")
                continue
            time.sleep(args.settle_s)

            frame = capture_frame(cap)
            raw = read_all_raw(arm)
            T, nc, err = detect_board_pose(frame, config, intrinsics)
            if (T is not None and nc >= args.min_corners
                    and err is not None and err < args.max_reproj):
                captures.append({'raw': raw, 'T_board_in_cam': T})
                print(f"  [{move_idx}/{total_moves}] J{joint+1} {joint_name} "
                      f"+: {nc} corners, reproj={err:.2f}px ✓ ({len(captures)})")
            else:
                status = f"{nc}c" if T is not None else "no board"
                err_s = f" e={err:.1f}" if err is not None else ""
                print(f"  [{move_idx}/{total_moves}] J{joint+1} {joint_name} "
                      f"+: skip ({status}{err_s})")

        # Return to start (no captures during return)
        for step in range(n_steps):
            move_idx += 1
            arm.jog_joint(joint, -1, step_deg=jog, speed=80)
            time.sleep(0.3)

        # Sweep negative: -jog, -2*jog, -3*jog...
        for step in range(n_steps):
            move_idx += 1
            try:
                arm.jog_joint(joint, -1, step_deg=jog, speed=80)
            except ValueError as e:
                print(f"  [{move_idx}/{total_moves}] J{joint+1} {joint_name} "
                      f"-: SAFETY {e}")
                continue
            time.sleep(args.settle_s)

            frame = capture_frame(cap)
            raw = read_all_raw(arm)
            T, nc, err = detect_board_pose(frame, config, intrinsics)
            if (T is not None and nc >= args.min_corners
                    and err is not None and err < args.max_reproj):
                captures.append({'raw': raw, 'T_board_in_cam': T})
                print(f"  [{move_idx}/{total_moves}] J{joint+1} {joint_name} "
                      f"-: {nc} corners, reproj={err:.2f}px ✓ ({len(captures)})")
            else:
                status = f"{nc}c" if T is not None else "no board"
                err_s = f" e={err:.1f}" if err is not None else ""
                print(f"  [{move_idx}/{total_moves}] J{joint+1} {joint_name} "
                      f"-: skip ({status}{err_s})")

        # Return to start
        for step in range(n_steps):
            move_idx += 1
            arm.jog_joint(joint, +1, step_deg=jog, speed=80)
            time.sleep(0.3)

    # Done collecting
    arm.disable_torque()
    cap.release()

    print(f"\n  Collection complete: {len(captures)} captures")

    if len(captures) < 6:
        print(f"  ERROR: Need at least 6 captures, got {len(captures)}.")
        print(f"  Try:")
        print(f"    - Reducing --jog-deg to keep board in view")
        print(f"    - Increasing --n-steps for more data points")
        print(f"    - Relaxing --max-reproj or --min-corners")
        arm.disconnect()
        return 1

    # ---------------------------------------------------------------
    # Phase 3: Run brute-force sign solver
    # ---------------------------------------------------------------
    print(f"\n--- Phase 3: Brute-force sign solver ({len(captures)} captures) ---")

    result = _brute_force_signs(captures, solver, offsets_raw, verbose=True)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Signs:      {result['signs_str']}")
    print(f"  Mean error: {result['mean_err_mm']:.2f}mm")
    print(f"  Offsets:    {result['offsets_raw'].astype(int).tolist()}")

    for i, name in enumerate(MOTOR_NAMES[:5]):
        s = '+' if result['signs'][i] > 0 else '-'
        print(f"    {name:<16}: sign={s}  offset={int(result['offsets_raw'][i])}")

    if result['ambiguous_joints']:
        amb = [MOTOR_NAMES[j] for j in result['ambiguous_joints']]
        print(f"\n  Ambiguous joints: {', '.join(amb)}")
        print(f"  The error spread between sign combos is too small.")
        print(f"  Try larger --jog-deg or more diverse arm positions.")

    if args.save:
        print(f"\n--- Saving calibration ---")
        save_calibration_results(
            result['signs'],
            result['offsets_raw'],
            result['T_cam_in_tcp'],
        )
        print(f"  Restart control panel to use new calibration.")
    else:
        print(f"\n  (Add --save to write results to servo_offsets.yaml)")

    arm.disconnect()
    return 0


if __name__ == '__main__':
    sys.exit(main())
