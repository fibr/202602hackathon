#!/usr/bin/env python3
"""GUI tool for servo calibration and manual hand-eye calibration.

Shows a live camera feed while you move the arm by hand (torque off).
Two modes:

  SERVO CALIBRATION (default):
    Move the arm to its URDF zero pose and press SPACE to save offsets.
    Live readout of raw servo positions and angles overlaid on camera.

  HAND-EYE CALIBRATION (--handeye):
    Move arm to diverse poses, press SPACE to capture each correspondence
    (joint angles via FK + yellow tape pixel detection). Press S to solve
    when enough points are collected.

Usage:
    ./run.sh scripts/calibration_gui.py                 # Servo calibration
    ./run.sh scripts/calibration_gui.py --handeye       # Hand-eye calibration

Pure algorithm helpers have been extracted to src/calibration/calib_helpers.py.
This script re-exports them for backward compatibility.
"""

import sys
import os
import time
import argparse
import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config_loader import load_config
from robot.lerobot_arm101 import LeRobotArm101
from kinematics.arm101_ik_solver import Arm101IKSolver

# Re-export all algorithm helpers from the canonical module so that any code
# still importing from this script continues to work unchanged.
from calibration.calib_helpers import (  # noqa: F401
    MOTOR_NAMES,
    ADDR_PRESENT_POSITION,
    YELLOW_HSV_LOW,
    YELLOW_HSV_HIGH,
    DEG_PER_POS,
    OFFSET_FILE,
    HANDEYE_FILE,
    read_all_raw,
    find_yellow_tape,
    draw_servo_overlay,
    draw_handeye_overlay,
    load_offsets,
    save_offsets,
    save_offsets_dict,
    solve_pnp,
    save_handeye_calibration,
    solve_and_save_handeye,
    joint_solve,
)


def main():
    parser = argparse.ArgumentParser(description="GUI calibration tool for arm101")
    parser.add_argument('--handeye', action='store_true',
                        help='Hand-eye calibration mode (default: servo calibration)')
    parser.add_argument('--camera', type=int, default=None,
                        help='Camera device index (default: from config)')
    args = parser.parse_args()

    config = load_config()

    # Camera setup
    cam_cfg = config.get('camera', {})
    cam_idx = args.camera if args.camera is not None else cam_cfg.get('device_index', 4)
    print(f"Opening camera /dev/video{cam_idx}...")
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {cam_idx}")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    autofocus = cam_cfg.get('autofocus', False)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
    # Flush initial frames
    for _ in range(10):
        cap.read()
        time.sleep(0.03)
    print("  Camera ready.")

    # Connect arm (no torque)
    ac = config.get('arm101', {})
    port = ac.get('port', '') or LeRobotArm101.find_port()
    print(f"Connecting to arm on {port}...")
    arm = LeRobotArm101(port=port, baudrate=ac.get('baudrate', 1_000_000),
                         motor_ids=ac.get('motor_ids', [1, 2, 3, 4, 5, 6]),
                         speed=ac.get('speed', 200))
    arm.connect()
    print("  Disabling torque — you can move the arm by hand.")
    arm.disable_torque()

    # FK solver
    solver = Arm101IKSolver()

    # Load camera intrinsics for handeye mode
    K, dist_coeffs = None, None
    if args.handeye:
        cam_yaml = os.path.join(os.path.dirname(__file__), '..', 'config', 'cameras.yaml')
        if os.path.exists(cam_yaml):
            with open(cam_yaml) as f:
                cdata = yaml.safe_load(f)
            for cname, cinfo in cdata.get('cameras', {}).items():
                if cinfo.get('device_index') == cam_idx:
                    intr = cinfo['intrinsics']
                    K = np.array(intr['camera_matrix'], dtype=np.float64)
                    dist_coeffs = np.array(intr['dist_coeffs'], dtype=np.float64)
                    print(f"  Intrinsics from {cname}: fx={K[0,0]:.1f}")
                    break
        if K is None:
            K = np.array([[554.3, 0, 320], [0, 554.3, 240], [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.zeros(5, dtype=np.float64)
            print("  Using default intrinsics (estimated)")

    # State
    offsets = load_offsets()
    status_msg = ""
    status_time = 0
    # Hand-eye state
    pts_3d_robot = []
    pts_2d = []
    raw_positions_list = []  # raw servo positions per capture (for joint solve)

    mode_name = "Hand-Eye Calibration" if args.handeye else "Servo Calibration"
    win = f"arm101 {mode_name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print(f"\n=== {mode_name} GUI ===")
    print("Torque is OFF. Move the arm by hand.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Read servo state
            raw_positions = read_all_raw(arm)
            angles = arm.get_angles()

            # FK
            tcp_pos = None
            if angles is not None:
                try:
                    tcp_pos, _ = solver.forward_kin(np.array(angles[:5]))
                except Exception:
                    pass

            if args.handeye:
                # Yellow tape detection
                cx, cy, mask = find_yellow_tape(frame)
                draw_handeye_overlay(frame, tcp_pos, (cx, cy), len(pts_2d))

                # Show small mask in corner
                mask_small = cv2.resize(mask, (160, 120))
                mask_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
                frame[0:120, frame.shape[1]-160:frame.shape[1]] = mask_bgr
            else:
                draw_servo_overlay(frame, raw_positions, offsets, angles)

            # Status message (fades after 3s)
            if status_msg and time.time() - status_time < 3.0:
                cv2.putText(frame, status_msg, (10, frame.shape[0] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow(win, frame)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # ESC
                break

            try:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            if key == ord(' '):
                if args.handeye:
                    # Capture hand-eye point (raw positions + pixel)
                    if tcp_pos is not None and cx is not None:
                        pts_3d_robot.append(tcp_pos.copy())
                        pts_2d.append([float(cx), float(cy)])
                        raw_positions_list.append(dict(raw_positions))
                        status_msg = f"Captured #{len(pts_2d)}: TCP=[{tcp_pos[0]:.0f},{tcp_pos[1]:.0f},{tcp_pos[2]:.0f}] px=({cx},{cy})"
                        status_time = time.time()
                        print(f"  {status_msg}")
                    else:
                        status_msg = "SKIP: no TCP or no yellow tape"
                        status_time = time.time()
                else:
                    # Save servo offsets
                    offsets = save_offsets(raw_positions)
                    status_msg = f"Saved offsets to {os.path.basename(OFFSET_FILE)}"
                    status_time = time.time()
                    print(f"  {status_msg}")
                    for mid, pos in sorted(raw_positions.items()):
                        name = MOTOR_NAMES[mid - 1]
                        print(f"    {name}: raw={pos}")

            elif key == ord('s') and args.handeye:
                # Joint solve: optimize servo offsets + camera extrinsics together
                if len(pts_2d) < 6:
                    status_msg = f"Need >= 6 points for joint solve (have {len(pts_2d)})"
                    status_time = time.time()
                else:
                    print(f"\nJoint solve: {len(pts_2d)} points, optimizing offsets + extrinsics...")
                    opt_offsets, T_c2b = joint_solve(
                        raw_positions_list, pts_2d, K, dist_coeffs, solver)
                    if opt_offsets is not None:
                        # Save both servo offsets and hand-eye calibration
                        save_offsets_dict(opt_offsets)
                        save_handeye_calibration(T_c2b, HANDEYE_FILE)
                        offsets = load_offsets()  # reload for display
                        status_msg = "Joint solve OK — saved offsets + extrinsics"
                    else:
                        status_msg = "Joint solve FAILED"
                    status_time = time.time()
                    print(f"  {status_msg}")

            elif key == ord('u') and args.handeye:
                # Undo last capture
                if pts_2d:
                    pts_2d.pop()
                    pts_3d_robot.pop()
                    raw_positions_list.pop()
                    status_msg = f"Undone — {len(pts_2d)} points remaining"
                    status_time = time.time()

            elif key == ord('r') and not args.handeye:
                # Reload offsets
                offsets = load_offsets()
                status_msg = "Reloaded offsets from file"
                status_time = time.time()

    except KeyboardInterrupt:
        pass
    finally:
        arm.disconnect()
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
