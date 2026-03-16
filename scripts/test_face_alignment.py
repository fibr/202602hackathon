#!/usr/bin/env python3
"""Test cube face alignment — validate gripper-to-cube yaw alignment.

Tests the CubeFaceAligner's ability to compute optimal wrist_roll (J5) for
aligning gripper jaws with cube edges.  Can run offline (pure math) or
with the real robot + camera.

Usage:
    ./run.sh scripts/test_face_alignment.py                     # Offline math test
    ./run.sh scripts/test_face_alignment.py --with-robot         # Live camera + robot
    ./run.sh scripts/test_face_alignment.py --sweep              # Sweep all yaw angles
    ./run.sh scripts/test_face_alignment.py --yaw 15.3           # Single yaw test
    ./run.sh scripts/test_face_alignment.py --yaw 15.3 --j5 -20  # Custom J5 start
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cube_face_aligner import CubeFaceAligner, batch_alignment_test


def run_offline_test(aligner, yaw_values=None, j5_start=0.0):
    """Run alignment tests for various yaw angles without hardware."""
    if yaw_values is None:
        # Test the full range of possible cube yaw detections
        yaw_values = list(range(-45, 46, 5))

    print(f'\n=== Offline Alignment Test (J5 start = {j5_start:.1f} deg) ===\n')
    print(batch_alignment_test(aligner, yaw_values, current_j5=j5_start))

    # Verify key properties
    print(f'\n=== Verification ===')
    errors = 0

    # 1. Zero yaw should produce zero (or near-zero) delta
    plan = aligner.compute_alignment(0.0, j5_start)
    if plan.valid and abs(plan.delta_deg) > aligner.deadband_deg:
        # With mount_angle=0, zero yaw should give small delta
        if abs(aligner.mount_angle_deg) < 1.0:
            print(f'  WARN: zero yaw gives delta={plan.delta_deg:.1f} deg')
    print(f'  Zero yaw: delta={plan.delta_deg:+.1f} deg  (face {plan.face_index})')

    # 2. Yaw +45 and -45 should give same result (due to 90-deg symmetry)
    p1 = aligner.compute_alignment(45.0, j5_start)
    p2 = aligner.compute_alignment(-45.0, j5_start)
    diff = abs(abs(p1.delta_deg) - abs(p2.delta_deg))
    if diff > 1.0:
        print(f'  WARN: +45 and -45 yaw give different deltas: '
              f'{p1.delta_deg:+.1f} vs {p2.delta_deg:+.1f}')
    print(f'  Symmetry: yaw=+45 delta={p1.delta_deg:+.1f}, '
          f'yaw=-45 delta={p2.delta_deg:+.1f}')

    # 3. All deltas should be <= 45 degrees (optimal face selection)
    max_delta = 0.0
    for yaw in range(-45, 46):
        plan = aligner.compute_alignment(float(yaw), j5_start)
        if plan.valid:
            max_delta = max(max_delta, abs(plan.delta_deg))
    within_45 = max_delta <= 46.0  # small margin for mount_angle
    print(f'  Max delta across all yaws: {max_delta:.1f} deg '
          f'({"OK" if within_45 else "FAIL"} — should be <= 45)')
    if not within_45:
        errors += 1

    # 4. Test with various starting J5 positions
    print(f'\n=== J5 Start Position Sweep ===\n')
    for j5 in [-120, -60, 0, 60, 120]:
        plan = aligner.compute_alignment(30.0, float(j5))
        print(f'  J5={j5:>5.0f}, yaw=30: J5_new={plan.selected_j5_deg:>7.1f}, '
              f'delta={plan.delta_deg:>+6.1f}, face={plan.face_index}')

    return errors


def run_sweep_test(aligner, j5_values=None):
    """Sweep all yaw x J5 combinations to find failure cases."""
    if j5_values is None:
        j5_values = [-120, -90, -60, -30, 0, 30, 60, 90, 120]

    yaw_values = list(range(-45, 46, 5))

    print(f'\n=== Sweep Test: {len(yaw_values)} yaws x {len(j5_values)} J5 starts ===\n')
    fail_count = 0
    big_delta_count = 0
    limit_constrained = 0  # delta > 45 but caused by joint limits
    total = 0

    for j5 in j5_values:
        for yaw in yaw_values:
            plan = aligner.compute_alignment(float(yaw), float(j5))
            total += 1
            if not plan.valid:
                fail_count += 1
            elif abs(plan.delta_deg) > 45.0:
                # Check if this is caused by joint limits (the ideal candidate
                # would be outside limits, so we had to pick a further one)
                ideal = j5 + plan.robot_yaw_deg
                if ideal < aligner.j5_min_deg or ideal > aligner.j5_max_deg:
                    limit_constrained += 1
                else:
                    big_delta_count += 1

    print(f'  Total tests:       {total}')
    print(f'  Valid plans:       {total - fail_count}')
    print(f'  Invalid plans:     {fail_count}')
    print(f'  Delta > 45 (bug):  {big_delta_count}')
    print(f'  Delta > 45 (limit):{limit_constrained} (expected, near joint limits)')
    real_issues = fail_count + big_delta_count
    print(f'  Result:            {"PASS" if real_issues == 0 else f"FAIL ({real_issues} issues)"}')
    return real_issues


def run_with_robot(aligner, config):
    """Live test: detect cubes, align gripper, repeat."""
    import cv2
    from config_loader import connect_robot
    from vision import create_camera
    from vision.green_cube_detector import detect_green_cubes, annotate_frame
    from calibration import CoordinateTransform
    from config_loader import config_path

    print('\nStarting camera...')
    camera = create_camera(config)
    camera.start()
    # Warm up
    for _ in range(15):
        camera.get_frames()
        time.sleep(0.03)

    print('Connecting robot...')
    robot = connect_robot(config)

    # Check for gripper camera
    gc = config.get('gripper_camera', {})
    gripper_cam_idx = gc.get('device_index', 8)
    use_gripper_cam = True

    print(f'Using gripper camera (device {gripper_cam_idx})')
    print('Controls:')
    print('  a = auto-align to detected cube')
    print('  1-4 = select face 0-3 manually')
    print('  h = home position')
    print('  q/ESC = quit')
    print()

    cv2.namedWindow('Face Alignment', cv2.WINDOW_NORMAL)

    try:
        # Try opening gripper camera
        gcap = cv2.VideoCapture(gripper_cam_idx, cv2.CAP_V4L2)
        if gcap.isOpened():
            gcap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            gcap.set(cv2.CAP_PROP_FRAME_WIDTH, gc.get('width', 640))
            gcap.set(cv2.CAP_PROP_FRAME_HEIGHT, gc.get('height', 480))
            # Warm up gripper cam
            for _ in range(10):
                gcap.read()
        else:
            print('WARNING: Gripper camera not available, using overview cam')
            use_gripper_cam = False
            gcap = None

        while True:
            # Read from chosen camera
            if use_gripper_cam and gcap is not None:
                ret, frame = gcap.read()
                if not ret or frame is None:
                    continue
            else:
                color, _, _ = camera.get_frames()
                if color is None:
                    continue
                frame = color

            # Detect cubes
            cubes, info = detect_green_cubes(frame, min_area=100 if use_gripper_cam else 300)
            display = annotate_frame(frame, cubes, target_idx=0 if cubes else -1)

            # Show current J5
            angles = robot.get_angles()
            j5 = angles[4] if angles else 0.0
            h, w = display.shape[:2]
            cv2.putText(display, f'J5={j5:.1f} deg', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if cubes:
                cube = cubes[0]
                cv2.putText(display, f'Cube yaw={cube.yaw_deg:.1f} deg',
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

                # Compute alignment plan (display only)
                plan = aligner.compute_alignment(cube.yaw_deg, j5)
                if plan.valid:
                    cv2.putText(display,
                                f'Plan: J5->{plan.selected_j5_deg:.1f} '
                                f'(delta={plan.delta_deg:+.1f})',
                                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 200, 255), 1)

                    # Draw alignment indicator
                    # Green circle if aligned, red if not
                    aligned = abs(plan.delta_deg) < aligner.deadband_deg
                    ind_color = (0, 255, 0) if aligned else (0, 0, 255)
                    cv2.circle(display, (w - 30, 30), 15, ind_color, -1)
                    cv2.putText(display,
                                'ALIGNED' if aligned else f'{plan.delta_deg:+.0f}' + chr(176),
                                (w - 100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                ind_color, 1)
            else:
                cv2.putText(display, 'No cube detected', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow('Face Alignment', display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('a') and cubes:
                # Auto-align
                plan = aligner.align_robot(robot, cubes[0].yaw_deg, speed=80)
                print(f'  Aligned: {plan.summary()}')
                time.sleep(0.5)
            elif key == ord('h'):
                # Home
                home = [0.0, 0.0, 90.0, 90.0, 0.0, 0.0]
                robot.move_joints(home, speed=80)
                print('  Homed')
                time.sleep(1.0)
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')] and cubes:
                # Select specific face
                face_idx = key - ord('1')
                plan = aligner.compute_alignment(cubes[0].yaw_deg, j5)
                if plan.valid and face_idx < len(plan.candidate_j5_degs):
                    target_j5 = plan.candidate_j5_degs[face_idx]
                    if aligner.j5_min_deg <= target_j5 <= aligner.j5_max_deg:
                        cmd = list(angles)
                        cmd[4] = target_j5
                        robot.move_joints(cmd, speed=80)
                        print(f'  Face {face_idx}: J5 -> {target_j5:.1f}')
                        time.sleep(0.5)
                    else:
                        print(f'  Face {face_idx}: J5={target_j5:.1f} out of limits')

    except KeyboardInterrupt:
        print('\nStopped.')
    finally:
        if gcap is not None:
            gcap.release()
        camera.stop()
        cv2.destroyAllWindows()
        try:
            robot.disconnect()
        except Exception:
            pass
        print('Done.')


def main():
    parser = argparse.ArgumentParser(description='Test cube face alignment')
    parser.add_argument('--with-robot', action='store_true',
                        help='Live test with robot + camera')
    parser.add_argument('--sweep', action='store_true',
                        help='Sweep all yaw x J5 combinations')
    parser.add_argument('--yaw', type=float, default=None,
                        help='Test a specific yaw angle (degrees)')
    parser.add_argument('--j5', type=float, default=0.0,
                        help='Starting J5 angle (degrees, default: 0)')
    parser.add_argument('--mount-angle', type=float, default=0.0,
                        help='Camera mount angle (degrees, default: 0)')
    args = parser.parse_args()

    if args.with_robot:
        from config_loader import load_config
        config = load_config()
        aligner = CubeFaceAligner.from_config(config)
        run_with_robot(aligner, config)
    else:
        aligner = CubeFaceAligner(mount_angle_deg=args.mount_angle)
        print(f'CubeFaceAligner: mount_angle={aligner.mount_angle_deg}' + chr(176)
              + f', deadband={aligner.deadband_deg}' + chr(176))

        if args.yaw is not None:
            # Single yaw test
            plan = aligner.compute_alignment(args.yaw, args.j5)
            print(f'\nInput: yaw={args.yaw}' + chr(176) + f', J5={args.j5}' + chr(176))
            print(f'Plan:  {plan.summary()}')
            print(f'  Robot yaw:    {plan.robot_yaw_deg:.1f}' + chr(176))
            print(f'  Candidates:   {[f"{c:.1f}" for c in plan.candidate_j5_degs]}')
            print(f'  Selected J5:  {plan.selected_j5_deg:.1f}' + chr(176))
            print(f'  Delta:        {plan.delta_deg:+.1f}' + chr(176))
            print(f'  Face index:   {plan.face_index}')
            print(f'  Status:       {plan.status}')
        elif args.sweep:
            errors = run_sweep_test(aligner)
            print(f'\nSweep result: {"PASS" if errors == 0 else f"FAIL ({errors} issues)"}')
        else:
            errors = run_offline_test(aligner, j5_start=args.j5)
            print(f'\nResult: {"PASS" if errors == 0 else f"FAIL ({errors} issues)"}')


if __name__ == '__main__':
    main()
