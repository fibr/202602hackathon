#!/usr/bin/env python3
"""Verify servo signs for J3 (elbow_flex), J4 (wrist_flex), J5 (wrist_roll).

Strategy:
1. Read the arm's natural resting position under gravity (low torque safe mode)
2. Compute FK with all 8 sign combinations for J3/J4/J5
3. For each joint, jog it in the gravity-friendly direction (large command),
   read back the actual position, compute FK, and check consistency
4. Use FK sanity constraints to eliminate wrong sign combinations:
   - The arm must be physically reachable (within workspace)
   - Z should be plausible (arm doesn't clip through base)
   - Position should match rough observation

Usage:
  ./run.sh scripts/verify_joint_signs.py
"""

import sys
import os
import time
import itertools
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config_loader import load_config, connect_robot, acquire_rig_lock
from kinematics.arm101_ik_solver import Arm101IKSolver


def compute_fk_with_signs(solver, motor_angles_5, signs):
    """Compute FK with specific joint signs."""
    orig = solver.signs.copy()
    solver.signs[:] = signs
    pos_mm, rpy_deg = solver.forward_kin(np.array(motor_angles_5, dtype=float))
    solver.signs[:] = orig
    return pos_mm, rpy_deg


def main():
    print("=" * 60)
    print("  J3/J4/J5 Servo Sign Verification (v2)")
    print("=" * 60)

    config = load_config()
    solver = Arm101IKSolver()

    names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
    print(f"\nCurrent signs: {solver.signs.tolist()}")
    for i, n in enumerate(names):
        print(f"  J{i+1} ({n}): sign={solver.signs[i]:+.0f}")

    # Acquire rig lock and connect
    print("\nAcquiring rig lock...")
    lock = acquire_rig_lock(holder="verify_joint_signs", force=True)
    print("Connecting to arm101 in safe mode...")
    robot = connect_robot(config, safe_mode=True)

    try:
        # ====================================================================
        # PHASE 1: Read resting position and analyze all sign combinations
        # ====================================================================
        print(f"\n{'='*60}")
        print("  PHASE 1: Resting position analysis")
        print(f"{'='*60}")

        angles_rest = robot.get_angles()
        raw_rest = robot.read_all_positions()
        if angles_rest is None:
            print("ERROR: Could not read joint angles")
            return 1

        print(f"\nResting motor angles: {[f'{a:.1f}' for a in angles_rest]}")
        print(f"Resting raw positions: {raw_rest}")

        # J1=-1, J2=+1 are fixed (verified). Test all combos of J3, J4, J5.
        fixed_signs = [-1.0, +1.0]  # J1, J2
        test_combos = list(itertools.product([-1.0, +1.0], repeat=3))

        print(f"\nFK for all 8 sign combinations (J1=-1, J2=+1 fixed):")
        print(f"{'J3':>4} {'J4':>4} {'J5':>4} | {'X':>8} {'Y':>8} {'Z':>8} | {'Rx':>8} {'Ry':>8} {'Rz':>8}")
        print("-" * 70)

        fk_results = {}
        for combo in test_combos:
            signs = np.array(fixed_signs + list(combo))
            pos, rpy = compute_fk_with_signs(solver, angles_rest[:5], signs)
            fk_results[combo] = (pos, rpy)
            print(f"{combo[0]:+4.0f} {combo[1]:+4.0f} {combo[2]:+4.0f} | "
                  f"{pos[0]:8.1f} {pos[1]:8.1f} {pos[2]:8.1f} | "
                  f"{rpy[0]:8.1f} {rpy[1]:8.1f} {rpy[2]:8.1f}")

        # ====================================================================
        # PHASE 2: Move each joint and measure FK change
        # ====================================================================
        print(f"\n{'='*60}")
        print("  PHASE 2: Individual joint movement tests")
        print(f"{'='*60}")

        # Strategy: command each joint with large angles in both directions.
        # Even in safe mode, some movement should occur.
        # We'll read the actual position to see what really happened.

        # First, try to go to home position
        home = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        print(f"\nCommanding home position {home}...")
        robot.set_z_safety(enabled=False)  # Disable Z check for testing
        robot.move_joints(home, speed=80)
        time.sleep(3.0)

        angles_home = robot.get_angles()
        raw_home = robot.read_all_positions()
        print(f"Home motor angles: {[f'{a:.1f}' for a in angles_home]}")
        print(f"Home raw positions: {raw_home}")

        fk_home, rpy_home = solver.forward_kin(np.array(angles_home[:5], dtype=float))
        print(f"FK at home: ({fk_home[0]:.1f}, {fk_home[1]:.1f}, {fk_home[2]:.1f}) mm")

        # Test each joint with LARGE movements in both directions
        test_joints = {
            2: ('elbow_flex', 80.0),   # J3: try ±80° to overcome gravity
            3: ('wrist_flex', 80.0),   # J4: try ±80°
            4: ('wrist_roll', 60.0),   # J5: try ±60° (lighter, should move easier)
        }

        joint_results = {}

        for jidx, (jname, delta) in test_joints.items():
            jnum = jidx + 1
            print(f"\n--- J{jnum} ({jname}): Testing with ±{delta}° ---")

            # Return to home between tests
            robot.move_joints(home, speed=80)
            time.sleep(2.0)
            angles_baseline = robot.get_angles()
            raw_baseline = robot.read_all_positions()

            # Move POSITIVE
            cmd_pos = list(home)
            cmd_pos[jidx] = +delta
            print(f"  Commanding J{jnum} → +{delta}° ...")
            robot.move_joints(cmd_pos, speed=100)
            time.sleep(2.5)
            angles_pos = robot.get_angles()
            raw_pos = robot.read_all_positions()
            actual_delta_pos = angles_pos[jidx] - angles_baseline[jidx]
            raw_delta_pos = raw_pos[jidx] - raw_baseline[jidx]
            print(f"  Baseline: motor={angles_baseline[jidx]:.1f}° raw={raw_baseline[jidx]}")
            print(f"  After +{delta}°: motor={angles_pos[jidx]:.1f}° raw={raw_pos[jidx]}")
            print(f"  Actual delta: {actual_delta_pos:.1f}° (raw: {raw_delta_pos})")

            # Move NEGATIVE
            cmd_neg = list(home)
            cmd_neg[jidx] = -delta
            print(f"  Commanding J{jnum} → -{delta}° ...")
            robot.move_joints(cmd_neg, speed=100)
            time.sleep(2.5)
            angles_neg = robot.get_angles()
            raw_neg = robot.read_all_positions()
            actual_delta_neg = angles_neg[jidx] - angles_baseline[jidx]
            raw_delta_neg = raw_neg[jidx] - raw_baseline[jidx]
            print(f"  After -{delta}°: motor={angles_neg[jidx]:.1f}° raw={raw_neg[jidx]}")
            print(f"  Actual delta: {actual_delta_neg:.1f}° (raw: {raw_delta_neg})")

            # Total range of motion achieved
            total_motor_range = angles_pos[jidx] - angles_neg[jidx]
            total_raw_range = raw_pos[jidx] - raw_neg[jidx]
            print(f"  Total range achieved: {total_motor_range:.1f}° (raw: {total_raw_range})")

            # Compute FK at both extremes
            fk_pos_val, rpy_pos_val = solver.forward_kin(
                np.array(angles_pos[:5], dtype=float))
            fk_neg_val, rpy_neg_val = solver.forward_kin(
                np.array(angles_neg[:5], dtype=float))

            print(f"  FK at +pos: ({fk_pos_val[0]:.1f}, {fk_pos_val[1]:.1f}, {fk_pos_val[2]:.1f}) mm")
            print(f"  FK at -pos: ({fk_neg_val[0]:.1f}, {fk_neg_val[1]:.1f}, {fk_neg_val[2]:.1f}) mm")
            print(f"  FK delta: dX={fk_pos_val[0]-fk_neg_val[0]:.1f} "
                  f"dY={fk_pos_val[1]-fk_neg_val[1]:.1f} "
                  f"dZ={fk_pos_val[2]-fk_neg_val[2]:.1f}")

            # Also compute FK with FLIPPED sign for this joint
            flipped_signs = solver.signs.copy()
            flipped_signs[jidx] *= -1
            fk_pos_flip, _ = compute_fk_with_signs(solver, angles_pos[:5], flipped_signs)
            fk_neg_flip, _ = compute_fk_with_signs(solver, angles_neg[:5], flipped_signs)
            print(f"  FK (flipped sign) at +pos: ({fk_pos_flip[0]:.1f}, {fk_pos_flip[1]:.1f}, {fk_pos_flip[2]:.1f})")
            print(f"  FK (flipped sign) at -pos: ({fk_neg_flip[0]:.1f}, {fk_neg_flip[1]:.1f}, {fk_neg_flip[2]:.1f})")
            print(f"  FK (flipped) delta: dX={fk_pos_flip[0]-fk_neg_flip[0]:.1f} "
                  f"dY={fk_pos_flip[1]-fk_neg_flip[1]:.1f} "
                  f"dZ={fk_pos_flip[2]-fk_neg_flip[2]:.1f}")

            joint_results[jname] = {
                'motor_range': total_motor_range,
                'raw_range': total_raw_range,
                'angles_pos': angles_pos[:5],
                'angles_neg': angles_neg[:5],
                'fk_pos': fk_pos_val.tolist(),
                'fk_neg': fk_neg_val.tolist(),
                'fk_pos_flipped': fk_pos_flip.tolist(),
                'fk_neg_flipped': fk_neg_flip.tolist(),
            }

        # ====================================================================
        # PHASE 3: Cross-validation with gravity direction
        # ====================================================================
        print(f"\n{'='*60}")
        print("  PHASE 3: Gravity consistency check")
        print(f"{'='*60}")

        # When motors are weak (safe mode), joints sag under gravity.
        # The direction of sag depends on the joint configuration.
        # If the arm droops, FK should show the gripper going DOWN (decreasing Z).

        # Disable torque, let arm rest under gravity, read position
        print("\nDisabling torque to let arm rest under gravity...")
        robot.disable_torque()
        time.sleep(3.0)  # Let arm settle

        angles_gravity = robot.get_angles()
        raw_gravity = robot.read_all_positions()
        print(f"Gravity rest angles: {[f'{a:.1f}' for a in angles_gravity]}")
        print(f"Gravity rest raw: {raw_gravity}")

        fk_gravity, rpy_gravity = solver.forward_kin(
            np.array(angles_gravity[:5], dtype=float))
        print(f"FK at gravity rest (current signs): "
              f"({fk_gravity[0]:.1f}, {fk_gravity[1]:.1f}, {fk_gravity[2]:.1f}) mm")

        # Try all 8 sign combos at gravity-rest position
        print(f"\nFK at gravity rest for all sign combos:")
        print(f"{'J3':>4} {'J4':>4} {'J5':>4} | {'X':>8} {'Y':>8} {'Z':>8} | "
              f"{'dist':>8} {'Notes'}")
        print("-" * 75)

        for combo in test_combos:
            signs = np.array(fixed_signs + list(combo))
            pos, rpy = compute_fk_with_signs(solver, angles_gravity[:5], signs)
            dist = np.linalg.norm(pos)
            # Sanity flags
            notes = []
            if pos[2] < -100:
                notes.append("Z<<0 IMPOSSIBLE")
            if pos[2] > 400:
                notes.append("Z>>400 UNLIKELY")
            if dist > 500:
                notes.append("TOO FAR")
            if dist < 20:
                notes.append("TOO CLOSE")
            is_current = all(
                signs[i+2] == solver.signs[i+2] for i in range(3))
            if is_current:
                notes.append("← CURRENT")
            print(f"{combo[0]:+4.0f} {combo[1]:+4.0f} {combo[2]:+4.0f} | "
                  f"{pos[0]:8.1f} {pos[1]:8.1f} {pos[2]:8.1f} | "
                  f"{dist:8.1f}  {'  '.join(notes)}")

        # Re-enable torque and return home
        print("\nRe-enabling torque and returning home...")
        robot.enable_torque()
        time.sleep(0.5)
        robot.move_joints(home, speed=80)
        time.sleep(2.0)

        # ====================================================================
        # SUMMARY
        # ====================================================================
        print(f"\n{'='*60}")
        print("  FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"\nCurrent signs: J1={solver.signs[0]:+.0f} J2={solver.signs[1]:+.0f} "
              f"J3={solver.signs[2]:+.0f} J4={solver.signs[3]:+.0f} J5={solver.signs[4]:+.0f}")
        print(f"\nGravity rest position: {[f'{a:.1f}' for a in angles_gravity]}")
        print(f"FK (current signs): "
              f"({fk_gravity[0]:.1f}, {fk_gravity[1]:.1f}, {fk_gravity[2]:.1f}) mm")

        for jname, res in joint_results.items():
            print(f"\n{jname}:")
            print(f"  Motor range achieved: {res['motor_range']:.1f}° "
                  f"(raw: {res['raw_range']})")
            if abs(res['motor_range']) < 5:
                print(f"  ⚠ Motor barely moved — cannot determine sign from motion")
            else:
                print(f"  FK current sign: {res['fk_pos']} → {res['fk_neg']}")
                print(f"  FK flipped sign: {res['fk_pos_flipped']} → {res['fk_neg_flipped']}")

        return 0

    finally:
        try:
            robot.enable_torque()
            robot.move_joints([0, 0, 0, 0, 0, 0], speed=80)
            time.sleep(1.0)
        except Exception:
            pass
        robot.disconnect()
        lock.release()
        print("Done.")


if __name__ == '__main__':
    sys.exit(main())
