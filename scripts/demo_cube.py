#!/usr/bin/env python3
"""Demo: trace the 8 corners of a cube using local IK + joint control.

The arm visits all 8 corners of a cube in 3D space, moving along edges.
Uses local IK to convert Cartesian targets to joint angles.

Usage: ./run.sh scripts/demo_cube.py [--size 80] [--center 350,0,350]
"""

import sys
import os
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config_loader import load_config
from robot import DobotNova5
from kinematics import IKSolver


# Cube corner visit order — traces edges, visiting all 8 corners.
# Hamiltonian path on a cube graph (each corner visited once):
#   0 -> 1 -> 3 -> 2 -> 6 -> 7 -> 5 -> 4
# Then back to 0.
# Corner indices: bits encode (x, y, z) offsets from cube origin.
CORNER_ORDER = [
    (0, 0, 0),  # 0: near-left-bottom
    (1, 0, 0),  # 1: far-left-bottom
    (1, 1, 0),  # 3: far-right-bottom
    (0, 1, 0),  # 2: near-right-bottom
    (0, 1, 1),  # 6: near-right-top
    (1, 1, 1),  # 7: far-right-top
    (1, 0, 1),  # 5: far-left-top
    (0, 0, 1),  # 4: near-left-top
]


def main():
    parser = argparse.ArgumentParser(description="Trace cube corners with robot arm")
    parser.add_argument('--size', type=float, default=80.0,
                        help='Cube edge length in mm (default: 80)')
    parser.add_argument('--center', type=str, default='350,0,350',
                        help='Cube center x,y,z in mm (default: 350,0,350)')
    parser.add_argument('--speed', type=int, default=30,
                        help='Robot speed percent (default: 30)')
    parser.add_argument('--loops', type=int, default=2,
                        help='Number of loops around the cube (default: 2)')
    parser.add_argument('--rx', type=float, default=180.0,
                        help='Gripper rx orientation in degrees (default: 180, pointing down)')
    args = parser.parse_args()

    center = np.array([float(x) for x in args.center.split(',')])
    half = args.size / 2.0

    # Generate corner positions
    corners = []
    for cx, cy, cz in CORNER_ORDER:
        pos = center + np.array([
            (cx - 0.5) * args.size,
            (cy - 0.5) * args.size,
            (cz - 0.5) * args.size,
        ])
        corners.append(pos)

    print(f"Cube demo: center={center}, size={args.size}mm, {len(corners)} corners")
    print(f"  Corners span: X=[{center[0]-half:.0f},{center[0]+half:.0f}] "
          f"Y=[{center[1]-half:.0f},{center[1]+half:.0f}] "
          f"Z=[{center[2]-half:.0f},{center[2]+half:.0f}]")

    # Orientation: gripper pointing down (or user-specified)
    rpy = np.array([args.rx, 0.0, 0.0])

    # Load config and init
    config = load_config()
    rc = config.get('robot', {})
    gc = config.get('gripper', {})

    ik = IKSolver(tool_length_mm=gc.get('tool_length_mm', 100.0))

    # Pre-solve all IK to check feasibility before moving
    print("\nPre-solving IK for all corners...")
    seed = None
    corner_joints = []
    for i, pos in enumerate(corners):
        joints = ik.solve_ik(pos, rpy, seed_joints_deg=seed)
        if joints is None:
            print(f"  FAIL: corner {i} at {pos} — IK has no solution!")
            print(f"  Try adjusting --center or --size to stay within reach.")
            return
        corner_joints.append(joints)
        max_change = np.max(np.abs(joints - seed)) if seed is not None else 0
        print(f"  Corner {i}: pos=({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f}) "
              f"joints=[{','.join(f'{j:.1f}' for j in joints)}] "
              f"max_change={max_change:.1f}deg")
        seed = joints

    print(f"\nAll {len(corners)} corners reachable. Connecting to robot...")

    robot = DobotNova5(
        ip=rc.get('ip', '192.168.5.1'),
        dashboard_port=rc.get('dashboard_port', 29999),
    )

    try:
        robot.connect()
        robot.clear_error()
        robot.enable()
        robot.set_speed(args.speed)
        time.sleep(1)

        # Get current joints for initial seed
        current = robot.get_joint_angles()
        print(f"Current joints: [{','.join(f'{j:.1f}' for j in current)}]")

        # Move to first corner (might be a big move, use joint motion)
        print(f"\nMoving to start corner...")
        first_joints = ik.solve_ik(corners[0], rpy, seed_joints_deg=current)
        if not robot.movj_joints(*first_joints):
            print("ERROR: Failed to move to start corner")
            return

        # Trace the cube
        for loop in range(args.loops):
            print(f"\n=== Loop {loop + 1}/{args.loops} ===")
            prev_joints = first_joints

            for i, (pos, pre_joints) in enumerate(zip(corners, corner_joints)):
                # Re-solve with current seed for continuity
                joints = ik.solve_ik(pos, rpy, seed_joints_deg=prev_joints)
                if joints is None:
                    print(f"  IK failed at corner {i}, skipping")
                    continue

                label = f"C{i}({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f})"
                print(f"  -> {label}", end="", flush=True)

                ok = robot.movj_joints(*joints)
                if ok:
                    print(" OK")
                else:
                    print(" FAIL")
                    robot.clear_error()
                    robot.enable()
                    time.sleep(1)

                prev_joints = joints

            # Return to first corner to close the loop
            joints = ik.solve_ik(corners[0], rpy, seed_joints_deg=prev_joints)
            if joints is not None:
                print(f"  -> Close loop", end="", flush=True)
                ok = robot.movj_joints(*joints)
                print(" OK" if ok else " FAIL")

        print("\nCube demo complete!")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        robot.disable()
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
