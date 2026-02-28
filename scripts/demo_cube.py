#!/usr/bin/env python3
"""Demo: move to random poses in 3D space using local IK + smooth trajectories.

Generates random Cartesian targets within a spherical reach envelope,
solves IK, and executes smooth quintic trajectories between them.
Can also trace cube corners as a structured pattern.

Usage:
  ./run.sh scripts/demo_cube.py                     # random poses (default)
  ./run.sh scripts/demo_cube.py --mode cube          # trace cube corners
  ./run.sh scripts/demo_cube.py --n 20 --reach 600   # 20 random poses, 600mm max reach
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
from planner.trajectory import execute_trajectory


# Cube corner visit order (Hamiltonian path)
CUBE_CORNERS = [
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 1, 1), (1, 1, 1), (1, 0, 1), (0, 0, 1),
]


def generate_cube_targets(center, size, rpy):
    """Generate cube corner positions and orientations."""
    targets = []
    for cx, cy, cz in CUBE_CORNERS:
        pos = center + np.array([
            (cx - 0.5) * size,
            (cy - 0.5) * size,
            (cz - 0.5) * size,
        ])
        targets.append((pos, rpy.copy()))
    return targets


def generate_random_targets(n, max_reach_mm, min_z_mm, ik, rpy_base,
                            rpy_vary_deg=30.0):
    """Generate random reachable poses within a sphere of max_reach_mm.

    Rejection-samples: generates random positions in a sphere, checks they're
    above min_z_mm and IK-solvable, then keeps up to n targets.
    """
    targets = []
    attempts = 0
    max_attempts = n * 20
    seed = None

    while len(targets) < n and attempts < max_attempts:
        attempts += 1

        # Random point in sphere (uniform distribution via rejection)
        while True:
            p = np.random.uniform(-max_reach_mm, max_reach_mm, 3)
            if np.linalg.norm(p) <= max_reach_mm:
                break

        # Enforce min Z (don't go below table)
        p[2] = max(p[2], min_z_mm)

        # Vary orientation slightly around base
        rpy = rpy_base.copy()
        rpy += np.random.uniform(-rpy_vary_deg, rpy_vary_deg, 3)

        # Check IK feasibility
        joints = ik.solve_ik(p, rpy, seed_joints_deg=seed)
        if joints is not None:
            targets.append((p, rpy))
            seed = joints

    return targets


def main():
    parser = argparse.ArgumentParser(description="Robot arm motion demo")
    parser.add_argument('--mode', choices=['random', 'cube'], default='random',
                        help='Demo mode (default: random)')
    parser.add_argument('--n', type=int, default=10,
                        help='Number of random poses to visit (default: 10)')
    parser.add_argument('--reach', type=float, default=750.0,
                        help='Max reach from origin in mm (default: 750)')
    parser.add_argument('--min-z', type=float, default=100.0,
                        help='Minimum Z height in mm (default: 100)')
    parser.add_argument('--size', type=float, default=80.0,
                        help='Cube edge length in mm, cube mode only (default: 80)')
    parser.add_argument('--center', type=str, default='350,0,350',
                        help='Cube center x,y,z in mm, cube mode only')
    parser.add_argument('--speed', type=int, default=30,
                        help='Robot speed percent (default: 30)')
    parser.add_argument('--loops', type=int, default=2,
                        help='Number of loops (default: 2)')
    parser.add_argument('--rx', type=float, default=180.0,
                        help='Base gripper rx orientation in degrees (default: 180)')
    parser.add_argument('--max-step', type=float, default=5.0,
                        help='Max joint change per trajectory step in deg (default: 5)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    config = load_config()
    rc = config.get('robot', {})
    gc = config.get('gripper', {})
    ik = IKSolver(tool_length_mm=gc.get('tool_length_mm', 100.0))
    rpy_base = np.array([args.rx, 0.0, 0.0])

    # Generate targets
    if args.mode == 'cube':
        center = np.array([float(x) for x in args.center.split(',')])
        targets = generate_cube_targets(center, args.size, rpy_base)
        print(f"Cube mode: {len(targets)} corners, size={args.size}mm, "
              f"center={center}")
    else:
        print(f"Generating {args.n} random reachable poses "
              f"(reach<={args.reach}mm, z>={args.min_z}mm)...")
        targets = generate_random_targets(
            args.n, args.reach, args.min_z, ik, rpy_base)
        print(f"  Found {len(targets)} reachable poses")
        if len(targets) == 0:
            print("  No reachable poses found. Try increasing --reach.")
            return

    # Pre-solve all IK
    print("\nPre-solving IK...")
    target_joints = []
    seed = None
    for i, (pos, rpy) in enumerate(targets):
        joints = ik.solve_ik(pos, rpy, seed_joints_deg=seed)
        if joints is None:
            print(f"  Target {i}: ({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f}) — SKIP (IK fail)")
            continue
        target_joints.append((pos, rpy, joints))
        dist = np.linalg.norm(pos)
        print(f"  Target {i}: ({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f}) "
              f"reach={dist:.0f}mm OK")
        seed = joints

    if not target_joints:
        print("No valid targets!")
        return

    print(f"\n{len(target_joints)} targets ready. Connecting to robot...")

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

        current = robot.get_joint_angles()
        print(f"Current joints: [{','.join(f'{j:.1f}' for j in current)}]")

        # Move to first target
        pos0, rpy0, joints0 = target_joints[0]
        first_joints = ik.solve_ik(pos0, rpy0, seed_joints_deg=current)
        print(f"\nMoving to first target ({pos0[0]:.0f},{pos0[1]:.0f},{pos0[2]:.0f})...")
        if not execute_trajectory(robot, current, first_joints,
                                  max_step_deg=args.max_step):
            print("ERROR: Failed to reach first target")
            return

        prev_joints = first_joints
        move_count = 0
        fail_count = 0

        for loop in range(args.loops):
            print(f"\n=== Loop {loop + 1}/{args.loops} ===")

            for i, (pos, rpy, pre_joints) in enumerate(target_joints):
                # Re-solve with current seed for continuity
                joints = ik.solve_ik(pos, rpy, seed_joints_deg=prev_joints)
                if joints is None:
                    continue

                dist = np.linalg.norm(pos)
                travel = np.max(np.abs(joints - prev_joints))
                label = f"T{i}({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f}) " \
                        f"r={dist:.0f}mm Δ={travel:.0f}°"
                print(f"  -> {label}", end="", flush=True)

                ok = execute_trajectory(robot, prev_joints, joints,
                                        max_step_deg=args.max_step)
                if ok:
                    print(" OK")
                    move_count += 1
                else:
                    print(" FAIL")
                    fail_count += 1

                prev_joints = joints

        print(f"\nDemo complete! {move_count} moves OK, {fail_count} failed.")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        robot.disable()
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
