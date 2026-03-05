#!/usr/bin/env python3
"""Diagnose arm101 joint sign mapping by showing live FK.

Reads motor angles, computes FK with current signs, and prints
continuously. Jog each joint one at a time (keyboard 1-6 in control
panel) and check if the FK position moves in the expected direction.

For each joint, press the + key and observe:
  J1+ (shoulder_pan):  should rotate TCP in +Y (counterclockwise from above)
  J2+ (shoulder_lift): should raise TCP (+Z) or tilt arm back (+X)
  J3+ (elbow_flex):    should move TCP forward/up
  J4+ (wrist_flex):    should tilt wrist
  J5+ (wrist_roll):    should roll wrist

If a joint moves the FK position opposite to expected, its sign needs flipping.

Usage:
    ./run.sh scripts/test_arm101_fk.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config_loader import load_config
from kinematics.arm101_ik_solver import Arm101IKSolver, JOINT_SIGNS

# Connect to arm
config = load_config()
ac = config.get('arm101', {})

from robot.lerobot_arm101 import LeRobotArm101
arm = LeRobotArm101(
    port=ac.get('port', ''),
    baudrate=ac.get('baudrate', 1_000_000),
    safe_mode=True)
arm.connect()

solver = Arm101IKSolver()

print(f"\nCurrent JOINT_SIGNS: {JOINT_SIGNS}")
print(f"\nJog joints one at a time and watch FK position change.")
print(f"If direction is wrong, that joint's sign needs flipping.\n")
print(f"{'Motor angles (deg)':>45s}  |  {'FK position (mm)':>30s}  |  {'delta (mm)':>20s}")
print("-" * 105)

prev_pos = None
try:
    while True:
        angles = arm.get_angles()
        if angles is None:
            time.sleep(0.1)
            continue

        motor_deg = np.array(angles[:5])
        pos_mm, rpy_deg = solver.forward_kin(motor_deg)

        delta_str = ""
        if prev_pos is not None:
            delta = pos_mm - prev_pos
            if np.linalg.norm(delta) > 0.5:
                delta_str = f"dx={delta[0]:+.1f} dy={delta[1]:+.1f} dz={delta[2]:+.1f}"
                prev_pos = pos_mm.copy()
        else:
            prev_pos = pos_mm.copy()

        angles_str = " ".join(f"{a:+7.1f}" for a in motor_deg)
        pos_str = f"[{pos_mm[0]:.1f}, {pos_mm[1]:.1f}, {pos_mm[2]:.1f}]"
        print(f"\r  {angles_str}  |  {pos_str:>30s}  |  {delta_str:>20s}", end="", flush=True)
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n\nDone.")
finally:
    arm.close()
