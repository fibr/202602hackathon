#!/usr/bin/env python3
"""Calibrate servo zero offsets for SO-ARM101.

The STS3215 servos use raw position 0-4095 (12-bit), with 2048 typically
considered center. However, each servo horn is installed at a different
physical angle, so 2048 does NOT correspond to 0° for all joints.

This script helps determine the correct zero offset for each motor:

  1. Disables torque so the arm can be manually positioned
  2. User moves arm to a known "zero" configuration:
     - All links straight / aligned (or a defined reference pose)
  3. Records the raw servo positions as zero offsets
  4. Saves to config/servo_offsets.yaml

Once calibrated, the driver should use:
  angle_deg = (raw_pos - offset) * DEG_PER_POS

Usage:
  ./run.sh scripts/calibrate_servos.py                    # Interactive calibration
  ./run.sh scripts/calibrate_servos.py --read-only         # Just read current positions
  ./run.sh scripts/calibrate_servos.py --show-offsets       # Show saved offsets
"""

import sys
import os
import time
import argparse
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

OFFSET_FILE = os.path.join(os.path.dirname(__file__), '..', 'config', 'servo_offsets.yaml')
MOTOR_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
               'wrist_flex', 'wrist_roll', 'gripper']

# STS3215 registers
ADDR_TORQUE_ENABLE = 40
ADDR_MAX_TORQUE = 48
ADDR_PRESENT_POSITION = 56


def read_all_raw(arm):
    """Read raw positions for all motors."""
    positions = {}
    for mid in arm.motor_ids:
        pos, result, error = arm.packet_handler.read2ByteTxRx(
            arm.port_handler, mid, ADDR_PRESENT_POSITION)
        positions[mid] = pos
    return positions


def print_positions(positions):
    """Print positions in a nice table."""
    print(f"\n{'Motor':<20} {'ID':>3} {'Raw':>5} {'Deg (raw-2048)':>15}")
    print("-" * 50)
    for mid, pos in sorted(positions.items()):
        deg = (pos - 2048) * 360.0 / 4096.0
        name = MOTOR_NAMES[mid - 1] if mid <= len(MOTOR_NAMES) else f"motor_{mid}"
        print(f"  {name:<20} {mid:>3} {pos:>5}  {deg:>+10.1f}°")


def load_offsets():
    """Load saved zero offsets."""
    path = os.path.normpath(OFFSET_FILE)
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return data.get('zero_offsets', {})
    return {}


def save_offsets(offsets, description=""):
    """Save zero offsets to file."""
    path = os.path.normpath(OFFSET_FILE)
    data = {
        'description': description or 'Servo zero offsets for SO-ARM101',
        'zero_offsets': offsets,
        'notes': {
            'usage': 'angle_deg = (raw_pos - zero_offset) * 360/4096',
            'default': '2048 (servo center) if no offset defined',
        }
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"\nSaved offsets to {path}")


def run_read_only(arm):
    """Just read and display current positions."""
    print("\n=== Current Servo Positions ===")
    positions = read_all_raw(arm)
    print_positions(positions)

    offsets = load_offsets()
    if offsets:
        print(f"\n=== Saved Zero Offsets ===")
        print(f"{'Motor':<20} {'Offset':>6} {'Cur-Offset':>12} {'Calibrated°':>12}")
        print("-" * 55)
        for mid, pos in sorted(positions.items()):
            name = MOTOR_NAMES[mid - 1]
            offset = offsets.get(name, {}).get('raw_position', 2048)
            cal_deg = (pos - offset) * 360.0 / 4096.0
            print(f"  {name:<20} {offset:>6} {pos - offset:>+12} {cal_deg:>+12.1f}°")
    else:
        print("\n(No saved offsets found)")


def run_interactive_calibration(arm):
    """Interactive calibration procedure."""
    print("\n" + "=" * 60)
    print("  SERVO ZERO-OFFSET CALIBRATION")
    print("=" * 60)

    # Step 1: Read current positions with torque on
    print("\n--- Step 1: Current positions (torque ON) ---")
    positions = read_all_raw(arm)
    print_positions(positions)

    # Step 2: Disable torque for manual positioning
    print("\n--- Step 2: Disabling torque ---")
    for mid in arm.motor_ids:
        try:
            arm._write1(mid, ADDR_TORQUE_ENABLE, 0)
        except IOError:
            pass
    arm._enabled = False
    print("Torque disabled. You can now move the arm by hand.")

    # Step 3: Interactive loop
    print("\n--- Step 3: Position the arm ---")
    print("Move the arm to its PHYSICAL ZERO configuration:")
    print("  • All links should be in their 'neutral' or 'straight' position")
    print("  • The specific pose depends on the URDF convention:")
    print("    - J1 (shoulder_pan): arm pointing forward (away from base)")
    print("    - J2 (shoulder_lift): upper arm horizontal or as defined in URDF")
    print("    - J3 (elbow_flex): forearm aligned with upper arm")
    print("    - J4 (wrist_flex): wrist straight")
    print("    - J5 (wrist_roll): wrist not rotated")
    print("    - J6 (gripper): open or neutral")
    print()
    print("Continuously reading positions. Press Ctrl+C when done.")

    try:
        while True:
            positions = read_all_raw(arm)
            line_parts = []
            for mid in sorted(positions.keys()):
                name = MOTOR_NAMES[mid - 1][:5]
                pos = positions[mid]
                deg = (pos - 2048) * 360.0 / 4096.0
                line_parts.append(f"{name}:{pos}({deg:+.0f}°)")
            print(f"\r  {' | '.join(line_parts)}", end="", flush=True)
            time.sleep(0.3)
    except KeyboardInterrupt:
        pass

    print()  # newline after the live display

    # Step 4: Record final positions
    print("\n--- Step 4: Recording zero positions ---")
    final_positions = read_all_raw(arm)
    print_positions(final_positions)

    # Build offset dict
    offsets = {}
    for mid, pos in sorted(final_positions.items()):
        name = MOTOR_NAMES[mid - 1]
        offsets[name] = {
            'motor_id': mid,
            'raw_position': pos,
            'deviation_from_2048': pos - 2048,
        }

    # Step 5: Ask to save
    print(f"\nThese will be used as zero offsets:")
    for name, data in offsets.items():
        dev = data['deviation_from_2048']
        print(f"  {name:<20}: raw={data['raw_position']}  "
              f"(deviation={dev:+d} from 2048, = {dev * 360.0/4096.0:+.1f}°)")

    save_offsets(offsets, "Calibrated zero offsets for SO-ARM101")

    # Also print what to add to config
    print("\n=== Integration Notes ===")
    print("To use these offsets in the driver, modify _pos_to_deg/_deg_to_pos")
    print("to use per-motor offsets instead of hardcoded POS_CENTER=2048.")
    print(f"Offset file: {os.path.normpath(OFFSET_FILE)}")

    return offsets


def main():
    parser = argparse.ArgumentParser(description="Calibrate ARM101 servo zero offsets")
    parser.add_argument('--read-only', action='store_true',
                        help='Just read current positions')
    parser.add_argument('--show-offsets', action='store_true',
                        help='Show saved offsets')
    args = parser.parse_args()

    if args.show_offsets:
        offsets = load_offsets()
        if offsets:
            print("Saved zero offsets:")
            for name, data in offsets.items():
                print(f"  {name:<20}: raw={data.get('raw_position', '?')} "
                      f"(dev={data.get('deviation_from_2048', '?'):+d})")
        else:
            print("No offsets saved yet.")
        return

    from robot.lerobot_arm101 import LeRobotArm101

    port = LeRobotArm101.find_port()
    print(f"Connecting to arm on {port}...")
    arm = LeRobotArm101(port=port, safe_mode=False)
    arm.connect()

    try:
        if args.read_only:
            run_read_only(arm)
        else:
            run_interactive_calibration(arm)
    except KeyboardInterrupt:
        print("\nAborted.")
    finally:
        try:
            arm.disconnect()
        except Exception:
            pass


if __name__ == '__main__':
    main()
