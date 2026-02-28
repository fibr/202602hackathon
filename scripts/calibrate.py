#!/usr/bin/env python3
"""Run hand-eye calibration for fixed camera + robot arm.

For the PoC, this uses manual measurements. Measure the camera position
relative to the robot base with a tape measure and enter the values below.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from calibration.calibrate import calibrate_manual


def main():
    print("=== Hand-Eye Calibration (Manual) ===")
    print()
    print("Measure the camera's position and orientation relative to the robot base.")
    print("The camera is fixed (eye-to-hand configuration).")
    print()

    # TODO: Replace these with actual measurements
    # Camera position in robot base frame (meters)
    # Example: camera is 0.5m in front, 0.3m to the left, 0.8m above the robot base
    camera_position = [0.5, 0.3, 0.8]

    # Camera orientation relative to robot base (Euler angles in degrees)
    # Example: camera points down at 45 degrees
    # rx=180 means camera Z axis points opposite to base Z (i.e., looking down)
    # Adjust based on actual mounting
    camera_rotation = [180.0, 0.0, 0.0]

    print(f"Camera position (in base frame): {camera_position} meters")
    print(f"Camera rotation (Euler XYZ):     {camera_rotation} degrees")
    print()

    transform = calibrate_manual(camera_position, camera_rotation)

    # Save calibration
    output_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'calibration.yaml')
    transform.save(output_path)
    print(f"Calibration saved to: {output_path}")
    print()
    print("Transform matrix (camera -> base):")
    print(transform.T_camera_to_base)
    print()

    # Quick validation: camera origin should map to the camera position
    origin_in_base = transform.camera_to_base(np.array([0, 0, 0]))
    print(f"Validation: camera origin in base = {origin_in_base} (should be ~{camera_position})")


if __name__ == "__main__":
    main()
