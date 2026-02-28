"""Hand-eye calibration routine for fixed camera + robot arm.

This module provides both manual measurement-based calibration
and automated ArUco-based calibration approaches.
"""

import numpy as np
import cv2
from .transform import CoordinateTransform


def calibrate_manual(camera_position_in_base: list[float],
                     camera_rotation_euler_deg: list[float]) -> CoordinateTransform:
    """Create calibration from manual measurements.

    For the PoC, measure the camera's position and orientation relative
    to the robot base manually with a tape measure.

    Args:
        camera_position_in_base: [x, y, z] of camera in robot base frame (meters)
        camera_rotation_euler_deg: [rx, ry, rz] Euler angles in degrees
            describing how camera frame is rotated relative to base frame

    Returns:
        CoordinateTransform with the measured transform
    """
    rx, ry, rz = np.radians(camera_rotation_euler_deg)

    # Build rotation matrix from Euler angles (XYZ convention)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx

    transform = CoordinateTransform()
    transform.set_manual(
        translation=np.array(camera_position_in_base),
        rotation_matrix=R
    )
    return transform


def calibrate_automated(robot_poses: list[np.ndarray],
                        camera_observations: list[np.ndarray]) -> CoordinateTransform:
    """Automated hand-eye calibration using known correspondences.

    Move the robot to N positions with a known target (e.g., ArUco marker
    on the gripper), record robot TCP pose and camera observation at each.

    Args:
        robot_poses: List of 4x4 homogeneous transforms (base to TCP)
        camera_observations: List of 4x4 homogeneous transforms (camera to target)

    Returns:
        CoordinateTransform with the computed transform
    """
    # Convert to rotation vectors and translation vectors for OpenCV
    R_gripper2base = [pose[:3, :3] for pose in robot_poses]
    t_gripper2base = [pose[:3, 3] for pose in robot_poses]
    R_target2cam = [obs[:3, :3] for obs in camera_observations]
    t_target2cam = [obs[:3, 3] for obs in camera_observations]

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    transform = CoordinateTransform()
    T = np.eye(4)
    T[:3, :3] = R_cam2base
    T[:3, 3] = t_cam2base.flatten()
    transform.T_camera_to_base = T
    return transform
