"""Camera-to-robot coordinate frame transformation."""

import numpy as np
import yaml
import os


class CoordinateTransform:
    """Handles transformation between camera frame and robot base frame.

    For a fixed camera (eye-to-hand), we need T_camera_to_base such that:
        P_base = T_camera_to_base @ P_camera_homogeneous
    """

    def __init__(self):
        self.T_camera_to_base = np.eye(4)

    def load(self, filepath: str):
        """Load calibration from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        self.T_camera_to_base = np.array(data['T_camera_to_base'])

    def save(self, filepath: str):
        """Save calibration to YAML file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            'T_camera_to_base': self.T_camera_to_base.tolist()
        }
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    def set_manual(self, translation: np.ndarray, rotation_matrix: np.ndarray):
        """Set transform from manual measurements.

        Args:
            translation: [tx, ty, tz] camera origin in robot base frame (meters)
            rotation_matrix: 3x3 rotation from camera frame to robot base frame
        """
        self.T_camera_to_base = np.eye(4)
        self.T_camera_to_base[:3, :3] = rotation_matrix
        self.T_camera_to_base[:3, 3] = translation

    def camera_to_base(self, point_camera: np.ndarray) -> np.ndarray:
        """Transform a 3D point from camera frame to robot base frame.

        Args:
            point_camera: [x, y, z] in camera frame (meters)

        Returns:
            [x, y, z] in robot base frame (meters)
        """
        p_hom = np.append(point_camera, 1.0)
        p_base_hom = self.T_camera_to_base @ p_hom
        return p_base_hom[:3]

    def camera_axis_to_base(self, axis_camera: np.ndarray) -> np.ndarray:
        """Transform a direction vector from camera frame to robot base frame.

        Args:
            axis_camera: unit vector in camera frame

        Returns:
            unit vector in robot base frame
        """
        R = self.T_camera_to_base[:3, :3]
        axis_base = R @ axis_camera
        norm = np.linalg.norm(axis_base)
        if norm > 0:
            axis_base /= norm
        return axis_base
