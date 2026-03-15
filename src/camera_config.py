"""Load and query the camera registry (config/cameras.yaml).

Provides a structured interface to access camera intrinsics, extrinsics,
mount information, and capabilities from the unified camera config file.

Usage:
    from camera_config import CameraRegistry

    registry = CameraRegistry.load()          # from default path
    cam = registry.get('realsense_d435i')     # by logical name
    cam = registry.find_by_mount('gripper')   # first gripper-mounted camera
    cam = registry.find_by_type('realsense')  # first RealSense camera

    # Access fields
    print(cam.intrinsics_matrix)   # 3x3 numpy array
    print(cam.mount_type)          # 'fixed', 'gripper', etc.
    print(cam.T_cam_to_link)       # 4x4 numpy array or None
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import yaml


# Default config path
from config_loader import get_config_dir
_CONFIG_DIR = get_config_dir()
_CAMERAS_YAML = os.path.join(_CONFIG_DIR, 'cameras.yaml')


@dataclass
class CameraInfo:
    """Structured representation of a single camera entry from cameras.yaml."""

    name: str
    device_path: str = ''
    device_index: int = -1
    serial: str = 'unknown'
    driver: str = ''
    card: str = ''
    camera_type: str = 'webcam'     # 'realsense' or 'webcam'
    resolution: list = field(default_factory=lambda: [640, 480])
    fps: int = 30

    # Mount info
    mount_type: str = 'other'       # 'fixed', 'gripper', 'overhead', 'other'
    parent_link: Optional[str] = None  # URDF link name
    mount_notes: str = ''

    # Raw config dicts for advanced access
    _intrinsics_raw: dict = field(default_factory=dict, repr=False)
    _extrinsics_raw: dict = field(default_factory=dict, repr=False)
    _mount_raw: dict = field(default_factory=dict, repr=False)
    _capabilities_raw: dict = field(default_factory=dict, repr=False)

    @property
    def intrinsics_matrix(self) -> Optional[np.ndarray]:
        """3x3 camera matrix as numpy array, or None."""
        mtx = self._intrinsics_raw.get('camera_matrix')
        if mtx is None:
            return None
        return np.array(mtx, dtype=np.float64)

    @property
    def dist_coeffs(self) -> Optional[np.ndarray]:
        """Distortion coefficients as numpy array, or None."""
        coeffs = self._intrinsics_raw.get('dist_coeffs')
        if coeffs is None:
            return None
        return np.array(coeffs, dtype=np.float64)

    @property
    def intrinsics_source(self) -> str:
        """How intrinsics were obtained: 'factory', 'calibrated', 'estimated', 'unknown'."""
        return self._intrinsics_raw.get('source', 'unknown')

    @property
    def intrinsics_image_size(self) -> list:
        """[width, height] for which intrinsics are valid."""
        return self._intrinsics_raw.get('image_size', self.resolution)

    @property
    def T_cam_to_link(self) -> Optional[np.ndarray]:
        """4x4 camera-to-parent-link transform, or None."""
        mtx = self._mount_raw.get('T_cam_to_link')
        if mtx is None:
            return None
        return np.array(mtx, dtype=np.float64)

    @property
    def T_cam_to_base(self) -> Optional[np.ndarray]:
        """4x4 camera-to-robot-base transform (for fixed cameras), or None."""
        mtx = self._extrinsics_raw.get('T_cam_to_base')
        if mtx is None:
            return None
        return np.array(mtx, dtype=np.float64)

    @property
    def extrinsics_source(self) -> str:
        """How extrinsics were obtained: 'calibrated', 'estimated', 'none'."""
        return self._extrinsics_raw.get('source', 'none')

    @property
    def has_depth(self) -> bool:
        return self._capabilities_raw.get('depth', False)

    @property
    def has_color(self) -> bool:
        return self._capabilities_raw.get('color', True)

    @property
    def has_imu(self) -> bool:
        return self._capabilities_raw.get('imu', False)

    @property
    def is_gripper_mounted(self) -> bool:
        return self.mount_type == 'gripper'

    @property
    def is_fixed(self) -> bool:
        return self.mount_type == 'fixed'

    def to_camera_intrinsics(self):
        """Convert to a CameraIntrinsics object (compatible with vision.camera).

        Returns:
            CameraIntrinsics instance, or None if intrinsics are unavailable.
        """
        from vision.camera import CameraIntrinsics

        mtx = self.intrinsics_matrix
        if mtx is None:
            return None

        coeffs = self.dist_coeffs
        intr = CameraIntrinsics(
            fx=float(mtx[0, 0]),
            fy=float(mtx[1, 1]),
            ppx=float(mtx[0, 2]),
            ppy=float(mtx[1, 2]),
            coeffs=coeffs.tolist() if coeffs is not None else None,
        )
        size = self.intrinsics_image_size
        intr.width = size[0]
        intr.height = size[1]
        return intr


class CameraRegistry:
    """Registry of all known cameras loaded from config/cameras.yaml.

    Provides lookup by name, mount type, camera type, etc.
    """

    def __init__(self, cameras: dict[str, CameraInfo]):
        self._cameras = cameras

    @classmethod
    def load(cls, path: str = None) -> 'CameraRegistry':
        """Load camera registry from YAML file.

        Args:
            path: Path to cameras.yaml. Defaults to config/cameras.yaml.

        Returns:
            CameraRegistry instance (may be empty if file doesn't exist).
        """
        if path is None:
            path = os.path.normpath(_CAMERAS_YAML)

        if not os.path.exists(path):
            return cls({})

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        if not data or not data.get('cameras'):
            return cls({})

        cameras = {}
        for name, entry in data['cameras'].items():
            if not isinstance(entry, dict):
                continue

            mount = entry.get('mount', {}) or {}
            intrinsics = entry.get('intrinsics', {}) or {}
            extrinsics = entry.get('extrinsics', {}) or {}
            capabilities = entry.get('capabilities', {}) or {}

            cameras[name] = CameraInfo(
                name=name,
                device_path=entry.get('device_path', ''),
                device_index=entry.get('device_index', -1),
                serial=entry.get('serial', 'unknown'),
                driver=entry.get('driver', ''),
                card=entry.get('card', ''),
                camera_type=entry.get('type', 'webcam'),
                resolution=entry.get('resolution', [640, 480]),
                fps=entry.get('fps', 30),
                mount_type=mount.get('type', 'other'),
                parent_link=mount.get('parent_link'),
                mount_notes=mount.get('notes', ''),
                _intrinsics_raw=intrinsics,
                _extrinsics_raw=extrinsics,
                _mount_raw=mount,
                _capabilities_raw=capabilities,
            )

        return cls(cameras)

    def get(self, name: str) -> Optional[CameraInfo]:
        """Look up a camera by its logical name."""
        return self._cameras.get(name)

    def find_by_mount(self, mount_type: str) -> Optional[CameraInfo]:
        """Return the first camera with the given mount type."""
        for cam in self._cameras.values():
            if cam.mount_type == mount_type:
                return cam
        return None

    def find_by_type(self, camera_type: str) -> Optional[CameraInfo]:
        """Return the first camera with the given type ('realsense' or 'webcam')."""
        for cam in self._cameras.values():
            if cam.camera_type == camera_type:
                return cam
        return None

    def find_gripper_camera(self) -> Optional[CameraInfo]:
        """Return the gripper-mounted camera, or None."""
        return self.find_by_mount('gripper')

    def find_fixed_camera(self) -> Optional[CameraInfo]:
        """Return the first fixed-mount camera, or None."""
        return self.find_by_mount('fixed')

    def all(self) -> list[CameraInfo]:
        """Return all cameras as a list."""
        return list(self._cameras.values())

    def names(self) -> list[str]:
        """Return all logical camera names."""
        return list(self._cameras.keys())

    def __len__(self) -> int:
        return len(self._cameras)

    def __contains__(self, name: str) -> bool:
        return name in self._cameras

    def __repr__(self) -> str:
        names = ', '.join(self._cameras.keys()) if self._cameras else '(empty)'
        return f"CameraRegistry({names})"

    def summary(self) -> str:
        """Return a human-readable summary of all cameras."""
        if not self._cameras:
            return "No cameras registered."

        lines = [f"Camera Registry ({len(self._cameras)} camera(s)):"]
        for name, cam in self._cameras.items():
            intr = cam.intrinsics_source
            ext = cam.extrinsics_source
            lines.append(
                f"  {name}: type={cam.camera_type}, mount={cam.mount_type}, "
                f"res={cam.resolution}, intrinsics={intr}, extrinsics={ext}"
            )
            if cam.parent_link:
                lines.append(f"    URDF link: {cam.parent_link}")
        return '\n'.join(lines)
