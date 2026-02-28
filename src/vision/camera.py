"""RealSense D435i camera wrapper for aligned RGB + depth streaming."""

import os
import numpy as np
import yaml

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None
    print("WARNING: pyrealsense2 not installed. Camera will not work.")


# Default path for calibrated intrinsics
_INTRINSICS_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'config', 'camera_intrinsics.yaml')


class CameraIntrinsics:
    """Camera intrinsics with same interface as pyrealsense2 intrinsics.

    Properties: fx, fy, ppx, ppy, coeffs (list of 5 floats).
    Also provides camera_matrix (3x3 numpy) and dist_coeffs (numpy array).
    """

    def __init__(self, fx, fy, ppx, ppy, coeffs=None):
        self.fx = fx
        self.fy = fy
        self.ppx = ppx
        self.ppy = ppy
        self.coeffs = coeffs or [0.0, 0.0, 0.0, 0.0, 0.0]
        self.width = 0
        self.height = 0

    @property
    def camera_matrix(self):
        return np.array([
            [self.fx, 0, self.ppx],
            [0, self.fy, self.ppy],
            [0, 0, 1]
        ], dtype=np.float64)

    @property
    def dist_coeffs(self):
        return np.array(self.coeffs, dtype=np.float64)

    @classmethod
    def from_rs_intrinsics(cls, rs_intr):
        """Create from pyrealsense2 intrinsics object."""
        obj = cls(rs_intr.fx, rs_intr.fy, rs_intr.ppx, rs_intr.ppy,
                  list(rs_intr.coeffs))
        obj.width = rs_intr.width
        obj.height = rs_intr.height
        return obj

    @classmethod
    def load(cls, filepath):
        """Load calibrated intrinsics from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        mtx = data['camera_matrix']
        coeffs = data['dist_coeffs']
        size = data.get('image_size', [0, 0])
        obj = cls(fx=mtx[0][0], fy=mtx[1][1], ppx=mtx[0][2], ppy=mtx[1][2],
                  coeffs=list(coeffs))
        obj.width = size[0]
        obj.height = size[1]
        return obj

    def save(self, filepath):
        """Save calibrated intrinsics to YAML file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': list(self.coeffs),
            'image_size': [self.width, self.height],
        }
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)


class RealSenseCamera:
    """Manages RealSense D435i pipeline for aligned color and depth frames."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 15):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.align = None
        self.intrinsics = None

    def start(self):
        """Initialize and start the RealSense pipeline."""
        if rs is None:
            raise RuntimeError("pyrealsense2 is not installed")

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # Get camera intrinsics for 3D projection
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        rs_intr = color_stream.get_intrinsics()
        self._rs_intrinsics = rs_intr  # keep raw for depth deprojection

        # Try loading calibrated intrinsics
        intr_path = os.path.normpath(_INTRINSICS_PATH)
        if os.path.exists(intr_path):
            try:
                calib = CameraIntrinsics.load(intr_path)
                if calib.width == self.width and calib.height == self.height:
                    self.intrinsics = calib
                    print(f"Using calibrated intrinsics from {intr_path}")
                else:
                    self.intrinsics = CameraIntrinsics.from_rs_intrinsics(rs_intr)
                    print(f"Calibrated intrinsics size mismatch "
                          f"({calib.width}x{calib.height} vs {self.width}x{self.height}), "
                          f"using factory")
            except Exception as e:
                self.intrinsics = CameraIntrinsics.from_rs_intrinsics(rs_intr)
                print(f"Failed to load calibrated intrinsics: {e}, using factory")
        else:
            self.intrinsics = CameraIntrinsics.from_rs_intrinsics(rs_intr)
            print("Using factory intrinsics (no camera_intrinsics.yaml)")

    def get_frames(self):
        """Capture aligned color and depth frames.

        Returns:
            tuple: (color_image, depth_image, depth_frame)
                - color_image: np.ndarray (H, W, 3) BGR
                - depth_image: np.ndarray (H, W) uint16 in mm
                - depth_frame: rs.depth_frame for point queries
        """
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        return color_image, depth_image, depth_frame

    def pixel_to_3d(self, x: int, y: int, depth_frame) -> np.ndarray:
        """Convert a 2D pixel + depth to a 3D point in camera frame.

        Args:
            x: pixel column
            y: pixel row
            depth_frame: RealSense depth frame

        Returns:
            np.ndarray: [x, y, z] in meters in camera coordinate frame
        """
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        depth_m = depth_frame.get_distance(x, y)
        point = rs.rs2_deproject_pixel_to_point(self._rs_intrinsics, [x, y], depth_m)
        return np.array(point)

    def stop(self):
        """Stop the RealSense pipeline."""
        if self.pipeline:
            self.pipeline.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
