"""Camera wrappers for aligned RGB + depth streaming.

Supports:
  - RealSenseCamera: Intel RealSense D435i (RGB + depth, hardware-aligned)
  - WebcamCamera: Any OpenCV-compatible webcam (RGB only, no depth)

Use create_camera(config) to instantiate the correct class from config.
"""

import os
import numpy as np
import yaml
import cv2

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


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


class WebcamCamera:
    """OpenCV-based webcam wrapper with the same interface as RealSenseCamera.

    Provides RGB frames only — depth_image and depth_frame are always None.
    pixel_to_3d() uses a fixed assumed depth (table-plane assumption) so that
    the rod detector can still compute approximate 3D positions.

    Args:
        device_index: cv2.VideoCapture device index (default 0)
        width: desired frame width
        height: desired frame height
        fps: desired frame rate
        assumed_depth_m: fixed depth assumed for pixel_to_3d (meters)
    """

    def __init__(self, device_index: int = 0, width: int = 640,
                 height: int = 480, fps: int = 30,
                 assumed_depth_m: float = 1.0):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.assumed_depth_m = assumed_depth_m
        self.intrinsics = None
        self._cap = None

    def start(self):
        """Open the webcam and set up intrinsics."""
        # Use V4L2 backend on Linux for reliable resolution control
        backend = cv2.CAP_V4L2 if hasattr(cv2, 'CAP_V4L2') else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self.device_index, backend)
        if not self._cap.isOpened():
            # Fallback to default backend
            self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open webcam device {self.device_index}. "
                "Check that no other process is using it."
            )

        # Set MJPEG codec first — many cameras only support higher
        # resolutions in MJPEG mode (not raw YUYV)
        self._cap.set(cv2.CAP_PROP_FOURCC,
                      cv2.VideoWriter_fourcc(*'MJPG'))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Read back actual resolution
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._needs_resize = (actual_w != self.width or actual_h != self.height)
        if self._needs_resize:
            print(f"Webcam: got {actual_w}x{actual_h}, will resize to {self.width}x{self.height}")
        else:
            print(f"Webcam: opened at {self.width}x{self.height}")

        # Try loading calibrated intrinsics; fall back to a pinhole estimate
        intr_path = os.path.normpath(_INTRINSICS_PATH)
        if os.path.exists(intr_path):
            try:
                calib = CameraIntrinsics.load(intr_path)
                if calib.width == self.width and calib.height == self.height:
                    self.intrinsics = calib
                    print(f"Webcam: using calibrated intrinsics from {intr_path}")
                else:
                    self.intrinsics = self._estimate_intrinsics()
                    print(
                        f"Webcam: calibrated intrinsics size mismatch "
                        f"({calib.width}x{calib.height} vs "
                        f"{self.width}x{self.height}), using estimate"
                    )
            except Exception as e:
                self.intrinsics = self._estimate_intrinsics()
                print(f"Webcam: failed to load calibrated intrinsics ({e}), "
                      "using estimate")
        else:
            self.intrinsics = self._estimate_intrinsics()
            print("Webcam: no camera_intrinsics.yaml — using estimated intrinsics")

    def _estimate_intrinsics(self) -> CameraIntrinsics:
        """Estimate pinhole intrinsics assuming a 60-degree horizontal FOV."""
        fov_h_deg = 60.0
        fx = self.width / (2.0 * np.tan(np.radians(fov_h_deg / 2.0)))
        fy = fx
        ppx = self.width / 2.0
        ppy = self.height / 2.0
        intr = CameraIntrinsics(fx=fx, fy=fy, ppx=ppx, ppy=ppy)
        intr.width = self.width
        intr.height = self.height
        return intr

    def get_frames(self):
        """Capture a color frame from the webcam.

        Returns:
            tuple: (color_image, None, None)
                - color_image: np.ndarray (H, W, 3) BGR, or None on failure
                - depth_image: always None (no depth sensor)
                - depth_frame: always None (no depth sensor)
        """
        if self._cap is None:
            return None, None, None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None, None, None

        if self._needs_resize:
            frame = cv2.resize(frame, (self.width, self.height))

        return frame, None, None

    def pixel_to_3d(self, x: int, y: int, depth_frame=None) -> np.ndarray:
        """Convert a 2D pixel to a 3D point using a fixed assumed depth.

        Without a depth sensor, all 3D coordinates are computed by back-
        projecting the pixel through the calibrated (or estimated) pinhole
        model at a fixed assumed depth.  The results are approximate and
        depend on how well assumed_depth_m matches reality.

        Args:
            x: pixel column
            y: pixel row
            depth_frame: ignored (kept for API compatibility)

        Returns:
            np.ndarray: [x, y, z] in meters in camera coordinate frame
        """
        intr = self.intrinsics
        x_cam = (x - intr.ppx) / intr.fx * self.assumed_depth_m
        y_cam = (y - intr.ppy) / intr.fy * self.assumed_depth_m
        z_cam = self.assumed_depth_m
        return np.array([x_cam, y_cam, z_cam])

    def stop(self):
        """Release the webcam."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def create_camera(config: dict):
    """Factory: create the right camera from config.

    Config keys (under 'camera'):
        type: 'realsense' (default) or 'webcam'
        width, height, fps: resolution / frame rate
        device_index: webcam device index (default 0, webcam only)
        assumed_depth_m: fixed depth for pixel_to_3d (default 1.0, webcam only)

    Returns:
        RealSenseCamera or WebcamCamera instance (not yet started).
    """
    cam_cfg = config.get('camera', {})
    cam_type = cam_cfg.get('type', 'realsense').lower()
    width = cam_cfg.get('width', 640)
    height = cam_cfg.get('height', 480)
    fps = cam_cfg.get('fps', 15)

    if cam_type == 'webcam':
        return WebcamCamera(
            device_index=cam_cfg.get('device_index', 0),
            width=width,
            height=height,
            fps=fps,
            assumed_depth_m=cam_cfg.get('assumed_depth_m', 1.0),
        )
    else:
        return RealSenseCamera(width=width, height=height, fps=fps)
