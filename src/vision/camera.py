"""RealSense D435i camera wrapper for aligned RGB + depth streaming."""

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None
    print("WARNING: pyrealsense2 not installed. Camera will not work.")


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
        self.intrinsics = color_stream.get_intrinsics()

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
        point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], depth_m)
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
