"""Rod detection using depth segmentation and contour analysis."""

import cv2
import numpy as np


class RodDetection:
    """Result of rod detection."""

    def __init__(self, center_3d: np.ndarray, axis_3d: np.ndarray,
                 center_2d: tuple, contour: np.ndarray, confidence: float):
        self.center_3d = center_3d      # (x, y, z) in camera frame, meters
        self.axis_3d = axis_3d          # unit vector along rod axis in camera frame
        self.center_2d = center_2d      # (x, y) pixel coordinates
        self.contour = contour          # OpenCV contour for visualization
        self.confidence = confidence    # 0.0 - 1.0


class RodDetector:
    """Detects a black metal rod lying on a surface using depth + color segmentation."""

    def __init__(self, min_aspect_ratio: float = 3.0, min_area: int = 500,
                 depth_min_mm: int = 100, depth_max_mm: int = 1000,
                 table_tolerance_mm: int = 30):
        self.min_aspect_ratio = min_aspect_ratio
        self.min_area = min_area
        self.depth_min_mm = depth_min_mm
        self.depth_max_mm = depth_max_mm
        self.table_tolerance_mm = table_tolerance_mm

    def detect(self, color_image: np.ndarray, depth_image: np.ndarray,
               depth_frame, camera) -> RodDetection | None:
        """Detect a rod in the current frame.

        Strategy:
        1. Filter depth to workspace range
        2. Find table plane (dominant depth)
        3. Isolate objects above table
        4. Filter by color (dark/black)
        5. Find elongated contours (high aspect ratio)
        6. Compute 3D center and orientation

        Args:
            color_image: BGR image (H, W, 3)
            depth_image: Depth in mm (H, W) uint16
            depth_frame: RealSense depth frame for 3D queries
            camera: RealSenseCamera instance for pixel_to_3d

        Returns:
            RodDetection or None if no rod found
        """
        # Step 1: Create workspace depth mask
        workspace_mask = (
            (depth_image > self.depth_min_mm) &
            (depth_image < self.depth_max_mm)
        ).astype(np.uint8) * 255

        # Step 2: Estimate table depth (mode of depth histogram in workspace)
        valid_depths = depth_image[workspace_mask > 0]
        if len(valid_depths) == 0:
            return None
        hist, bin_edges = np.histogram(valid_depths, bins=100)
        table_depth_mm = int(bin_edges[np.argmax(hist)])

        # Step 3: Isolate objects above table (closer to camera = smaller depth)
        object_mask = (
            (depth_image > self.depth_min_mm) &
            (depth_image < table_depth_mm - self.table_tolerance_mm)
        ).astype(np.uint8) * 255

        # Step 4: Filter by color (dark objects)
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        dark_mask = (hsv[:, :, 2] < 80).astype(np.uint8) * 255  # Low value = dark

        # Combine depth and color masks
        combined_mask = cv2.bitwise_and(object_mask, dark_mask)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # Step 5: Find elongated contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_detection = None
        best_score = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            # Fit minimum bounding rectangle
            rect = cv2.minAreaRect(contour)
            (cx, cy), (w, h), angle = rect

            # Ensure w is the longer dimension
            if w < h:
                w, h = h, w
                angle += 90

            if h == 0:
                continue

            aspect_ratio = w / h
            if aspect_ratio < self.min_aspect_ratio:
                continue

            # Score based on aspect ratio and area
            score = aspect_ratio * np.sqrt(area)
            if score > best_score:
                best_score = score
                cx_int, cy_int = int(cx), int(cy)

                # Get 3D center point
                center_3d = camera.pixel_to_3d(cx_int, cy_int, depth_frame)

                # Compute 3D axis direction from endpoints of the bounding rect
                angle_rad = np.radians(angle)
                dx, dy = int(w / 2 * np.cos(angle_rad)), int(w / 2 * np.sin(angle_rad))
                p1 = camera.pixel_to_3d(cx_int - dx, cy_int - dy, depth_frame)
                p2 = camera.pixel_to_3d(cx_int + dx, cy_int + dy, depth_frame)
                axis = p2 - p1
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 0:
                    axis = axis / axis_norm
                else:
                    axis = np.array([1, 0, 0])  # fallback

                confidence = min(1.0, aspect_ratio / 10.0)  # Heuristic confidence

                best_detection = RodDetection(
                    center_3d=center_3d,
                    axis_3d=axis,
                    center_2d=(cx_int, cy_int),
                    contour=contour,
                    confidence=confidence,
                )

        return best_detection

    def draw_detection(self, image: np.ndarray, detection: RodDetection) -> np.ndarray:
        """Draw detection overlay on the image for visualization."""
        vis = image.copy()
        cv2.drawContours(vis, [detection.contour], -1, (0, 255, 0), 2)
        cx, cy = detection.center_2d
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
        label = f"Rod ({detection.confidence:.1%}) z={detection.center_3d[2]:.3f}m"
        cv2.putText(vis, label, (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return vis
