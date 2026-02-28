"""Rod detection using FastSAM segmentation, shape filtering, and color scoring."""

import cv2
import numpy as np
from ultralytics import FastSAM


class RodDetection:
    """Result of rod detection."""

    def __init__(self, center_3d: np.ndarray, axis_3d: np.ndarray,
                 center_2d: tuple, contour: np.ndarray, confidence: float,
                 mask: np.ndarray = None):
        self.center_3d = center_3d      # (x, y, z) in camera frame, meters
        self.axis_3d = axis_3d          # unit vector along rod axis in camera frame
        self.center_2d = center_2d      # (x, y) pixel coordinates
        self.contour = contour          # OpenCV contour for visualization
        self.confidence = confidence    # 0.0 - 1.0
        self.mask = mask                # Binary mask of the segment


class RodDetector:
    """Detects a metal rod using FastSAM segmentation + shape + color filtering.

    Scoring pipeline:
    1. FastSAM generates all segment masks
    2. Filter by workspace ROI overlap (>=50%)
    3. Filter by minimum contour area
    4. Compute shape score: aspect_ratio (elongation)
    5. Compute darkness score: how dark the segment is (low HSV V)
    6. Compute uniformity score: how consistent the color is (low V std dev)
    7. Combined score picks the best rod-like segment
    """

    def __init__(self, min_aspect_ratio: float = 3.0, min_area: int = 500,
                 depth_min_mm: int = 6000, depth_max_mm: int = 19000,
                 max_brightness: int = 120, max_brightness_std: float = 40.0,
                 min_convexity: float = 0.85,
                 workspace_roi: dict = None, fastsam_conf: float = 0.4,
                 fastsam_iou: float = 0.9, fastsam_imgsz: int = 1024):
        self.min_aspect_ratio = min_aspect_ratio
        self.min_area = min_area
        self.depth_min_mm = depth_min_mm
        self.depth_max_mm = depth_max_mm
        self.max_brightness = max_brightness      # Max mean V to be "dark"
        self.max_brightness_std = max_brightness_std  # Max V std for uniformity
        self.min_convexity = min_convexity        # contour_area / convex_hull_area
        self.workspace_roi = workspace_roi
        self.fastsam_conf = fastsam_conf
        self.fastsam_iou = fastsam_iou
        self.fastsam_imgsz = fastsam_imgsz
        self._model = None

    def _get_model(self):
        """Lazy-load FastSAM model (auto-downloads ~23MB on first run)."""
        if self._model is None:
            self._model = FastSAM('FastSAM-s.pt')
        return self._model

    def _build_roi_mask(self, h: int, w: int) -> np.ndarray | None:
        """Build a binary mask from the workspace ROI config.

        Handles resolution scaling if the ROI was defined at a different resolution.
        """
        if self.workspace_roi is None:
            return None

        roi = self.workspace_roi
        roi_res = roi.get('resolution', [w, h])
        sx = w / roi_res[0]
        sy = h / roi_res[1]

        mask = np.zeros((h, w), dtype=np.uint8)

        if roi.get('type') == 'rectangle':
            rect = roi['rect']
            x1, y1, x2, y2 = [int(v * s) for v, s in
                               zip(rect, [sx, sy, sx, sy])]
            mask[y1:y2, x1:x2] = 255
        elif roi.get('type') == 'polygon':
            pts = np.array(roi['points'], dtype=np.float64)
            pts[:, 0] *= sx
            pts[:, 1] *= sy
            pts = pts.astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
        else:
            return None

        return mask

    def _color_stats(self, hsv_image: np.ndarray,
                     binary_mask: np.ndarray) -> dict:
        """Compute color statistics for pixels under a mask.

        Returns:
            Dict with mean_v, std_v, mean_s, darkness_score (0-1), uniformity_score (0-1)
        """
        v_channel = hsv_image[:, :, 2]
        s_channel = hsv_image[:, :, 1]
        pixels_v = v_channel[binary_mask > 0]
        pixels_s = s_channel[binary_mask > 0]

        if len(pixels_v) == 0:
            return {'mean_v': 255, 'std_v': 255, 'mean_s': 0,
                    'darkness_score': 0.0, 'uniformity_score': 0.0}

        mean_v = float(np.mean(pixels_v))
        std_v = float(np.std(pixels_v))
        mean_s = float(np.mean(pixels_s))

        # Darkness score: 1.0 when mean_v=0, 0.0 when mean_v>=max_brightness
        darkness_score = max(0.0, 1.0 - mean_v / self.max_brightness)

        # Uniformity score: 1.0 when std_v=0, 0.0 when std_v>=max_brightness_std
        uniformity_score = max(0.0, 1.0 - std_v / self.max_brightness_std)

        return {
            'mean_v': mean_v, 'std_v': std_v, 'mean_s': mean_s,
            'darkness_score': darkness_score,
            'uniformity_score': uniformity_score,
        }

    def detect(self, color_image: np.ndarray, depth_image: np.ndarray,
               depth_frame, camera) -> RodDetection | None:
        """Detect a rod in the current frame using FastSAM segmentation.

        Pipeline:
        1. Run FastSAM to get all segment masks
        2. Filter segments by overlap with workspace ROI (>=50%)
        3. Filter by minimum contour area
        4. Score by shape (aspect ratio), darkness, and uniformity
        5. Pick best candidate, compute 3D center + axis from depth

        Args:
            color_image: BGR image (H, W, 3)
            depth_image: Depth in mm (H, W) uint16
            depth_frame: RealSense depth frame for 3D queries
            camera: RealSenseCamera instance for pixel_to_3d

        Returns:
            RodDetection or None if no rod found
        """
        h, w = color_image.shape[:2]
        roi_mask = self._build_roi_mask(h, w)
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # Run FastSAM
        model = self._get_model()
        results = model(color_image, device='cpu', retina_masks=True,
                        imgsz=self.fastsam_imgsz,
                        conf=self.fastsam_conf,
                        iou=self.fastsam_iou,
                        verbose=False)

        if not results or results[0].masks is None:
            self._last_candidates = []
            self._last_masks = None
            self._last_roi_mask = roi_mask
            return None

        masks = results[0].masks.data.cpu().numpy()  # (N, H, W)

        # Store debug info
        self._last_candidates = []
        self._last_masks = masks
        self._last_roi_mask = roi_mask

        best_detection = None
        best_score = 0.0

        for i, seg_mask in enumerate(masks):
            # Resize mask to image dimensions if needed
            if seg_mask.shape != (h, w):
                seg_mask = cv2.resize(seg_mask, (w, h),
                                      interpolation=cv2.INTER_NEAREST)

            binary_mask = (seg_mask > 0.5).astype(np.uint8)
            mask_area = int(binary_mask.sum())

            if mask_area < self.min_area:
                continue

            # Check ROI overlap
            if roi_mask is not None:
                overlap = int((binary_mask & (roi_mask > 0)).sum())
                overlap_ratio = overlap / mask_area
                if overlap_ratio < 0.5:
                    continue
            else:
                overlap_ratio = 1.0

            # Fit minimum bounding rectangle to the mask contour
            contours, _ = cv2.findContours(binary_mask * 255,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            rect = cv2.minAreaRect(contour)
            (cx, cy), (rw, rh), angle = rect

            if rw < rh:
                rw, rh = rh, rw
                angle += 90

            if rh == 0:
                continue

            aspect_ratio = rw / rh

            # Convexity: contour area / convex hull area
            # Straight rod ~0.9+, curvy cable ~0.5-0.7
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            convexity = area / hull_area if hull_area > 0 else 0.0

            # Color analysis
            color_stats = self._color_stats(hsv_image, binary_mask)

            # Reject reason tracking
            reject_reason = None
            if aspect_ratio < self.min_aspect_ratio - 0.01:  # tolerance for float rounding
                reject_reason = f"aspect {aspect_ratio:.1f} < {self.min_aspect_ratio}"
            elif convexity < self.min_convexity:
                reject_reason = f"convexity {convexity:.2f} < {self.min_convexity}"
            elif color_stats['mean_v'] > self.max_brightness:
                reject_reason = f"bright V={color_stats['mean_v']:.0f} > {self.max_brightness}"

            candidate_info = {
                'index': i, 'area': area, 'aspect_ratio': aspect_ratio,
                'convexity': convexity,
                'center': (int(cx), int(cy)), 'size': (rw, rh),
                'angle': angle, 'overlap_ratio': overlap_ratio,
                'mean_v': color_stats['mean_v'],
                'std_v': color_stats['std_v'],
                'mean_s': color_stats['mean_s'],
                'darkness_score': color_stats['darkness_score'],
                'uniformity_score': color_stats['uniformity_score'],
                'reject_reason': reject_reason,
                'passed': reject_reason is None,
            }
            self._last_candidates.append(candidate_info)

            if reject_reason is not None:
                continue

            # Combined score:
            #   shape:      aspect_ratio (higher = more rod-like)
            #   size:       sqrt(area) (prefer larger)
            #   darkness:   0-1 (darker = better)
            #   uniformity: 0-1 (more uniform = better, rods are consistent)
            #   roi:        overlap fraction
            shape_score = aspect_ratio * np.sqrt(area)
            color_score = (0.6 * color_stats['darkness_score'] +
                           0.4 * color_stats['uniformity_score'])
            score = shape_score * (0.3 + 0.7 * color_score) * overlap_ratio

            candidate_info['score'] = score

            # Validate depth is in workspace range before accepting as best
            cx_int, cy_int = int(cx), int(cy)
            depth_at_center = depth_image[
                min(cy_int, h - 1), min(cx_int, w - 1)]
            if depth_at_center > 0 and (
                depth_at_center < self.depth_min_mm or
                depth_at_center > self.depth_max_mm
            ):
                candidate_info['reject_reason'] = (
                    f"depth {depth_at_center}mm outside "
                    f"{self.depth_min_mm}-{self.depth_max_mm}")
                candidate_info['passed'] = False
                continue

            if score > best_score:
                best_score = score

                # Get 3D center point
                center_3d = camera.pixel_to_3d(cx_int, cy_int, depth_frame)

                # Compute 3D axis direction from endpoints of bounding rect
                angle_rad = np.radians(angle)
                dx = int(rw / 2 * np.cos(angle_rad))
                dy = int(rw / 2 * np.sin(angle_rad))
                p1 = camera.pixel_to_3d(cx_int - dx, cy_int - dy, depth_frame)
                p2 = camera.pixel_to_3d(cx_int + dx, cy_int + dy, depth_frame)
                axis = p2 - p1
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 0:
                    axis = axis / axis_norm
                else:
                    axis = np.array([1, 0, 0])  # fallback

                # Confidence from shape + color combined
                shape_conf = min(1.0, aspect_ratio / 8.0)
                color_conf = color_score
                confidence = 0.5 * shape_conf + 0.5 * color_conf

                best_detection = RodDetection(
                    center_3d=center_3d,
                    axis_3d=axis,
                    center_2d=(cx_int, cy_int),
                    contour=contour,
                    confidence=confidence,
                    mask=binary_mask,
                )

        return best_detection

    def draw_detection(self, image: np.ndarray, detection: RodDetection) -> np.ndarray:
        """Draw detection overlay on the image for visualization."""
        vis = image.copy()

        # Draw segment mask as semi-transparent overlay
        if detection.mask is not None:
            overlay = vis.copy()
            overlay[detection.mask > 0] = (0, 255, 0)
            cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)

        cv2.drawContours(vis, [detection.contour], -1, (0, 255, 0), 2)

        # Bounding rect
        rect = cv2.minAreaRect(detection.contour)
        box = cv2.boxPoints(rect).astype(int)
        cv2.drawContours(vis, [box], 0, (255, 0, 255), 2)

        cx, cy = detection.center_2d
        cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)

        # Axis direction arrow
        if detection.contour is not None:
            rect = cv2.minAreaRect(detection.contour)
            (_, _), (rw, rh), angle = rect
            if rw < rh:
                rw, rh = rh, rw
                angle += 90
            angle_rad = np.radians(angle)
            dx = int(rw / 2 * np.cos(angle_rad))
            dy = int(rw / 2 * np.sin(angle_rad))
            cv2.arrowedLine(vis, (cx - dx, cy - dy), (cx + dx, cy + dy),
                            (0, 255, 255), 2, tipLength=0.15)

        label = f"Rod ({detection.confidence:.0%}) z={detection.center_3d[2]:.3f}m"
        cv2.putText(vis, label, (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return vis

    def draw_all_segments(self, image: np.ndarray) -> np.ndarray:
        """Draw all FastSAM segments as a colored overlay with index labels."""
        vis = image.copy()
        if not hasattr(self, '_last_masks') or self._last_masks is None:
            return vis

        h, w = image.shape[:2]
        overlay = vis.copy()
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
            (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
        ]
        for i, mask in enumerate(self._last_masks):
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h),
                                  interpolation=cv2.INTER_NEAREST)
            color = colors[i % len(colors)]
            region = mask > 0.5
            overlay[region] = color
            # Label each segment with its index
            ys, xs = np.where(region)
            if len(xs) > 0:
                cx_label = int(np.mean(xs))
                cy_label = int(np.mean(ys))
                cv2.putText(vis, str(i), (cx_label - 5, cy_label + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
        return vis
