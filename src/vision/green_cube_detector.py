"""Green cube detector using HSV color segmentation.

Detects green cubes on a white/light surface using color thresholding
in HSV space, contour detection, and geometric filtering.

Returns a list of CubeDetection dataclasses with pixel coordinates,
bounding boxes, and estimated centroids.
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class CubeDetection:
    """A single detected green cube."""
    cx: int              # centroid x (pixels)
    cy: int              # centroid y (pixels)
    area: float          # contour area (pixels^2)
    bbox: tuple          # (x, y, w, h) bounding box
    contour: np.ndarray  # original contour points
    yaw_deg: float = 0.0       # estimated yaw from minAreaRect (degrees)
    aspect_ratio: float = 1.0  # bbox width/height ratio
    solidity: float = 1.0      # contour area / convex hull area


# Default HSV range for green cubes (tuned for bright green on white table)
DEFAULT_HSV_LOW = np.array([35, 60, 60])
DEFAULT_HSV_HIGH = np.array([85, 255, 255])

# Minimum contour area to consider (filters noise)
DEFAULT_MIN_AREA = 300

# Aspect ratio range for cubes (should be roughly square-ish from most views)
DEFAULT_MAX_ASPECT = 3.0


def detect_green_cubes(
    frame: np.ndarray,
    hsv_low: np.ndarray = None,
    hsv_high: np.ndarray = None,
    min_area: float = DEFAULT_MIN_AREA,
    max_aspect: float = DEFAULT_MAX_ASPECT,
    debug: bool = False,
) -> tuple[list[CubeDetection], dict]:
    """Detect green cubes in a BGR image.

    Args:
        frame: BGR image (numpy array).
        hsv_low: Lower HSV threshold (default: [35, 60, 60]).
        hsv_high: Upper HSV threshold (default: [85, 255, 255]).
        min_area: Minimum contour area in pixels.
        max_aspect: Maximum bbox aspect ratio (width/height or height/width).
        debug: If True, return debug images in the info dict.

    Returns:
        (detections, info): List of CubeDetection, and a dict with debug info.
    """
    if hsv_low is None:
        hsv_low = DEFAULT_HSV_LOW
    if hsv_high is None:
        hsv_high = DEFAULT_HSV_HIGH

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create green mask
    mask = cv2.inRange(hsv, hsv_low, hsv_high)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Bounding box and aspect ratio check
        x, y, w, h = cv2.boundingRect(contour)
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > max_aspect:
            continue

        # Centroid via moments
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # Yaw from minimum-area rotated rectangle.
        # OpenCV minAreaRect returns angle in [-90, 0) for the rotation of
        # the rect's width-edge relative to the horizontal.  For a square
        # (or near-square) cube, orientation repeats every 90°, so we
        # normalize to [-45, +45) — the smallest rotation needed to align
        # the gripper with a cube edge.
        yaw_deg = 0.0
        if len(contour) >= 5:
            rect = cv2.minAreaRect(contour)
            rw, rh = rect[1]
            raw_angle = rect[2]  # in [-90, 0)
            # OpenCV convention: angle is between the width-edge and
            # horizontal.  If width < height, the rect is "tall" and the
            # angle refers to the short side — add 90° to get the long-
            # side angle.  For cubes (rw ≈ rh) this doesn't matter much,
            # but handle it correctly for robustness.
            if rw < rh:
                raw_angle += 90.0
            # Normalize to [-45, +45) for square symmetry
            yaw_deg = ((raw_angle + 45.0) % 90.0) - 45.0

        # Solidity (how square-like the contour is)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        detections.append(CubeDetection(
            cx=cx, cy=cy, area=area,
            bbox=(x, y, w, h), contour=contour,
            yaw_deg=yaw_deg,
            aspect_ratio=max(w, h) / max(min(w, h), 1),
            solidity=solidity,
        ))

    # Sort by area (largest first)
    detections.sort(key=lambda d: d.area, reverse=True)

    info = {'mask': mask, 'contour_count': len(contours)}
    if debug:
        info['hsv'] = hsv

    return detections, info


def select_target_cube(
    detections: list[CubeDetection],
    mode: str = 'largest',
    click_xy: tuple = None,
    reference_pos: np.ndarray = None,
) -> int:
    """Select the best target cube from a list of detections.

    Args:
        detections: List of CubeDetection objects.
        mode: Selection mode - 'largest', 'closest_to_click', 'closest_to_center',
              'closest_to_ref' (closest to reference_pos in pixel space).
        click_xy: (x, y) pixel position for 'closest_to_click' mode.
        reference_pos: (x, y) reference position for 'closest_to_ref' mode.

    Returns:
        Index of the selected cube, or -1 if no detections.
    """
    if not detections:
        return -1

    if mode == 'largest':
        # Already sorted by area descending
        return 0

    if mode == 'closest_to_click' and click_xy is not None:
        cx, cy = click_xy
        dists = [(d.cx - cx)**2 + (d.cy - cy)**2 for d in detections]
        return int(np.argmin(dists))

    if mode == 'closest_to_center':
        # Assumes 640x480 if no frame size available
        cx, cy = 320, 240
        dists = [(d.cx - cx)**2 + (d.cy - cy)**2 for d in detections]
        return int(np.argmin(dists))

    if mode == 'closest_to_ref' and reference_pos is not None:
        rx, ry = reference_pos[0], reference_pos[1]
        dists = [(d.cx - rx)**2 + (d.cy - ry)**2 for d in detections]
        return int(np.argmin(dists))

    return 0  # fallback to largest


def annotate_frame(
    frame: np.ndarray,
    detections: list[CubeDetection],
    label_prefix: str = "Cube",
    target_idx: int = -1,
) -> np.ndarray:
    """Draw detection annotations on a copy of the frame.

    Args:
        frame: BGR image.
        detections: List of CubeDetection.
        label_prefix: Prefix for labels.
        target_idx: Index of the target cube to highlight (-1 = none).

    Returns:
        Annotated BGR image (copy of original).
    """
    vis = frame.copy()
    for i, det in enumerate(detections):
        x, y, w, h = det.bbox
        is_target = (i == target_idx)

        # Draw bounding box - target in cyan, others in green
        color = (0, 255, 255) if is_target else (0, 255, 0)
        thickness = 3 if is_target else 1
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)

        # Draw centroid
        cv2.circle(vis, (det.cx, det.cy), 6 if is_target else 4,
                   (0, 0, 255) if is_target else (0, 128, 255), -1)

        # Target marker - crosshair
        if is_target:
            cv2.drawMarker(vis, (det.cx, det.cy), (0, 255, 255),
                           cv2.MARKER_CROSS, 20, 2)

        # Draw orientation line (yaw) through the centroid
        if abs(det.yaw_deg) > 0.01 or is_target:
            import math
            yaw_rad = math.radians(det.yaw_deg)
            line_len = max(w, h) // 2
            dx = int(line_len * math.cos(yaw_rad))
            dy = int(line_len * math.sin(yaw_rad))
            cv2.line(vis, (det.cx - dx, det.cy - dy),
                     (det.cx + dx, det.cy + dy),
                     (0, 165, 255), 2)  # orange line for yaw

        # Label
        tag = " [TARGET]" if is_target else ""
        yaw_str = f" y={det.yaw_deg:.0f}°" if abs(det.yaw_deg) > 0.5 else ""
        label = f"{label_prefix} {i+1} A={det.area:.0f}{yaw_str}{tag}"
        cv2.putText(vis, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Detection count
    cv2.putText(vis, f"{len(detections)} cube(s) detected",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return vis
