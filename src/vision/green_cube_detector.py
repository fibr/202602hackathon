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

        detections.append(CubeDetection(
            cx=cx, cy=cy, area=area,
            bbox=(x, y, w, h), contour=contour,
        ))

    # Sort by area (largest first)
    detections.sort(key=lambda d: d.area, reverse=True)

    info = {'mask': mask, 'contour_count': len(contours)}
    if debug:
        info['hsv'] = hsv

    return detections, info


def annotate_frame(
    frame: np.ndarray,
    detections: list[CubeDetection],
    label_prefix: str = "Cube",
) -> np.ndarray:
    """Draw detection annotations on a copy of the frame.

    Args:
        frame: BGR image.
        detections: List of CubeDetection.
        label_prefix: Prefix for labels.

    Returns:
        Annotated BGR image (copy of original).
    """
    vis = frame.copy()
    for i, det in enumerate(detections):
        x, y, w, h = det.bbox
        # Draw bounding box
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Draw centroid
        cv2.circle(vis, (det.cx, det.cy), 5, (0, 0, 255), -1)
        # Label
        label = f"{label_prefix} {i+1} ({det.cx},{det.cy})"
        cv2.putText(vis, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return vis
