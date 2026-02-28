#!/usr/bin/env python3
"""Capture HD frame, run FastSAM rod detection, save debug images.

Saves numbered debug images to /tmp/rod_debug/ showing each pipeline stage:
  {timestamp}_1_color.png       -- raw color frame
  {timestamp}_2_depth.png       -- colorized depth map
  {timestamp}_3_roi.png         -- workspace ROI overlay
  {timestamp}_4_segments.png    -- all FastSAM segment masks
  {timestamp}_5_candidates.png  -- shape+color filtered candidates
  {timestamp}_6_detection.png   -- final detection result
  {timestamp}_info.txt          -- full detection log

Usage:
    ./run.sh scripts/snapshot_rod.py [--hd]
"""

import sys
import os
import time
from datetime import datetime
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from vision import RealSenseCamera
from vision.rod_detector import RodDetector, RodDetection
from config_loader import load_config

OUT_DIR = "/tmp/rod_debug"

# --- Drawing helpers ---

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
MAGENTA = (255, 0, 255)
CYAN = (255, 255, 0)


def put_title(image, title, subtitle=None):
    """Draw a title bar at the top of the image."""
    vis = image.copy()
    # Semi-transparent black bar
    bar_h = 40 if subtitle is None else 60
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (vis.shape[1], bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
    cv2.putText(vis, title, (10, 28), FONT, 0.8, WHITE, 2)
    if subtitle:
        cv2.putText(vis, subtitle, (10, 52), FONT, 0.5, YELLOW, 1)
    return vis


def draw_roi_overlay(image, roi_mask):
    """Draw the workspace ROI as a semi-transparent green zone with border."""
    vis = image.copy()
    if roi_mask is None:
        return put_title(vis, "3. ROI", "No workspace ROI configured")

    # Green fill
    overlay = vis.copy()
    overlay[roi_mask > 0] = (0, 180, 0)
    cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

    # Border
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, YELLOW, 2)

    roi_pct = 100.0 * np.count_nonzero(roi_mask) / roi_mask.size
    return put_title(vis, "3. Workspace ROI", f"ROI covers {roi_pct:.1f}% of frame")


def draw_candidates(image, detector, roi_mask):
    """Draw all candidate segments with per-candidate detail labels."""
    vis = image.copy()

    # Draw ROI boundary
    if roi_mask is not None:
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, YELLOW, 1)

    candidates = getattr(detector, '_last_candidates', [])
    masks = getattr(detector, '_last_masks', None)
    h, w = image.shape[:2]

    n_pass = sum(1 for c in candidates if c['passed'])
    n_fail = len(candidates) - n_pass

    for c in candidates:
        passed = c['passed']
        color = GREEN if passed else RED
        cx, cy = c['center']
        rw, rh = c['size']

        # Draw the mask outline for this segment
        if masks is not None and c['index'] < len(masks):
            seg_mask = masks[c['index']]
            if seg_mask.shape != (h, w):
                seg_mask = cv2.resize(seg_mask, (w, h),
                                      interpolation=cv2.INTER_NEAREST)
            binary = (seg_mask > 0.5).astype(np.uint8)
            seg_contours, _ = cv2.findContours(binary * 255,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, seg_contours, -1, color, 2 if passed else 1)

            # Semi-transparent fill for passed candidates
            if passed:
                overlay = vis.copy()
                overlay[binary > 0] = GREEN
                cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)

        # Label
        line1 = f"#{c['index']} ar={c['aspect_ratio']:.1f} cx={c['convexity']:.2f} V={c['mean_v']:.0f}+/-{c['std_v']:.0f}"
        if c['reject_reason']:
            line2 = c['reject_reason']
        else:
            line2 = f"score={c.get('score', 0):.1f}"

        # Background for readability
        for dy_off, text in [(- 22, line1), (-8, line2)]:
            (tw, th), _ = cv2.getTextSize(text, FONT, 0.35, 1)
            cv2.rectangle(vis, (cx - 2, cy + dy_off - th - 1),
                          (cx + tw + 2, cy + dy_off + 2), (0, 0, 0), -1)
            cv2.putText(vis, text, (cx, cy + dy_off), FONT, 0.35, color, 1)

        cv2.circle(vis, (cx, cy), 4, color, -1)

    subtitle = f"{n_pass} passed, {n_fail} rejected (ar>={detector.min_aspect_ratio} cx>={detector.min_convexity} V<={detector.max_brightness})"
    return put_title(vis, "5. Candidates: shape + color filter", subtitle)


def main():
    hd = '--hd' in sys.argv
    width, height = (1280, 720) if hd else (640, 480)

    os.makedirs(OUT_DIR, exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = f"{OUT_DIR}/{ts}"

    config = load_config()
    cam_cfg = config.get('camera', {})

    detector = RodDetector(
        min_aspect_ratio=cam_cfg.get('min_aspect_ratio', 3.0),
        min_area=cam_cfg.get('min_area', 500),
        depth_min_mm=cam_cfg.get('depth_min_mm', 6000),
        depth_max_mm=cam_cfg.get('depth_max_mm', 19000),
        max_brightness=cam_cfg.get('max_brightness', 120),
        max_brightness_std=cam_cfg.get('max_brightness_std', 40.0),
        min_convexity=cam_cfg.get('min_convexity', 0.85),
        workspace_roi=cam_cfg.get('workspace_roi'),
    )

    print(f"Starting camera ({width}x{height})...")
    camera = RealSenseCamera(width=width, height=height, fps=15)
    camera.start()

    # Let auto-exposure settle
    for _ in range(30):
        camera.get_frames()
    time.sleep(0.5)

    print("Capturing frame...")
    color_image, depth_image, depth_frame = camera.get_frames()
    camera.stop()

    if color_image is None:
        print("ERROR: No frame captured")
        return

    print(f"Frame: {color_image.shape[1]}x{color_image.shape[0]}")

    info_lines = [
        f"Timestamp: {ts}",
        f"Resolution: {width}x{height}",
        f"",
        f"--- Detector config ---",
        f"min_aspect_ratio: {detector.min_aspect_ratio}",
        f"min_area: {detector.min_area}",
        f"depth_range: {detector.depth_min_mm}-{detector.depth_max_mm} mm",
        f"max_brightness (V): {detector.max_brightness}",
        f"max_brightness_std: {detector.max_brightness_std}",
        f"min_convexity: {detector.min_convexity}",
        f"workspace_roi: {detector.workspace_roi is not None}",
        f"fastsam: conf={detector.fastsam_conf} iou={detector.fastsam_iou} imgsz={detector.fastsam_imgsz}",
        f"",
    ]

    # --- 1. Color ---
    color_vis = put_title(color_image, "1. Color input", f"{width}x{height}")
    cv2.imwrite(f"{prefix}_1_color.png", color_vis)

    # --- 2. Depth ---
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    valid = depth_image[depth_image > 0]
    depth_sub = ""
    if len(valid) > 0:
        depth_sub = f"range: {int(valid.min())}-{int(valid.max())} mm, median: {int(np.median(valid))} mm"
    depth_vis = put_title(depth_colormap, "2. Depth", depth_sub)
    cv2.imwrite(f"{prefix}_2_depth.png", depth_vis)

    # --- Run detection ---
    print("Running FastSAM detection...")
    t0 = time.time()
    detection = detector.detect(color_image, depth_image, depth_frame, camera)
    t_detect = time.time() - t0
    print(f"Inference time: {t_detect:.2f}s")
    info_lines.append(f"--- Pipeline results ---")
    info_lines.append(f"Inference time: {t_detect:.2f}s")

    # --- 3. ROI ---
    roi_mask = getattr(detector, '_last_roi_mask', None)
    roi_vis = draw_roi_overlay(color_image, roi_mask)
    cv2.imwrite(f"{prefix}_3_roi.png", roi_vis)

    # --- 4. Segments ---
    masks = getattr(detector, '_last_masks', None)
    n_masks = len(masks) if masks is not None else 0
    segments_vis = detector.draw_all_segments(color_image)
    segments_vis = put_title(segments_vis, "4. FastSAM segments",
                             f"{n_masks} segments found in {t_detect:.2f}s")
    cv2.imwrite(f"{prefix}_4_segments.png", segments_vis)
    info_lines.append(f"Total segments: {n_masks}")

    # --- 5. Candidates ---
    candidates_vis = draw_candidates(color_image, detector, roi_mask)
    cv2.imwrite(f"{prefix}_5_candidates.png", candidates_vis)

    candidates = getattr(detector, '_last_candidates', [])
    n_candidates = len(candidates)
    n_pass = sum(1 for c in candidates if c['passed'])
    info_lines.append(f"Candidates in ROI (area >= {detector.min_area}): {n_candidates}")
    info_lines.append(f"Passed shape+color filter: {n_pass}")
    info_lines.append(f"")

    for c in candidates:
        status = "PASS" if c['passed'] else "FAIL"
        reason = f"  ({c['reject_reason']})" if c['reject_reason'] else ""
        info_lines.append(
            f"  [{status}] seg#{c['index']}: area={c['area']:.0f} "
            f"ar={c['aspect_ratio']:.1f} cx={c['convexity']:.2f} "
            f"size=({c['size'][0]:.0f}x{c['size'][1]:.0f}) "
            f"V={c['mean_v']:.0f}+/-{c['std_v']:.0f} S={c['mean_s']:.0f} "
            f"dark={c['darkness_score']:.2f} unif={c['uniformity_score']:.2f} "
            f"overlap={c['overlap_ratio']:.0%} "
            f"score={c.get('score', '-')}{reason}")

    # --- 6. Detection ---
    if detection:
        info_lines.append(f"")
        info_lines.append(f"=== DETECTION ===")
        info_lines.append(f"Center 2D: {detection.center_2d}")
        info_lines.append(f"Center 3D: [{detection.center_3d[0]:.4f}, {detection.center_3d[1]:.4f}, {detection.center_3d[2]:.4f}] m")
        info_lines.append(f"Axis 3D:   [{detection.axis_3d[0]:.4f}, {detection.axis_3d[1]:.4f}, {detection.axis_3d[2]:.4f}]")
        info_lines.append(f"Confidence: {detection.confidence:.0%}")

        det_vis = detector.draw_detection(color_image, detection)
        det_vis = put_title(det_vis, "6. Detection",
                            f"conf={detection.confidence:.0%} z={detection.center_3d[2]:.3f}m @ {detection.center_2d}")

        cv2.imwrite(f"{prefix}_6_detection.png", det_vis)
        print(f"\nROD DETECTED!")
        print(f"  Center 2D: {detection.center_2d}")
        print(f"  Center 3D: [{detection.center_3d[0]:.4f}, {detection.center_3d[1]:.4f}, {detection.center_3d[2]:.4f}] m")
        print(f"  Axis:      [{detection.axis_3d[0]:.4f}, {detection.axis_3d[1]:.4f}, {detection.axis_3d[2]:.4f}]")
        print(f"  Confidence: {detection.confidence:.0%}")
    else:
        det_vis = put_title(color_image, "6. Detection", "NO ROD DETECTED")
        cv2.imwrite(f"{prefix}_6_detection.png", det_vis)
        info_lines.append(f"")
        info_lines.append(f"=== NO DETECTION ===")
        print("\nNo rod detected.")

    # --- Info file ---
    info_text = '\n'.join(info_lines)
    with open(f"{prefix}_info.txt", 'w') as f:
        f.write(info_text)
    print(f"\n{info_text}")
    print(f"\nDebug images: {prefix}_*.png")


if __name__ == "__main__":
    main()
