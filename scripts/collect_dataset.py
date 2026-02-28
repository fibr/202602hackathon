#!/usr/bin/env python3
"""Collect a dataset of rod images for testing detection.

Shows a live camera feed. Press SPACE to capture a frame pair (color + depth).
Images are saved to data/rod_dataset/ with sequential numbering.

Each capture saves:
  {NNN}_color.png   -- BGR color image
  {NNN}_depth.png   -- raw uint16 depth in mm (lossless)
  {NNN}_depth_vis.png -- colorized depth for quick browsing

Usage:
    ./run.sh scripts/collect_dataset.py [--hd]

Controls:
    SPACE       Capture current frame
    d           Toggle detection overlay (run detector on current frame)
    q / Esc     Quit
"""

import sys
import os
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from vision import RealSenseCamera
from vision.rod_detector import RodDetector
from config_loader import load_config

DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'rod_dataset')


def find_next_index(dataset_dir):
    """Find the next available index by scanning existing files."""
    if not os.path.exists(dataset_dir):
        return 0
    existing = [f for f in os.listdir(dataset_dir) if f.endswith('_color.png')]
    if not existing:
        return 0
    indices = []
    for f in existing:
        try:
            indices.append(int(f.split('_')[0]))
        except ValueError:
            pass
    return max(indices) + 1 if indices else 0


def main():
    hd = '--hd' in sys.argv
    width, height = (1280, 720) if hd else (640, 480)

    os.makedirs(DATASET_DIR, exist_ok=True)
    next_idx = find_next_index(DATASET_DIR)

    config = load_config()
    cam_cfg = config.get('camera', {})

    detector = RodDetector(
        min_aspect_ratio=cam_cfg.get('min_aspect_ratio', 2.2),
        min_area=cam_cfg.get('min_area', 500),
        depth_min_mm=cam_cfg.get('depth_min_mm', 6000),
        depth_max_mm=cam_cfg.get('depth_max_mm', 19000),
        max_brightness=cam_cfg.get('max_brightness', 120),
        max_brightness_std=cam_cfg.get('max_brightness_std', 40.0),
        min_convexity=cam_cfg.get('min_convexity', 0.85),
        workspace_roi=cam_cfg.get('workspace_roi'),
    )

    print("=== Rod Dataset Collection ===")
    print(f"Resolution: {width}x{height}")
    print(f"Save directory: {os.path.abspath(DATASET_DIR)}")
    print(f"Next index: {next_idx}")
    print()
    print("SPACE  = capture    d = toggle detection    q = quit")
    print()

    camera = RealSenseCamera(width=width, height=height, fps=15)
    camera.start()

    cv2.namedWindow('Collect Dataset')

    # Let auto-exposure settle
    for _ in range(30):
        camera.get_frames()

    show_detection = False
    count = 0

    try:
        while True:
            color, depth_img, depth_frame = camera.get_frames()
            if color is None:
                continue

            vis = color.copy()

            # Run detection overlay if toggled
            if show_detection:
                detection = detector.detect(color, depth_img, depth_frame, camera)
                if detection:
                    vis = detector.draw_detection(vis, detection)
                # Draw ROI
                roi_mask = getattr(detector, '_last_roi_mask', None)
                if roi_mask is not None:
                    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis, contours, -1, (0, 255, 255), 1)

            # Status bar
            h_img = vis.shape[0]
            status = f"#{next_idx}  |  {count} captured  |  det:{'ON' if show_detection else 'OFF'}  |  SPACE=capture  d=detect  q=quit"
            cv2.rectangle(vis, (0, h_img - 30), (vis.shape[1], h_img), (0, 0, 0), -1)
            cv2.putText(vis, status, (10, h_img - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow('Collect Dataset', vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break
            try:
                if cv2.getWindowProperty('Collect Dataset', cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            if key == ord('d'):
                show_detection = not show_detection
                print(f"  Detection overlay: {'ON' if show_detection else 'OFF'}")

            if key == ord(' '):
                idx_str = f"{next_idx:03d}"
                color_path = os.path.join(DATASET_DIR, f"{idx_str}_color.png")
                depth_path = os.path.join(DATASET_DIR, f"{idx_str}_depth.png")
                depth_vis_path = os.path.join(DATASET_DIR, f"{idx_str}_depth_vis.png")

                cv2.imwrite(color_path, color)
                cv2.imwrite(depth_path, depth_img)
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imwrite(depth_vis_path, depth_colormap)

                print(f"  [{idx_str}] Saved to {DATASET_DIR}/")
                next_idx += 1
                count += 1

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print(f"\nDone. {count} frames captured (indices up to {next_idx - 1}).")
        print(f"Dataset: {os.path.abspath(DATASET_DIR)}")


if __name__ == "__main__":
    main()
