#!/usr/bin/env python3
"""Run rod detection on saved dataset images (no camera needed).

Loads color + depth PNGs from data/rod_dataset/, runs the FastSAM
detector on each, and saves annotated results.

Output per image in data/rod_dataset/results/:
  {NNN}_result.png   -- 2x2 grid: color | segments | candidates | detection

Also prints a summary table and saves it to results/summary.txt.

Usage:
    ./run.sh scripts/eval_dataset.py [--dir data/rod_dataset] [--vis]

Options:
    --dir PATH   Dataset directory (default: data/rod_dataset)
    --vis        Show each result in a window (press any key to advance)
"""

import sys
import os
import time
import glob
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from vision.rod_detector import RodDetector
from config_loader import load_config


class OfflineCamera:
    """Fake camera that deprojects pixels using pinhole model + saved depth."""

    def __init__(self, width: int, height: int,
                 fx: float = None, fy: float = None,
                 cx: float = None, cy: float = None):
        self.width = width
        self.height = height
        # Default to approximate RealSense D435i intrinsics at 1280x720
        self.fx = fx or (width * 0.7)
        self.fy = fy or (width * 0.7)
        self.cx = cx or (width / 2.0)
        self.cy = cy or (height / 2.0)

    def pixel_to_3d(self, x: int, y: int, depth_image: np.ndarray) -> np.ndarray:
        """Deproject pixel to 3D using pinhole model.

        Args:
            x: pixel column
            y: pixel row
            depth_image: uint16 depth in mm (used as the "depth_frame" stand-in)

        Returns:
            np.ndarray: [x, y, z] in meters in camera coordinate frame
        """
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        z_m = float(depth_image[y, x]) / 1000.0
        x_m = (x - self.cx) * z_m / self.fx
        y_m = (y - self.cy) * z_m / self.fy
        return np.array([x_m, y_m, z_m])


def make_grid(images, titles, grid_w=2):
    """Arrange images into a labeled grid."""
    # Resize all to same size
    h, w = images[0].shape[:2]
    cells = []
    for img, title in zip(images, titles):
        cell = img.copy()
        if cell.shape[:2] != (h, w):
            cell = cv2.resize(cell, (w, h))
        if len(cell.shape) == 2:
            cell = cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)
        # Title bar
        overlay = cell.copy()
        cv2.rectangle(overlay, (0, 0), (w, 32), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, cell, 0.4, 0, cell)
        cv2.putText(cell, title, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cells.append(cell)

    # Pad to fill grid
    while len(cells) % grid_w != 0:
        cells.append(np.zeros_like(cells[0]))

    rows = []
    for i in range(0, len(cells), grid_w):
        rows.append(np.hstack(cells[i:i + grid_w]))
    return np.vstack(rows)


def draw_candidates_vis(image, detector, roi_mask):
    """Draw candidate contours with labels."""
    vis = image.copy()
    h, w = image.shape[:2]

    if roi_mask is not None:
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 255), 1)

    candidates = getattr(detector, '_last_candidates', [])
    masks = getattr(detector, '_last_masks', None)

    for c in candidates:
        passed = c['passed']
        color = (0, 255, 0) if passed else (0, 0, 255)
        cx, cy = c['center']

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

        label = f"#{c['index']} ar={c['aspect_ratio']:.1f} cx={c['convexity']:.2f} V={c['mean_v']:.0f}"
        cv2.putText(vis, label, (cx, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    return vis


def main():
    # Parse args
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'rod_dataset')
    show_vis = False
    args = sys.argv[1:]
    while args:
        if args[0] == '--dir' and len(args) > 1:
            dataset_dir = args[1]
            args = args[2:]
        elif args[0] == '--vis':
            show_vis = True
            args = args[1:]
        else:
            args = args[1:]

    dataset_dir = os.path.abspath(dataset_dir)
    results_dir = os.path.join(dataset_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Find all color images
    color_files = sorted(glob.glob(os.path.join(dataset_dir, '*_color.png')))
    if not color_files:
        print(f"No *_color.png files found in {dataset_dir}")
        return

    print(f"=== Offline Rod Detection Evaluation ===")
    print(f"Dataset: {dataset_dir}")
    print(f"Images:  {len(color_files)}")
    print()

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

    # Detect resolution from first image
    first = cv2.imread(color_files[0])
    h, w = first.shape[:2]
    camera = OfflineCamera(w, h)
    print(f"Resolution: {w}x{h}")
    print()

    summary_lines = []
    summary_lines.append(f"{'#':>3}  {'Det':>3}  {'Conf':>5}  {'AR':>4}  {'CX':>4}  {'V':>3}  {'Center 2D':>12}  {'Z(m)':>6}  {'Time':>5}  Candidates")
    summary_lines.append("-" * 90)

    n_detected = 0
    total_time = 0.0

    for color_path in color_files:
        basename = os.path.basename(color_path)
        idx_str = basename.split('_')[0]

        depth_path = color_path.replace('_color.png', '_depth.png')
        if not os.path.exists(depth_path):
            print(f"  [{idx_str}] SKIP — no depth image")
            continue

        color_img = cv2.imread(color_path)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        t0 = time.time()
        detection = detector.detect(color_img, depth_img, depth_img, camera)
        dt = time.time() - t0
        total_time += dt

        candidates = getattr(detector, '_last_candidates', [])
        n_pass = sum(1 for c in candidates if c['passed'])
        roi_mask = getattr(detector, '_last_roi_mask', None)

        # Build result grid
        seg_vis = detector.draw_all_segments(color_img)
        cand_vis = draw_candidates_vis(color_img, detector, roi_mask)

        if detection:
            n_detected += 1
            det_vis = detector.draw_detection(color_img, detection)
            cx, cy = detection.center_2d
            z = detection.center_3d[2]
            conf = detection.confidence

            # Find winning candidate's stats
            winning = next((c for c in candidates if c['passed'] and c.get('score')), {})
            ar = winning.get('aspect_ratio', 0)
            cvx = winning.get('convexity', 0)
            mv = winning.get('mean_v', 0)

            det_str = "YES"
            summary_lines.append(
                f"{idx_str:>3}  {det_str:>3}  {conf:>5.0%}  {ar:>4.1f}  {cvx:>4.2f}  {mv:>3.0f}  ({cx:>4},{cy:>4})  {z:>6.3f}  {dt:>4.2f}s  {n_pass}/{len(candidates)}")
            status = f"DETECTED conf={conf:.0%} z={z:.3f}m"
        else:
            det_vis = color_img.copy()
            cv2.putText(det_vis, "NO DETECTION", (w // 4, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            summary_lines.append(
                f"{idx_str:>3}   NO      -     -     -     -            -       -  {dt:>4.2f}s  {n_pass}/{len(candidates)}")
            status = "no detection"

        masks = getattr(detector, '_last_masks', None)
        n_segs = len(masks) if masks is not None else 0
        grid = make_grid(
            [color_img, seg_vis, cand_vis, det_vis],
            [f"[{idx_str}] Color", f"Segments ({n_segs})",
             f"Candidates ({n_pass} pass)", status],
        )

        result_path = os.path.join(results_dir, f"{idx_str}_result.png")
        cv2.imwrite(result_path, grid)
        print(f"  [{idx_str}] {status} ({dt:.2f}s)")

        if show_vis:
            cv2.imshow('Eval', grid)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q') or key == 27:
                break

    if show_vis:
        cv2.destroyAllWindows()

    # Summary
    print()
    print(f"=== Summary ===")
    print(f"Detected: {n_detected}/{len(color_files)} ({100*n_detected/len(color_files):.0f}%)")
    print(f"Total time: {total_time:.1f}s  Avg: {total_time/len(color_files):.2f}s/frame")
    print()
    for line in summary_lines:
        print(line)

    summary_text = '\n'.join(summary_lines)
    with open(os.path.join(results_dir, 'summary.txt'), 'w') as f:
        f.write(f"Detected: {n_detected}/{len(color_files)}\n")
        f.write(f"Total time: {total_time:.1f}s  Avg: {total_time/len(color_files):.2f}s/frame\n\n")
        f.write(summary_text)

    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()
