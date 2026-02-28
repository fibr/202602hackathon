#!/usr/bin/env python3
"""Click to define a workspace ROI on the live camera feed.

Click 4 corners (or 2 for a rectangle) to define the region where
the robot can reach and the rod might be. Saves to config/settings.yaml.

Usage:
    ./run.sh scripts/select_roi.py [--hd]

Controls:
    Click       Add corner point
    r           Reset points
    s           Save ROI to config/settings.yaml
    q / Esc     Quit
"""

import sys
import os
import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from vision import RealSenseCamera

points = []
current_frame = None


def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"  Point {len(points)}: ({x}, {y})")


def draw_overlay(frame):
    vis = frame.copy()
    # Draw existing points
    for i, pt in enumerate(points):
        cv2.circle(vis, pt, 5, (0, 0, 255), -1)
        cv2.putText(vis, str(i + 1), (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw polygon if 3+ points
    if len(points) >= 3:
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        # Semi-transparent fill
        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)
    elif len(points) == 2:
        # Show rectangle from 2 points
        cv2.rectangle(vis, points[0], points[1], (0, 255, 0), 2)

    # Instructions
    h = vis.shape[0]
    cv2.putText(vis, f"Points: {len(points)}  |  Click corners, 's' save, 'r' reset, 'q' quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return vis


def main():
    global points, current_frame
    hd = '--hd' in sys.argv
    width, height = (1280, 720) if hd else (640, 480)

    print("=== Workspace ROI Selection ===")
    print(f"Resolution: {width}x{height}")
    print()
    print("Click corners of the workspace area (table surface).")
    print("Use 2 points for a rectangle, or 3+ for a polygon.")
    print("Press 's' to save, 'r' to reset, 'q' to quit.")
    print()

    camera = RealSenseCamera(width=width, height=height, fps=15)
    camera.start()

    cv2.namedWindow('Select ROI')
    cv2.setMouseCallback('Select ROI', mouse_callback)

    # Let auto-exposure settle
    for _ in range(30):
        camera.get_frames()

    try:
        while True:
            color, depth_img, depth_frame = camera.get_frames()
            if color is None:
                continue

            current_frame = color
            vis = draw_overlay(color)

            # Show depth at each point
            if depth_img is not None:
                for i, pt in enumerate(points):
                    x, y = pt
                    x = max(0, min(x, width - 1))
                    y = max(0, min(y, height - 1))
                    d = depth_img[y, x]
                    cv2.putText(vis, f"{d}mm", (pt[0] + 8, pt[1] + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            cv2.imshow('Select ROI', vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break
            if cv2.getWindowProperty('Select ROI', cv2.WND_PROP_VISIBLE) < 1:
                break

            if key == ord('r'):
                points = []
                print("  Reset.")

            if key == ord('s'):
                if len(points) < 2:
                    print("  Need at least 2 points!")
                    continue

                if len(points) == 2:
                    # Rectangle: store as [x1,y1,x2,y2]
                    x1 = min(points[0][0], points[1][0])
                    y1 = min(points[0][1], points[1][1])
                    x2 = max(points[0][0], points[1][0])
                    y2 = max(points[0][1], points[1][1])
                    roi_data = {
                        'type': 'rectangle',
                        'rect': [x1, y1, x2, y2],
                        'resolution': [width, height],
                    }
                    print(f"\n  Rectangle ROI: [{x1}, {y1}] to [{x2}, {y2}]")
                else:
                    # Polygon
                    roi_data = {
                        'type': 'polygon',
                        'points': [[int(p[0]), int(p[1])] for p in points],
                        'resolution': [width, height],
                    }
                    print(f"\n  Polygon ROI: {len(points)} points")

                # Also compute depth stats within ROI
                if depth_img is not None:
                    mask = np.zeros(depth_img.shape, dtype=np.uint8)
                    if len(points) == 2:
                        mask[y1:y2, x1:x2] = 255
                    else:
                        pts = np.array(points, dtype=np.int32)
                        cv2.fillPoly(mask, [pts], 255)
                    roi_depths = depth_img[(mask > 0) & (depth_img > 0)]
                    if len(roi_depths) > 0:
                        roi_data['depth_min_mm'] = int(roi_depths.min())
                        roi_data['depth_max_mm'] = int(roi_depths.max())
                        roi_data['depth_median_mm'] = int(np.median(roi_depths))
                        print(f"  Depth in ROI: {roi_data['depth_min_mm']}-{roi_data['depth_max_mm']} mm (median {roi_data['depth_median_mm']})")

                # Save to settings.yaml
                settings_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.yaml')
                settings = {}
                if os.path.exists(settings_path):
                    with open(settings_path, 'r') as f:
                        settings = yaml.safe_load(f) or {}

                if 'camera' not in settings:
                    settings['camera'] = {}
                settings['camera']['workspace_roi'] = roi_data

                with open(settings_path, 'w') as f:
                    yaml.dump(settings, f, default_flow_style=False)

                print(f"  Saved to {settings_path}")

                # Also save annotated screenshot
                screenshot = draw_overlay(current_frame)
                cv2.imwrite('/tmp/rod_debug/roi_selection.png', screenshot)
                print(f"  Screenshot: /tmp/rod_debug/roi_selection.png")

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
