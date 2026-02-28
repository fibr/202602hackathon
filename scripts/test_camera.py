#!/usr/bin/env python3
"""Test RealSense camera connection and display RGB + depth streams."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2
import numpy as np
from vision import RealSenseCamera, RodDetector


def main():
    print("Starting RealSense camera test...")
    camera = RealSenseCamera(width=640, height=480, fps=15)
    detector = RodDetector()

    try:
        camera.start()
        print("Camera started. Press 'q' to quit, 'd' to toggle detection overlay.")

        show_detection = True

        while True:
            color_image, depth_image, depth_frame = camera.get_frames()
            if color_image is None:
                continue

            # Colorize depth for display
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

            # Run detection
            if show_detection:
                detection = detector.detect(color_image, depth_image, depth_frame, camera)
                if detection:
                    color_image = detector.draw_detection(color_image, detection)

            # Stack side by side
            display = np.hstack([color_image, depth_colormap])
            cv2.imshow('RealSense Test (RGB | Depth)', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break
            if cv2.getWindowProperty('RealSense Test (RGB | Depth)', cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord('d'):
                show_detection = not show_detection
                print(f"Detection overlay: {'ON' if show_detection else 'OFF'}")

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Camera stopped.")


if __name__ == "__main__":
    main()
