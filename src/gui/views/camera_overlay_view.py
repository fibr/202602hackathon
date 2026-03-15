"""Camera Overlay view: AR-style digital twin overlay on live camera feed.

Projects the FK skeleton of the ARM101 onto the live camera frame using the
hand-eye calibration transform (T_camera_to_base) and camera intrinsics.
This provides an augmented-reality view where the user can directly compare
the physical arm with the FK skeleton projected into camera space.

Requires:
  - A calibrated camera (config/calibration.yaml + config/camera_intrinsics.yaml)
  - A robot connection (for live joint angles)
  - An active camera (--no-camera disables the overlay)

Key bindings:
  O  — toggle overlay on/off
  A  — toggle alpha (solid vs translucent skeleton)
  T  — toggle angle table
  G  — toggle grid lines in overlay
  R  — reset / clear trail

Usage:
    ./run.sh src/unified_gui.py --view camera_overlay
"""

from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np
import yaml

from gui.views.base import BaseView, ViewRegistry

FONT = cv2.FONT_HERSHEY_SIMPLEX

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))

_CALIB_PATH = os.path.join(_PROJECT_ROOT, 'config', 'calibration.yaml')
_INTRINSICS_PATH = os.path.join(_PROJECT_ROOT, 'config', 'camera_intrinsics.yaml')


def _load_calibration():
    """Load T_camera_to_base from calibration.yaml.

    Returns:
        T_camera_to_base as a 4x4 np.ndarray, or None if not found.
    """
    if not os.path.exists(_CALIB_PATH):
        return None
    with open(_CALIB_PATH) as f:
        data = yaml.safe_load(f)
    mat = data.get('T_camera_to_base')
    if mat is None:
        return None
    return np.array(mat, dtype=np.float64)


def _load_intrinsics(camera=None):
    """Load camera intrinsics from camera_intrinsics.yaml or RealSense camera.

    Args:
        camera: Optional camera object with .intrinsics attribute.

    Returns:
        (K, dist_coeffs) where K is 3x3 and dist_coeffs is (5,), or (None, None).
    """
    # Try to get from camera object first (most accurate)
    if camera is not None and hasattr(camera, 'intrinsics') and camera.intrinsics is not None:
        intr = camera.intrinsics
        if hasattr(intr, 'camera_matrix') and intr.camera_matrix is not None:
            K = np.array(intr.camera_matrix, dtype=np.float64)
            dist = np.array(intr.dist_coeffs, dtype=np.float64) if intr.dist_coeffs else np.zeros(5)
            return K, dist

    # Fall back to config file
    if os.path.exists(_INTRINSICS_PATH):
        with open(_INTRINSICS_PATH) as f:
            data = yaml.safe_load(f)
        K = np.array(data['camera_matrix'], dtype=np.float64)
        dist = np.array(data.get('dist_coeffs', [0, 0, 0, 0, 0]), dtype=np.float64)
        return K, dist

    return None, None


def _try_create_renderer():
    """Create ArmRenderer, returning None if deps are missing."""
    try:
        from gui.arm_renderer import ArmRenderer
        return ArmRenderer()
    except Exception as e:
        print(f"  WARNING: Could not create ArmRenderer: {e}")
        return None


@ViewRegistry.register
class CameraOverlayView(BaseView):
    """AR-style digital twin overlay on the live camera feed."""

    view_id = 'camera_overlay'
    view_name = 'Camera Overlay'
    description = 'AR skeleton overlay on camera feed (requires calibration)'
    needs_camera = True
    needs_robot = True
    headless_ok = False

    def __init__(self, app):
        super().__init__(app)
        self._renderer = None
        self._K = None
        self._dist = None
        self._T_base_to_camera = None   # inverse of T_camera_to_base

        self._actual_angles = None
        self._lock = threading.Lock()
        self._poll_thread = None
        self._running = False
        self._poll_rate = 0.05   # 20 Hz

        # Display toggles
        self._show_overlay = True   # AR skeleton on/off
        self._alpha = 0.85          # skeleton opacity
        self._show_table = True     # angle readout table
        self._show_axes = True      # coordinate frame axes

        # Status messages
        self._calib_status = 'Not loaded'
        self._intrinsics_status = 'Not loaded'

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self):
        self.app.ensure_robot()
        self.app.ensure_camera()

        self._renderer = _try_create_renderer()

        # Load calibration
        T_cam_to_base = _load_calibration()
        if T_cam_to_base is not None:
            self._T_base_to_camera = np.linalg.inv(T_cam_to_base)
            self._calib_status = 'OK'
        else:
            self._calib_status = 'Missing (run hand-eye calib first)'

        # Load intrinsics (try camera object first)
        self._K, self._dist = _load_intrinsics(self.app.camera)
        if self._K is not None:
            self._intrinsics_status = 'OK'
        else:
            self._intrinsics_status = 'Missing (run checkerboard calib)'

        # Start polling thread
        if self._renderer is not None and self.app.robot is not None:
            self._running = True
            self._poll_thread = threading.Thread(
                target=self._poll_loop, daemon=True)
            self._poll_thread.start()

    def cleanup(self):
        self._running = False
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=1.0)

    # ------------------------------------------------------------------
    # Background polling
    # ------------------------------------------------------------------

    def _poll_loop(self):
        """Background thread: poll actual arm angles at ~20 Hz."""
        while self._running:
            if self.app.robot is not None:
                try:
                    angles = self.app.robot.get_angles()
                    if angles is not None:
                        with self._lock:
                            self._actual_angles = np.array(angles, dtype=float)
                except Exception:
                    pass
            time.sleep(self._poll_rate)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def update(self, canvas):
        vw = self.app.view_width
        vh = self.app.canvas_height

        # Get camera frame
        frame = self._get_camera_frame()

        if frame is None:
            # No camera — show dark background with message
            canvas[:vh, :vw] = (30, 30, 35)
            self._draw_no_camera(canvas, vw, vh)
            return

        # Resize/crop frame to fill view area
        fh, fw = frame.shape[:2]
        scale = min(vw / fw, vh / fh)
        disp_w = int(fw * scale)
        disp_h = int(fh * scale)
        frame_disp = cv2.resize(frame, (disp_w, disp_h))

        # Pad to fill area
        ox = (vw - disp_w) // 2
        oy = (vh - disp_h) // 2
        canvas[:vh, :vw] = (10, 10, 10)
        canvas[oy:oy + disp_h, ox:ox + disp_w] = frame_disp

        # Get current angles
        with self._lock:
            actual = self._actual_angles.copy() if self._actual_angles is not None else None

        # Overlay skeleton
        ready = (self._renderer is not None and
                 self._K is not None and
                 self._T_base_to_camera is not None and
                 actual is not None)

        if self._show_overlay and ready:
            # Create a working copy of the displayed frame area for blending
            roi = canvas[oy:oy + disp_h, ox:ox + disp_w].copy()

            # Build scaled intrinsics for the resized display image
            K_scaled = self._K.copy()
            K_scaled[0, 0] *= scale   # fx
            K_scaled[1, 1] *= scale   # fy
            K_scaled[0, 2] *= scale   # cx
            K_scaled[1, 2] *= scale   # cy

            # Project and draw onto display ROI
            self._renderer.draw_on_camera_frame(
                roi,
                actual[:5],
                K_scaled,
                self._dist,
                self._T_base_to_camera,
                alpha=self._alpha,
            )

            # Draw coordinate axes at base origin if axes enabled
            if self._show_axes:
                self._draw_origin_axes(roi, K_scaled)

            canvas[oy:oy + disp_h, ox:ox + disp_w] = roi

        # HUD overlays on full canvas
        self._draw_hud(canvas, actual, ready, vw, vh)

        if self._show_table and actual is not None and self._renderer is not None:
            self._renderer.render_angle_table(
                canvas, actual, x=10, y=vh - 115)

    def _get_camera_frame(self):
        """Grab current BGR frame from the app camera."""
        if self.app.camera is None:
            return None
        try:
            result = self.app.camera.get_frames()
            if result is None:
                return None
            color, *_ = result
            return color
        except Exception:
            return None

    def _draw_origin_axes(self, roi, K_scaled):
        """Draw XYZ axes at the robot base origin in the camera image."""
        axis_len = 0.05  # 50 mm in meters
        origin = np.array([[0., 0., 0.]])
        x_tip = np.array([[axis_len, 0., 0.]])
        y_tip = np.array([[0., axis_len, 0.]])
        z_tip = np.array([[0., 0., axis_len]])

        R = self._T_base_to_camera[:3, :3]
        t = self._T_base_to_camera[:3, 3]
        rvec, _ = cv2.Rodrigues(R)
        tvec = t.reshape(3, 1)

        def _proj(pt):
            p, _ = cv2.projectPoints(
                pt.reshape(1, 1, 3), rvec, tvec,
                K_scaled, self._dist)
            return tuple(map(int, p[0][0].round()))

        try:
            o = _proj(origin)
            px = _proj(x_tip)
            py = _proj(y_tip)
            pz = _proj(z_tip)
            h, w = roi.shape[:2]
            for pt, color, label in [(px, (0, 0, 255), 'X'),
                                     (py, (0, 255, 0), 'Y'),
                                     (pz, (255, 0, 0), 'Z')]:
                if (0 <= o[0] < w and 0 <= o[1] < h and
                        0 <= pt[0] < w and 0 <= pt[1] < h):
                    cv2.arrowedLine(roi, o, pt, color, 2, tipLength=0.3)
                    cv2.putText(roi, label, pt, FONT, 0.4, color, 1)
        except Exception:
            pass

    def _draw_no_camera(self, canvas, vw, vh):
        """Placeholder when camera is unavailable."""
        cy = vh // 2
        cv2.putText(canvas, 'No camera available', (20, cy),
                    FONT, 0.6, (0, 80, 220), 1)
        cv2.putText(canvas, 'Start without --no-camera flag',
                    (20, cy + 28), FONT, 0.4, (120, 120, 120), 1)
        self._draw_hud(canvas, None, False, vw, vh)

    def _draw_hud(self, canvas, actual, ready, vw, vh):
        """Draw status overlay and keyboard hints."""
        # Title bar
        cv2.putText(canvas, 'Camera Overlay', (10, 16),
                    FONT, 0.48, (255, 200, 100), 1)

        # Calibration status badges
        calib_ok = self._calib_status == 'OK'
        intr_ok = self._intrinsics_status == 'OK'

        def _badge(text, ok, x, y):
            color = (0, 200, 80) if ok else (0, 80, 220)
            cv2.putText(canvas, text, (x, y), FONT, 0.32, color, 1)

        _badge(f'Calib: {self._calib_status}', calib_ok, 10, 32)
        _badge(f'Intr:  {self._intrinsics_status}', intr_ok, 10, 46)

        # Robot connection
        robot_ok = self.app.robot is not None
        robot_txt = 'Robot: Connected' if robot_ok else 'Robot: Disconnected'
        robot_col = (0, 200, 80) if robot_ok else (0, 80, 220)
        cv2.putText(canvas, robot_txt, (vw - 160, 16),
                    FONT, 0.32, robot_col, 1)

        # Overlay on/off badge
        ov_txt = 'OVL:ON' if self._show_overlay else 'OVL:OFF'
        ov_col = (0, 255, 160) if self._show_overlay else (100, 100, 100)
        cv2.putText(canvas, ov_txt, (vw - 160, 32),
                    FONT, 0.32, ov_col, 1)

        # EE position from FK
        if actual is not None and self._renderer is not None:
            try:
                positions = self._renderer.get_joint_positions(actual[:5])
                if positions:
                    ee = positions[-1] * 1000   # m -> mm
                    cv2.putText(
                        canvas,
                        f'EE: ({ee[0]:.0f}, {ee[1]:.0f}, {ee[2]:.0f}) mm',
                        (10, 62), FONT, 0.33, (150, 200, 255), 1)
            except Exception:
                pass

        # Alpha readout
        cv2.putText(canvas, f'alpha={self._alpha:.0%}',
                    (vw - 160, 48), FONT, 0.30, (130, 130, 130), 1)

        # Help bar
        help_text = 'O=overlay  A=alpha  T=table  G=axes  R=reset'
        cv2.putText(canvas, help_text, (10, vh - 8),
                    FONT, 0.28, (80, 80, 80), 1)

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def handle_key(self, key):
        if key == ord('o') or key == ord('O'):
            self._show_overlay = not self._show_overlay
            return True

        if key == ord('a') or key == ord('A'):
            # Cycle through alpha levels: 1.0 -> 0.7 -> 0.4 -> 1.0
            if self._alpha >= 0.9:
                self._alpha = 0.7
            elif self._alpha >= 0.6:
                self._alpha = 0.4
            else:
                self._alpha = 1.0
            return True

        if key == ord('t') or key == ord('T'):
            self._show_table = not self._show_table
            return True

        if key == ord('g') or key == ord('G'):
            self._show_axes = not self._show_axes
            return True

        if key == ord('r') or key == ord('R'):
            with self._lock:
                self._actual_angles = None
            return True

        return False
