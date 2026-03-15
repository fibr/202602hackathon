"""Live Digital Twin view: real-time 3D arm visualization.

Shows a lightweight 3D skeleton of the ARM101 that mirrors the actual
arm's joint angles in real time.  Designed to help detect servo
misconfiguration (wrong signs, offsets, or mapping) by comparing the
digital twin's pose to the physical arm.

Features:
- Real-time FK-based 3D skeleton rendering (no Isaac Sim required)
- Side-by-side multi-view angles (front + side + top)
- Commanded vs actual angle comparison when commands are sent
- Angle table with color-coded deltas
- Rotate view with mouse drag or keyboard

Usage:
    ./run.sh src/unified_gui.py --view live_twin
"""

import time
import threading
import numpy as np
import cv2

from gui.views.base import BaseView, ViewRegistry

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _try_create_renderer():
    """Create ArmRenderer, returning None if deps are missing."""
    try:
        from gui.arm_renderer import ArmRenderer
        return ArmRenderer()
    except Exception as e:
        print(f"  WARNING: Could not create ArmRenderer: {e}")
        return None


@ViewRegistry.register
class LiveTwinView(BaseView):
    """Real-time digital twin visualization for ARM101."""

    view_id = 'live_twin'
    view_name = 'Live Twin'
    description = 'Real-time 3D arm visualization (lightweight, no Isaac Sim)'
    needs_camera = False
    needs_robot = True
    headless_ok = False

    # View presets: (name, azimuth, elevation)
    VIEW_PRESETS = [
        ('Front', 0.0, 20.0),
        ('Side', -90.0, 20.0),
        ('Top', 0.0, 89.0),
        ('Iso', -45.0, 25.0),
    ]

    def __init__(self, app):
        super().__init__(app)
        self._renderer = None
        self._actual_angles = None
        self._commanded_angles = None
        self._last_commanded_time = 0.0
        self._lock = threading.Lock()
        self._active_view = 3  # default to Iso
        self._dragging = False
        self._drag_start = (0, 0)
        self._drag_az_start = 0.0
        self._drag_el_start = 0.0
        self._show_multi = True  # show multi-view by default
        self._show_table = True  # show angle table
        self._show_camera = False  # optional camera overlay
        self._poll_rate = 0.05  # 20 Hz
        self._poll_thread = None
        self._running = False
        self._ee_trail = []  # end-effector position trail
        self._max_trail = 100

    def setup(self):
        self.app.ensure_robot()
        self._renderer = _try_create_renderer()

        if self._renderer is None:
            return

        # Start polling thread
        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

        # Optionally start camera for side-by-side
        if not getattr(self.app.args, 'no_camera', False):
            try:
                self.app.ensure_camera()
                if self.app.camera is not None:
                    self._show_camera = True
            except Exception:
                pass

        # Install command observer on the robot driver
        self._install_observer()

    def _install_observer(self):
        """Hook into robot.write_all_angles to capture commanded angles."""
        if self.app.robot is None:
            return
        robot = self.app.robot
        if not hasattr(robot, '_original_write_all_angles'):
            original = robot.write_all_angles

            def observed_write_all_angles(angles, speed=None):
                with self._lock:
                    self._commanded_angles = np.array(angles[:6], dtype=float)
                    self._last_commanded_time = time.time()
                return original(angles, speed)

            robot._original_write_all_angles = original
            robot.write_all_angles = observed_write_all_angles

        # Also hook move_joints for the control panel path
        if not hasattr(robot, '_original_move_joints'):
            original_mj = robot.move_joints

            def observed_move_joints(angles, speed=None):
                with self._lock:
                    self._commanded_angles = np.array(angles[:6], dtype=float)
                    self._last_commanded_time = time.time()
                return original_mj(angles, speed)

            robot._original_move_joints = original_mj
            robot.move_joints = observed_move_joints

    def _uninstall_observer(self):
        """Remove command observer hooks."""
        if self.app.robot is None:
            return
        robot = self.app.robot
        if hasattr(robot, '_original_write_all_angles'):
            robot.write_all_angles = robot._original_write_all_angles
            del robot._original_write_all_angles
        if hasattr(robot, '_original_move_joints'):
            robot.move_joints = robot._original_move_joints
            del robot._original_move_joints

    def _poll_loop(self):
        """Background thread: poll actual arm angles."""
        while self._running:
            if self.app.robot is not None:
                try:
                    angles = self.app.robot.get_angles()
                    if angles is not None:
                        with self._lock:
                            self._actual_angles = np.array(angles, dtype=float)
                            # Track EE trail
                            if self._renderer is not None:
                                positions = self._renderer.get_joint_positions(
                                    self._actual_angles[:5])
                                if positions:
                                    ee_pos = positions[-1].copy()
                                    self._ee_trail.append(ee_pos)
                                    if len(self._ee_trail) > self._max_trail:
                                        self._ee_trail = self._ee_trail[-self._max_trail:]
                except Exception:
                    pass
            time.sleep(self._poll_rate)

    def update(self, canvas):
        vw = self.app.view_width
        vh = self.app.canvas_height

        # Dark background
        canvas[:vh, :vw] = (30, 30, 35)

        if self._renderer is None:
            cv2.putText(canvas, 'ArmRenderer not available',
                        (20, vh // 2), FONT, 0.5, (0, 80, 220), 1)
            cv2.putText(canvas, 'Requires pinocchio + URDF',
                        (20, vh // 2 + 25), FONT, 0.4, (120, 120, 120), 1)
            return

        with self._lock:
            actual = self._actual_angles.copy() if self._actual_angles is not None else None
            commanded = self._commanded_angles.copy() if self._commanded_angles is not None else None
            cmd_age = time.time() - self._last_commanded_time
            trail = list(self._ee_trail)

        if actual is None:
            cv2.putText(canvas, 'Waiting for arm data...',
                        (20, vh // 2), FONT, 0.5, (150, 150, 150), 1)
            if self.app.robot is None:
                cv2.putText(canvas, 'No robot connected',
                            (20, vh // 2 + 25), FONT, 0.4, (0, 80, 220), 1)
            return

        # Fade out commanded overlay after 5 seconds
        if cmd_age > 5.0:
            commanded = None

        if self._show_multi:
            self._draw_multi_view(canvas, actual, commanded, trail, vw, vh)
        else:
            self._draw_single_view(canvas, actual, commanded, trail, vw, vh)

        # Angle table
        if self._show_table:
            table_x = 10
            table_y = vh - 120
            self._renderer.render_angle_table(
                canvas, actual, commanded, x=table_x, y=table_y)

        # HUD
        self._draw_hud(canvas, actual, commanded, vw, vh)

    def _draw_single_view(self, canvas, actual, commanded, trail, vw, vh):
        """Draw single large view."""
        r = self._renderer
        name, az, el = self.VIEW_PRESETS[self._active_view]
        r.azimuth = az
        r.elevation = el
        r.width = vw
        r.height = vh
        r.zoom = min(vw, vh) * 2.5

        # Draw ground reference grid
        self._draw_grid(canvas, r, vw, vh)

        # Draw EE trail
        self._draw_trail(canvas, r, trail)

        r.render_comparison(canvas, actual, commanded)

    def _draw_multi_view(self, canvas, actual, commanded, trail, vw, vh):
        """Draw 2x2 multi-view layout."""
        r = self._renderer
        half_w = vw // 2
        half_h = vh // 2

        views = [
            ('Front', 0.0, 20.0, 0, 0),
            ('Side', -90.0, 20.0, half_w, 0),
            ('Top', 0.0, 89.0, 0, half_h),
            ('Iso', -45.0, 25.0, half_w, half_h),
        ]

        for name, az, el, ox, oy in views:
            r.azimuth = az
            r.elevation = el
            r.width = half_w
            r.height = half_h
            r.zoom = min(half_w, half_h) * 2.2

            # Draw border
            cv2.rectangle(canvas, (ox, oy), (ox + half_w - 1, oy + half_h - 1),
                          (50, 50, 55), 1)

            # Label
            cv2.putText(canvas, name, (ox + 5, oy + 15),
                        FONT, 0.35, (120, 120, 120), 1)

            # Draw grid
            self._draw_grid(canvas, r, half_w, half_h, ox, oy)

            # Draw trail
            self._draw_trail(canvas, r, trail, ox, oy)

            # Render arm
            r.render_comparison(canvas, actual, commanded,
                                offset_x=ox, offset_y=oy)

    def _draw_grid(self, canvas, renderer, w, h, ox=0, oy=0):
        """Draw a simple ground reference grid."""
        # Draw a small grid at z=0 plane
        grid_size = 0.15  # meters
        steps = 5
        pts = []
        for i in range(-steps, steps + 1):
            # Lines parallel to X
            p1 = np.array([i * grid_size / steps, -grid_size, 0.0])
            p2 = np.array([i * grid_size / steps, grid_size, 0.0])
            pts.append((p1, p2))
            # Lines parallel to Y
            p1 = np.array([-grid_size, i * grid_size / steps, 0.0])
            p2 = np.array([grid_size, i * grid_size / steps, 0.0])
            pts.append((p1, p2))

        for p1, p2 in pts:
            s1 = renderer.project_3d_to_2d([p1])[0]
            s2 = renderer.project_3d_to_2d([p2])[0]
            s1 = (s1[0] + ox, s1[1] + oy)
            s2 = (s2[0] + ox, s2[1] + oy)
            cv2.line(canvas, s1, s2, (40, 40, 45), 1)

    def _draw_trail(self, canvas, renderer, trail, ox=0, oy=0):
        """Draw end-effector position trail."""
        if len(trail) < 2:
            return
        pts_2d = renderer.project_3d_to_2d(trail)
        pts_2d = [(x + ox, y + oy) for x, y in pts_2d]
        for i in range(1, len(pts_2d)):
            alpha = int(80 * i / len(pts_2d))  # fade in
            color = (alpha, alpha, 50 + alpha)
            cv2.line(canvas, pts_2d[i - 1], pts_2d[i], color, 1)

    def _draw_hud(self, canvas, actual, commanded, vw, vh):
        """Draw heads-up display with controls help."""
        # Title bar
        cv2.putText(canvas, 'Live Digital Twin', (10, 15),
                    FONT, 0.45, (255, 200, 100), 1)

        # Connection status
        status = 'Connected' if self.app.robot is not None else 'No Robot'
        color = (0, 255, 100) if self.app.robot is not None else (0, 80, 220)
        cv2.putText(canvas, status, (vw - 100, 15),
                    FONT, 0.35, color, 1)

        # FK position
        if actual is not None and self._renderer is not None:
            positions = self._renderer.get_joint_positions(actual[:5])
            if positions:
                ee = positions[-1] * 1000  # meters to mm
                cv2.putText(canvas,
                            f'EE: ({ee[0]:.0f}, {ee[1]:.0f}, {ee[2]:.0f}) mm',
                            (10, 30), FONT, 0.32, (150, 200, 255), 1)

        # Mismatch indicator
        if commanded is not None and actual is not None:
            n = min(len(actual), len(commanded), 5)
            max_delta = max(abs(actual[i] - commanded[i]) for i in range(n))
            if max_delta > 10:
                # Flashing red warning
                if int(time.time() * 4) % 2:
                    cv2.putText(canvas,
                                f'MISMATCH: {max_delta:.1f} deg',
                                (vw // 2 - 80, 30), FONT, 0.45,
                                (0, 0, 255), 1)

        # Controls help
        help_y = vh - 8
        help_text = 'M=multi/single  T=table  C=trail  1-4=view  R=reset'
        cv2.putText(canvas, help_text, (10, help_y),
                    FONT, 0.28, (80, 80, 80), 1)
        # Hint for camera overlay mode
        cv2.putText(canvas, 'AR overlay: use "camera_overlay" view',
                    (10, vh - 20), FONT, 0.27, (60, 60, 80), 1)

    def handle_key(self, key):
        if key == ord('m') or key == ord('M'):
            self._show_multi = not self._show_multi
            return True

        if key == ord('t') or key == ord('T'):
            self._show_table = not self._show_table
            return True

        if key == ord('c') or key == ord('C'):
            with self._lock:
                self._ee_trail.clear()
            return True

        if key == ord('r') or key == ord('R'):
            # Reset view
            self._active_view = 3
            with self._lock:
                self._ee_trail.clear()
                self._commanded_angles = None
            return True

        # View presets 1-4
        if ord('1') <= key <= ord('4'):
            self._active_view = key - ord('1')
            self._show_multi = False
            return True

        return False

    def handle_mouse(self, event, x, y, flags):
        if self._renderer is None:
            return False

        if not self._show_multi:
            # Single view: drag to rotate
            if event == cv2.EVENT_LBUTTONDOWN:
                self._dragging = True
                self._drag_start = (x, y)
                self._drag_az_start = self.VIEW_PRESETS[self._active_view][1]
                self._drag_el_start = self.VIEW_PRESETS[self._active_view][2]
                return True

            if event == cv2.EVENT_MOUSEMOVE and self._dragging:
                dx = x - self._drag_start[0]
                dy = y - self._drag_start[1]
                # Modify the preset tuple temporarily
                name = self.VIEW_PRESETS[self._active_view][0]
                new_az = self._drag_az_start + dx * 0.5
                new_el = max(-89, min(89, self._drag_el_start + dy * 0.5))
                self.VIEW_PRESETS[self._active_view] = (name, new_az, new_el)
                return True

            if event == cv2.EVENT_LBUTTONUP:
                self._dragging = False
                return True

        # Scroll to zoom
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self._renderer.zoom *= 1.1
            else:
                self._renderer.zoom /= 1.1
            return True

        return False

    def cleanup(self):
        self._running = False
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=1.0)
        self._uninstall_observer()
