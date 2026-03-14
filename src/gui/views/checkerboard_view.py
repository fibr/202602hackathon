"""Checkerboard hand-eye calibration view (embedded in unified GUI).

Replaces the subprocess-based launcher for detect_checkerboard.py.
Shares the app's camera and robot connections rather than opening new ones.

Workflow:
  1. Jog the arm so the TCP tip touches a spot on the checkerboard.
  2. Left-click on the TCP tip in the camera image.
  3. The click ray is intersected with the board plane for 3-D camera coords.
  4. Repeat 4+ times spread across the board.
  5. Press Enter (or the "Solve HandEye" button) to solve + save.

The right panel provides arm jog controls (identical to the Control Panel view)
plus buttons for intrinsics calibration, ground-plane capture, and hand-eye
solve.  Keyboard shortcuts mirror the standalone detect_checkerboard.py script.

Additional keys:
  i  = capture intrinsics frame
  g  = capture ground-plane sample
  u  = undo last hand-eye point
  n  = clear all hand-eye points
  p  = print robot pose to console
  Enter = solve hand-eye calibration
  Esc = leave view

Mouse:
  Left-click in camera area = record calibration point (need board detected)
  Panel area clicks route to the robot control panel as usual
"""

import os
import sys
import time

import cv2
import numpy as np

from gui.views.base import BaseView, ViewRegistry
from gui.robot_controls import RobotControlPanel, PANEL_WIDTH

FONT = cv2.FONT_HERSHEY_SIMPLEX

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, 'scripts')


def _import_dcb():
    """Lazy-import detect_checkerboard helpers.  Raises ImportError on failure."""
    if _SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, _SCRIPTS_DIR)
    import detect_checkerboard as _dcb  # noqa: F401
    return _dcb


# ---------------------------------------------------------------------------

@ViewRegistry.register
class CheckerboardCalibView(BaseView):
    """Interactive hand-eye calibration using a printed checkerboard."""

    view_id = 'checkerboard'
    view_name = 'Checkerboard'
    description = 'Hand-eye via checkerboard'
    needs_camera = True
    needs_robot = False   # camera-only mode is supported
    headless_ok = False
    show_in_sidebar = False  # reached via the Calibration menu

    def __init__(self, app):
        super().__init__(app)
        self._dcb = None
        self._panel = None
        self._cam_width = 640
        self._cam_height = 480
        self._click_point = None      # set by handle_mouse, consumed by update
        self._error_msg = ''

        # --- Calibration state ---
        self._pairs = []              # (p_cam_3d_m, p_robot_3d_m, pixel_xy)
        self._current_corners = None
        self._current_detection = None
        self._current_T_board = None
        self._current_reproj_err = None

        # Intrinsics calibration state
        self._intr_frames = []        # [(obj_pts, img_pts)] legacy path
        self._intr_detections = []    # [BoardDetection] for BoardDetector path
        self._intr_path = os.path.join(_PROJECT_ROOT, 'config',
                                       'camera_intrinsics.yaml')

        # Ground-plane calibration state
        self._plane_samples = []      # [(normal_vec, distance)]

        # Robot joint overlay (loaded from saved calibration if present)
        self._robot_overlay = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self):
        # Lazy import
        try:
            self._dcb = _import_dcb()
        except ImportError as exc:
            self._error_msg = f'Import error: {exc}'
            return

        # Initialise board detector from config
        try:
            self._dcb._board_detector = (
                self._dcb.BoardDetector.from_config(self.app.config))
            print(f'  Board: {self._dcb._board_detector.describe()}')
        except Exception as exc:
            print(f'  WARNING: BoardDetector init failed: {exc}')
            self._dcb._board_detector = None

        # Ensure camera
        self.app.ensure_camera()
        if self.app.camera is None:
            self._error_msg = 'No camera available for checkerboard calibration'
            return

        self._cam_width = self.app.camera.width
        self._cam_height = self.app.camera.height

        # Resize app view to camera + control panel (sidebar added by app)
        self.app.view_width = self._cam_width + PANEL_WIDTH
        self.app.view_height = self._cam_height

        # Ensure robot (optional)
        self.app.ensure_robot()
        robot = self.app.robot

        # Robot control panel
        self._panel = RobotControlPanel(
            robot,
            panel_x=self._cam_width,
            panel_height=self._cam_height)

        if robot:
            speed = self.app.config.get('robot', {}).get('speed_percent', 30)
            self._panel.speed = speed
            self._panel.status_msg = 'Touch TCP to board, then click on it'
        else:
            self._panel.status_msg = 'Camera-only mode'

        # Custom panel buttons (mirrors detect_checkerboard.py)
        self._panel.add_button(
            lambda: (
                f"Capture Intr "
                f"({len(self._intr_detections) or len(self._intr_frames)})"),
            self._capture_intrinsics_frame,
            color=(100, 80, 0))

        self._panel.add_button(
            lambda: (
                f"Calibrate Intr "
                f"({len(self._intr_detections) or len(self._intr_frames)})"),
            self._calibrate_intrinsics,
            color=(0, 100, 100))

        self._panel.add_button(
            lambda: (
                f"Visualize Intr "
                f"({len(self._intr_detections) or len(self._intr_frames)})"),
            self._visualize_intrinsics,
            color=(100, 0, 100))

        self._panel.add_button(
            lambda: f"Capture Plane ({len(self._plane_samples)})",
            self._capture_plane_sample,
            color=(80, 80, 0))

        self._panel.add_button(
            lambda: f"Save Plane ({len(self._plane_samples)})",
            self._save_plane,
            color=(0, 100, 100))

        self._panel.add_button(
            lambda: f"Solve HandEye ({len(self._pairs)})",
            self._solve_handeye,
            color=(0, 100, 0))

        # Load existing calibration for robot-joint overlay
        calib_path = os.path.join(_PROJECT_ROOT, 'config', 'calibration.yaml')
        if os.path.exists(calib_path):
            try:
                from calibration import CoordinateTransform
                from visualization import RobotOverlay
                transform = CoordinateTransform()
                transform.load(calib_path)
                gripper_cfg = self.app.config.get('gripper', {})
                self._robot_overlay = RobotOverlay(
                    T_camera_to_base=transform.T_camera_to_base,
                    tool_length_mm=gripper_cfg.get('tool_length_mm', 120.0),
                    base_offset_mm=transform.base_offset_mm,
                    base_rpy_deg=transform.base_rpy_deg,
                )
                print('  Loaded calibration for robot overlay')
            except Exception as exc:
                print(f'  WARNING: Could not load robot overlay: {exc}')

    # ------------------------------------------------------------------
    def update(self, canvas):
        if self._error_msg:
            vw = self.app.view_width
            vh = self.app.canvas_height
            canvas[:vh, :vw] = (30, 30, 35)
            cv2.putText(canvas, 'Checkerboard Calibration', (20, 35),
                        FONT, 0.6, (255, 200, 100), 1)
            cv2.putText(canvas, self._error_msg, (20, 80),
                        FONT, 0.5, (0, 80, 220), 1)
            cv2.putText(canvas, 'Press ESC or click another view to go back.',
                        (20, 110), FONT, 0.38, (130, 130, 130), 1)
            return

        color_image, _, _ = self.app.get_camera_frame()
        if color_image is None:
            return

        w, h = self._cam_width, self._cam_height
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        found, corners, detection = self._dcb.detect_corners(gray)

        display = color_image.copy()
        reproj_err = None

        if found:
            self._current_corners = corners
            self._current_detection = detection
            self._current_T_board, _, reproj_err = self._dcb.compute_board_pose(
                corners, self.app.camera.intrinsics, detection)
            self._current_reproj_err = reproj_err

            if (self._dcb._board_detector is not None
                    and detection is not None):
                self._dcb._board_detector.draw_corners(display, detection)
            else:
                cv2.drawChessboardCorners(
                    display,
                    (self._dcb.BOARD_COLS, self._dcb.BOARD_ROWS),
                    corners, found)
        else:
            self._current_corners = None
            self._current_detection = None
            self._current_T_board = None
            self._current_reproj_err = None

        # Process pending click (set by handle_mouse, consumed here)
        if self._click_point is not None:
            cx, cy = self._click_point
            self._click_point = None
            self._process_click(cx, cy)

        # Draw recorded calibration points
        for i, (_, _, px) in enumerate(self._pairs):
            cv2.circle(display, px, 8, (0, 255, 255), 2)
            cv2.putText(display, str(i + 1), (px[0] + 10, px[1] - 5),
                        FONT, 0.5, (0, 255, 255), 2)

        # Robot joint overlay
        if self._robot_overlay and self.app.camera.intrinsics is not None:
            robot = self.app.robot
            if robot:
                angles = robot.get_angles()
                if angles:
                    display = self._robot_overlay.draw_joints(
                        display, np.array(angles), self.app.camera.intrinsics)
                else:
                    display = self._robot_overlay.draw_base_marker(
                        display, self.app.camera.intrinsics)
            else:
                display = self._robot_overlay.draw_base_marker(
                    display, self.app.camera.intrinsics)

        # Top HUD bar
        if found and reproj_err is not None:
            board_status = f'Board: reproj {reproj_err:.2f}px'
        elif found:
            board_status = 'Board OK'
        else:
            board_status = 'No board'

        n_intr = (len(self._intr_detections)
                  if self._intr_detections else len(self._intr_frames))
        intr_str = f'  Intr:{n_intr}' if n_intr else ''
        plane_str = (f'  Plane:{len(self._plane_samples)}'
                     if self._plane_samples else '')
        jog_str = ' JOG' if self._panel and self._panel.jogging else ''
        spd = self._panel.speed if self._panel else 0
        bar_text = (f'{len(self._pairs)} pts |{intr_str}{plane_str} | '
                    f'Spd:{spd}%{jog_str} | {board_status}')
        cv2.rectangle(display, (0, 0), (w, 32), (0, 0, 0), -1)
        cv2.putText(display, bar_text, (10, 22), FONT, 0.5,
                    (0, 255, 0) if found else (0, 0, 255), 1)

        # Bottom help bar
        help_lines = [
            '[i] capture intrinsics  [g] capture plane  [click] hand-eye pt  [p] pose',
            '[Enter] solve hand-eye  [u] undo  [n] clear  [Esc] quit',
        ]
        line_h = 22
        bar_h = line_h * len(help_lines) + 8
        cv2.rectangle(display, (0, h - bar_h), (w, h), (0, 0, 0), -1)
        for i, line in enumerate(help_lines):
            y = h - bar_h + 4 + line_h * (i + 1)
            cv2.putText(display, line, (10, y), FONT, 0.4, (220, 220, 220), 1)

        # Compose: camera area + panel
        canvas[0:h, 0:w] = display
        if self._panel:
            self._panel.draw(canvas)

    # ------------------------------------------------------------------
    def handle_key(self, key):
        # Panel keys take priority (jogging, speed, gripper …)
        if self._panel and self._panel.handle_key(key):
            return True

        if key == ord('p') and self.app.robot:
            pose = self.app.robot.get_pose()
            angles = self.app.robot.get_angles()
            if pose:
                print(f'  Pose:   {", ".join(f"{v:.2f}" for v in pose)}')
            if angles:
                print(f'  Joints: {", ".join(f"{v:.2f}" for v in angles)}')
            if (pose or angles) and self._panel:
                self._panel.status_msg = 'Pose printed to console'
            return True

        if key == ord('u') and self._pairs:
            self._pairs.pop()
            msg = f'Undid -> {len(self._pairs)} pts remain'
            if self._panel:
                self._panel.status_msg = msg
            print(f'  {msg}')
            return True

        if key == ord('n'):
            self._pairs.clear()
            msg = 'Cleared all points'
            if self._panel:
                self._panel.status_msg = msg
            print(f'  {msg}')
            return True

        if key == ord('i'):
            self._capture_intrinsics_frame()
            return True

        if key == ord('g'):
            self._capture_plane_sample()
            return True

        if key == 13:  # Enter
            self._solve_handeye()
            return True

        return False

    # ------------------------------------------------------------------
    def handle_mouse(self, event, x, y, flags):
        # Panel area
        if self._panel and x >= self._cam_width:
            return self._panel.handle_mouse(event, x, y, flags)

        # Camera area: left-click records a calibration correspondence
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click_point = (x, y)
            return True

        return False

    # ------------------------------------------------------------------
    def cleanup(self):
        # Stop any ongoing jog for Dobot (arm101 handles its own stop)
        if self.app.robot:
            robot_type = self.app.config.get('robot_type')
            if robot_type != 'arm101':
                try:
                    self.app.robot.send('MoveJog()')
                except Exception:
                    pass

        # Restore default app view dimensions so other views aren't affected
        self.app.view_width = 640
        self.app.view_height = 480

    # ------------------------------------------------------------------
    # Private calibration helpers
    # ------------------------------------------------------------------

    def _process_click(self, cx, cy):
        """Ray-plane intersection + robot pose → hand-eye correspondence."""
        if self._current_T_board is None:
            msg = 'No board detected — need board for ray-plane intersection'
            if self._panel:
                self._panel.status_msg = msg
            print(f'  {msg}')
            return

        robot = self.app.robot
        if not robot:
            print('  No robot connected — cannot record calibration point')
            return

        p_cam = self._dcb.ray_plane_intersect(
            (cx, cy), self.app.camera.intrinsics, self._current_T_board)
        if p_cam is None:
            msg = 'Ray parallel to board plane — try a different angle'
            if self._panel:
                self._panel.status_msg = msg
            print(f'  {msg}')
            return

        pose = robot.get_pose()
        if pose is None:
            msg = "ERROR: can't read robot pose"
            if self._panel:
                self._panel.status_msg = msg
            print(f'  {msg}')
            return

        p_robot_m = np.array(pose[:3]) / 1000.0
        self._pairs.append((p_cam, p_robot_m, (cx, cy)))
        msg = (f'Pt {len(self._pairs)}: '
               f'cam=[{p_cam[0]:.3f},{p_cam[1]:.3f},{p_cam[2]:.3f}] '
               f'robot=[{pose[0]:.1f},{pose[1]:.1f},{pose[2]:.1f}]')
        if self._panel:
            self._panel.status_msg = msg
        print(f'  {msg}')

    # ------------------------------------------------------------------
    def _capture_intrinsics_frame(self):
        """Capture the current board detection for intrinsics calibration."""
        if self._current_corners is None:
            msg = "No board detected — can't capture"
            if self._panel:
                self._panel.status_msg = msg
            return

        dcb = self._dcb
        if dcb._board_detector is not None and self._current_detection is not None:
            self._intr_detections.append(self._current_detection)
            n = len(self._current_detection.corners)
            partial = ' (partial)' if self._current_detection.is_partial else ''
            msg = (f'Intrinsics frame {len(self._intr_detections)}: '
                   f'{n} corners{partial}')
        else:
            # Legacy checkerboard path
            n = len(self._current_corners)
            i_cols, i_rows = dcb.BOARD_COLS, dcb.BOARD_ROWS
            if n != dcb.BOARD_ROWS * dcb.BOARD_COLS:
                matched = False
                for cc, rr in [(dcb.BOARD_COLS, dcb.BOARD_ROWS - 2),
                               (dcb.BOARD_COLS - 2, dcb.BOARD_ROWS),
                               (dcb.BOARD_COLS - 2, dcb.BOARD_ROWS - 2)]:
                    if cc * rr == n:
                        i_cols, i_rows = cc, rr
                        matched = True
                        break
                if not matched:
                    msg = f'Unexpected corner count {n}, skipping'
                    if self._panel:
                        self._panel.status_msg = msg
                    return
            obj_pts = np.zeros((i_rows * i_cols, 3), dtype=np.float32)
            for rr in range(i_rows):
                for cc in range(i_cols):
                    obj_pts[rr * i_cols + cc] = [
                        cc * dcb.SQUARE_SIZE_M, rr * dcb.SQUARE_SIZE_M, 0]
            self._intr_frames.append((obj_pts, self._current_corners.copy()))
            msg = f'Intrinsics frame {len(self._intr_frames)} captured'

        if self._panel:
            self._panel.status_msg = msg
        print(f'  {msg}')

    # ------------------------------------------------------------------
    def _calibrate_intrinsics(self):
        """Run cv2.calibrateCamera (or BoardDetector path) and save results."""
        from vision import CameraIntrinsics

        n = (len(self._intr_detections) if self._intr_detections
             else len(self._intr_frames))
        if n < 5:
            msg = f'Need 5+ frames (have {n})'
            if self._panel:
                self._panel.status_msg = msg
            return

        w, h = self._cam_width, self._cam_height
        dcb = self._dcb

        if self._intr_detections and dcb._board_detector is not None:
            print(f'\n=== Calibrating intrinsics from '
                  f'{len(self._intr_detections)} frames '
                  f'({dcb._board_detector.describe()}) ===')
            try:
                ret, calib_intr = dcb._board_detector.calibrate_intrinsics(
                    self._intr_detections, (w, h))
                print(f'  Reprojection error: {ret:.4f} px')
                calib_intr.save(self._intr_path)
                self.app.camera.intrinsics = calib_intr
                msg = (f'Intrinsics saved! reproj={ret:.3f}px '
                       f'({len(self._intr_detections)} frames)')
            except Exception as exc:
                msg = f'Calibration failed: {exc}'
        else:
            obj_points = [f[0] for f in self._intr_frames]
            img_points = [f[1] for f in self._intr_frames]
            print(f'\n=== Calibrating intrinsics from '
                  f'{len(self._intr_frames)} frames ===')
            ret, mtx, dist, _, _ = cv2.calibrateCamera(
                obj_points, img_points, (w, h), None, None)
            print(f'  Reprojection error: {ret:.4f} px')
            calib_intr = CameraIntrinsics(
                fx=mtx[0, 0], fy=mtx[1, 1],
                ppx=mtx[0, 2], ppy=mtx[1, 2],
                coeffs=dist.ravel().tolist())
            calib_intr.width = w
            calib_intr.height = h
            calib_intr.save(self._intr_path)
            self.app.camera.intrinsics = calib_intr
            msg = f'Intrinsics saved! reproj={ret:.3f}px'

        if self._panel:
            self._panel.status_msg = msg
        print(f'  {msg}')

    # ------------------------------------------------------------------
    def _visualize_intrinsics(self):
        """Open an interactive zoom/pan window showing reprojection errors."""
        n_det = len(self._intr_detections)
        n_leg = len(self._intr_frames)
        if n_det == 0 and n_leg == 0:
            if self._panel:
                self._panel.status_msg = 'No intrinsics frames captured yet'
            return

        intr = self.app.camera.intrinsics
        cam_mtx = intr.camera_matrix
        dist_c = intr.dist_coeffs
        w, h = self._cam_width, self._cam_height
        dcb = self._dcb

        # Compute per-frame reprojection data
        frame_data = []
        if n_det and dcb._board_detector is not None:
            for i, det in enumerate(self._intr_detections):
                obj_pts = dcb._board_detector.get_object_points(det)
                corners_2d = det.corners.reshape(-1, 2)
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, corners_2d.astype(np.float64), cam_mtx, dist_c)
                if not ok:
                    continue
                proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, cam_mtx, dist_c)
                errors = np.linalg.norm(
                    corners_2d - proj.reshape(-1, 2), axis=1)
                frame_data.append((i, corners_2d, proj.reshape(-1, 2), errors))
        else:
            for i, (obj_pts, img_pts) in enumerate(self._intr_frames):
                corners_2d = img_pts.reshape(-1, 2)
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, corners_2d.astype(np.float64), cam_mtx, dist_c)
                if not ok:
                    continue
                proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, cam_mtx, dist_c)
                errors = np.linalg.norm(
                    corners_2d - proj.reshape(-1, 2), axis=1)
                frame_data.append((i, corners_2d, proj.reshape(-1, 2), errors))

        if not frame_data:
            if self._panel:
                self._panel.status_msg = 'No valid frames for visualization'
            return

        all_errors = np.concatenate([fd[3] for fd in frame_data])
        max_err = max(all_errors.max(), 1.0)
        mean_err = all_errors.mean()
        print(f'\n=== Intrinsics Visualization: {len(frame_data)} frames ===')
        print(f'  Mean reproj error: {mean_err:.3f}px, '
              f'Max: {max_err:.3f}px')

        # Build overlay canvas
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        frame_colors = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (255, 100, 255), (100, 255, 255),
            (200, 150, 100), (100, 200, 150), (150, 100, 200),
            (200, 200, 200),
        ]
        for fi, (idx, corners, projected, errors) in enumerate(frame_data):
            base_color = frame_colors[fi % len(frame_colors)]
            for j in range(len(corners)):
                err_ratio = min(errors[j] / max(max_err, 0.5), 1.0)
                r = int(255 * err_ratio)
                g = int(255 * (1.0 - err_ratio))
                color = (0, g, r)
                cx_p = int(corners[j][0])
                cy_p = int(corners[j][1])
                px_p = int(projected[j][0])
                py_p = int(projected[j][1])
                cv2.circle(vis, (cx_p, cy_p), 4, color, -1)
                cv2.circle(vis, (px_p, py_p), 4, color, 1)
                cv2.line(vis, (cx_p, cy_p), (px_p, py_p), color, 1)
            cx_mean = int(corners[:, 0].mean())
            cy_mean = int(corners[:, 1].mean())
            cv2.putText(vis, f'F{idx+1}', (cx_mean - 10, cy_mean),
                        FONT, 0.4, base_color, 1)

        cv2.rectangle(vis, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.putText(vis,
                    f'Intrinsics: {len(frame_data)} frames, '
                    f'mean={mean_err:.2f}px, max={max_err:.2f}px  '
                    '[filled=detected, hollow=reprojected, '
                    'green=low err, red=high]',
                    (10, 20), FONT, 0.4, (220, 220, 220), 1)
        cv2.putText(vis, 'Scroll=zoom  Drag=pan  Esc/q=close',
                    (10, 42), FONT, 0.4, (180, 180, 180), 1)

        # Interactive zoom/pan sub-loop
        win_name = 'Intrinsics Visualization'
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        zoom = 1.0
        pan_x, pan_y = 0, 0
        dragging = False
        drag_start = (0, 0)
        pan_start = (0, 0)

        def _vis_mouse(event, mx, my, flags_v, _param):
            nonlocal zoom, pan_x, pan_y, dragging, drag_start, pan_start
            if event == cv2.EVENT_MOUSEWHEEL:
                old_zoom = zoom
                zoom_new = (min(zoom * 1.2, 10.0) if flags_v > 0
                            else max(zoom / 1.2, 0.5))
                # Zoom toward cursor
                pan_x_new = int(mx - (mx - pan_x) * zoom_new / old_zoom)
                pan_y_new = int(my - (my - pan_y) * zoom_new / old_zoom)
                zoom = zoom_new
                pan_x = pan_x_new
                pan_y = pan_y_new
            elif event == cv2.EVENT_LBUTTONDOWN:
                dragging = True
                drag_start = (mx, my)
                pan_start = (pan_x, pan_y)
            elif event == cv2.EVENT_MOUSEMOVE and dragging:
                pan_x = pan_start[0] + (mx - drag_start[0])
                pan_y = pan_start[1] + (my - drag_start[1])
            elif event == cv2.EVENT_LBUTTONUP:
                dragging = False

        cv2.setMouseCallback(win_name, _vis_mouse)

        while True:
            zh = int(h * zoom)
            zw = int(w * zoom)
            zoomed = cv2.resize(vis, (zw, zh), interpolation=cv2.INTER_NEAREST)
            vx = max(0, -pan_x)
            vy = max(0, -pan_y)
            vx2 = min(zw, w - pan_x)
            vy2 = min(zh, h - pan_y)
            viewport = np.zeros((h, w, 3), dtype=np.uint8)
            dx_v = max(0, pan_x)
            dy_v = max(0, pan_y)
            cw_v = min(vx2 - vx, w - dx_v)
            ch_v = min(vy2 - vy, h - dy_v)
            if cw_v > 0 and ch_v > 0:
                viewport[dy_v:dy_v + ch_v, dx_v:dx_v + cw_v] = (
                    zoomed[vy:vy + ch_v, vx:vx + cw_v])
            cv2.imshow(win_name, viewport)
            k = cv2.waitKey(30) & 0xFF
            if k == 27 or k == ord('q'):
                break
            try:
                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

        cv2.destroyWindow(win_name)
        if self._panel:
            self._panel.status_msg = f'Vis closed (mean={mean_err:.2f}px)'

    # ------------------------------------------------------------------
    def _capture_plane_sample(self):
        """Record the current board's plane normal and distance."""
        if self._current_T_board is None:
            if self._panel:
                self._panel.status_msg = "No board — can't capture plane sample"
            return
        R_board = self._current_T_board[:3, :3]
        t_board = self._current_T_board[:3, 3]
        normal = R_board[:, 2]
        d = float(np.dot(normal, t_board))
        self._plane_samples.append((normal.copy(), d))
        msg = (f'Plane sample {len(self._plane_samples)}: d={d * 1000:.1f}mm')
        if self._panel:
            self._panel.status_msg = msg
        print(f'  {msg}  normal=[{normal[0]:.4f},{normal[1]:.4f},{normal[2]:.4f}]')

    # ------------------------------------------------------------------
    def _save_plane(self):
        """Average plane samples and save to config/ground_plane.yaml."""
        import yaml
        if not self._plane_samples:
            if self._panel:
                self._panel.status_msg = "No plane samples — press 'g' to capture"
            return

        normals = np.array([s[0] for s in self._plane_samples])
        distances = np.array([s[1] for s in self._plane_samples])

        # Flip normals that point opposite to the majority
        ref = normals[0]
        for idx in range(1, len(normals)):
            if np.dot(normals[idx], ref) < 0:
                normals[idx] = -normals[idx]
                distances[idx] = -distances[idx]

        avg_normal = normals.mean(axis=0)
        avg_normal /= np.linalg.norm(avg_normal)
        avg_d = float(distances.mean())
        std_d = float(distances.std() if len(distances) > 1 else 0.0)
        angles_deg = [
            np.degrees(np.arccos(np.clip(np.dot(n, avg_normal), -1, 1)))
            for n in normals]
        max_angle = float(max(angles_deg))

        plane_path = os.path.join(_PROJECT_ROOT, 'config', 'ground_plane.yaml')
        os.makedirs(os.path.dirname(plane_path), exist_ok=True)
        data = {
            'plane_normal': avg_normal.tolist(),
            'plane_d': avg_d,
            'num_samples': len(self._plane_samples),
            'std_d_mm': std_d * 1000,
            'max_angle_deg': max_angle,
        }
        with open(plane_path, 'w') as fh:
            yaml.dump(data, fh, default_flow_style=False)

        msg = (f'Plane saved: {len(self._plane_samples)}x, '
               f'd={avg_d * 1000:.1f}mm std={std_d * 1000:.1f}mm')
        if self._panel:
            self._panel.status_msg = msg
        self._plane_samples.clear()
        print(f'  {msg}')

    # ------------------------------------------------------------------
    def _solve_handeye(self):
        """RANSAC + least-squares solve of hand-eye transform; save to YAML."""
        from calibration import CoordinateTransform

        if len(self._pairs) < 3:
            msg = f'Need 3+ points (have {len(self._pairs)})'
            if self._panel:
                self._panel.status_msg = msg
            return

        pts_cam = [p[0] for p in self._pairs]
        pts_robot = [p[1] for p in self._pairs]
        T_cam2base, inlier_mask = self._dcb.solve_robust_transform(
            pts_cam, pts_robot)
        n_inliers = int(inlier_mask.sum())
        n_outliers = len(self._pairs) - n_inliers

        print(f'\n=== Calibration Result '
              f'({len(self._pairs)} pts, '
              f'{n_inliers} inliers, {n_outliers} outliers) ===')
        print('T_camera_to_base:')
        print(T_cam2base)
        cam_pos = T_cam2base[:3, 3] * 1000
        print(f'\nCamera in robot frame: '
              f'[{cam_pos[0]:.1f}, {cam_pos[1]:.1f}, {cam_pos[2]:.1f}] mm')

        errors = []
        for i, (p_cam, p_robot, _) in enumerate(self._pairs):
            p_hom = np.append(p_cam, 1.0)
            p_est = (T_cam2base @ p_hom)[:3]
            err_mm = float(np.linalg.norm(p_est - p_robot) * 1000)
            errors.append(err_mm)
            tag = '  ' if inlier_mask[i] else '* '
            print(f'  {tag}Point {i + 1}: {err_mm:.1f} mm'
                  f'{"  <-- OUTLIER" if not inlier_mask[i] else ""}')

        inlier_errors = [e for e, m in zip(errors, inlier_mask) if m]
        print(f'\nInlier mean: {np.mean(inlier_errors):.1f} mm, '
              f'max: {np.max(inlier_errors):.1f} mm')

        ct = CoordinateTransform()
        ct.T_camera_to_base = T_cam2base
        out_path = os.path.join(_PROJECT_ROOT, 'config', 'calibration.yaml')
        ct.save(out_path)
        print(f'Saved to {out_path}')

        msg = (f'Saved! {n_inliers}/{len(self._pairs)} inliers, '
               f'mean {np.mean(inlier_errors):.1f}mm')
        if self._panel:
            self._panel.status_msg = msg
