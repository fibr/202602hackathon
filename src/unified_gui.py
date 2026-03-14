#!/usr/bin/env python3
"""Unified GUI application for ArmRobotics.

Single entry point that provides a sidebar to switch between all available
scripts/views (control panel, calibration, dataset collection, etc.).

Usage:
    ./run.sh src/unified_gui.py                  # Launch GUI with home view
    ./run.sh src/unified_gui.py --view control    # Jump straight to control panel
    ./run.sh src/unified_gui.py --headless --view demo_cube   # Run demo_cube without GUI
    ./run.sh src/unified_gui.py --direct --view demo_cube     # Skip GUI, go to actuation
    ./run.sh src/unified_gui.py --list            # List available views

Flags:
    --view <id>    Start with a specific view (skip home screen)
    --headless     Run without any GUI window (view must support it)
    --direct       Skip interactive GUI, go directly to actuation
    --list         List available views and exit
    --no-camera    Don't initialize camera
    --no-robot     Don't initialize robot
    --safe         Safe mode (reduced torque/speed, arm101 only)
    --sd           Use 640x480 camera resolution
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

# Ensure src/ is on the path
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config_loader import load_config
from gui.views.base import BaseView, ViewRegistry

# --- Layout constants ---
SIDEBAR_WIDTH = 220
MIN_VIEW_WIDTH = 640
MIN_VIEW_HEIGHT = 480
WINDOW_NAME = 'ArmRobotics'

# Sidebar colors
SB_BG = (25, 25, 30)
SB_HEADER_BG = (40, 35, 50)
SB_ITEM_BG = (35, 35, 40)
SB_ITEM_HOVER = (50, 50, 60)
SB_ITEM_ACTIVE = (70, 50, 30)
SB_TEXT = (200, 200, 200)
SB_TEXT_DIM = (120, 120, 120)
SB_TEXT_ACTIVE = (255, 200, 100)
SB_ACCENT = (0, 160, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class UnifiedApp:
    """Main application that hosts views with a sidebar for navigation."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = load_config()

        # Shared resources (lazy-initialized)
        self.robot = None
        self.camera = None
        self._robot_connected = False
        self._camera_started = False

        # View dimensions (camera area; sidebar is separate)
        sd = getattr(args, 'sd', False)
        self.view_width = 640 if sd else 640  # default, views can override
        self.view_height = 480

        # Sidebar state
        self._sidebar_items = []  # [(view_id, view_name, description)]
        self._hover_idx = -1

        # Active view
        self._active_view: BaseView | None = None
        self._active_view_id: str = ''

        # Discover all views
        ViewRegistry.discover()

    @property
    def canvas_width(self) -> int:
        return self.view_width + SIDEBAR_WIDTH

    @property
    def canvas_height(self) -> int:
        return max(self.view_height, MIN_VIEW_HEIGHT)

    # --- Resource management ---

    def ensure_camera(self) -> bool:
        """Start camera if not already running. Returns True on success."""
        if self._camera_started:
            return self.camera is not None
        if getattr(self.args, 'no_camera', False):
            self._camera_started = True
            return False
        try:
            from vision import create_camera
            cam_cfg = self.config.get('camera', {})
            cam_config = dict(self.config)
            sd = getattr(self.args, 'sd', False)
            w, h = (640, 480) if sd else (cam_cfg.get('width', 640), cam_cfg.get('height', 480))
            cam_config['camera'] = dict(cam_cfg, width=w, height=h)
            cam_type = cam_cfg.get('type', 'realsense')
            print(f"Starting {cam_type} camera ({w}x{h})...")
            self.camera = create_camera(cam_config)
            self.camera.start()
            self.view_width = max(self.camera.width, MIN_VIEW_WIDTH)
            self.view_height = max(self.camera.height, MIN_VIEW_HEIGHT)
            print(f"  Camera started ({self.camera.width}x{self.camera.height}).")
            self._camera_started = True
            return True
        except Exception as e:
            print(f"  WARNING: Camera not available: {e}")
            self._camera_started = True
            return False

    def ensure_robot(self) -> bool:
        """Connect robot if not already connected. Returns True on success."""
        if self._robot_connected:
            return self.robot is not None
        if getattr(self.args, 'no_robot', False):
            self._robot_connected = True
            return False
        try:
            from config_loader import connect_robot
            safe = getattr(self.args, 'safe', False)
            self.robot = connect_robot(self.config, safe_mode=safe)
            print("  Robot connected.")
            self._robot_connected = True
            return True
        except Exception as e:
            print(f"  WARNING: Robot not available: {e}")
            self._robot_connected = True
            return False

    def get_camera_frame(self):
        """Get current camera frame. Returns (color_bgr, depth, depth_frame) or Nones."""
        if self.camera is not None:
            try:
                return self.camera.get_frames()
            except Exception:
                pass
        return None, None, None

    # --- View management ---

    def switch_view(self, view_id: str) -> bool:
        """Switch to a different view by ID. Returns True on success."""
        if view_id == self._active_view_id:
            return True

        view_cls = ViewRegistry.get(view_id)
        if view_cls is None:
            print(f"  ERROR: Unknown view '{view_id}'")
            return False

        # Cleanup old view
        if self._active_view is not None:
            try:
                self._active_view.cleanup()
            except Exception as e:
                print(f"  Warning: cleanup error: {e}")

        # Create and setup new view
        print(f"  Switching to view: {view_cls.view_name}")
        view = view_cls(self)
        try:
            view.setup()
        except Exception as e:
            print(f"  ERROR setting up view '{view_id}': {e}")
            import traceback
            traceback.print_exc()
            return False

        self._active_view = view
        self._active_view_id = view_id
        return True

    # --- Sidebar ---

    def _build_sidebar_items(self):
        """Build sidebar item list from registry."""
        self._sidebar_items = []
        # Home is always first
        self._sidebar_items.append(('home', 'Home', 'Config & utilities'))
        for vcls in ViewRegistry.list_views():
            if vcls.view_id != 'home':
                self._sidebar_items.append(
                    (vcls.view_id, vcls.view_name, vcls.description))

    def _draw_sidebar(self, canvas: np.ndarray):
        """Draw the sidebar on the right edge of the canvas."""
        sx = self.view_width  # sidebar X start
        h = self.canvas_height

        # Background
        canvas[:h, sx:sx + SIDEBAR_WIDTH] = SB_BG

        # Header
        header_h = 50
        canvas[:header_h, sx:sx + SIDEBAR_WIDTH] = SB_HEADER_BG
        cv2.putText(canvas, 'ArmRobotics', (sx + 12, 22),
                    FONT, 0.55, SB_ACCENT, 1)
        cv2.putText(canvas, 'Unified GUI', (sx + 12, 42),
                    FONT, 0.38, SB_TEXT_DIM, 1)

        # Separator
        cv2.line(canvas, (sx, header_h), (sx + SIDEBAR_WIDTH, header_h),
                 (60, 60, 70), 1)

        # View items
        item_h = 48
        y = header_h + 4
        for i, (vid, vname, vdesc) in enumerate(self._sidebar_items):
            iy = y + i * item_h
            if iy + item_h > h:
                break

            # Background
            is_active = (vid == self._active_view_id)
            is_hover = (i == self._hover_idx)
            if is_active:
                bg = SB_ITEM_ACTIVE
            elif is_hover:
                bg = SB_ITEM_HOVER
            else:
                bg = SB_ITEM_BG
            canvas[iy:iy + item_h - 2, sx + 4:sx + SIDEBAR_WIDTH - 4] = bg

            # Active indicator bar
            if is_active:
                cv2.rectangle(canvas, (sx, iy), (sx + 3, iy + item_h - 2),
                              SB_ACCENT, -1)

            # Text
            text_col = SB_TEXT_ACTIVE if is_active else SB_TEXT
            cv2.putText(canvas, vname, (sx + 14, iy + 20),
                        FONT, 0.42, text_col, 1)
            cv2.putText(canvas, vdesc[:28], (sx + 14, iy + 36),
                        FONT, 0.32, SB_TEXT_DIM, 1)

        # Bottom status
        status_y = h - 60
        cv2.line(canvas, (sx, status_y), (sx + SIDEBAR_WIDTH, status_y),
                 (60, 60, 70), 1)
        robot_type = self.config.get('robot_type', '?')
        robot_status = 'connected' if self.robot else 'off'
        cam_status = 'on' if self.camera else 'off'
        cv2.putText(canvas, f'Robot: {robot_type} ({robot_status})',
                    (sx + 10, status_y + 20), FONT, 0.33, SB_TEXT_DIM, 1)
        cv2.putText(canvas, f'Camera: {cam_status}',
                    (sx + 10, status_y + 38), FONT, 0.33, SB_TEXT_DIM, 1)
        cv2.putText(canvas, 'ESC=quit  TAB=sidebar',
                    (sx + 10, status_y + 54), FONT, 0.28, SB_TEXT_DIM, 1)

    def _sidebar_click(self, x: int, y: int) -> bool:
        """Handle click in sidebar area. Returns True if consumed."""
        sx = self.view_width
        if x < sx or x >= sx + SIDEBAR_WIDTH:
            return False

        header_h = 50
        item_h = 48
        idx = (y - header_h - 4) // item_h
        if 0 <= idx < len(self._sidebar_items):
            vid = self._sidebar_items[idx][0]
            self.switch_view(vid)
            return True
        return False

    def _sidebar_hover(self, x: int, y: int):
        """Update hover state."""
        sx = self.view_width
        if x < sx or x >= sx + SIDEBAR_WIDTH:
            self._hover_idx = -1
            return
        header_h = 50
        item_h = 48
        idx = (y - header_h - 4) // item_h
        if 0 <= idx < len(self._sidebar_items):
            self._hover_idx = idx
        else:
            self._hover_idx = -1

    # --- Main loop ---

    def _on_mouse(self, event, x, y, flags, param):
        """Global mouse callback."""
        # Sidebar hover
        if event == cv2.EVENT_MOUSEMOVE:
            self._sidebar_hover(x, y)

        # Sidebar click
        if event == cv2.EVENT_LBUTTONDOWN:
            if self._sidebar_click(x, y):
                return

        # Forward to active view
        if self._active_view is not None:
            self._active_view.handle_mouse(event, x, y, flags)

    def run(self) -> int:
        """Run the unified GUI main loop. Returns exit code."""
        self._build_sidebar_items()

        # Start with requested view or home
        start_view = getattr(self.args, 'view', None) or 'home'
        if not self.switch_view(start_view):
            print(f"Failed to start view '{start_view}'. Falling back to home.")
            if start_view != 'home':
                self.switch_view('home')

        # Create window
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

        try:
            while True:
                # Create canvas
                canvas = np.zeros(
                    (self.canvas_height, self.canvas_width, 3), dtype=np.uint8)

                # Update active view
                if self._active_view is not None:
                    try:
                        self._active_view.update(canvas)
                    except Exception as e:
                        # Draw error on canvas
                        cv2.putText(canvas, f"View error: {e}",
                                    (20, self.canvas_height // 2),
                                    FONT, 0.5, (0, 0, 200), 1)

                # Draw sidebar
                self._draw_sidebar(canvas)

                cv2.imshow(WINDOW_NAME, canvas)
                key = cv2.waitKey(30) & 0xFF

                # Global keys
                if key == 27:  # ESC
                    break
                try:
                    if cv2.getWindowProperty(WINDOW_NAME,
                                             cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break

                # Forward key to active view
                if key != 255 and self._active_view is not None:
                    self._active_view.handle_key(key)

        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

        return 0

    def run_headless(self, view_id: str) -> int:
        """Run a specific view in headless mode (no GUI)."""
        view_cls = ViewRegistry.get(view_id)
        if view_cls is None:
            print(f"ERROR: Unknown view '{view_id}'")
            return 1
        if not view_cls.headless_ok:
            print(f"ERROR: View '{view_id}' does not support --headless mode.")
            print("Views with headless support:")
            for v in ViewRegistry.list_views():
                if v.headless_ok:
                    print(f"  {v.view_id:20s} {v.view_name}")
            return 1

        view = view_cls(self)
        try:
            view.setup()
            return view.run_headless()
        except Exception as e:
            print(f"ERROR in headless mode: {e}")
            import traceback
            traceback.print_exc()
            return 1
        finally:
            view.cleanup()
            self._shutdown()

    def run_direct(self, view_id: str) -> int:
        """Run a view in direct actuation mode (skip interactive GUI)."""
        view_cls = ViewRegistry.get(view_id)
        if view_cls is None:
            print(f"ERROR: Unknown view '{view_id}'")
            return 1

        view = view_cls(self)
        try:
            view.setup()
            if hasattr(view, 'run_direct'):
                return view.run_direct()
            elif view.headless_ok:
                print(f"  View '{view_id}' has no direct mode; using headless.")
                return view.run_headless()
            else:
                print(f"  View '{view_id}' has no direct/headless mode.")
                return 1
        except Exception as e:
            print(f"ERROR in direct mode: {e}")
            import traceback
            traceback.print_exc()
            return 1
        finally:
            view.cleanup()
            self._shutdown()

    def _shutdown(self):
        """Clean up all resources."""
        if self._active_view is not None:
            try:
                self._active_view.cleanup()
            except Exception:
                pass
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass
        if self.robot is not None:
            try:
                if hasattr(self.robot, 'close'):
                    self.robot.close()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("  Shutdown complete.")


def parse_args():
    parser = argparse.ArgumentParser(
        description='ArmRobotics Unified GUI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--view', type=str, default=None,
                        help='Start with a specific view (e.g. control, calibration)')
    parser.add_argument('--headless', action='store_true',
                        help='Run without GUI (view must support it)')
    parser.add_argument('--direct', action='store_true',
                        help='Skip GUI, go directly to actuation')
    parser.add_argument('--list', action='store_true',
                        help='List available views and exit')
    parser.add_argument('--no-camera', action='store_true',
                        help='Don\'t initialize camera')
    parser.add_argument('--no-robot', action='store_true',
                        help='Don\'t initialize robot')
    parser.add_argument('--safe', action='store_true',
                        help='Safe mode (reduced torque/speed, arm101)')
    parser.add_argument('--sd', action='store_true',
                        help='Use 640x480 camera resolution')
    return parser.parse_args()


def main():
    args = parse_args()

    # Discover views early for --list
    _SRC_DIR_local = os.path.dirname(os.path.abspath(__file__))
    if _SRC_DIR_local not in sys.path:
        sys.path.insert(0, _SRC_DIR_local)
    ViewRegistry.discover()

    if args.list:
        print("Available views:")
        print(f"  {'ID':<24s} {'Name':<20s} {'Headless':>8s}  Description")
        print(f"  {'-'*24} {'-'*20} {'-'*8}  {'-'*30}")
        for v in ViewRegistry.list_views():
            hl = 'yes' if v.headless_ok else 'no'
            print(f"  {v.view_id:<24s} {v.view_name:<20s} {hl:>8s}  {v.description}")
        return 0

    app = UnifiedApp(args)

    if args.headless:
        if not args.view:
            print("ERROR: --headless requires --view <view_id>")
            return 1
        return app.run_headless(args.view)

    if args.direct:
        if not args.view:
            print("ERROR: --direct requires --view <view_id>")
            return 1
        return app.run_direct(args.view)

    return app.run()


if __name__ == '__main__':
    sys.exit(main())
