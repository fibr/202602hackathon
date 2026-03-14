"""Base class and registry for unified GUI views.

Each script/tool is wrapped as a View subclass with a standard lifecycle:
  setup()        -> initialize resources (camera, robot, etc.)
  update(canvas) -> draw one frame onto the shared canvas
  handle_key(k)  -> process keyboard input; return True if consumed
  handle_mouse(event, x, y, flags) -> process mouse input
  cleanup()      -> release resources

Views declare metadata via class attributes:
  view_id:      short identifier (e.g. 'control_panel')
  view_name:    human-readable name for sidebar
  description:  one-line tooltip
  needs_camera: whether this view requires a camera
  needs_robot:  whether this view requires a robot connection
  headless_ok:  whether this view can run without any GUI
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, List, Optional, Type


class BaseView:
    """Abstract base class for unified GUI views."""

    # --- Metadata (override in subclasses) ---
    view_id: str = ''
    view_name: str = 'Unnamed'
    description: str = ''
    needs_camera: bool = False
    needs_robot: bool = False
    headless_ok: bool = False

    def __init__(self, app: 'UnifiedApp'):
        """Initialize with a reference to the unified app.

        Args:
            app: The UnifiedApp instance, providing access to shared
                 resources (config, robot, camera, etc.).
        """
        self.app = app

    def setup(self) -> None:
        """Called once when this view becomes active.

        Use this to initialize view-specific resources. The shared
        camera/robot are available via self.app.camera / self.app.robot.
        """
        pass

    def update(self, canvas: np.ndarray) -> None:
        """Draw one frame onto the provided canvas.

        Args:
            canvas: BGR image (H x W x 3) to draw on. The left portion
                    is the main area; the right portion is the sidebar
                    (drawn by the app). Views should only draw in the
                    main area: canvas[:, :self.app.view_width].
        """
        pass

    def handle_key(self, key: int) -> bool:
        """Process a keyboard event.

        Args:
            key: OpenCV key code (from waitKey & 0xFF).

        Returns:
            True if the key was consumed by this view.
        """
        return False

    def handle_mouse(self, event: int, x: int, y: int, flags: int) -> bool:
        """Process a mouse event.

        Args:
            event: OpenCV mouse event type.
            x, y: Coordinates in canvas space.
            flags: OpenCV mouse flags.

        Returns:
            True if the event was consumed.
        """
        return False

    def cleanup(self) -> None:
        """Called when leaving this view. Release view-specific resources."""
        pass

    def run_headless(self) -> int:
        """Run this view's logic without GUI (for --headless mode).

        Returns:
            Exit code (0 = success).

        Raises:
            NotImplementedError if headless mode is not supported.
        """
        raise NotImplementedError(
            f"View '{self.view_id}' does not support headless mode.")


class ViewRegistry:
    """Registry of available views, auto-populated by import."""

    _views: Dict[str, Type[BaseView]] = {}

    @classmethod
    def register(cls, view_cls: Type[BaseView]) -> Type[BaseView]:
        """Register a view class. Used as a decorator."""
        if view_cls.view_id:
            cls._views[view_cls.view_id] = view_cls
        return view_cls

    @classmethod
    def get(cls, view_id: str) -> Optional[Type[BaseView]]:
        """Look up a view class by ID."""
        return cls._views.get(view_id)

    @classmethod
    def all_views(cls) -> Dict[str, Type[BaseView]]:
        """Return all registered views."""
        return dict(cls._views)

    @classmethod
    def list_views(cls) -> List[Type[BaseView]]:
        """Return registered views sorted by name."""
        return sorted(cls._views.values(), key=lambda v: v.view_name)

    @classmethod
    def discover(cls) -> None:
        """Import all view modules to trigger registration."""
        import importlib
        import pkgutil
        import gui.views as pkg
        for _importer, modname, _ispkg in pkgutil.iter_modules(pkg.__path__):
            if modname != 'base':
                importlib.import_module(f'gui.views.{modname}')
