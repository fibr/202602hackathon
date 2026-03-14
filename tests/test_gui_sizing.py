"""Tests for unified GUI window sizing.

Verifies that the RobotControlPanel reports correct minimum heights
and that views using the panel set canvas dimensions large enough to
display all controls (including custom buttons and the status area).
"""

import os
import sys

import pytest

# Ensure src/ is importable
_SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from gui.robot_controls import (
    RobotControlPanel,
    PANEL_WIDTH,
    PAD_SIZE,
    BTN_H,
    BTN_GAP,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeRobot:
    """Minimal duck-typed robot for RobotControlPanel without hardware."""
    robot_type = 'nova5'

    def send(self, cmd):
        return '0,{},OK;'

    def get_pose(self):
        return [0.0] * 6

    def get_angles(self):
        return [0.0] * 6


class _FakeArm101(_FakeRobot):
    robot_type = 'arm101'
    safe_mode = False
    speed = 200
    _enabled = False

    def get_mode(self):
        return 0

    def jog_joint(self, *a, **kw):
        pass

    def move_joints(self, *a, **kw):
        pass

    def gripper_open(self):
        pass

    def gripper_close(self):
        pass

    def enable_torque(self):
        pass

    def disable_torque(self):
        pass

    def set_safe_mode(self, v):
        self.safe_mode = v


# ---------------------------------------------------------------------------
# RobotControlPanel.min_height tests
# ---------------------------------------------------------------------------

class TestPanelMinHeight:
    """RobotControlPanel.min_height must cover all elements."""

    def test_nova5_base_min_height(self):
        """Nova5 panel without custom buttons has a sane min_height."""
        panel = RobotControlPanel(_FakeRobot(), panel_x=0, panel_height=480)
        # Must be at least status_y (all buttons drawn) + some status space
        assert panel.min_height > panel.status_y
        # Should fit in a reasonable window (< 800px without custom buttons)
        assert panel.min_height <= 800, (
            f"Nova5 base panel min_height={panel.min_height} unreasonably large"
        )

    def test_arm101_base_min_height(self):
        """Arm101 panel (extra wrist/safe buttons) needs more height."""
        panel = RobotControlPanel(_FakeArm101(), panel_x=0, panel_height=800)
        # arm101 has extra wrist buttons (J4/J5/J6) and safe mode toggle
        # Should be taller than nova5 base
        nova5_panel = RobotControlPanel(_FakeRobot(), panel_x=0, panel_height=800)
        assert panel.min_height > nova5_panel.min_height

    def test_custom_buttons_increase_min_height(self):
        """Adding custom buttons must increase min_height."""
        panel = RobotControlPanel(_FakeRobot(), panel_x=0, panel_height=800)
        base_h = panel.min_height

        panel.add_button("Test 1", lambda: None)
        assert panel.min_height > base_h

        h_after_1 = panel.min_height
        panel.add_button("Test 2", lambda: None)
        assert panel.min_height > h_after_1

    def test_six_custom_buttons_arm101(self):
        """Arm101 + 6 custom buttons (like checkerboard view) needs >480px."""
        panel = RobotControlPanel(_FakeArm101(), panel_x=0, panel_height=800)
        for i in range(6):
            panel.add_button(f"Btn {i}", lambda: None)
        # This is the case that was broken: 480px camera height is too small
        assert panel.min_height > 480, (
            f"arm101 + 6 buttons min_height={panel.min_height} must exceed "
            f"480px camera height"
        )

    def test_min_height_covers_status_area(self):
        """min_height must be >= status_y + reasonable status space."""
        for robot in [_FakeRobot(), _FakeArm101()]:
            panel = RobotControlPanel(robot, panel_x=0, panel_height=800)
            for i in range(6):
                panel.add_button(f"Btn {i}", lambda: None)
            # At least 100px for status lines below status_y
            assert panel.min_height >= panel.status_y + 100, (
                f"min_height={panel.min_height} too small for "
                f"status_y={panel.status_y}"
            )


# ---------------------------------------------------------------------------
# View sizing integration tests
# ---------------------------------------------------------------------------

class _FakeCamera:
    """Minimal camera stand-in."""
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.intrinsics = None

    def start(self):
        pass

    def stop(self):
        pass

    def get_frames(self):
        return None, None, None


class _FakeApp:
    """Minimal UnifiedApp stand-in for view setup tests."""
    def __init__(self, robot=None, camera=None):
        self.robot = robot
        self.camera = camera
        self.view_width = 640
        self.view_height = 480
        self.config = {'robot_type': 'nova5'}
        if robot and getattr(robot, 'robot_type', None) == 'arm101':
            self.config['robot_type'] = 'arm101'

        class _Args:
            no_camera = camera is None
            no_robot = robot is None
            safe = False
            sd = False
            dry_run = False
        self.args = _Args()

    def ensure_camera(self):
        return self.camera is not None

    def ensure_robot(self):
        return self.robot is not None

    @property
    def canvas_width(self):
        return self.view_width + 220  # SIDEBAR_WIDTH

    @property
    def canvas_height(self):
        return max(self.view_height, 480)


class TestViewSizing:
    """Views with robot control panels must set view_height >= panel.min_height."""

    def test_control_view_nova5_height(self):
        """ControlPanelView with nova5 must have adequate height."""
        from gui.views.control_view import ControlPanelView
        app = _FakeApp(robot=_FakeRobot(), camera=_FakeCamera())
        view = ControlPanelView(app)
        view.setup()

        panel = view._panel
        assert app.view_height >= panel.min_height, (
            f"view_height={app.view_height} < panel.min_height={panel.min_height}"
        )
        view.cleanup()

    def test_control_view_arm101_height(self):
        """ControlPanelView with arm101 must have adequate height."""
        from gui.views.control_view import ControlPanelView
        app = _FakeApp(robot=_FakeArm101(), camera=_FakeCamera())
        view = ControlPanelView(app)
        view.setup()

        panel = view._panel
        assert app.view_height >= panel.min_height, (
            f"view_height={app.view_height} < panel.min_height={panel.min_height}"
        )
        view.cleanup()

    def test_control_view_no_camera_arm101(self):
        """ControlPanelView without camera (arm101) must have adequate height."""
        from gui.views.control_view import ControlPanelView
        app = _FakeApp(robot=_FakeArm101(), camera=None)
        view = ControlPanelView(app)
        view.setup()

        panel = view._panel
        assert app.view_height >= panel.min_height, (
            f"view_height={app.view_height} < panel.min_height={panel.min_height}"
        )
        view.cleanup()

    def test_checkerboard_view_height(self):
        """Checkerboard view with 6 custom buttons must have adequate height."""
        from gui.views.checkerboard_view import CheckerboardCalibView
        cam = _FakeCamera()
        app = _FakeApp(robot=_FakeArm101(), camera=cam)
        view = CheckerboardCalibView(app)
        view.setup()

        panel = view._panel
        assert panel is not None, "Panel was not created"
        assert app.view_height >= panel.min_height, (
            f"view_height={app.view_height} < panel.min_height={panel.min_height} "
            f"(panel has {len(panel._custom_buttons)} custom buttons)"
        )
        view.cleanup()

    def test_canvas_height_at_least_view_height(self):
        """canvas_height must be >= view_height (via max with MIN_VIEW_HEIGHT)."""
        from gui.views.control_view import ControlPanelView
        app = _FakeApp(robot=_FakeArm101(), camera=_FakeCamera())
        view = ControlPanelView(app)
        view.setup()

        assert app.canvas_height >= app.view_height
        view.cleanup()

    def test_panel_height_matches_view_height(self):
        """Panel's panel_height should match the computed view_height."""
        from gui.views.checkerboard_view import CheckerboardCalibView
        cam = _FakeCamera()
        app = _FakeApp(robot=_FakeArm101(), camera=cam)
        view = CheckerboardCalibView(app)
        view.setup()

        panel = view._panel
        assert panel.panel_height >= panel.min_height, (
            f"panel_height={panel.panel_height} < min_height={panel.min_height}"
        )
        view.cleanup()


# ---------------------------------------------------------------------------
# Regression: ensure 640x480 camera + arm101 + 6 buttons doesn't clip
# ---------------------------------------------------------------------------

class TestRegressionWindowClipping:
    """Regression tests: window must not clip panel content."""

    def test_480p_camera_arm101_six_buttons(self):
        """480p camera + arm101 + 6 custom buttons previously clipped to 480px."""
        panel = RobotControlPanel(_FakeArm101(), panel_x=640, panel_height=480)
        for i in range(6):
            panel.add_button(f"Button {i}", lambda: None)

        # The old behavior set panel_height = cam_height = 480,
        # which cut off buttons.  Now min_height must exceed 480.
        assert panel.min_height > 480, (
            f"min_height={panel.min_height} should exceed 480 for "
            f"arm101 + 6 buttons"
        )

    def test_480p_camera_nova5_six_buttons(self):
        """480p camera + nova5 + 6 custom buttons should also be checked."""
        panel = RobotControlPanel(_FakeRobot(), panel_x=640, panel_height=480)
        for i in range(6):
            panel.add_button(f"Button {i}", lambda: None)

        # Nova5 doesn't have wrist/safe buttons, but 6 custom buttons
        # still need significant space
        assert panel.min_height > 480, (
            f"min_height={panel.min_height} should exceed 480 for "
            f"nova5 + 6 buttons"
        )
