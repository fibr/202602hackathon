"""PyQt GUI integration tests: widget creation and view switching.

Uses pytest-qt (pytestqt) for a headless QApplication fixture.
All tests run with QT_QPA_PLATFORM=offscreen so no display is needed.

Run with:
    ./run.sh -m pytest tests/test_gui_views.py -v
"""

import os
import sys
from types import SimpleNamespace

import pytest

# Force offscreen rendering before any Qt import.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**overrides):
    """Build a minimal argparse-compatible namespace for UnifiedPyQtApp."""
    defaults = dict(
        no_camera=True,
        no_robot=True,
        safe=False,
        sd=False,
        dry_run=False,
        view=None,
        list=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def app_window(qapp):
    """Create a single UnifiedPyQtApp for the whole module (expensive to build)."""
    from unified_gui_pyqt import UnifiedPyQtApp
    window = UnifiedPyQtApp(_make_args())
    yield window
    # Do not call window.close() — let pytest-qt / qapp handle teardown.


# ---------------------------------------------------------------------------
# Widget creation tests
# ---------------------------------------------------------------------------

class TestWidgetCreation:
    """Verify that the main window and all view widgets are created properly."""

    def test_main_window_created(self, app_window):
        """UnifiedPyQtApp is a QMainWindow and has a title."""
        from PyQt5.QtWidgets import QMainWindow
        assert isinstance(app_window, QMainWindow)
        assert 'ArmRobotics' in app_window.windowTitle()

    def test_minimum_size(self, app_window):
        """Window respects the minimum size constraint (900×600)."""
        ms = app_window.minimumSize()
        assert ms.width() >= 900
        assert ms.height() >= 600

    def test_all_views_registered(self, app_window):
        """All expected view IDs are present in the view registry."""
        expected = {
            'home', 'control', 'calibration',
            'dataset', 'demo_cube', 'pipeline', 'discover', 'extras',
            'digital_twin', 'live_twin', 'camera_overlay',
            # sub-views
            'checkerboard', 'servo_calib', 'handeye_yellow',
            'servo_direction', 'verify_calib',
        }
        assert expected.issubset(set(app_window._views.keys()))

    def test_sidebar_shows_top_level_views(self, app_window):
        """Sidebar list contains exactly the 11 top-level (show_in_sidebar=True) views."""
        assert app_window._nav_list.count() == 11

    def test_stacked_widget_has_all_views(self, app_window):
        """QStackedWidget has one page per registered view."""
        from PyQt5.QtWidgets import QStackedWidget
        stack = app_window._stack
        assert isinstance(stack, QStackedWidget)
        assert stack.count() == len(app_window._views)

    def test_home_view_has_table(self, app_window):
        """HomeView contains a QTableWidget for config display."""
        from PyQt5.QtWidgets import QTableWidget
        home = app_window._views['home']
        assert hasattr(home, '_table')
        assert isinstance(home._table, QTableWidget)
        # Should have at least 3 columns (Section / Key / Value).
        assert home._table.columnCount() == 3

    def test_control_view_has_camera_label(self, app_window):
        """ControlPanelView contains a camera feed label."""
        from PyQt5.QtWidgets import QLabel
        ctrl = app_window._views['control']
        assert hasattr(ctrl, '_cam_label')
        assert isinstance(ctrl._cam_label, QLabel)

    def test_control_view_has_status_label(self, app_window):
        """ControlPanelView contains a status label."""
        from PyQt5.QtWidgets import QLabel
        ctrl = app_window._views['control']
        assert hasattr(ctrl, '_status_label')
        assert isinstance(ctrl._status_label, QLabel)

    def test_base_view_attributes(self, app_window):
        """Every registered view has the required class-level attributes."""
        for view_id, widget in app_window._views.items():
            assert widget.view_id == view_id, (
                f'{type(widget).__name__}.view_id mismatch: '
                f'expected {view_id!r}, got {widget.view_id!r}'
            )
            assert isinstance(widget.view_name, str) and widget.view_name, (
                f'{type(widget).__name__} has empty view_name'
            )


# ---------------------------------------------------------------------------
# View switching tests
# ---------------------------------------------------------------------------

class TestViewSwitching:
    """Verify that switch_view() correctly transitions between views."""

    def test_switch_to_home(self, app_window):
        """Switching to 'home' sets the active view id and shows HomeView."""
        from unified_gui_pyqt import HomeView
        app_window.switch_view('home')
        assert app_window._active_view_id == 'home'
        assert isinstance(app_window._stack.currentWidget(), HomeView)

    def test_switch_to_control(self, app_window):
        """Switching to 'control' shows ControlPanelView."""
        from unified_gui_pyqt import ControlPanelView
        app_window.switch_view('control')
        assert app_window._active_view_id == 'control'
        assert isinstance(app_window._stack.currentWidget(), ControlPanelView)

    def test_switch_to_calibration(self, app_window):
        """Switching to 'calibration' shows CalibrationView."""
        from unified_gui_pyqt import CalibrationView
        app_window.switch_view('calibration')
        assert app_window._active_view_id == 'calibration'
        assert isinstance(app_window._stack.currentWidget(), CalibrationView)

    def test_switch_to_subview(self, app_window):
        """Switching to a sub-view (checkerboard) works even though it's not in sidebar."""
        from unified_gui_pyqt import CheckerboardCalibView
        app_window.switch_view('checkerboard')
        assert app_window._active_view_id == 'checkerboard'
        assert isinstance(app_window._stack.currentWidget(), CheckerboardCalibView)

    def test_switch_back_restores_previous(self, app_window):
        """Switching away from a view and back restores it."""
        app_window.switch_view('home')
        app_window.switch_view('dataset')
        app_window.switch_view('home')
        assert app_window._active_view_id == 'home'

    def test_switching_to_same_view_is_noop(self, app_window):
        """Calling switch_view() with the current view_id is a no-op."""
        app_window.switch_view('home')
        # Should not raise and active view must remain unchanged.
        app_window.switch_view('home')
        assert app_window._active_view_id == 'home'

    def test_switching_to_unknown_view_is_safe(self, app_window):
        """switch_view() with an unknown id prints a message but does not crash."""
        before = app_window._active_view_id
        app_window.switch_view('nonexistent_view_xyz')
        # Active view must be unchanged.
        assert app_window._active_view_id == before

    def test_nav_list_selection_triggers_switch(self, app_window):
        """Selecting a sidebar item switches the active view."""
        from PyQt5.QtCore import Qt
        # Find the row for 'control'
        nav = app_window._nav_list
        for row in range(nav.count()):
            item = nav.item(row)
            if item.data(Qt.UserRole) == 'control':
                nav.setCurrentRow(row)
                break
        assert app_window._active_view_id == 'control'

    def test_cycle_through_all_sidebar_views(self, app_window):
        """Every sidebar view can be switched to without raising an exception."""
        from PyQt5.QtCore import Qt
        nav = app_window._nav_list
        for row in range(nav.count()):
            item = nav.item(row)
            view_id = item.data(Qt.UserRole)
            app_window.switch_view(view_id)
            assert app_window._active_view_id == view_id, (
                f'switch_view({view_id!r}) did not update _active_view_id'
            )

    def test_status_bar_updated_on_switch(self, app_window):
        """Status bar message is updated when switching views."""
        app_window.switch_view('home')
        msg = app_window.statusBar().currentMessage()
        assert msg, 'Status bar should be non-empty after switch'
