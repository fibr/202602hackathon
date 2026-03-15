#!/usr/bin/env python3
"""PyQt5 reimplementation of the ArmRobotics Unified GUI.

Every action has a corresponding GUI button — no keyboard-only actions.
Sidebar navigation, view switching, and resource management all use PyQt5.

Usage:
    ./run.sh src/unified_gui_pyqt.py                    # Launch with home view
    ./run.sh src/unified_gui_pyqt.py --view control      # Jump to control panel
    ./run.sh src/unified_gui_pyqt.py --list              # List views
    ./run.sh src/unified_gui_pyqt.py --no-camera         # Skip camera init
    ./run.sh src/unified_gui_pyqt.py --no-robot          # Skip robot init
    ./run.sh src/unified_gui_pyqt.py --safe              # Safe mode (arm101)
    ./run.sh src/unified_gui_pyqt.py --sd                # 640x480 camera
    ./run.sh src/unified_gui_pyqt.py --dry-run           # Dry-run mode
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Optional

# Qt plugin conflict prevention:
# We use opencv-python-headless (no bundled Qt) to avoid clashing with PyQt5.
# If someone accidentally installs opencv-python (non-headless), clear any
# inherited Qt env vars so PyQt5's own plugins take precedence.
os.environ.pop('QT_PLUGIN_PATH', None)
os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)

import cv2
import numpy as np

# Ensure src/ is on the path
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

_PROJECT_ROOT = os.path.abspath(os.path.join(_SRC_DIR, '..'))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QListWidget, QListWidgetItem, QPushButton,
    QLabel, QFrame, QSplitter, QGroupBox, QGridLayout, QScrollArea,
    QTextEdit, QLineEdit, QComboBox, QSlider, QSpinBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy,
    QTreeWidget, QTreeWidgetItem, QProgressBar, QToolButton,
    QMessageBox, QStatusBar,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QIcon, QPalette

from config_loader import load_config, config_path


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def cv_to_qimage(frame: np.ndarray) -> QImage:
    """Convert an OpenCV BGR frame to QImage."""
    if frame is None:
        return QImage()
    h, w = frame.shape[:2]
    if len(frame.shape) == 2:
        return QImage(frame.data, w, h, w, QImage.Format_Grayscale8)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)


def make_button(text, callback, tooltip='', color=None, min_w=80, min_h=32):
    """Create a styled QPushButton."""
    btn = QPushButton(text)
    btn.setMinimumSize(min_w, min_h)
    if tooltip:
        btn.setToolTip(tooltip)
    if callback:
        btn.clicked.connect(callback)
    if color:
        btn.setStyleSheet(
            f'QPushButton {{ background-color: {color}; color: white; '
            f'border: 1px solid #555; border-radius: 4px; padding: 4px 8px; }}'
            f'QPushButton:hover {{ background-color: {_lighten(color)}; }}'
        )
    else:
        btn.setStyleSheet(
            'QPushButton { background-color: #3c3c3c; color: #ddd; '
            'border: 1px solid #555; border-radius: 4px; padding: 4px 8px; }'
            'QPushButton:hover { background-color: #505050; }'
        )
    return btn


def _lighten(hex_color: str) -> str:
    """Lighten a hex color by 20%."""
    c = hex_color.lstrip('#')
    if len(c) < 6:
        c = c.ljust(6, '0')
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    r = min(255, int(r * 1.2))
    g = min(255, int(g * 1.2))
    b = min(255, int(b * 1.2))
    return f'#{r:02x}{g:02x}{b:02x}'


def section_label(text):
    lbl = QLabel(text)
    lbl.setStyleSheet('color: #aaa; font-size: 11px; margin-top: 6px;')
    return lbl


# ---------------------------------------------------------------------------
# Clickable camera label
# ---------------------------------------------------------------------------

class ClickableLabel(QLabel):
    """A QLabel camera feed that emits clicked(x, y) in source-image coordinates.

    When the user left-clicks anywhere on the displayed image (not the
    letterbox bars), the widget converts the label-local click position to
    the corresponding pixel in the *original* camera frame and emits
    ``clicked(img_x, img_y)``.

    Usage::

        label = ClickableLabel('Camera')
        label.clicked.connect(self._on_cam_click)

        # In _on_frame, tell the label the source resolution so the
        # coordinate mapping stays accurate:
        def _on_frame(self, frame):
            h, w = frame.shape[:2]
            pix = QPixmap.fromImage(cv_to_qimage(frame)).scaled(
                label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.set_source_size(w, h)
            label.setPixmap(pix)
    """

    # Emits (image_x, image_y) in original source-frame pixel coordinates.
    clicked = pyqtSignal(int, int)

    def __init__(self, text: str = '', parent=None):
        super().__init__(text, parent)
        self._source_w: int = 0
        self._source_h: int = 0
        # Show a crosshair to hint the widget is interactive.
        self.setCursor(Qt.CrossCursor)

    def set_source_size(self, w: int, h: int) -> None:
        """Record the original image resolution for coordinate mapping."""
        self._source_w = w
        self._source_h = h

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton and self._source_w > 0:
            pix = self.pixmap()
            if pix is not None and not pix.isNull():
                # The pixmap is drawn centred inside the label (AlignCenter).
                lw, lh = self.width(), self.height()
                pw, ph = pix.width(), pix.height()
                ox = (lw - pw) // 2  # horizontal offset of pixmap within label
                oy = (lh - ph) // 2  # vertical offset
                px = event.x() - ox   # x inside the pixmap
                py = event.y() - oy   # y inside the pixmap
                if 0 <= px < pw and 0 <= py < ph:
                    # Scale from displayed-pixmap coords → original frame coords.
                    ix = int(px * self._source_w / pw)
                    iy = int(py * self._source_h / ph)
                    self.clicked.emit(ix, iy)
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# Camera feed thread
# ---------------------------------------------------------------------------

class CameraThread(QThread):
    """Polls camera frames and emits them as signals."""
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.camera = camera
        self._running = True

    def run(self):
        while self._running:
            try:
                color, depth, depth_frame = self.camera.get_frames()
                if color is not None:
                    self.frame_ready.emit(color)
            except Exception:
                pass
            time.sleep(0.033)  # ~30 fps

    def stop(self):
        self._running = False
        self.wait(2000)


# ---------------------------------------------------------------------------
# Subprocess runner thread
# ---------------------------------------------------------------------------

class SubprocessThread(QThread):
    """Runs a script in a subprocess, emitting output line by line."""
    output_line = pyqtSignal(str)
    finished_signal = pyqtSignal(int)  # exit code

    def __init__(self, cmd, cwd=None, timeout=120, parent=None):
        super().__init__(parent)
        self.cmd = cmd
        self.cwd = cwd or _PROJECT_ROOT
        self.timeout = timeout
        self._proc = None

    def run(self):
        try:
            self._proc = subprocess.Popen(
                self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=self.cwd)
            for line in self._proc.stdout:
                self.output_line.emit(line.rstrip('\n'))
            self._proc.wait(timeout=self.timeout)
            self.finished_signal.emit(self._proc.returncode)
        except subprocess.TimeoutExpired:
            if self._proc:
                self._proc.kill()
            self.output_line.emit('ERROR: Process timed out')
            self.finished_signal.emit(-1)
        except Exception as e:
            self.output_line.emit(f'ERROR: {e}')
            self.finished_signal.emit(-1)

    def stop_process(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()


# ---------------------------------------------------------------------------
# Robot poll thread
# ---------------------------------------------------------------------------

class RobotPollThread(QThread):
    """Polls robot state at a configurable rate."""
    state_updated = pyqtSignal(object, object, object)  # pose, angles, mode

    def __init__(self, robot, interval=0.1, parent=None):
        super().__init__(parent)
        self.robot = robot
        self.interval = interval
        self._running = True

    def run(self):
        while self._running:
            pose = angles = mode = None
            try:
                pose = self.robot.get_pose()
                angles = self.robot.get_angles()
                arm101 = getattr(self.robot, 'robot_type', None) == 'arm101'
                if arm101:
                    mode = self.robot.get_mode()
                else:
                    resp = self.robot.send('RobotMode()')
                    if '{' in resp:
                        val = resp.split('{')[1].split('}')[0]
                        mode = int(float(val))
            except Exception:
                pass
            self.state_updated.emit(pose, angles, mode)
            time.sleep(self.interval)

    def stop(self):
        self._running = False
        self.wait(2000)


# ---------------------------------------------------------------------------
# Base view widget
# ---------------------------------------------------------------------------

class BaseViewWidget(QWidget):
    """Base class for all view widgets in the PyQt GUI."""

    view_id: str = ''
    view_name: str = 'Unnamed'
    description: str = ''
    show_in_sidebar: bool = True
    parent_view_id: str = ''  # non-empty for sub-views

    def __init__(self, app: 'UnifiedPyQtApp', parent=None):
        super().__init__(parent)
        self.app = app

    def on_activate(self):
        """Called when this view becomes active."""
        pass

    def on_deactivate(self):
        """Called when leaving this view."""
        pass


# ---------------------------------------------------------------------------
# HOME VIEW
# ---------------------------------------------------------------------------

class HomeView(BaseViewWidget):
    view_id = 'home'
    view_name = 'Home'
    description = 'Config editor & utilities'

    def __init__(self, app, parent=None):
        super().__init__(app, parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title = QLabel('ArmRobotics Configuration')
        title.setStyleSheet('font-size: 18px; font-weight: bold; color: #ffc864;')
        layout.addWidget(title)

        robot_type = self.app.config.get('robot_type', '?')
        cam_type = self.app.config.get('camera', {}).get('type', '?')
        subtitle = QLabel(f'Robot: {robot_type}  |  Camera: {cam_type}')
        subtitle.setStyleSheet('color: #999; font-size: 12px;')
        layout.addWidget(subtitle)

        # Config table
        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(['Section', 'Key', 'Value'])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._table.setStyleSheet(
            'QTableWidget { background-color: #252528; color: #ddd; gridline-color: #444; }'
            'QHeaderView::section { background-color: #333; color: #aaa; padding: 4px; }'
        )
        self._table.cellDoubleClicked.connect(self._on_cell_edit)
        self._load_config()
        layout.addWidget(self._table)

        # Utility buttons
        util_frame = QGroupBox('Utilities')
        util_frame.setStyleSheet(
            'QGroupBox { color: #aaa; border: 1px solid #444; border-radius: 4px; '
            'margin-top: 8px; padding-top: 12px; }')
        util_layout = QHBoxLayout(util_frame)
        btn_discover = make_button('Discover Cameras', lambda: self._run_utility(
            'scripts/discover_cameras.py', []), 'Detect cameras and write config')
        btn_discover_dry = make_button('Discover (dry-run)', lambda: self._run_utility(
            'scripts/discover_cameras.py', ['--dry-run']), 'Preview without writing')
        util_layout.addWidget(btn_discover)
        util_layout.addWidget(btn_discover_dry)
        util_layout.addStretch()
        layout.addWidget(util_frame)

        # Output area (hidden by default)
        self._output = QTextEdit()
        self._output.setReadOnly(True)
        self._output.setMaximumHeight(200)
        self._output.setStyleSheet('background-color: #1a1a1a; color: #ccc; font-family: monospace;')
        self._output.hide()
        layout.addWidget(self._output)

    def _load_config(self):
        config = self.app.config
        entries = []
        for section, values in config.items():
            if isinstance(values, dict):
                for key, val in values.items():
                    entries.append((section, key, val))
            else:
                entries.append(('', section, values))

        self._table.setRowCount(len(entries))
        for i, (section, key, val) in enumerate(entries):
            self._table.setItem(i, 0, QTableWidgetItem(section))
            self._table.setItem(i, 1, QTableWidgetItem(key))
            item = QTableWidgetItem(str(val))
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self._table.setItem(i, 2, item)
        self._entries = entries

    def _on_cell_edit(self, row, col):
        """Handle double-click to edit a value in column 2."""
        if col != 2:
            return
        # Qt's default editing handles the edit; we save on cell change
        self._table.cellChanged.connect(self._on_cell_changed)

    def _on_cell_changed(self, row, col):
        if col != 2:
            return
        try:
            self._table.cellChanged.disconnect(self._on_cell_changed)
        except Exception:
            pass

        section = self._table.item(row, 0).text()
        key = self._table.item(row, 1).text()
        val_str = self._table.item(row, 2).text().strip()

        # Parse value
        try:
            if '.' in val_str:
                parsed = float(val_str)
            else:
                parsed = int(val_str)
        except ValueError:
            if val_str.lower() in ('true', 'false'):
                parsed = val_str.lower() == 'true'
            else:
                parsed = val_str

        # Save to settings.yaml
        import yaml
        settings_path = config_path('settings.yaml')
        settings = {}
        if os.path.exists(settings_path):
            with open(settings_path, 'r') as f:
                settings = yaml.safe_load(f) or {}

        if section:
            if section not in settings:
                settings[section] = {}
            settings[section][key] = parsed
        else:
            settings[key] = parsed

        with open(settings_path, 'w') as f:
            yaml.dump(settings, f, default_flow_style=False)

        self.app.config = load_config()
        print(f'  Saved: {section}.{key} = {parsed}')

    def _run_utility(self, script, args):
        self._output.show()
        self._output.clear()
        self._output.append(f'Running: {script} {" ".join(args)}')

        def _run():
            try:
                result = subprocess.run(
                    [sys.executable, os.path.join(_PROJECT_ROOT, script)] + args,
                    capture_output=True, text=True, timeout=30, cwd=_PROJECT_ROOT)
                self._output.append(result.stdout)
                if result.stderr:
                    self._output.append('--- stderr ---\n' + result.stderr)
                self._output.append(f'\nExit code: {result.returncode}')
            except Exception as e:
                self._output.append(f'ERROR: {e}')

        threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# CONTROL PANEL VIEW
# ---------------------------------------------------------------------------

class ControlPanelView(BaseViewWidget):
    view_id = 'control'
    view_name = 'Control Panel'
    description = 'Camera + robot jog controls'

    def __init__(self, app, parent=None):
        super().__init__(app, parent)
        self._poll_thread = None
        self._cam_thread = None
        self._cached_pose = None
        self._cached_angles = None
        self._cached_mode = None
        self._ik_solver = None
        self._pos_log_path = None
        self._pos_log_count = 0
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Left: camera feed
        self._cam_label = QLabel('No Camera')
        self._cam_label.setAlignment(Qt.AlignCenter)
        self._cam_label.setMinimumSize(320, 240)
        self._cam_label.setStyleSheet('background-color: #1a1a1a; color: #666;')
        layout.addWidget(self._cam_label, stretch=3)

        # Right: controls
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setMaximumWidth(320)
        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)
        ctrl_layout.setSpacing(4)

        # Status display
        self._status_label = QLabel('Connecting...')
        self._status_label.setStyleSheet('color: #ffa; font-family: monospace; font-size: 11px;')
        self._status_label.setWordWrap(True)
        ctrl_layout.addWidget(self._status_label)

        # XY Jog
        ctrl_layout.addWidget(section_label('Cartesian Jog (XY)'))
        xy_grid = QGridLayout()
        xy_grid.addWidget(make_button('Y+', lambda: self._cart_step(1, +1), 'Move Y+'), 0, 1)
        xy_grid.addWidget(make_button('X-', lambda: self._cart_step(0, -1), 'Move X-'), 1, 0)
        xy_grid.addWidget(make_button('X+', lambda: self._cart_step(0, +1), 'Move X+'), 1, 2)
        xy_grid.addWidget(make_button('Y-', lambda: self._cart_step(1, -1), 'Move Y-'), 2, 1)
        ctrl_layout.addLayout(xy_grid)

        # Z Jog
        ctrl_layout.addWidget(section_label('Z Axis'))
        z_row = QHBoxLayout()
        z_row.addWidget(make_button('Z +', lambda: self._cart_step(2, +1), 'Move Z up'))
        z_row.addWidget(make_button('Z -', lambda: self._cart_step(2, -1), 'Move Z down'))
        ctrl_layout.addLayout(z_row)

        # Wrist joints (J4, J5, J6) — always shown, disabled if not arm101
        ctrl_layout.addWidget(section_label('Wrist Joints'))
        for j in [4, 5, 6]:
            row = QHBoxLayout()
            row.addWidget(make_button(f'J{j} -', lambda checked, jj=j: self._joint_step(jj - 1, -1),
                                      color='#5a3770'))
            row.addWidget(make_button(f'J{j} +', lambda checked, jj=j: self._joint_step(jj - 1, +1),
                                      color='#5a3770'))
            ctrl_layout.addLayout(row)

        # Joint jog (J1-J6)
        ctrl_layout.addWidget(section_label('Joint Jog'))
        jog_grid = QGridLayout()
        for i in range(6):
            jog_grid.addWidget(
                make_button(f'J{i+1}-', lambda checked, idx=i: self._jog(idx, -1)),
                i, 0)
            jog_grid.addWidget(
                make_button(f'J{i+1}+', lambda checked, idx=i: self._jog(idx, +1)),
                i, 1)
        ctrl_layout.addLayout(jog_grid)

        stop_btn = make_button('STOP Jog', self._stop_jog, color='#992222')
        ctrl_layout.addWidget(stop_btn)

        # Gripper
        ctrl_layout.addWidget(section_label('Gripper'))
        grip_row = QHBoxLayout()
        grip_row.addWidget(make_button('Open', self._gripper_open, color='#006600'))
        grip_row.addWidget(make_button('Close', self._gripper_close, color='#660000'))
        ctrl_layout.addLayout(grip_row)

        # Speed
        ctrl_layout.addWidget(section_label('Speed'))
        spd_row = QHBoxLayout()
        self._spd_label = QLabel('30')
        self._spd_label.setAlignment(Qt.AlignCenter)
        self._spd_label.setStyleSheet('color: #ffc800; font-weight: bold;')
        spd_row.addWidget(make_button('<<', lambda: self._speed_change(-10)))
        spd_row.addWidget(self._spd_label)
        spd_row.addWidget(make_button('>>', lambda: self._speed_change(+10)))
        ctrl_layout.addLayout(spd_row)

        # Enable / Home / Safe
        ctrl_layout.addWidget(section_label('Robot Control'))
        eh_row = QHBoxLayout()
        self._enable_btn = make_button('Servos ON', self._do_enable, color='#006400')
        eh_row.addWidget(self._enable_btn)
        eh_row.addWidget(make_button('Home', self._do_home, color='#644800'))
        eh_row.addWidget(make_button('Set Home', self._set_home, color='#4a3200',
                                     tooltip='Save current position as home'))
        ctrl_layout.addLayout(eh_row)

        self._safe_btn = make_button('Safe: OFF', self._toggle_safe, color='#503232')
        ctrl_layout.addWidget(self._safe_btn)

        # J1 rotate
        ctrl_layout.addWidget(make_button('J1 +30 deg', self._j1_rotate, color='#643c00'))

        # Logging
        ctrl_layout.addWidget(section_label('Logging'))
        log_row = QHBoxLayout()
        log_row.addWidget(make_button('Log Position', self._log_position,
                                      'Log current joint angles + pose to CSV'))
        log_row.addWidget(make_button('Print Pose', self._print_pose,
                                      'Print joint angles and pose to console'))
        ctrl_layout.addLayout(log_row)

        ctrl_layout.addStretch()
        ctrl_scroll.setWidget(ctrl_widget)
        layout.addWidget(ctrl_scroll, stretch=1)

    def on_activate(self):
        self.app.ensure_robot()
        self.app.ensure_camera()

        # Speed init
        speed = 30
        if self.app.config.get('robot_type') == 'arm101':
            speed = self.app.config.get('arm101', {}).get('speed', 200)
        self._speed = speed
        self._spd_label.setText(str(speed))

        # Start camera thread
        if self.app.camera is not None:
            self._cam_thread = CameraThread(self.app.camera)
            self._cam_thread.frame_ready.connect(self._on_frame)
            self._cam_thread.start()

        # Start robot poll thread
        if self.app.robot is not None:
            interval = 0.1 if self._is_arm101() else 0.5
            self._poll_thread = RobotPollThread(self.app.robot, interval)
            self._poll_thread.state_updated.connect(self._on_robot_state)
            self._poll_thread.start()

        # Position log
        log_dir = os.path.join(_PROJECT_ROOT, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._pos_log_path = os.path.join(log_dir, f'positions_{ts}.csv')
        with open(self._pos_log_path, 'w') as f:
            f.write('timestamp,j1,j2,j3,j4,j5,j6,x,y,z,rx,ry,rz,label\n')

    def on_deactivate(self):
        if self._cam_thread:
            self._cam_thread.stop()
            self._cam_thread = None
        if self._poll_thread:
            self._poll_thread.stop()
            self._poll_thread = None

    def _is_arm101(self):
        return getattr(self.app.robot, 'robot_type', None) == 'arm101'

    def _on_frame(self, frame):
        img = cv_to_qimage(frame)
        pix = QPixmap.fromImage(img).scaled(
            self._cam_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._cam_label.setPixmap(pix)

    def _on_robot_state(self, pose, angles, mode):
        self._cached_pose = pose
        self._cached_angles = angles
        self._cached_mode = mode
        lines = []
        if pose:
            lines.append(f'TCP: {pose[0]:.1f}, {pose[1]:.1f}, {pose[2]:.1f}')
            lines.append(f'     {pose[3]:.1f}, {pose[4]:.1f}, {pose[5]:.1f}')
        if angles:
            lines.append(f'J: {", ".join(f"{a:.1f}" for a in angles)}')
        MODE_NAMES = {1: 'init', 2: 'brake', 3: 'init', 4: 'disabled', 5: 'enabled',
                      6: 'backdrive', 7: 'running', 9: 'error', 10: 'pause', 11: 'jog'}
        if mode is not None:
            name = MODE_NAMES.get(mode, '?')
            lines.append(f'Mode: {mode} ({name})')
        # Update servo on/off button label
        if self._is_arm101():
            enabled = getattr(self.app.robot, '_enabled', False)
            self._enable_btn.setText('Servos OFF' if enabled else 'Servos ON')
            self._enable_btn.setStyleSheet(
                f'background-color: {"#640000" if enabled else "#006400"}; '
                f'color: white; padding: 4px; border-radius: 3px;')
        self._status_label.setText('\n'.join(lines) if lines else 'No robot')

    def _cart_step(self, axis_idx, sign):
        if self.app.robot is None:
            return
        if self._is_arm101():
            step_mm = 5.0
            self._do_cart_step_ik(axis_idx, sign, step_mm)
        else:
            from robot.motion_utils import move_to_pose
            pose = self.app.robot.get_pose()
            if not pose:
                return
            target = list(pose)
            step = 10.0 if axis_idx < 2 else 10.0
            target[axis_idx] += sign * step
            move_to_pose(self.app.robot, *target[:6], speed=self._speed)

    def _do_cart_step_ik(self, axis_idx, sign, step_mm):
        if self._ik_solver is None:
            try:
                from kinematics.arm101_ik_solver import Arm101IKSolver
                self._ik_solver = Arm101IKSolver()
            except Exception as e:
                print(f'IK init error: {e}')
                return
        angles = self.app.robot.get_angles()
        if angles is None:
            return
        motor_deg = np.array(angles[:5], dtype=float)
        pos_mm, _ = self._ik_solver.forward_kin(motor_deg)
        target_pos = pos_mm.copy()
        target_pos[axis_idx] += sign * step_mm
        result = self._ik_solver.solve_ik_position(target_pos, seed_motor_deg=motor_deg)
        if result is None:
            return
        full = list(result) + [angles[5]]
        self.app.robot.move_joints(full)

    def _joint_step(self, joint_idx, direction):
        if self.app.robot is None:
            return
        if self._is_arm101():
            self.app.robot.jog_joint(joint_idx, direction, 5.0)

    def _jog(self, joint_idx, direction):
        if self.app.robot is None:
            return
        if self._is_arm101():
            self.app.robot.jog_joint(joint_idx, direction)
        else:
            axis = f'J{joint_idx + 1}{"+" if direction > 0 else "-"}'
            self.app.robot.send(f'MoveJog({axis})')

    def _stop_jog(self):
        if self.app.robot is None:
            return
        if not self._is_arm101():
            self.app.robot.send('MoveJog()')

    def _gripper_open(self):
        if self.app.robot is None:
            return
        if self._is_arm101():
            self.app.robot.gripper_open()
        else:
            self.app.robot.send('ToolDOInstant(1,0)')
            self.app.robot.send('ToolDOInstant(2,1)')

    def _gripper_close(self):
        if self.app.robot is None:
            return
        if self._is_arm101():
            self.app.robot.gripper_close()
        else:
            self.app.robot.send('ToolDOInstant(2,0)')
            self.app.robot.send('ToolDOInstant(1,1)')

    def _speed_change(self, delta):
        if self._is_arm101():
            self._speed = max(10, min(1000, self._speed + delta * 5))
            if self.app.robot:
                self.app.robot.speed = self._speed
        else:
            self._speed = max(1, min(100, self._speed + delta))
            if self.app.robot:
                self.app.robot.send(f'SpeedFactor({self._speed})')
        self._spd_label.setText(str(self._speed))

    def _do_enable(self):
        if self.app.robot is None:
            return
        if self._is_arm101():
            if getattr(self.app.robot, '_enabled', False):
                self.app.robot.disable_torque()
            else:
                self.app.robot.enable_torque()
        else:
            self.app.robot.send('DisableRobot()')
            time.sleep(1)
            self.app.robot.send('ClearError()')
            self.app.robot.send('EnableRobot()')

    def _toggle_safe(self):
        if self.app.robot is None or not self._is_arm101():
            return
        is_safe = getattr(self.app.robot, 'safe_mode', False)
        self.app.robot.set_safe_mode(not is_safe)
        self._speed = self.app.robot.speed
        self._spd_label.setText(str(self._speed))
        self._safe_btn.setText(f'Safe: {"ON" if self.app.robot.safe_mode else "OFF"}')

    def _do_home(self):
        if self.app.robot is None:
            return

        def _home():
            if self._is_arm101():
                home = self.app.config.get('arm101', {}).get(
                    'home_angles', [0.0, 0.0, 90.0, 90.0, 0.0, 0.0])
                self.app.robot.move_joints(home, speed=100)
            else:
                self.app.robot.send('SpeedFactor(10)')
                self.app.robot.send('MovJ(joint={43.5,-13.9,-85.4,196.3,-90.0,43.5})')
                time.sleep(3)
                self.app.robot.send(f'SpeedFactor({self._speed})')
        threading.Thread(target=_home, daemon=True).start()

    def _set_home(self):
        """Save current joint angles as the home position in robot_config.yaml."""
        if self.app.robot is None:
            return
        angles = self.app.robot.get_angles()
        if not angles or len(angles) < 6:
            return
        home = [round(a, 1) for a in angles]

        # Update in-memory config
        if 'arm101' not in self.app.config:
            self.app.config['arm101'] = {}
        self.app.config['arm101']['home_angles'] = home

        # Write to config file
        import yaml
        from config_loader import config_path
        cfg_path = config_path('robot_config.yaml')
        with open(cfg_path, 'r') as f:
            lines = f.readlines()

        # Find and replace home_angles line
        found = False
        for i, line in enumerate(lines):
            if 'home_angles:' in line and not line.lstrip().startswith('#'):
                indent = len(line) - len(line.lstrip())
                lines[i] = f"{' ' * indent}home_angles: {home}  # Safe home position\n"
                found = True
                break

        if found:
            with open(cfg_path, 'w') as f:
                f.writelines(lines)
            print(f'  Set home position: {home}')
        else:
            print(f'  WARNING: home_angles not found in {cfg_path}')

    def _j1_rotate(self):
        if self.app.robot is None:
            return
        angles = self.app.robot.get_angles()
        if not angles:
            return
        target = list(angles)
        target[0] += 30.0

        def _move():
            if self._is_arm101():
                self.app.robot.move_joints(target)
            else:
                jstr = ','.join(f'{v:.2f}' for v in target)
                self.app.robot.send(f'MovJ(joint={{{jstr}}})')
        threading.Thread(target=_move, daemon=True).start()

    def _log_position(self):
        if self.app.robot is None:
            return
        angles = self.app.robot.get_angles()
        if angles and self._pos_log_path:
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            row = ','.join(f'{v:.3f}' for v in angles)
            pose = self.app.robot.get_pose()
            if pose:
                row += ',' + ','.join(f'{v:.3f}' for v in pose)
            else:
                row += ',,,,,,'
            with open(self._pos_log_path, 'a') as f:
                f.write(f'{ts},{row},\n')
            self._pos_log_count += 1
            print(f'  Logged #{self._pos_log_count}')

    def _print_pose(self):
        if self.app.robot is None:
            return
        angles = self.app.robot.get_angles()
        if angles:
            print(f"  Joints: {', '.join(f'{v:.2f}' for v in angles)}")
            pose = self.app.robot.get_pose()
            if pose:
                print(f"  Pose:   {', '.join(f'{v:.2f}' for v in pose)}")


# ---------------------------------------------------------------------------
# CALIBRATION MENU VIEW
# ---------------------------------------------------------------------------

class CalibrationView(BaseViewWidget):
    view_id = 'calibration'
    view_name = 'Calibration'
    description = 'Servo & hand-eye calibration'

    OPTIONS = [
        ('checkerboard', 'Checkerboard Calibration',
         'Intrinsics + click TCP on board, ray-plane solve'),
        ('servo_calib', 'Servo Calibration',
         'Move arm to zero pose, save offsets (arm101)'),
        ('handeye_yellow', 'Hand-Eye Calibration (Yellow Tape)',
         'Capture FK+pixel poses, joint solve (arm101)'),
        ('verify_calib', 'Verify Checkerboard',
         'Move arm above board corners to verify calibration'),
        ('servo_direction', 'Servo Direction Auto-Calib',
         'Auto-detect servo signs + offsets via ChArUco board (arm101)'),
    ]

    def __init__(self, app, parent=None):
        super().__init__(app, parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel('Calibration Tools')
        title.setStyleSheet('font-size: 18px; font-weight: bold; color: #ffc864;')
        layout.addWidget(title)
        layout.addWidget(QLabel('Click a tool to open it'))

        for view_id, label, desc in self.OPTIONS:
            btn = QPushButton(f'{label}\n{desc}')
            btn.setMinimumHeight(50)
            btn.setStyleSheet(
                'QPushButton { background-color: #28283c; color: #b4dcff; '
                'border: 1px solid #465a5a; border-radius: 6px; text-align: left; '
                'padding: 8px 14px; font-size: 13px; }'
                'QPushButton:hover { background-color: #373750; border-color: #78a0dc; }'
            )
            btn.clicked.connect(lambda checked, vid=view_id: self.app.switch_view(vid))
            layout.addWidget(btn)

        layout.addStretch()


# ---------------------------------------------------------------------------
# CALIBRATION SUB-VIEWS  (checkerboard, servo, handeye, verify, servo_direction)
# ---------------------------------------------------------------------------

class CheckerboardCalibView(BaseViewWidget):
    view_id = 'checkerboard'
    view_name = 'Checkerboard'
    description = 'Intrinsics + hand-eye via checkerboard'
    show_in_sidebar = False
    parent_view_id = 'calibration'

    def __init__(self, app, parent=None):
        super().__init__(app, parent)
        self._cam_thread = None
        self._poll_thread = None
        self._he_points = []  # hand-eye correspondences: list of (p_robot_m, p_cam, pixel_x, pixel_y)
        self._intr_frames = []
        self._ground_samples = []
        self._last_frame = None  # most recent camera frame (for click handler)
        # Gripper camera intrinsics
        self._gripper_intr_frames = []  # captured frames for gripper intrinsics calib
        self._gripper_cap = None        # cv2.VideoCapture for the gripper camera
        self._gripper_timer = None      # QTimer for live gripper preview
        self._last_gripper_frame = None  # most recent gripper frame
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)

        # Camera feed — ClickableLabel so users can click to record H-E points
        cam_col = QVBoxLayout()
        self._cam_label = ClickableLabel('Camera')
        self._cam_label.setAlignment(Qt.AlignCenter)
        self._cam_label.setMinimumSize(320, 240)
        self._cam_label.setStyleSheet('background-color: #1a1a1a;')
        self._cam_label.clicked.connect(self._on_cam_click)
        cam_col.addWidget(self._cam_label)
        hint = QLabel('Left-click image to record hand-eye point (board must be visible)')
        hint.setStyleSheet('color: #666; font-size: 10px;')
        hint.setAlignment(Qt.AlignCenter)
        cam_col.addWidget(hint)
        layout.addLayout(cam_col, stretch=3)

        # Controls
        ctrl = QWidget()
        ctrl_layout = QVBoxLayout(ctrl)
        ctrl_layout.setSpacing(4)

        # Camera switch
        self._cam_switch_btn = make_button('Camera: Overview', self._switch_camera,
                                           'Switch between overview and gripper camera', '#4a3c5a')
        ctrl_layout.addWidget(self._cam_switch_btn)

        ctrl_layout.addWidget(section_label('Intrinsics'))
        self._gripper_intr_status = QLabel('')
        self._gripper_intr_status.setStyleSheet('color: #aaa; font-size: 10px;')
        ctrl_layout.addWidget(self._gripper_intr_status)
        ctrl_layout.addWidget(make_button('Capture Frame', self._capture_intr,
                                          'Capture checkerboard frame for intrinsics', '#3c5a70'))
        ctrl_layout.addWidget(make_button('Calibrate Intrinsics', self._calibrate_intr,
                                          'Run camera calibration', '#3c5a70'))
        ctrl_layout.addWidget(make_button('Visualize Intrinsics', self._visualize_intr,
                                          '', '#3c5a70'))
        ctrl_layout.addWidget(make_button('Clear Frames', self._clear_intr_frames,
                                          'Discard all captured frames', '#643232'))

        ctrl_layout.addWidget(section_label('Ground Plane'))
        ctrl_layout.addWidget(make_button('Capture Plane Sample', self._capture_ground,
                                          'Capture ground-plane board detection', '#705a3c'))
        ctrl_layout.addWidget(make_button('Save Plane', self._save_ground, '', '#705a3c'))

        ctrl_layout.addWidget(section_label('Hand-Eye'))
        self._he_status = QLabel('Points: 0')
        self._he_status.setStyleSheet('color: #aaa;')
        ctrl_layout.addWidget(self._he_status)
        ctrl_layout.addWidget(make_button('Solve Hand-Eye', self._solve_handeye,
                                          'Solve camera-to-base transform', '#506430'))
        ctrl_layout.addWidget(make_button('Undo Last Point', self._undo_he_point,
                                          'Remove last hand-eye correspondence', '#644832'))
        ctrl_layout.addWidget(make_button('Clear All Points', self._clear_he_points,
                                          'Remove all correspondences', '#643232'))
        ctrl_layout.addWidget(make_button('Print Pose', self._print_pose, '', '#3c3c3c'))

        # Robot controls (if robot present)
        ctrl_layout.addWidget(section_label('Robot'))
        for j in range(1, 7):
            row = QHBoxLayout()
            row.addWidget(make_button(f'J{j}-', lambda ch, jj=j: self._jog(jj-1, -1), min_w=50))
            row.addWidget(make_button(f'J{j}+', lambda ch, jj=j: self._jog(jj-1, +1), min_w=50))
            ctrl_layout.addLayout(row)
        ctrl_layout.addWidget(make_button('Stop Jog', self._stop_jog, color='#992222'))

        grip_row = QHBoxLayout()
        grip_row.addWidget(make_button('Grip Open', self._grip_open, color='#006600'))
        grip_row.addWidget(make_button('Grip Close', self._grip_close, color='#660000'))
        ctrl_layout.addLayout(grip_row)

        self._calib_enable_btn = make_button('Servos ON', self._enable_robot, color='#006400')
        ctrl_layout.addWidget(self._calib_enable_btn)

        ctrl_layout.addWidget(make_button('< Back to Calibration',
                                          lambda: self.app.switch_view('calibration'), color='#444'))
        ctrl_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(300)
        scroll.setWidget(ctrl)
        layout.addWidget(scroll, stretch=1)

    def on_activate(self):
        self.app.ensure_camera()
        self.app.ensure_robot()
        if self.app.camera:
            self._cam_thread = CameraThread(self.app.camera)
            self._cam_thread.frame_ready.connect(self._on_frame)
            self._cam_thread.start()

    def on_deactivate(self):
        if self._cam_thread:
            self._cam_thread.stop()
            self._cam_thread = None
        self._stop_gripper_preview()

    def _is_gripper_active(self):
        return self._gripper_cap is not None

    def _switch_camera(self):
        """Toggle between overview and gripper camera."""
        if self._is_gripper_active():
            self._stop_gripper_preview()
            self._cam_switch_btn.setText('Camera: Overview')
            self._cam_switch_btn.setStyleSheet(
                'background-color: #4a3c5a; color: white; padding: 4px; border-radius: 3px;')
        else:
            self._start_gripper_preview()
            self._cam_switch_btn.setText('Camera: Gripper')
            self._cam_switch_btn.setStyleSheet(
                'background-color: #5a3c4a; color: white; padding: 4px; border-radius: 3px;')
        self._update_intr_status()

    def _update_intr_status(self):
        if self._is_gripper_active():
            n = len(self._gripper_intr_frames)
            self._gripper_intr_status.setText(f'Gripper camera  |  Frames: {n}')
        else:
            n = len(self._intr_frames)
            self._gripper_intr_status.setText(f'Overview camera  |  Frames: {n}')

    def _clear_intr_frames(self):
        """Clear captured intrinsics frames for the active camera."""
        if self._is_gripper_active():
            self._gripper_intr_frames.clear()
            print('  Cleared gripper intrinsics frames')
        else:
            self._intr_frames.clear()
            print('  Cleared overview intrinsics frames')
        self._update_intr_status()

    def _on_frame(self, frame):
        # Skip overview camera updates while gripper preview is active
        if self._gripper_cap is not None:
            return
        h, w = frame.shape[:2]
        # Apply numbered markers for hand-eye click points
        frame_with_markers = self._draw_he_markers(frame)
        img = cv_to_qimage(frame_with_markers)
        pix = QPixmap.fromImage(img).scaled(
            self._cam_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._cam_label.set_source_size(w, h)
        self._cam_label.setPixmap(pix)
        self._last_frame = frame.copy()

    def _draw_he_markers(self, frame):
        """Draw numbered markers at each recorded hand-eye click point on the frame.

        Args:
            frame: BGR numpy array from camera

        Returns:
            Frame with numbered markers drawn at each recorded pixel location
        """
        frame_marked = frame.copy()
        h, w = frame.shape[:2]

        # Draw a marker for each recorded hand-eye point
        for idx, point in enumerate(self._he_points, start=1):
            # point is now (p_robot_m, p_cam, x, y)
            if len(point) >= 4:
                x, y = int(point[2]), int(point[3])
                # Ensure coordinates are within frame bounds
                if 0 <= x < w and 0 <= y < h:
                    # Draw a circle at the clicked position
                    radius = 15
                    color = (0, 255, 0)  # Green circle
                    thickness = 2
                    cv2.circle(frame_marked, (x, y), radius, color, thickness)

                    # Draw the point number inside or near the circle
                    text = str(idx)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    text_thickness = 2
                    text_color = (0, 255, 0)

                    # Get text size to center it
                    text_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
                    text_x = x - text_size[0] // 2
                    text_y = y + text_size[1] // 2

                    cv2.putText(frame_marked, text, (text_x, text_y), font,
                               font_scale, text_color, text_thickness)

        return frame_marked

    def _on_cam_click(self, x: int, y: int) -> None:
        """Handle left-click on the camera image: record a hand-eye correspondence.

        Grabs the most recent camera frame, runs board detection to get the
        board-in-camera transform, then casts a ray through the clicked pixel
        and intersects it with the checkerboard plane.  The 3-D point in camera
        frame is paired with the current robot TCP position (in metres) and
        appended to ``self._he_points``.
        """
        import yaml as _yaml
        frame = getattr(self, '_last_frame', None)
        if frame is None and self.app.camera is not None:
            color, _, _ = self.app.camera.get_frames()
            frame = color
        if frame is None:
            print('  [HE click] No frame available')
            return
        if self.app.robot is None:
            print('  [HE click] No robot connected')
            return

        try:
            from calibration.calib_helpers import detect_corners, compute_board_pose, ray_plane_intersect

            # Load camera intrinsics (camera_intrinsics.yaml written by checkerboard calib).
            intr_yaml = config_path('camera_intrinsics.yaml')
            if not os.path.exists(intr_yaml):
                print('  [HE click] No camera_intrinsics.yaml — calibrate intrinsics first')
                return
            with open(intr_yaml) as f:
                d = _yaml.safe_load(f)

            # Build a lightweight intrinsics object matching the calib_helpers API.
            class _Intr:
                pass
            intr = _Intr()
            intr.fx = float(d['fx'])
            intr.fy = float(d['fy'])
            intr.ppx = float(d['ppx'])
            intr.ppy = float(d['ppy'])
            intr.coeffs = list(d.get('dist', [0.0] * 5))

            # Try to load a BoardDetector from config (supports ChArUco/ArUco).
            board_detector = None
            try:
                from vision.board_detector import BoardDetector as _BD
                board_detector = _BD.from_config(self.app.config)
            except Exception:
                pass

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners, detection = detect_corners(gray, board_detector)
            if not found:
                print(f'  [HE click] No board detected — make sure the checkerboard is visible')
                self._he_status.setText(
                    f'Points: {len(self._he_points)} (no board at last click)')
                return

            T_board, _, reproj_err = compute_board_pose(
                corners, intr, detection, board_detector)
            if T_board is None:
                print('  [HE click] Could not compute board pose')
                return

            p_cam = ray_plane_intersect((x, y), intr, T_board)
            if p_cam is None:
                print('  [HE click] Ray is parallel to the board plane — try a different angle')
                return

            # Read the current robot TCP pose (mm → m).
            pose = self.app.robot.get_pose()
            if pose is None:
                print('  [HE click] Cannot read robot pose')
                return
            p_robot_m = np.array(pose[:3], dtype=float) / 1000.0

            self._he_points.append((p_robot_m, p_cam, x, y))  # Store pixel coords (x, y) as well
            self._he_status.setText(f'Points: {len(self._he_points)}')
            print(
                f'  [HE click] Point {len(self._he_points)}: '
                f'cam=[{p_cam[0]:.4f}, {p_cam[1]:.4f}, {p_cam[2]:.4f}] '
                f'robot=[{pose[0]:.1f}, {pose[1]:.1f}, {pose[2]:.1f}] mm  '
                f'pixel=({x}, {y}) '
                f'(reproj {reproj_err:.2f} px)'
            )

        except Exception as exc:
            import traceback
            print(f'  [HE click] Error: {exc}')
            traceback.print_exc()

    def _capture_intr(self):
        """Capture an intrinsics frame from the active camera."""
        if self._is_gripper_active():
            frame = self._last_gripper_frame
            if frame is not None:
                self._gripper_intr_frames.append(frame.copy())
                print(f'  Captured gripper intrinsics frame '
                      f'#{len(self._gripper_intr_frames)}')
        else:
            if self.app.camera is None:
                return
            color, _, _ = self.app.camera.get_frames()
            if color is not None:
                self._intr_frames.append(color.copy())
                print(f'  Captured overview intrinsics frame '
                      f'#{len(self._intr_frames)}')
        self._update_intr_status()

    def _calibrate_intr(self):
        print('  Running intrinsics calibration...')
        # Delegate to the existing calibration logic
        threading.Thread(target=self._run_intr_calib, daemon=True).start()

    def _run_intr_calib(self):
        try:
            is_gripper = self._is_gripper_active()
            frames = self._gripper_intr_frames if is_gripper else self._intr_frames
            cam_name = 'gripper' if is_gripper else 'overview'

            from vision.board_detector import BoardDetector
            detector = BoardDetector.from_config(self.app.config)
            print(f'  Calibrating {cam_name} camera ({len(frames)} frames, '
                  f'{detector.describe()})')

            all_obj_pts, all_img_pts = [], []
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detection = detector.detect(gray)
                if detection is not None:
                    obj_pts = detector.get_object_points(detection)
                    all_obj_pts.append(obj_pts)
                    all_img_pts.append(detection.corners)
            if len(all_obj_pts) < 3:
                print(f'  Need >=3 valid frames, got {len(all_obj_pts)}')
                return
            h, w = frames[0].shape[:2]
            ret, K, dist, _, _ = cv2.calibrateCamera(
                all_obj_pts, all_img_pts, (w, h), None, None)
            print(f'  {cam_name} intrinsics RMS error: {ret:.3f}')

            import yaml
            if is_gripper:
                intr_path = config_path('gripper_intrinsics.yaml')
            else:
                intr_path = config_path('camera_intrinsics.yaml')
            data = {
                'fx': float(K[0, 0]), 'fy': float(K[1, 1]),
                'ppx': float(K[0, 2]), 'ppy': float(K[1, 2]),
                'dist': [float(d) for d in dist[0]],
                'width': w, 'height': h,
            }
            with open(intr_path, 'w') as f:
                yaml.dump(data, f)
            print(f'  Saved {cam_name} intrinsics to {intr_path}')
        except Exception as e:
            print(f'  Intrinsics calibration error: {e}')

    def _visualize_intr(self):
        """Visualize intrinsics by undistorting a captured frame."""
        import yaml as _yaml
        intr_path = config_path('camera_intrinsics.yaml')
        if not os.path.exists(intr_path):
            print('  No intrinsics file found — run Calibrate Intrinsics first')
            return
        try:
            with open(intr_path) as f:
                d = _yaml.safe_load(f)
            K = np.array([
                [d['fx'], 0, d['ppx']],
                [0, d['fy'], d['ppy']],
                [0, 0, 1],
            ], dtype=np.float64)
            dist = np.array(d.get('dist', [0]*5), dtype=np.float64)
            w, h = d.get('width', 640), d.get('height', 480)
        except Exception as e:
            print(f'  Error loading intrinsics: {e}')
            return
        # Grab a live frame (or use last captured frame)
        frame = None
        if self.app.camera:
            color, _, _ = self.app.camera.get_frames()
            if color is not None:
                frame = color.copy()
        if frame is None and self._intr_frames:
            frame = self._intr_frames[-1].copy()
        if frame is None:
            print('  No frame available for visualisation')
            return
        # Undistort and show side by side
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        undistorted = cv2.undistort(frame, K, dist, None, new_K)
        # Draw grid lines on both to make distortion visible
        for img, label in [(frame, 'Original'), (undistorted, 'Undistorted')]:
            cv2.putText(img, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        combined = np.hstack([frame, undistorted])
        # Show in the camera label
        img_q = cv_to_qimage(combined)
        pix = QPixmap.fromImage(img_q).scaled(
            self._cam_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._cam_label.setPixmap(pix)
        print(f'  Visualised intrinsics: fx={K[0,0]:.1f} fy={K[1,1]:.1f} '
              f'ppx={K[0,2]:.1f} ppy={K[1,2]:.1f}')

    # ------------------------------------------------------------------
    # Gripper Camera Intrinsics
    # ------------------------------------------------------------------

    def _get_gripper_device_index(self):
        """Return the device index for the gripper-mounted camera.

        Priority:
        1. cameras.yaml gripper camera entry
        2. robot_config.yaml gripper_camera.device_index
        3. Fallback: 8
        """
        try:
            from camera_config import CameraRegistry
            registry = CameraRegistry.load()
            gripper_cam = registry.find_gripper_camera()
            if gripper_cam is not None and gripper_cam.device_index >= 0:
                return gripper_cam.device_index, gripper_cam.name
        except Exception:
            pass
        idx = self.app.config.get('gripper_camera', {}).get('device_index', 8)
        return idx, None

    def _update_gripper_status(self):
        """Refresh the gripper intrinsics status label."""
        n = len(self._gripper_intr_frames)
        preview = 'on' if self._gripper_cap is not None else 'off'
        self._gripper_intr_status.setText(f'Frames: {n}  |  Preview: {preview}')

    def _gripper_preview_toggle(self):
        """Toggle live preview of the gripper-mounted camera."""
        if self._gripper_cap is not None:
            self._stop_gripper_preview()
        else:
            self._start_gripper_preview()
        self._update_gripper_status()

    def _start_gripper_preview(self):
        """Open the gripper camera and start the live preview timer."""
        dev_idx, cam_name = self._get_gripper_device_index()
        gc = self.app.config.get('gripper_camera', {})
        w = gc.get('width', 640)
        h = gc.get('height', 480)
        print(f'  Opening gripper camera /dev/video{dev_idx} ({w}x{h})...')
        cap = cv2.VideoCapture(dev_idx)
        if not cap.isOpened():
            print(f'  Cannot open gripper camera at /dev/video{dev_idx}')
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        # Flush stale buffer frames
        for _ in range(10):
            cap.read()
        self._gripper_cap = cap
        self._gripper_timer = QTimer()
        self._gripper_timer.timeout.connect(self._gripper_timer_tick)
        self._gripper_timer.start(100)  # ~10 fps preview
        label = cam_name or f'/dev/video{dev_idx}'
        print(f'  Gripper camera preview started ({label})')

    def _stop_gripper_preview(self):
        """Stop the gripper preview timer and release the camera."""
        if self._gripper_timer is not None:
            self._gripper_timer.stop()
            self._gripper_timer = None
        if self._gripper_cap is not None:
            self._gripper_cap.release()
            self._gripper_cap = None

    def _gripper_timer_tick(self):
        """Called by QTimer to update the camera label with a gripper frame."""
        if self._gripper_cap is None:
            return
        # Flush one stale frame, then read fresh
        self._gripper_cap.read()
        ok, frame = self._gripper_cap.read()
        if not ok or frame is None:
            return
        self._last_gripper_frame = frame.copy()
        annotated = frame.copy()
        n = len(self._gripper_intr_frames)
        cv2.putText(annotated, f'Gripper cam  |  Frames: {n}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        img = cv_to_qimage(annotated)
        pix = QPixmap.fromImage(img).scaled(
            self._cam_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._cam_label.setPixmap(pix)

    def _capture_gripper_frame(self):
        """Capture a checkerboard frame from the gripper camera for intrinsics calibration."""
        if self._gripper_cap is None:
            print('  Start gripper preview first (click "Preview Gripper Cam")')
            return
        # Flush stale frames then grab a fresh one
        for _ in range(3):
            self._gripper_cap.read()
        ok, frame = self._gripper_cap.read()
        if not ok or frame is None:
            print('  Failed to read frame from gripper camera')
            return
        self._gripper_intr_frames.append(frame.copy())
        n = len(self._gripper_intr_frames)
        print(f'  Captured gripper intrinsics frame #{n}')
        # Flash captured annotation on the label
        annotated = frame.copy()
        cv2.putText(annotated, f'Captured #{n}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        img = cv_to_qimage(annotated)
        pix = QPixmap.fromImage(img).scaled(
            self._cam_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._cam_label.setPixmap(pix)
        self._update_gripper_status()

    def _calibrate_gripper_intr(self):
        """Run intrinsics calibration on captured gripper frames (threaded)."""
        if len(self._gripper_intr_frames) < 3:
            print(f'  Need at least 3 gripper frames, have {len(self._gripper_intr_frames)}')
            return
        print('  Running gripper camera intrinsics calibration...')
        threading.Thread(target=self._run_gripper_intr_calib, daemon=True).start()

    def _run_gripper_intr_calib(self):
        """Worker: detect board corners in gripper frames and calibrate."""
        try:
            from vision.board_detector import BoardDetector
            detector = BoardDetector.from_config(self.app.config)
            print(f'  Using board: {detector.describe()}')
            all_obj_pts, all_img_pts = [], []
            for i, frame in enumerate(self._gripper_intr_frames):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detection = detector.detect(gray)
                if detection is not None:
                    obj_pts = detector.get_object_points(detection)
                    all_obj_pts.append(obj_pts)
                    all_img_pts.append(detection.corners)
                else:
                    print(f'  Frame {i+1}: no board detected — skipped')
            if len(all_obj_pts) < 3:
                print(f'  Need >=3 valid frames, got {len(all_obj_pts)} with board')
                return
            h, w = self._gripper_intr_frames[0].shape[:2]
            rms, K, dist, _, _ = cv2.calibrateCamera(
                all_obj_pts, all_img_pts, (w, h), None, None)
            print(f'  Gripper intrinsics RMS error: {rms:.3f} px  '
                  f'(fx={K[0,0]:.1f} fy={K[1,1]:.1f} '
                  f'ppx={K[0,2]:.1f} ppy={K[1,2]:.1f})')
            self._save_gripper_intrinsics(K, dist, (w, h), rms)
        except Exception as exc:
            import traceback
            print(f'  Gripper calibration error: {exc}')
            traceback.print_exc()

    def _save_gripper_intrinsics(self, K, dist, image_size, rms):
        """Persist calibrated gripper intrinsics to YAML files.

        Saves to:
          - config/gripper_camera_intrinsics.yaml  (dedicated file)
          - config/cameras.yaml  (updates the gripper camera entry)
        """
        import yaml as _yaml
        from datetime import datetime as _dt

        fx = float(K[0, 0])
        fy = float(K[1, 1])
        ppx = float(K[0, 2])
        ppy = float(K[1, 2])
        dist_list = [float(d) for d in dist.ravel()]
        w, h = int(image_size[0]), int(image_size[1])
        calibrated_at = _dt.now().isoformat()

        # ── gripper_camera_intrinsics.yaml ──
        gripper_intr_path = config_path('gripper_camera_intrinsics.yaml')
        data = {
            'camera_matrix': [[fx, 0.0, ppx], [0.0, fy, ppy], [0.0, 0.0, 1.0]],
            'dist_coeffs': dist_list,
            'image_size': [w, h],
            # Legacy flat keys for compatibility
            'fx': fx, 'fy': fy, 'ppx': ppx, 'ppy': ppy,
            'dist': dist_list,
            'width': w, 'height': h,
            'rms_error': float(rms),
            'calibrated_at': calibrated_at,
        }
        with open(gripper_intr_path, 'w') as f:
            _yaml.dump(data, f, default_flow_style=False)
        print(f'  Saved gripper intrinsics → {gripper_intr_path}')

        # ── cameras.yaml — update gripper camera entry ──
        cameras_path = config_path('cameras.yaml')
        if os.path.exists(cameras_path):
            with open(cameras_path, 'r') as f:
                cameras_data = _yaml.safe_load(f) or {}
            # Find gripper camera by mount type
            updated_name = None
            for cam_name, cam_entry in (cameras_data.get('cameras') or {}).items():
                mount = cam_entry.get('mount', {}) or {}
                if mount.get('type') == 'gripper':
                    cam_entry['intrinsics'] = {
                        'camera_matrix': [[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]],
                        'dist_coeffs': dist_list,
                        'image_size': [w, h],
                        'source': 'calibrated',
                        'rms_error': float(rms),
                        'calibrated_at': calibrated_at,
                    }
                    updated_name = cam_name
                    break
            if updated_name:
                with open(cameras_path, 'w') as f:
                    _yaml.dump(cameras_data, f, default_flow_style=False)
                print(f'  Updated cameras.yaml ({updated_name}) → source: calibrated')
            else:
                print('  Warning: no gripper-mounted camera found in cameras.yaml; '
                      'intrinsics saved to gripper_camera_intrinsics.yaml only')

        QTimer.singleShot(0, self._update_gripper_status)

    def _visualize_gripper_intr(self):
        """Show undistorted gripper camera frame to verify calibration."""
        import yaml as _yaml
        gripper_intr_path = config_path('gripper_camera_intrinsics.yaml')
        if not os.path.exists(gripper_intr_path):
            print('  No gripper_camera_intrinsics.yaml — run "Calibrate Gripper Camera" first')
            return
        try:
            with open(gripper_intr_path) as f:
                d = _yaml.safe_load(f)
            K = np.array([
                [d['fx'], 0, d['ppx']],
                [0, d['fy'], d['ppy']],
                [0, 0, 1],
            ], dtype=np.float64)
            dist = np.array(d.get('dist', [0] * 5), dtype=np.float64)
            w, h = d.get('width', 640), d.get('height', 480)
        except Exception as exc:
            print(f'  Error loading gripper intrinsics: {exc}')
            return
        # Use last gripper frame, or grab a fresh one
        frame = None
        if self._last_gripper_frame is not None:
            frame = self._last_gripper_frame.copy()
        elif self._gripper_cap is not None:
            for _ in range(3):
                self._gripper_cap.read()
            ok, frame = self._gripper_cap.read()
            if not ok:
                frame = None
        if frame is None and self._gripper_intr_frames:
            frame = self._gripper_intr_frames[-1].copy()
        if frame is None:
            print('  No gripper frame available for visualisation — start preview first')
            return
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        undistorted = cv2.undistort(frame, K, dist, None, new_K)
        for img, label in [(frame, 'Gripper-Orig'), (undistorted, 'Gripper-Undist')]:
            cv2.putText(img, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        combined = np.hstack([frame, undistorted])
        img_q = cv_to_qimage(combined)
        pix = QPixmap.fromImage(img_q).scaled(
            self._cam_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._cam_label.setPixmap(pix)
        rms = d.get('rms_error', float('nan'))
        print(f'  Visualised gripper intrinsics: fx={K[0,0]:.1f} fy={K[1,1]:.1f} '
              f'ppx={K[0,2]:.1f} ppy={K[1,2]:.1f}  rms={rms:.3f}px')

    def _clear_gripper_frames(self):
        """Discard all captured gripper intrinsics frames."""
        self._gripper_intr_frames.clear()
        self._last_gripper_frame = None
        print('  Cleared all gripper intrinsics frames')
        self._update_gripper_status()

    def _capture_ground(self):
        if self.app.camera is None:
            return
        color, _, _ = self.app.camera.get_frames()
        if color is not None:
            self._ground_samples.append(color.copy())
            print(f'  Captured ground sample #{len(self._ground_samples)}')

    def _save_ground(self):
        """Compute and save ground plane from captured board samples."""
        import yaml as _yaml
        if len(self._ground_samples) < 1:
            print('  Need at least 1 ground plane sample')
            return
        try:
            from calibration.calib_helpers import detect_corners, compute_board_pose
            # Load intrinsics
            intr_path = config_path('camera_intrinsics.yaml')
            intrinsics = None
            if os.path.exists(intr_path):
                with open(intr_path) as f:
                    d = _yaml.safe_load(f)
                K = np.array([
                    [d['fx'], 0, d['ppx']],
                    [0, d['fy'], d['ppy']],
                    [0, 0, 1],
                ], dtype=np.float64)
                dist = np.array(d.get('dist', [0]*5), dtype=np.float64)
                intrinsics = {'camera_matrix': K, 'dist_coeffs': dist}
            if intrinsics is None:
                print('  No intrinsics available — calibrate intrinsics first')
                return

            # Try to load BoardDetector from config
            board_detector = None
            try:
                from vision.board_detector import BoardDetector
                board_detector = BoardDetector.from_config(self.app.config)
            except Exception:
                pass

            # Detect board in each sample and compute pose
            planes = []
            for i, frame in enumerate(self._ground_samples):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found, corners, detection = detect_corners(gray, board_detector)
                if not found:
                    print(f'  Sample {i+1}: no board detected — skipped')
                    continue
                T_board_in_cam, obj_pts, err = compute_board_pose(
                    corners, intrinsics, detection, board_detector)
                if T_board_in_cam is not None:
                    # Ground plane normal = z-axis of board in camera frame
                    normal = T_board_in_cam[:3, 2]
                    origin = T_board_in_cam[:3, 3]
                    planes.append({
                        'normal': normal.tolist(),
                        'origin_m': origin.tolist(),
                        'reproj_err_px': float(err),
                    })
                    print(f'  Sample {i+1}: board at z={origin[2]*1000:.0f}mm, err={err:.2f}px')

            if not planes:
                print('  No valid board detections in ground samples')
                return

            # Average the plane parameters
            avg_normal = np.mean([p['normal'] for p in planes], axis=0)
            avg_normal /= np.linalg.norm(avg_normal)
            avg_origin = np.mean([p['origin_m'] for p in planes], axis=0)

            ground_data = {
                'ground_plane': {
                    'normal': avg_normal.tolist(),
                    'origin_m': avg_origin.tolist(),
                    'n_samples': len(planes),
                },
                'samples': planes,
            }
            out_path = config_path('ground_plane.yaml')
            with open(out_path, 'w') as f:
                _yaml.dump(ground_data, f, default_flow_style=False)
            print(f'  Saved ground plane to {out_path} ({len(planes)} samples)')
        except Exception as e:
            print(f'  Ground plane error: {e}')

    def _solve_handeye(self):
        if len(self._he_points) < 3:
            print(f'  Need >=3 points, have {len(self._he_points)}')
            return
        print('  Solving hand-eye calibration...')
        threading.Thread(target=self._run_handeye_solve, daemon=True).start()

    def _run_handeye_solve(self):
        try:
            from calibration.handeye_solver import solve_robust_transform
            robot_pts = np.array([p[0] for p in self._he_points])
            cam_pts = np.array([p[1] for p in self._he_points])
            T, inliers = solve_robust_transform(robot_pts, cam_pts)
            if T is not None:
                import yaml
                cal_path = config_path('calibration.yaml')
                data = {'T_camera_to_base': T.tolist()}
                with open(cal_path, 'w') as f:
                    yaml.dump(data, f)
                print(f'  Saved calibration to {cal_path} (inliers: {inliers})')
        except Exception as e:
            print(f'  Hand-eye solve error: {e}')

    def _undo_he_point(self):
        if self._he_points:
            self._he_points.pop()
        self._he_status.setText(f'Points: {len(self._he_points)}')

    def _clear_he_points(self):
        self._he_points.clear()
        self._he_status.setText('Points: 0')

    def _print_pose(self):
        if self.app.robot:
            angles = self.app.robot.get_angles()
            pose = self.app.robot.get_pose()
            if angles:
                print(f"  Joints: {', '.join(f'{v:.2f}' for v in angles)}")
            if pose:
                print(f"  Pose:   {', '.join(f'{v:.2f}' for v in pose)}")

    def _jog(self, idx, direction):
        if self.app.robot is None:
            return
        arm101 = getattr(self.app.robot, 'robot_type', None) == 'arm101'
        if arm101:
            self.app.robot.jog_joint(idx, direction)
        else:
            axis = f'J{idx + 1}{"+" if direction > 0 else "-"}'
            self.app.robot.send(f'MoveJog({axis})')

    def _stop_jog(self):
        if self.app.robot and not (getattr(self.app.robot, 'robot_type', None) == 'arm101'):
            self.app.robot.send('MoveJog()')

    def _grip_open(self):
        if self.app.robot is None:
            return
        if getattr(self.app.robot, 'robot_type', None) == 'arm101':
            self.app.robot.gripper_open()
        else:
            self.app.robot.send('ToolDOInstant(1,0)')
            self.app.robot.send('ToolDOInstant(2,1)')

    def _grip_close(self):
        if self.app.robot is None:
            return
        if getattr(self.app.robot, 'robot_type', None) == 'arm101':
            self.app.robot.gripper_close()
        else:
            self.app.robot.send('ToolDOInstant(2,0)')
            self.app.robot.send('ToolDOInstant(1,1)')

    def _enable_robot(self):
        if self.app.robot is None:
            return
        if getattr(self.app.robot, 'robot_type', None) == 'arm101':
            if getattr(self.app.robot, '_enabled', False):
                self.app.robot.disable_torque()
            else:
                self.app.robot.enable_torque()
            enabled = getattr(self.app.robot, '_enabled', False)
            btn = getattr(self, '_calib_enable_btn', None)
            if btn:
                btn.setText('Servos OFF' if enabled else 'Servos ON')
                btn.setStyleSheet(
                    f'background-color: {"#640000" if enabled else "#006400"}; '
                    f'color: white; padding: 4px; border-radius: 3px;')
        else:
            self.app.robot.send('DisableRobot()')
            time.sleep(1)
            self.app.robot.send('ClearError()')
            self.app.robot.send('EnableRobot()')


class ServoCalibView(BaseViewWidget):
    view_id = 'servo_calib'
    view_name = 'Servo Calibration'
    description = 'Zero offsets (arm101)'
    show_in_sidebar = False
    parent_view_id = 'calibration'

    def __init__(self, app, parent=None):
        super().__init__(app, parent)
        self._cam_thread = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel('Servo Calibration')
        title.setStyleSheet('font-size: 16px; font-weight: bold; color: #ffc864;')
        layout.addWidget(title)

        layout.addWidget(QLabel(
            'Move arm to zero pose by hand (torque is disabled).\n'
            'Click "Save Zero Offsets" when all joints are at 0 degrees.'))

        self._status = QLabel('Status: Ready')
        self._status.setStyleSheet('color: #aaa; font-family: monospace;')
        layout.addWidget(self._status)

        # Camera feed
        self._cam_label = QLabel('Camera')
        self._cam_label.setAlignment(Qt.AlignCenter)
        self._cam_label.setMinimumSize(320, 240)
        self._cam_label.setStyleSheet('background-color: #1a1a1a;')
        layout.addWidget(self._cam_label)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addWidget(make_button('Save Zero Offsets', self._save_offsets,
                                      'Save current raw positions as zero', '#506430'))
        btn_row.addWidget(make_button('Reload Offsets', self._reload_offsets,
                                      'Reload offsets from file', '#3c5a70'))
        btn_row.addWidget(make_button('< Back', lambda: self.app.switch_view('calibration'),
                                      color='#444'))
        layout.addLayout(btn_row)
        layout.addStretch()

    def on_activate(self):
        self.app.ensure_camera()
        self.app.ensure_robot()
        if self.app.robot and getattr(self.app.robot, 'robot_type', None) == 'arm101':
            try:
                self.app.robot.disable_torque()
                self._status.setText('Status: Torque OFF (freedrive)')
            except Exception as e:
                self._status.setText(f'Status: Error disabling torque: {e}')
        if self.app.camera:
            self._cam_thread = CameraThread(self.app.camera)
            self._cam_thread.frame_ready.connect(self._on_frame)
            self._cam_thread.start()

    def on_deactivate(self):
        if self._cam_thread:
            self._cam_thread.stop()
            self._cam_thread = None
        if self.app.robot and getattr(self.app.robot, 'robot_type', None) == 'arm101':
            try:
                self.app.robot.enable_torque()
            except Exception:
                pass

    def _on_frame(self, frame):
        img = cv_to_qimage(frame)
        pix = QPixmap.fromImage(img).scaled(
            self._cam_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._cam_label.setPixmap(pix)

    def _save_offsets(self):
        if self.app.robot is None:
            return
        try:
            raw = self.app.robot.get_raw_positions()
            if raw:
                import yaml
                offsets_path = config_path('servo_offsets.yaml')
                data = {}
                if os.path.exists(offsets_path):
                    with open(offsets_path, 'r') as f:
                        data = yaml.safe_load(f) or {}
                data['zero_raw'] = [int(r) for r in raw]
                with open(offsets_path, 'w') as f:
                    yaml.dump(data, f)
                self._status.setText(f'Saved zero offsets: {[int(r) for r in raw]}')
                print(f'  Saved servo offsets to {offsets_path}')
        except Exception as e:
            self._status.setText(f'Error: {e}')

    def _reload_offsets(self):
        try:
            import yaml
            offsets_path = config_path('servo_offsets.yaml')
            if os.path.exists(offsets_path):
                with open(offsets_path, 'r') as f:
                    data = yaml.safe_load(f)
                self._status.setText(f'Loaded: {data}')
            else:
                self._status.setText('No offsets file found')
        except Exception as e:
            self._status.setText(f'Error: {e}')


class HandEyeYellowView(BaseViewWidget):
    view_id = 'handeye_yellow'
    view_name = 'Hand-Eye Yellow'
    description = 'FK+pixel calibration (arm101)'
    show_in_sidebar = False
    parent_view_id = 'calibration'

    # Signal emitted by worker thread to update progress bar
    progress_updated = pyqtSignal(int, int)  # (current, total)

    def __init__(self, app, parent=None):
        super().__init__(app, parent)
        self._cam_thread = None
        # Capture data: list of dicts with 'raw', 'tcp', 'pixel'
        self._captures = []
        self._raw_positions_list = []
        self._pts_2d = []
        self._pts_3d_robot = []
        self._solver = None
        self._K = None
        self._dist_coeffs = None
        self._last_frame = None
        self._last_yellow = (None, None)
        self._last_tcp = None
        # Manual click override: set when user clicks camera; None = use auto-detect
        self._manual_pixel = None
        # Per-capture reprojection errors (px) set after joint_solve; None before first solve
        self._reproj_errors = None
        self._build_ui()
        # Connect progress signal to update progress bar
        self.progress_updated.connect(self._on_progress_update)

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Camera feed (left) — ClickableLabel so user can manually set pixel
        self._cam_label = ClickableLabel('Camera')
        self._cam_label.setAlignment(Qt.AlignCenter)
        self._cam_label.setMinimumSize(480, 360)
        self._cam_label.setStyleSheet('background-color: #1a1a1a;')
        self._cam_label.clicked.connect(self._on_cam_click)
        layout.addWidget(self._cam_label, stretch=3)

        # Controls (right)
        ctrl = QWidget()
        ctrl_layout = QVBoxLayout(ctrl)

        title = QLabel('Hand-Eye Calibration (Yellow Tape)')
        title.setStyleSheet('font-size: 14px; font-weight: bold; color: #ffc864;')
        ctrl_layout.addWidget(title)
        ctrl_layout.addWidget(QLabel(
            'Move arm to diverse poses with\n'
            'yellow tape visible in camera.\n'
            'Capture FK+pixel, then solve.'))

        self._status = QLabel('Captures: 0 (need >= 6)')
        self._status.setStyleSheet('color: #aaa; font-family: monospace;')
        ctrl_layout.addWidget(self._status)

        # Scrollable table of all captures (7 columns: 6 data + reprojection error after solve)
        self._capture_table = QTableWidget(0, 7)
        self._capture_table.setHorizontalHeaderLabels(
            ['#', 'TCP X', 'TCP Y', 'TCP Z', 'Px', 'Py', 'Err(px)'])
        self._capture_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._capture_table.verticalHeader().setVisible(False)
        self._capture_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._capture_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._capture_table.setMinimumHeight(160)
        self._capture_table.setMaximumHeight(260)
        self._capture_table.setStyleSheet(
            'QTableWidget { background-color: #1a1a1a; color: #ccc;'
            '  font-family: monospace; font-size: 10px; gridline-color: #444; }'
            'QHeaderView::section { background-color: #333; color: #ffc864;'
            '  font-size: 10px; padding: 2px; }'
            'QTableWidget::item:selected { background-color: #3a5a8a; }')
        ctrl_layout.addWidget(self._capture_table)

        # Progress bar for joint solve optimization
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        self._progress_bar.setStyleSheet('''
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                background-color: #222;
            }
            QProgressBar::chunk {
                background-color: #3c705a;
            }
        ''')
        ctrl_layout.addWidget(self._progress_bar)

        # Pixel mode indicator (auto-detect vs. manual click override)
        self._pixel_mode_label = QLabel('Pixel: auto-detect')
        self._pixel_mode_label.setStyleSheet(
            'color: #aaa; font-family: monospace; font-size: 11px;')
        self._pixel_mode_label.setWordWrap(True)
        ctrl_layout.addWidget(self._pixel_mode_label)

        ctrl_layout.addWidget(make_button('Capture Pose', self._capture,
                                          'Capture FK + yellow tape position', '#506430'))
        ctrl_layout.addWidget(make_button('Clear Click Override', self._clear_manual_pixel,
                                          'Revert to auto-detect yellow tape', '#4a4a22'))
        ctrl_layout.addWidget(make_button('Solve', self._solve,
                                          'Run joint solve (>= 6 captures)', '#3c705a'))
        ctrl_layout.addWidget(make_button('Undo', self._undo, 'Remove last capture', '#644832'))
        ctrl_layout.addWidget(make_button('Clear All', self._clear_all,
                                          'Remove all captures', '#643232'))
        ctrl_layout.addWidget(make_button('< Back', lambda: self.app.switch_view('calibration'),
                                          color='#444'))
        ctrl_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(300)
        scroll.setWidget(ctrl)
        layout.addWidget(scroll, stretch=1)

    def _refresh_table(self):
        """Rebuild the capture table from self._captures with colour-coded rows."""
        t = self._capture_table
        t.setRowCount(0)

        # Collect all pixels to detect duplicates
        all_pixels = [c.get('pixel') for c in self._captures]

        # Pre-compute reprojection error threshold for outlier highlighting.
        # Use 2 × median so that rows notably worse than the rest stand out.
        reproj_threshold = None
        if (self._reproj_errors is not None
                and len(self._reproj_errors) >= 2):
            median_err = float(np.median(self._reproj_errors))
            reproj_threshold = 2.0 * median_err

        for i, cap in enumerate(self._captures):
            tcp = cap.get('tcp')
            px, py = cap.get('pixel', (None, None))
            fk_ok = cap.get('fk_ok', True)

            # Duplicate pixel detection: same pixel as any earlier capture
            is_duplicate = (px, py) in all_pixels[:i]

            # Per-row reprojection error (available after joint_solve)
            err_px = None
            if (self._reproj_errors is not None
                    and i < len(self._reproj_errors)):
                err_px = float(self._reproj_errors[i])
            err_str = f'{err_px:.1f}' if err_px is not None else '-'

            # High-error flag: err > 2 × median (only meaningful post-solve)
            is_high_error = (reproj_threshold is not None
                             and err_px is not None
                             and err_px > reproj_threshold)

            # Build row values (7 columns)
            if tcp is not None:
                vals = [
                    str(i + 1),
                    f'{tcp[0]:.0f}',
                    f'{tcp[1]:.0f}',
                    f'{tcp[2]:.0f}',
                    str(int(px)) if px is not None else '?',
                    str(int(py)) if py is not None else '?',
                    err_str,
                ]
            else:
                vals = [str(i + 1), '?', '?', '?',
                        str(int(px)) if px is not None else '?',
                        str(int(py)) if py is not None else '?',
                        err_str]

            row = t.rowCount()
            t.insertRow(row)
            for col, val in enumerate(vals):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)

                # Colour coding priority:
                #   1. red  — FK solve failed
                #   2. red  — high reprojection error (outlier after solve)
                #   3. yellow — duplicate pixel
                #   4. green  — good
                if not fk_ok:
                    item.setForeground(QColor('#ff6060'))
                    item.setToolTip('FK solve failed for this pose')
                elif is_high_error:
                    item.setForeground(QColor('#ff6060'))
                    item.setToolTip(
                        f'High reprojection error ({err_px:.1f} px) — '
                        f'outlier (threshold {reproj_threshold:.1f} px); '
                        f'consider deleting and re-solving')
                elif is_duplicate:
                    item.setForeground(QColor('#ffcc44'))
                    item.setToolTip('Duplicate pixel — same as an earlier capture')
                else:
                    item.setForeground(QColor('#88dd88'))

                t.setItem(row, col, item)

        # Scroll to last row so newest capture is always visible
        if t.rowCount() > 0:
            t.scrollToBottom()

        # Update status colour based on readiness
        n = len(self._captures)
        fk_fails = sum(1 for c in self._captures if not c.get('fk_ok', True))
        dups = sum(1 for i, c in enumerate(self._captures)
                   if c.get('pixel') in [self._captures[j].get('pixel') for j in range(i)])
        issues = fk_fails + dups
        status_txt = f'Captures: {n} (need >= 6)'
        if issues:
            status_txt += f'  ⚠ {issues} issue(s)'
        self._status.setText(status_txt)
        if n >= 6 and issues == 0:
            self._status.setStyleSheet('color: #88dd88; font-family: monospace;')
        elif n >= 6:
            self._status.setStyleSheet('color: #ffcc44; font-family: monospace;')
        else:
            self._status.setStyleSheet('color: #aaa; font-family: monospace;')

    def _init_solver_and_intrinsics(self):
        """Lazily initialise IK solver and load camera intrinsics."""
        if self._solver is None:
            try:
                from kinematics.arm101_ik_solver import Arm101IKSolver
                self._solver = Arm101IKSolver()
            except Exception as e:
                print(f'  ERROR: Cannot init IK solver: {e}')
                return False
        if self._K is None:
            self._K, self._dist_coeffs = self._load_intrinsics()
        return self._solver is not None and self._K is not None

    def _load_intrinsics(self):
        """Load camera intrinsics from cameras.yaml or use defaults."""
        import yaml as _yaml
        cam_yaml = config_path('cameras.yaml')
        cam_cfg = self.app.config.get('camera', {})
        cam_idx = cam_cfg.get('device_index', 4)
        if os.path.exists(cam_yaml):
            try:
                with open(cam_yaml) as f:
                    cdata = _yaml.safe_load(f)
                for cname, cinfo in cdata.get('cameras', {}).items():
                    if cinfo.get('device_index') == cam_idx:
                        intr = cinfo['intrinsics']
                        K = np.array(intr['camera_matrix'], dtype=np.float64)
                        dist = np.array(intr['dist_coeffs'], dtype=np.float64)
                        print(f'  Intrinsics from {cname}: fx={K[0,0]:.1f}')
                        return K, dist
            except Exception as e:
                print(f'  Warning: failed to load intrinsics: {e}')
        # Also try camera_intrinsics.yaml (saved by checkerboard calib)
        intr_yaml = config_path('camera_intrinsics.yaml')
        if os.path.exists(intr_yaml):
            try:
                with open(intr_yaml) as f:
                    d = _yaml.safe_load(f)
                K = np.array([
                    [d['fx'], 0, d['ppx']],
                    [0, d['fy'], d['ppy']],
                    [0, 0, 1],
                ], dtype=np.float64)
                dist = np.array(d.get('dist', [0]*5), dtype=np.float64)
                print(f'  Intrinsics from camera_intrinsics.yaml: fx={K[0,0]:.1f}')
                return K, dist
            except Exception as e:
                print(f'  Warning: failed to load camera_intrinsics.yaml: {e}')
        # Fallback defaults
        K = np.array([[554.3, 0, 320], [0, 554.3, 240], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
        print('  Using default intrinsics (estimated)')
        return K, dist

    def on_activate(self):
        self.app.ensure_camera()
        self.app.ensure_robot()
        if self.app.robot and getattr(self.app.robot, 'robot_type', None) == 'arm101':
            try:
                self.app.robot.disable_torque()
            except Exception:
                pass
        if self.app.camera:
            self._cam_thread = CameraThread(self.app.camera)
            self._cam_thread.frame_ready.connect(self._on_frame)
            self._cam_thread.start()
        # Pre-init solver/intrinsics
        self._init_solver_and_intrinsics()

    def on_deactivate(self):
        if self._cam_thread:
            self._cam_thread.stop()
            self._cam_thread = None
        if self.app.robot and getattr(self.app.robot, 'robot_type', None) == 'arm101':
            try:
                self.app.robot.enable_torque()
            except Exception:
                pass

    def _on_frame(self, frame):
        """Show camera feed with yellow tape overlay."""
        from calibration.calib_helpers import find_yellow_tape, draw_handeye_overlay
        self._last_frame = frame.copy()
        h, w = frame.shape[:2]
        # Detect yellow tape for live overlay
        cx, cy, mask = find_yellow_tape(frame)
        self._last_yellow = (cx, cy)
        # Compute FK for overlay
        tcp_pos = None
        if self.app.robot and self._solver:
            try:
                angles = self.app.robot.get_angles()
                if angles is not None:
                    tcp_pos, _ = self._solver.forward_kin(np.array(angles[:5]))
                    self._last_tcp = tcp_pos
            except Exception:
                pass
        # Choose which pixel to show in the overlay: manual override takes priority
        overlay_pixel = (cx, cy)
        if self._manual_pixel is not None:
            overlay_pixel = self._manual_pixel
        # Draw overlay
        draw_handeye_overlay(frame, tcp_pos, overlay_pixel, len(self._pts_2d))
        # Draw a prominent cyan crosshair when manual override is active
        if self._manual_pixel is not None:
            mx, my = self._manual_pixel
            cv2.drawMarker(frame, (mx, my), (0, 255, 255),
                           cv2.MARKER_CROSS, 24, 3, cv2.LINE_AA)
            cv2.putText(frame, 'MANUAL', (mx + 14, my - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        # Show small yellow mask in corner
        if mask is not None:
            try:
                mask_small = cv2.resize(mask, (160, 120))
                mask_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
                frame[0:120, frame.shape[1]-160:frame.shape[1]] = mask_bgr
            except Exception:
                pass
        img = cv_to_qimage(frame)
        pix = QPixmap.fromImage(img).scaled(
            self._cam_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._cam_label.set_source_size(w, h)
        self._cam_label.setPixmap(pix)

    def _capture(self):
        """Capture FK + yellow tape pixel + raw servo positions."""
        from calibration.calib_helpers import read_all_raw, find_yellow_tape
        if not self._init_solver_and_intrinsics():
            self._status.setText('ERROR: solver or intrinsics not available')
            return
        arm = self.app.robot
        if arm is None or getattr(arm, 'robot_type', None) != 'arm101':
            self._status.setText('ERROR: arm101 robot not connected')
            return
        # Read raw servo positions
        raw_positions = read_all_raw(arm)
        # Compute FK from current angles
        angles = arm.get_angles()
        tcp_pos = None
        if angles is not None:
            try:
                tcp_pos, _ = self._solver.forward_kin(np.array(angles[:5]))
            except Exception as e:
                print(f'  FK error: {e}')
        if tcp_pos is None:
            self._status.setText('SKIP: cannot compute FK')
            return
        # Pixel source: manual click override takes priority over auto-detect
        if self._manual_pixel is not None:
            cx, cy = self._manual_pixel
        else:
            cx, cy = self._last_yellow
        if cx is None:
            self._status.setText('SKIP: no yellow tape detected — click camera to set manually')
            return
        # Store capture
        self._raw_positions_list.append(dict(raw_positions))
        self._pts_2d.append([float(cx), float(cy)])
        self._pts_3d_robot.append(tcp_pos.copy())
        self._captures.append({
            'raw': dict(raw_positions),
            'tcp': tcp_pos.copy(),
            'pixel': (cx, cy),
            'fk_ok': True,
        })
        n = len(self._pts_2d)
        detail = (f'#{n}: TCP=[{tcp_pos[0]:.0f},{tcp_pos[1]:.0f},{tcp_pos[2]:.0f}] '
                  f'px=({cx},{cy})')
        print(f'  Captured pose {detail}')
        # New capture invalidates previous solve errors
        self._reproj_errors = None
        self._refresh_table()

    def _on_cam_click(self, img_x: int, img_y: int):
        """Store a manual pixel override when the user clicks the camera feed."""
        self._manual_pixel = (img_x, img_y)
        self._pixel_mode_label.setText(
            f'Pixel: MANUAL ({img_x}, {img_y})\nClick camera to update')
        self._pixel_mode_label.setStyleSheet(
            'color: #00ffff; font-family: monospace; font-size: 11px; font-weight: bold;')

    def _clear_manual_pixel(self):
        """Remove manual pixel override; revert to auto-detect."""
        self._manual_pixel = None
        self._pixel_mode_label.setText('Pixel: auto-detect')
        self._pixel_mode_label.setStyleSheet(
            'color: #aaa; font-family: monospace; font-size: 11px;')

    def _solve(self):
        """Run joint_solve to optimise servo offsets + camera extrinsics."""
        from calibration.calib_helpers import joint_solve, save_offsets_dict, \
            save_handeye_calibration, load_offsets, HANDEYE_FILE
        n = len(self._pts_2d)
        if n < 6:
            self._status.setText(f'Need >= 6 captures, have {n}')
            return
        if not self._init_solver_and_intrinsics():
            self._status.setText('ERROR: solver or intrinsics not available')
            return
        self._status.setText('Solving...')
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        print(f'\n  Joint solve: {n} points, optimising offsets + extrinsics...')
        # Run in thread to avoid blocking GUI
        threading.Thread(target=self._run_joint_solve, daemon=True).start()

    def _run_joint_solve(self):
        """Worker thread for joint solve."""
        from calibration.calib_helpers import joint_solve, save_offsets_dict, \
            save_handeye_calibration, load_offsets, HANDEYE_FILE
        try:
            # Create progress callback that emits the signal
            def progress_callback(current, total):
                """Called from optimization loop with progress updates."""
                self.progress_updated.emit(current, total)

            opt_offsets, T_c2b, errs_px = joint_solve(
                self._raw_positions_list, self._pts_2d,
                self._K, self._dist_coeffs, self._solver,
                progress_callback=progress_callback)
            if opt_offsets is not None:
                save_offsets_dict(opt_offsets)
                save_handeye_calibration(T_c2b, HANDEYE_FILE)
                # Store per-capture reprojection errors for table display
                self._reproj_errors = errs_px.tolist()
                msg = 'Joint solve OK — saved offsets + extrinsics'
                print(f'  {msg}')
                # Update status and table from main thread
                saved_fname = os.path.basename(HANDEYE_FILE)
                mean_err = float(np.mean(errs_px))
                status_msg = (f'Joint solve OK — saved offsets + {saved_fname}'
                              f'  (mean reproj {mean_err:.1f} px)')
                QTimer.singleShot(0, lambda m=status_msg: self._status.setText(m))
                QTimer.singleShot(0, lambda: self._status.setStyleSheet(
                    'color: #88dd88; font-family: monospace;'))
                QTimer.singleShot(0, lambda: self._progress_bar.setVisible(False))
                # Refresh table to show per-row reprojection errors
                QTimer.singleShot(0, self._refresh_table)
            else:
                self._reproj_errors = None
                msg = 'Joint solve FAILED — check captures'
                print(f'  {msg}')
                QTimer.singleShot(0, lambda m=msg: self._status.setText(m))
                QTimer.singleShot(0, lambda: self._status.setStyleSheet(
                    'color: #ff6060; font-family: monospace;'))
                QTimer.singleShot(0, lambda: self._progress_bar.setVisible(False))
        except Exception as e:
            msg = f'Solve error: {e}'
            print(f'  {msg}')
            QTimer.singleShot(0, lambda m=msg: self._status.setText(m))
            QTimer.singleShot(0, lambda: self._status.setStyleSheet(
                'color: #ff6060; font-family: monospace;'))
            QTimer.singleShot(0, lambda: self._progress_bar.setVisible(False))

    def _undo(self):
        if self._captures:
            self._captures.pop()
            self._raw_positions_list.pop()
            self._pts_2d.pop()
            self._pts_3d_robot.pop()
        # Capture list changed — reproj errors are stale
        self._reproj_errors = None
        self._refresh_table()

    def _clear_all(self):
        self._captures.clear()
        self._raw_positions_list.clear()
        self._pts_2d.clear()
        self._pts_3d_robot.clear()
        self._reproj_errors = None
        self._refresh_table()

    def _on_progress_update(self, current, total):
        """Slot to update progress bar from worker thread signal."""
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(current)
        # Update percentage in status
        if total > 0:
            pct = int(100 * current / total)
            self._status.setText(f'Solving... {pct}%')


class GripperCameraThread(QThread):
    """Polls gripper camera (cv2.VideoCapture) and emits frames."""
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, cap, parent=None):
        super().__init__(parent)
        self._cap = cap
        self._running = True

    def run(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                self.frame_ready.emit(frame)
            time.sleep(0.033)

    def stop(self):
        self._running = False
        self.wait(2000)


class ServoDirectionCalibView(BaseViewWidget):
    """Auto servo calibration using gripper camera + ChArUco board.

    Opens the gripper-mounted camera (not the main RealSense) and detects
    a ChArUco board.  The user moves the arm by hand to diverse poses
    keeping the board visible, and captures raw servo positions + board pose.
    The solver brute-forces all 32 sign combinations to find the best
    offset/sign configuration.
    """
    view_id = 'servo_direction'
    view_name = 'Servo Direction'
    description = 'Auto-detect servo signs + offsets via ChArUco (arm101)'
    show_in_sidebar = False
    parent_view_id = 'calibration'

    MIN_CAPTURES = 6
    GOOD_CAPTURES = 10

    def __init__(self, app, parent=None):
        super().__init__(app, parent)
        self._gripper_cap = None
        self._gripper_cam_thread = None
        self._board_detector = None
        self._solver = None
        self._gripper_K = None
        self._gripper_dist = None
        self._captures = []  # list of {raw, T_board_in_cam, centroid_px}
        self._result = None
        self._current_frame = None
        self._current_T_board = None
        self._current_detection = None
        self._current_raw = None
        self._error_msg = ''
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Left: camera feed
        cam_panel = QVBoxLayout()
        self._cam_label = QLabel('Gripper Camera')
        self._cam_label.setAlignment(Qt.AlignCenter)
        self._cam_label.setMinimumSize(480, 360)
        self._cam_label.setStyleSheet('background-color: #1a1a1a;')
        cam_panel.addWidget(self._cam_label)

        self._detection_label = QLabel('')
        self._detection_label.setStyleSheet('color: #aaa; font-family: monospace; font-size: 10px;')
        cam_panel.addWidget(self._detection_label)
        layout.addLayout(cam_panel, stretch=3)

        # Right: controls
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setMaximumWidth(320)
        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)

        title = QLabel('Servo Direction Auto-Calibration')
        title.setStyleSheet('font-size: 14px; font-weight: bold; color: #ffc864;')
        ctrl_layout.addWidget(title)

        ctrl_layout.addWidget(QLabel(
            'Uses gripper camera + ChArUco board.\n'
            'Move arm by hand to diverse poses,\n'
            'keep board visible, press Capture.'))

        self._status = QLabel(f'Captures: 0 (need >= {self.MIN_CAPTURES}, '
                              f'{self.GOOD_CAPTURES}+ recommended)')
        self._status.setStyleSheet('color: #aaa; font-family: monospace; font-size: 11px;')
        self._status.setWordWrap(True)
        ctrl_layout.addWidget(self._status)

        self._error_label = QLabel('')
        self._error_label.setStyleSheet('color: #ff4444; font-size: 11px;')
        self._error_label.setWordWrap(True)
        ctrl_layout.addWidget(self._error_label)

        # Action buttons
        ctrl_layout.addWidget(section_label('Actions'))
        ctrl_layout.addWidget(make_button('Capture Pose', self._capture,
                                          'Capture raw servo positions + board pose', '#506430'))
        ctrl_layout.addWidget(make_button('Solve (Brute-force)', self._solve,
                                          'Test all 32 sign combos (need >= 6 captures)', '#3c705a'))
        ctrl_layout.addWidget(make_button('Undo Last', self._undo,
                                          'Remove last capture', '#644832'))
        ctrl_layout.addWidget(make_button('Reset All', self._reset,
                                          'Clear all captures', '#643232'))

        # Results area
        ctrl_layout.addWidget(section_label('Results'))
        self._result_label = QLabel('No results yet')
        self._result_label.setStyleSheet('color: #aaa; font-family: monospace; font-size: 10px;')
        self._result_label.setWordWrap(True)
        ctrl_layout.addWidget(self._result_label)

        ctrl_layout.addWidget(make_button('< Back to Calibration',
                                          lambda: self.app.switch_view('calibration'),
                                          color='#444'))
        ctrl_layout.addStretch()

        ctrl_scroll.setWidget(ctrl_widget)
        layout.addWidget(ctrl_scroll, stretch=1)

    def on_activate(self):
        self._error_msg = ''
        self._error_label.setText('')

        # Robot check
        self.app.ensure_robot()
        robot = self.app.robot
        if robot is None:
            self._show_error('No robot connected')
            return
        if getattr(robot, 'robot_type', None) != 'arm101':
            self._show_error('Servo direction calibration requires arm101')
            return

        try:
            robot.disable_torque()
            print('  Servo direction calib: torque disabled, move arm by hand')
        except Exception as exc:
            print(f'  WARNING: Could not disable torque: {exc}')

        # FK solver
        try:
            from kinematics.arm101_ik_solver import Arm101IKSolver
            self._solver = Arm101IKSolver()
        except Exception as exc:
            self._show_error(f'FK solver not available: {exc}')
            return

        # Board detector
        try:
            from vision.board_detector import BoardDetector
            self._board_detector = BoardDetector.from_config(self.app.config)
        except Exception as exc:
            self._show_error(f'Board detector error: {exc}')
            return

        # Open gripper camera
        gc = self.app.config.get('gripper_camera', {})
        dev_idx = gc.get('device_index', 0)
        width = gc.get('width', 640)
        height = gc.get('height', 480)
        print(f'  Opening gripper camera /dev/video{dev_idx}...')
        self._gripper_cap = cv2.VideoCapture(dev_idx)
        if not self._gripper_cap.isOpened():
            self._show_error(f'Cannot open gripper camera (device {dev_idx})')
            return
        self._gripper_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._gripper_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Flush initial frames
        for _ in range(10):
            self._gripper_cap.read()
        print('  Gripper camera ready.')

        # Load gripper camera intrinsics
        self._gripper_K, self._gripper_dist = self._load_gripper_intrinsics()

        # Start camera thread
        self._gripper_cam_thread = GripperCameraThread(self._gripper_cap)
        self._gripper_cam_thread.frame_ready.connect(self._on_frame)
        self._gripper_cam_thread.start()

        # Read-servo timer (10Hz)
        self._servo_timer = QTimer()
        self._servo_timer.timeout.connect(self._poll_servos)
        self._servo_timer.start(100)

    def on_deactivate(self):
        if hasattr(self, '_servo_timer'):
            self._servo_timer.stop()
        if self._gripper_cam_thread:
            self._gripper_cam_thread.stop()
            self._gripper_cam_thread = None
        if self._gripper_cap is not None:
            self._gripper_cap.release()
            self._gripper_cap = None
            print('  Gripper camera released')
        if self.app.robot and getattr(self.app.robot, 'robot_type', None) == 'arm101':
            try:
                self.app.robot.enable_torque()
                print('  Servo direction calib: torque re-enabled')
            except Exception:
                pass

    def _show_error(self, msg):
        self._error_msg = msg
        self._error_label.setText(msg)

    def _load_gripper_intrinsics(self):
        """Load gripper camera intrinsics from cameras.yaml or estimate."""
        import yaml
        import math
        gc = self.app.config.get('gripper_camera', {})
        dev_idx = gc.get('device_index', 0)
        cam_yaml = config_path('cameras.yaml')

        if os.path.exists(cam_yaml):
            with open(cam_yaml) as fh:
                cdata = yaml.safe_load(fh)
            for cname, cinfo in (cdata or {}).get('cameras', {}).items():
                if cinfo.get('device_index') == dev_idx:
                    intr = cinfo.get('intrinsics', {})
                    cm = intr.get('camera_matrix')
                    dc = intr.get('dist_coeffs')
                    if cm is not None:
                        K = np.array(cm, dtype=np.float64)
                        dist = np.array(dc or [0, 0, 0, 0, 0], dtype=np.float64)
                        print(f'  Gripper intrinsics from {cname}: fx={K[0, 0]:.1f}')
                        return K, dist

        # Estimate from HFOV
        w = gc.get('width', 640)
        h = gc.get('height', 480)
        hfov = gc.get('hfov_deg', 60.0)
        fx = w / (2.0 * math.tan(math.radians(hfov / 2.0)))
        K = np.array([[fx, 0, w / 2.0],
                       [0, fx, h / 2.0],
                       [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
        print(f'  Gripper intrinsics estimated: fx={fx:.1f} (hfov={hfov} deg)')
        return K, dist

    def _poll_servos(self):
        """Read raw servo positions at 10Hz."""
        if self.app.robot is None:
            return
        try:
            from calibration.calib_helpers import read_all_raw
            self._current_raw = read_all_raw(self.app.robot)
        except Exception:
            self._current_raw = None

    def _on_frame(self, frame):
        """Process each gripper camera frame: detect board, update display."""
        self._current_frame = frame
        display = frame.copy()

        # Detect ChArUco board
        from calibration.calib_helpers import detect_corners, compute_board_pose
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners, detection = detect_corners(
            gray, board_detector=self._board_detector)

        self._current_detection = detection
        self._current_T_board = None
        n_corners = 0
        reproj = None

        if found and corners is not None:
            n_corners = len(corners)
            # Compute board pose
            try:
                T, _, reproj = compute_board_pose(
                    corners, _SimpleIntrinsics(self._gripper_K, self._gripper_dist),
                    detection, board_detector=self._board_detector)
                self._current_T_board = T
            except Exception:
                pass

            # Draw detected corners
            if self._board_detector is not None and detection is not None:
                self._board_detector.draw_corners(display, detection)
            else:
                corners_draw = corners.reshape(-1, 2).astype(int)
                for pt in corners_draw:
                    cv2.circle(display, tuple(pt), 3, (0, 255, 0), -1)

        # Draw capture markers
        for i, cap in enumerate(self._captures):
            cx, cy = int(cap.get('centroid_px', [0, 0])[0]), \
                     int(cap.get('centroid_px', [0, 0])[1])
            cv2.drawMarker(display, (cx, cy), (0, 200, 0),
                           cv2.MARKER_DIAMOND, 10, 1)
            cv2.putText(display, str(i + 1), (cx + 8, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 0), 1)

        # Detection status text
        det_text = (f'Board: {n_corners} corners' if found
                    else 'No board detected')
        if reproj is not None:
            det_text += f', reproj={reproj:.1f}px'
        self._detection_label.setText(det_text)

        # Convert and display
        img = cv_to_qimage(display)
        pix = QPixmap.fromImage(img).scaled(
            self._cam_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._cam_label.setPixmap(pix)

    def _capture(self):
        """Capture current pose: raw servos + board pose in camera frame."""
        if self._error_msg:
            return

        T_board = self._current_T_board
        raw = self._current_raw

        if T_board is None:
            self._status.setText('SKIP: No board detected - point gripper camera at board')
            return
        if raw is None:
            self._status.setText('SKIP: Cannot read servo positions')
            return

        # Duplicate check: board must have moved enough in camera frame
        for prev in self._captures:
            prev_t = prev['T_board_in_cam'][:3, 3]
            cur_t = T_board[:3, 3]
            dist_mm = float(np.linalg.norm(cur_t - prev_t) * 1000)
            if dist_mm < 10:
                self._status.setText(
                    f'SKIP: Too similar (board moved {dist_mm:.0f}mm, need >10mm)')
                return

        # Corner centroid for visualization
        detection = self._current_detection
        if detection is not None and hasattr(detection, 'corners'):
            corners_2d = detection.corners.reshape(-1, 2)
        else:
            corners_2d = np.zeros((0, 2))
        centroid = corners_2d.mean(axis=0).tolist() if len(corners_2d) else [0, 0]

        self._captures.append({
            'raw': dict(raw),
            'T_board_in_cam': T_board.copy(),
            'centroid_px': centroid,
        })
        n = len(self._captures)
        z_mm = T_board[2, 3] * 1000
        self._status.setText(
            f'Captured #{n}: board at z={z_mm:.0f}mm | '
            f'{"Ready to solve! Press Solve" if n >= self.MIN_CAPTURES else f"need {self.MIN_CAPTURES - n} more"}')
        self._result = None
        print(f'  Captured #{n}: raw=[{",".join(str(raw[m]) for m in range(1, 6))}]')

    def _solve(self):
        """Run brute-force sign detection across all 32 combinations."""
        n = len(self._captures)
        if n < self.MIN_CAPTURES:
            self._status.setText(
                f'Need >= {self.MIN_CAPTURES} captures (have {n})')
            return

        if self._solver is None:
            self._status.setText('FK solver not available')
            return

        self._status.setText('Solving... (testing 32 sign combinations)')
        QApplication.processEvents()

        from calibration.calib_helpers import load_offsets
        from calibration.sign_solver import (
            _brute_force_signs, save_calibration_results, MOTOR_NAMES)

        offsets = load_offsets()
        current_offsets_raw = np.array([
            offsets.get(name, {}).get('zero_raw', 2048)
            for name in MOTOR_NAMES
        ], dtype=float)

        print(f'\n  Starting auto-calibration with {n} captures...')
        result = _brute_force_signs(
            self._captures, self._solver, current_offsets_raw, verbose=True)
        self._result = result

        # Display results
        lines = [f'Signs: {result["signs_str"]}',
                 f'Mean error: {result["mean_err_mm"]:.1f}mm']

        quality = ('EXCELLENT' if result['mean_err_mm'] < 5 else
                   'GOOD' if result['mean_err_mm'] < 15 else
                   'OK' if result['mean_err_mm'] < 30 else 'POOR')
        lines.append(f'Quality: {quality}')

        ambiguous = result.get('ambiguous_joints', set())
        if ambiguous:
            names = [MOTOR_NAMES[j] for j in sorted(ambiguous)]
            lines.append(f'Ambiguous: {", ".join(names)}')

        try:
            from kinematics.arm101_ik_solver import JOINT_SIGNS
            for i, name in enumerate(MOTOR_NAMES):
                old_s = '+' if JOINT_SIGNS[i] > 0 else '-'
                new_s = '+' if result['signs'][i] > 0 else '-'
                status = ('AMBIG' if i in ambiguous
                          else 'CHANGED' if JOINT_SIGNS[i] != result['signs'][i]
                          else 'ok')
                offset = int(round(result['offsets_raw'][i]))
                lines.append(f'  {name[:14]:<14} {old_s}->{new_s} [{status}] off={offset}')
        except Exception:
            pass

        self._result_label.setText('\n'.join(lines))

        # Save if quality is acceptable
        if result['mean_err_mm'] < 30:
            save_calibration_results(
                result['signs'], result['offsets_raw'],
                result.get('T_cam_in_tcp', np.eye(4)))
            self._status.setText(
                f'Solved & SAVED! signs={result["signs_str"]} '
                f'err={result["mean_err_mm"]:.1f}mm')
        else:
            self._status.setText(
                f'Solved but error is high ({result["mean_err_mm"]:.0f}mm). '
                f'Collect more diverse poses.')

    def _undo(self):
        if self._captures:
            self._captures.pop()
            self._result = None
            self._result_label.setText('No results yet')
        n = len(self._captures)
        self._status.setText(
            f'Captures: {n} (need >= {self.MIN_CAPTURES})')

    def _reset(self):
        self._captures.clear()
        self._result = None
        self._result_label.setText('No results yet')
        self._status.setText(
            f'Captures: 0 (need >= {self.MIN_CAPTURES}, '
            f'{self.GOOD_CAPTURES}+ recommended)')


class _SimpleIntrinsics:
    """Minimal intrinsics wrapper for compute_board_pose()."""
    def __init__(self, K, dist):
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.ppx = K[0, 2]
        self.ppy = K[1, 2]
        self.coeffs = dist.tolist() if hasattr(dist, 'tolist') else list(dist)


class VerifyCalibView(BaseViewWidget):
    view_id = 'verify_calib'
    view_name = 'Verify Calibration'
    description = 'Verify hand-eye accuracy'
    show_in_sidebar = False
    parent_view_id = 'calibration'

    HOVER_HEIGHT_MM = 50.0  # hover 50mm above each corner

    def __init__(self, app, parent=None):
        super().__init__(app, parent)
        self._cam_thread = None
        self._state = 'detect'  # detect, review, moving, at_corner, done
        self._dry_run = getattr(app.args, 'dry_run', False)
        self._transform = None
        self._board_detector = None
        self._corners_base = []  # list of [x,y,z] mm in robot base frame
        self._corner_labels = []
        self._corner_idx = -1
        self._last_frame = None
        self._T_board_in_cam = None
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)

        # Camera
        self._cam_label = QLabel('Camera')
        self._cam_label.setAlignment(Qt.AlignCenter)
        self._cam_label.setMinimumSize(320, 240)
        self._cam_label.setStyleSheet('background-color: #1a1a1a;')
        layout.addWidget(self._cam_label, stretch=3)

        # Controls
        ctrl = QWidget()
        ctrl_layout = QVBoxLayout(ctrl)

        title = QLabel('Verify Calibration')
        title.setStyleSheet('font-size: 14px; font-weight: bold; color: #ffc864;')
        ctrl_layout.addWidget(title)

        self._state_label = QLabel('State: DETECT\nCapture board, then step through corners')
        self._state_label.setStyleSheet('color: #aaa; font-family: monospace;')
        self._state_label.setWordWrap(True)
        ctrl_layout.addWidget(self._state_label)

        self._detail_label = QLabel('')
        self._detail_label.setStyleSheet('color: #888; font-family: monospace; font-size: 11px;')
        self._detail_label.setWordWrap(True)
        ctrl_layout.addWidget(self._detail_label)

        ctrl_layout.addWidget(make_button('Capture Board', self._capture_board,
                                          'Detect checkerboard in current frame', '#3c5a70'))
        ctrl_layout.addWidget(make_button('Move to Next Corner', self._advance,
                                          'Move arm above next board corner', '#506430'))
        self._dryrun_btn = make_button(
            f'Dry-Run: {"ON" if self._dry_run else "OFF"}', self._toggle_dryrun,
            'Toggle dry-run mode (no arm movement)')
        ctrl_layout.addWidget(self._dryrun_btn)

        # Robot controls
        ctrl_layout.addWidget(section_label('Robot'))
        for j in range(1, 7):
            row = QHBoxLayout()
            row.addWidget(make_button(f'J{j}-', lambda ch, jj=j: self._jog(jj-1, -1), min_w=50))
            row.addWidget(make_button(f'J{j}+', lambda ch, jj=j: self._jog(jj-1, +1), min_w=50))
            ctrl_layout.addLayout(row)
        ctrl_layout.addWidget(make_button('Stop', self._stop_jog, color='#992222'))

        grip_row = QHBoxLayout()
        grip_row.addWidget(make_button('Open', self._grip_open, color='#006600', min_w=50))
        grip_row.addWidget(make_button('Close', self._grip_close, color='#660000', min_w=50))
        ctrl_layout.addLayout(grip_row)

        ctrl_layout.addWidget(make_button('< Back', lambda: self.app.switch_view('calibration'),
                                          color='#444'))
        ctrl_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(280)
        scroll.setWidget(ctrl)
        layout.addWidget(scroll, stretch=1)

    def _load_calibration(self):
        """Load the camera-to-base transform."""
        if self._transform is not None:
            return True
        try:
            from calibration.transform import CoordinateTransform
            calib_path = config_path('calibration.yaml')
            # Also check arm101-specific calibration
            arm101_path = config_path('calibration_arm101.yaml')
            path = arm101_path if os.path.exists(arm101_path) else calib_path
            if not os.path.exists(path):
                print(f'  No calibration file found')
                return False
            self._transform = CoordinateTransform()
            self._transform.load(path)
            print(f'  Loaded calibration from {os.path.basename(path)}')
            return True
        except Exception as e:
            print(f'  Error loading calibration: {e}')
            return False

    def on_activate(self):
        self.app.ensure_camera()
        self.app.ensure_robot()
        if self.app.camera:
            self._cam_thread = CameraThread(self.app.camera)
            self._cam_thread.frame_ready.connect(self._on_frame)
            self._cam_thread.start()
        # Try to load board detector
        try:
            from vision.board_detector import BoardDetector
            self._board_detector = BoardDetector.from_config(self.app.config)
        except Exception:
            pass
        self._load_calibration()
        self._state = 'detect'
        self._corners_base = []
        self._corner_idx = -1
        self._state_label.setText('State: DETECT\nCapture board to begin')

    def on_deactivate(self):
        if self._cam_thread:
            self._cam_thread.stop()
            self._cam_thread = None

    def _on_frame(self, frame):
        self._last_frame = frame.copy()
        # Live board detection overlay
        from calibration.calib_helpers import detect_corners
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners, detection = detect_corners(gray, self._board_detector)
        if found:
            if self._board_detector and detection:
                self._board_detector.draw_corners(frame, detection)
            else:
                cv2.drawChessboardCorners(frame, (7, 9), corners, found)
            cv2.putText(frame, f'{len(corners)} corners detected',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No board detected',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # Show current corner target info
        if self._state in ('at_corner', 'moving') and 0 <= self._corner_idx < len(self._corners_base):
            tgt = self._corners_base[self._corner_idx]
            label = self._corner_labels[self._corner_idx] if self._corner_idx < len(self._corner_labels) else ''
            cv2.putText(frame, f'Corner {self._corner_idx}: {label}  target=[{tgt[0]:.0f},{tgt[1]:.0f},{tgt[2]:.0f}]mm',
                        (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        img = cv_to_qimage(frame)
        pix = QPixmap.fromImage(img).scaled(
            self._cam_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._cam_label.setPixmap(pix)

    def _capture_board(self):
        """Detect board in current frame, compute corner positions in robot base frame."""
        from calibration.calib_helpers import detect_corners, compute_board_pose, \
            _get_board_outer_corners_cam
        if not self._load_calibration():
            self._state_label.setText('ERROR: No calibration loaded')
            return
        if self._last_frame is None:
            self._state_label.setText('ERROR: No camera frame')
            return
        # Get camera intrinsics
        intrinsics = None
        if self.app.camera and hasattr(self.app.camera, 'intrinsics'):
            intrinsics = self.app.camera.intrinsics
        if intrinsics is None:
            self._state_label.setText('ERROR: No camera intrinsics')
            return

        gray = cv2.cvtColor(self._last_frame, cv2.COLOR_BGR2GRAY)
        found, corners, detection = detect_corners(gray, self._board_detector)
        if not found:
            self._state_label.setText('State: DETECT\nNo board found — reposition and retry')
            return

        T_board_in_cam, obj_pts, reproj_err = compute_board_pose(
            corners, intrinsics, detection, self._board_detector)
        if T_board_in_cam is None:
            self._state_label.setText('State: DETECT\nBoard pose solve failed')
            return
        self._T_board_in_cam = T_board_in_cam

        # Get outer corners in camera frame
        corners_cam, labels = _get_board_outer_corners_cam(
            T_board_in_cam, board_detector=self._board_detector)

        # Transform to robot base frame
        self._corners_base = []
        self._corner_labels = labels
        detail_lines = []
        for i, (p_cam, label) in enumerate(zip(corners_cam, labels)):
            p_base = self._transform.camera_to_base(p_cam)
            p_base_mm = p_base * 1000.0
            hover_mm = p_base_mm.copy()
            hover_mm[2] += self.HOVER_HEIGHT_MM
            self._corners_base.append(hover_mm)
            detail_lines.append(
                f'{label}: [{hover_mm[0]:.0f},{hover_mm[1]:.0f},{hover_mm[2]:.0f}]')

        self._corner_idx = -1
        self._state = 'review'
        n = len(self._corners_base)
        self._state_label.setText(
            f'State: REVIEW\n{n} corners found (reproj: {reproj_err:.2f}px)\n'
            f'Click "Move to Next Corner" to begin')
        self._detail_label.setText('Hover targets (mm):\n' + '\n'.join(detail_lines))
        print(f'  Board captured: {n} outer corners, reproj={reproj_err:.2f}px')

    def _advance(self):
        """Move arm to next corner (or report position if dry-run)."""
        if self._state == 'detect':
            self._state_label.setText('State: DETECT\nCapture board first')
            return
        if not self._corners_base:
            self._state_label.setText('No corners — capture board first')
            return

        self._corner_idx += 1
        if self._corner_idx >= len(self._corners_base):
            self._state = 'done'
            self._state_label.setText('State: DONE\nAll corners visited!')
            self._detail_label.setText('Verification complete')
            print('  Verification complete — all corners visited')
            return

        tgt = self._corners_base[self._corner_idx]
        label = self._corner_labels[self._corner_idx] if self._corner_idx < len(self._corner_labels) else ''
        i = self._corner_idx

        if self._dry_run:
            self._state = 'at_corner'
            self._state_label.setText(
                f'State: AT_CORNER (dry-run)\n'
                f'Corner {i} ({label})\n'
                f'Target: [{tgt[0]:.1f}, {tgt[1]:.1f}, {tgt[2]:.1f}] mm')
            print(f'  [DRY RUN] Corner {i} ({label}): [{tgt[0]:.1f},{tgt[1]:.1f},{tgt[2]:.1f}]mm')
            return

        # Actually move the robot
        self._state = 'moving'
        self._state_label.setText(
            f'State: MOVING\nCorner {i} ({label})...')
        print(f'  Moving to corner {i} ({label}): [{tgt[0]:.1f},{tgt[1]:.1f},{tgt[2]:.1f}]mm')
        threading.Thread(target=self._move_to_corner, args=(i, tgt, label),
                         daemon=True).start()

    def _move_to_corner(self, idx, target_mm, label):
        """Worker thread: move robot to target position."""
        robot = self.app.robot
        if robot is None:
            QTimer.singleShot(0, lambda: self._state_label.setText('ERROR: No robot'))
            return
        try:
            robot_type = getattr(robot, 'robot_type', None)
            if robot_type == 'arm101':
                # For arm101, use IK solver to compute joint angles
                from kinematics.arm101_ik_solver import Arm101IKSolver
                solver = Arm101IKSolver()
                target_m = target_mm / 1000.0
                angles = robot.get_angles()
                seed = np.array(angles[:5]) if angles else None
                result = solver.solve_ik_position(target_m, seed_motor_deg=seed)
                if result is not None and result.success:
                    robot.move_to_angles(result.joint_angles_deg.tolist())
                    time.sleep(1.0)
                else:
                    QTimer.singleShot(0, lambda: self._state_label.setText(
                        f'IK failed for corner {idx}'))
                    return
            else:
                # Nova5: use MovJ with pose
                pose = robot.get_pose()
                rx, ry, rz = (pose[3], pose[4], pose[5]) if pose else (180, 0, 0)
                ok = robot.movj(target_mm[0], target_mm[1], target_mm[2], rx, ry, rz)
                if not ok:
                    QTimer.singleShot(0, lambda: self._state_label.setText(
                        f'Move failed for corner {idx}'))
                    return
                time.sleep(0.5)

            # Read actual position and compute error
            actual = robot.get_pose() if hasattr(robot, 'get_pose') else None
            err_str = ''
            if actual:
                err = np.linalg.norm(np.array(actual[:3]) - target_mm)
                err_str = f'\nPosition error: {err:.1f}mm'

            def _update():
                self._state = 'at_corner'
                self._state_label.setText(
                    f'State: AT_CORNER\nCorner {idx} ({label}){err_str}')
            QTimer.singleShot(0, _update)
        except Exception as e:
            msg = f'Move error: {e}'
            print(f'  {msg}')
            QTimer.singleShot(0, lambda: self._state_label.setText(msg))

    def _toggle_dryrun(self):
        self._dry_run = not self._dry_run
        self._dryrun_btn.setText(f'Dry-Run: {"ON" if self._dry_run else "OFF"}')

    def _jog(self, idx, direction):
        if self.app.robot is None:
            return
        arm101 = getattr(self.app.robot, 'robot_type', None) == 'arm101'
        if arm101:
            self.app.robot.jog_joint(idx, direction)
        else:
            axis = f'J{idx + 1}{"+" if direction > 0 else "-"}'
            self.app.robot.send(f'MoveJog({axis})')

    def _stop_jog(self):
        if self.app.robot and not (getattr(self.app.robot, 'robot_type', None) == 'arm101'):
            self.app.robot.send('MoveJog()')

    def _grip_open(self):
        if self.app.robot:
            if getattr(self.app.robot, 'robot_type', None) == 'arm101':
                self.app.robot.gripper_open()
            else:
                self.app.robot.send('ToolDOInstant(1,0)')
                self.app.robot.send('ToolDOInstant(2,1)')

    def _grip_close(self):
        if self.app.robot:
            if getattr(self.app.robot, 'robot_type', None) == 'arm101':
                self.app.robot.gripper_close()
            else:
                self.app.robot.send('ToolDOInstant(2,0)')
                self.app.robot.send('ToolDOInstant(1,1)')


# ---------------------------------------------------------------------------
# SUBPROCESS LAUNCHER VIEWS  (Dataset, Demo Cube, Pipeline, Discover, Extras)
# ---------------------------------------------------------------------------

class SubprocessLauncherView(BaseViewWidget):
    """Base for views that launch external scripts."""

    def __init__(self, app, parent=None):
        super().__init__(app, parent)
        self._proc_thread = None
        self._build_launcher_ui()

    def _build_launcher_ui(self):
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(20, 20, 20, 20)

        self._output = QTextEdit()
        self._output.setReadOnly(True)
        self._output.setStyleSheet(
            'background-color: #1a1a1a; color: #ccc; font-family: monospace; font-size: 11px;')

        self._running_label = QLabel('')
        self._running_label.setStyleSheet('color: #0cf; font-weight: bold;')

    def _launch(self, cmd, timeout=120):
        if self._proc_thread and self._proc_thread.isRunning():
            return
        self._output.clear()
        self._output.append(f'$ {" ".join(cmd)}')
        self._running_label.setText('Running...')
        self._proc_thread = SubprocessThread(cmd, timeout=timeout)
        self._proc_thread.output_line.connect(self._on_output)
        self._proc_thread.finished_signal.connect(self._on_finished)
        self._proc_thread.start()

    def _stop_running(self):
        if self._proc_thread:
            self._proc_thread.stop_process()

    def _on_output(self, line):
        self._output.append(line)

    def _on_finished(self, code):
        self._running_label.setText(f'Done (exit code {code})')

    def on_deactivate(self):
        self._stop_running()


class DatasetView(SubprocessLauncherView):
    view_id = 'dataset'
    view_name = 'Collect Dataset'
    description = 'Capture detection dataset'

    def _build_launcher_ui(self):
        super()._build_launcher_ui()
        title = QLabel('Collect Dataset')
        title.setStyleSheet('font-size: 16px; font-weight: bold; color: #ffc864;')
        self._main_layout.addWidget(title)

        self._main_layout.addWidget(make_button('Live Collection',
            lambda: self._launch([sys.executable,
                os.path.join(_PROJECT_ROOT, 'scripts/collect_dataset.py')], 600),
            'Run live dataset collection with robot + camera', '#3c5a70'))
        self._main_layout.addWidget(make_button('Camera Only (no robot)',
            lambda: self._launch([sys.executable,
                os.path.join(_PROJECT_ROOT, 'scripts/collect_dataset.py'), '--no-robot'], 600),
            'Camera feed with detection overlay', '#3c5a70'))
        self._main_layout.addWidget(make_button('Snapshot Debug',
            lambda: self._launch([sys.executable,
                os.path.join(_PROJECT_ROOT, 'scripts/collect_dataset.py'), '--snapshot'], 60),
            'Single-frame detection debug', '#3c5a70'))
        self._main_layout.addWidget(make_button('Stop', self._stop_running, color='#992222'))
        self._main_layout.addWidget(self._running_label)
        self._main_layout.addWidget(self._output)


class DemoCubeView(SubprocessLauncherView):
    view_id = 'demo_cube'
    view_name = 'Demo Cube'
    description = 'Random poses / cube trace'

    def _build_launcher_ui(self):
        super()._build_launcher_ui()
        title = QLabel('Demo Cube')
        title.setStyleSheet('font-size: 16px; font-weight: bold; color: #ffc864;')
        self._main_layout.addWidget(title)

        self._main_layout.addWidget(make_button('Random Poses',
            lambda: self._launch([sys.executable,
                os.path.join(_PROJECT_ROOT, 'scripts/demo_cube.py'), '--mode', 'random'], 120),
            'Run random reachable poses demo', '#3c5a70'))
        self._main_layout.addWidget(make_button('Cube Corners',
            lambda: self._launch([sys.executable,
                os.path.join(_PROJECT_ROOT, 'scripts/demo_cube.py'), '--mode', 'cube'], 120),
            'Trace cube corners', '#3c5a70'))
        self._main_layout.addWidget(make_button('Stop', self._stop_running, color='#992222'))
        self._main_layout.addWidget(self._running_label)
        self._main_layout.addWidget(self._output)


class PipelineView(SubprocessLauncherView):
    view_id = 'pipeline'
    view_name = 'Pipeline'
    description = 'Full pick-and-stand'

    def _build_launcher_ui(self):
        super()._build_launcher_ui()
        title = QLabel('Pick & Stand Pipeline')
        title.setStyleSheet('font-size: 16px; font-weight: bold; color: #ffc864;')
        self._main_layout.addWidget(title)

        self._main_layout.addWidget(make_button('Run Pipeline',
            lambda: self._launch([sys.executable,
                os.path.join(_PROJECT_ROOT, 'src/main.py')], 300),
            'Run the full pick-and-stand state machine', '#506430'))
        self._main_layout.addWidget(make_button('Stop', self._stop_running, color='#992222'))
        self._main_layout.addWidget(self._running_label)
        self._main_layout.addWidget(self._output)


class DiscoverView(SubprocessLauncherView):
    view_id = 'discover'
    view_name = 'Discover Cameras'
    description = 'Find & configure cameras'

    def _build_launcher_ui(self):
        super()._build_launcher_ui()
        title = QLabel('Discover Cameras')
        title.setStyleSheet('font-size: 16px; font-weight: bold; color: #ffc864;')
        self._main_layout.addWidget(title)

        script = os.path.join(_PROJECT_ROOT, 'scripts/discover_cameras.py')
        self._main_layout.addWidget(make_button('Detect & Write Config',
            lambda: self._launch([sys.executable, script], 30),
            'Detect cameras and write to config', '#3c5a70'))
        self._main_layout.addWidget(make_button('Preview / Dry-run',
            lambda: self._launch([sys.executable, script, '--dry-run'], 30),
            'Preview without writing', '#705a3c'))
        self._main_layout.addWidget(make_button('Merge with Existing',
            lambda: self._launch([sys.executable, script, '--merge'], 30),
            'Merge new cameras into existing config', '#3c705a'))
        self._main_layout.addWidget(make_button('Stop', self._stop_running, color='#992222'))
        self._main_layout.addWidget(self._running_label)
        self._main_layout.addWidget(self._output)


class ExtrasView(SubprocessLauncherView):
    view_id = 'extras'
    view_name = 'Extra Scripts'
    description = 'Utility scripts'

    SCRIPTS = [
        ('Visual Servo Test', 'scripts/test_visual_servo.py'),
        ('Green Cube Point', 'scripts/green_cube_point.py'),
        ('Visit Cubes', 'scripts/visit_cubes.py'),
        ('Visit Cubes (calibrated)', 'scripts/visit_cubes_calibrated.py'),
        ('Select ROI', 'scripts/select_roi.py'),
        ('Evaluate Dataset', 'scripts/eval_dataset.py'),
        ('Test arm101 FK', 'scripts/test_arm101_fk.py'),
    ]

    def _build_launcher_ui(self):
        super()._build_launcher_ui()
        title = QLabel('Extra Scripts')
        title.setStyleSheet('font-size: 16px; font-weight: bold; color: #ffc864;')
        self._main_layout.addWidget(title)

        for name, script in self.SCRIPTS:
            script_path = os.path.join(_PROJECT_ROOT, script)
            exists = os.path.exists(script_path)
            btn = make_button(
                name if exists else f'{name} (missing)',
                (lambda sp=script_path: self._launch([sys.executable, sp], 300)) if exists else None,
                f'Run {script}',
                '#3c5a70' if exists else '#444')
            if not exists:
                btn.setEnabled(False)
            self._main_layout.addWidget(btn)

        self._main_layout.addWidget(make_button('Stop', self._stop_running, color='#992222'))
        self._main_layout.addWidget(self._running_label)
        self._main_layout.addWidget(self._output)


# ---------------------------------------------------------------------------
# DIGITAL TWIN VIEW
# ---------------------------------------------------------------------------

class DigitalTwinView(SubprocessLauncherView):
    view_id = 'digital_twin'
    view_name = 'Digital Twin'
    description = 'Isaac Sim launcher'

    MODES = [
        ('GUI Mode', []),
        ('Headless', ['--headless']),
        ('With Cameras', ['--enable_cameras']),
        ('Cameras + Save', ['--enable_cameras', '--save_images']),
        ('Mirror Arm', ['--mirror']),
        ('Mirror + Cameras', ['--mirror', '--enable_cameras']),
    ]

    def _build_launcher_ui(self):
        super()._build_launcher_ui()
        title = QLabel('Digital Twin (Isaac Sim)')
        title.setStyleSheet('font-size: 16px; font-weight: bold; color: #ffc864;')
        self._main_layout.addWidget(title)

        script = os.path.join(_PROJECT_ROOT, 'scripts/run_digital_twin.sh')
        for name, args in self.MODES:
            cmd = ['/bin/bash', script] + args
            self._main_layout.addWidget(make_button(name,
                lambda checked, c=cmd: self._launch(c, 600),
                f'Launch Isaac Sim: {name}', '#3c5a70'))

        self._main_layout.addWidget(make_button('Stop / Detach', self._stop_running,
                                                 color='#992222'))
        self._main_layout.addWidget(self._running_label)
        self._main_layout.addWidget(self._output)


# ---------------------------------------------------------------------------
# LIVE TWIN VIEW
# ---------------------------------------------------------------------------

class LiveTwinView(BaseViewWidget):
    view_id = 'live_twin'
    view_name = 'Live Twin'
    description = 'Real-time 3D FK skeleton'

    def __init__(self, app, parent=None):
        super().__init__(app, parent)
        self._poll_thread = None
        self._renderer = None
        self._multi_view = True
        self._show_table = True
        self._trail = []
        self._actual_angles = None
        self._commanded_angles = None
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)

        # 3D view
        self._render_label = QLabel('Connecting to robot...')
        self._render_label.setAlignment(Qt.AlignCenter)
        self._render_label.setMinimumSize(400, 300)
        self._render_label.setStyleSheet('background-color: #1a1a1a;')
        layout.addWidget(self._render_label, stretch=3)

        # Controls
        ctrl = QWidget()
        ctrl_layout = QVBoxLayout(ctrl)

        ctrl_layout.addWidget(section_label('View Controls'))
        self._mv_btn = make_button('Toggle Multi-View', self._toggle_multiview)
        ctrl_layout.addWidget(self._mv_btn)
        ctrl_layout.addWidget(make_button('Toggle Angle Table', self._toggle_table))
        ctrl_layout.addWidget(make_button('Clear Trail', self._clear_trail))
        ctrl_layout.addWidget(make_button('Reset View', self._reset_view))

        ctrl_layout.addWidget(section_label('View Presets'))
        for i, name in enumerate(['Front', 'Side', 'Top', 'Iso']):
            ctrl_layout.addWidget(make_button(name, lambda ch, n=name: self._set_preset(n)))

        self._angle_label = QLabel('')
        self._angle_label.setStyleSheet('color: #aaa; font-family: monospace; font-size: 10px;')
        self._angle_label.setWordWrap(True)
        ctrl_layout.addWidget(self._angle_label)

        ctrl_layout.addStretch()
        ctrl.setMaximumWidth(200)
        layout.addWidget(ctrl, stretch=0)

    def on_activate(self):
        self.app.ensure_robot()
        try:
            from gui.arm_renderer import ArmRenderer
            self._renderer = ArmRenderer()
        except Exception as e:
            print(f'  ArmRenderer init error: {e}')

        if self.app.robot:
            self._poll_thread = RobotPollThread(self.app.robot, 0.05)
            self._poll_thread.state_updated.connect(self._on_state)
            self._poll_thread.start()

        # Render timer
        self._render_timer = QTimer()
        self._render_timer.timeout.connect(self._render)
        self._render_timer.start(50)  # 20 fps

    def on_deactivate(self):
        if self._poll_thread:
            self._poll_thread.stop()
            self._poll_thread = None
        if hasattr(self, '_render_timer'):
            self._render_timer.stop()

    def _on_state(self, pose, angles, mode):
        self._actual_angles = angles
        if angles:
            lines = [f'J{i+1}: {a:.1f}' for i, a in enumerate(angles)]
            self._angle_label.setText('\n'.join(lines))

    def _render(self):
        if self._renderer is None or self._actual_angles is None:
            return
        try:
            w = max(400, self._render_label.width())
            h = max(300, self._render_label.height())
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            self._renderer.render_comparison(
                canvas, self._actual_angles, self._commanded_angles)
            if self._show_table and self._actual_angles:
                self._renderer.render_angle_table(
                    canvas, self._actual_angles, self._commanded_angles)
            img = cv_to_qimage(canvas)
            self._render_label.setPixmap(QPixmap.fromImage(img))
        except Exception:
            pass

    def _toggle_multiview(self):
        self._multi_view = not self._multi_view

    def _toggle_table(self):
        self._show_table = not self._show_table

    def _clear_trail(self):
        self._trail.clear()

    def _reset_view(self):
        self._trail.clear()
        self._commanded_angles = None

    def _set_preset(self, name):
        if self._renderer:
            presets = {'Front': (0, 0), 'Side': (90, 0), 'Top': (0, 90), 'Iso': (45, 30)}
            az, el = presets.get(name, (45, 30))
            self._renderer.azimuth = az
            self._renderer.elevation = el


# ---------------------------------------------------------------------------
# CAMERA OVERLAY VIEW
# ---------------------------------------------------------------------------

class CameraOverlayView(BaseViewWidget):
    view_id = 'camera_overlay'
    view_name = 'Camera Overlay'
    description = 'AR skeleton on camera feed'

    def __init__(self, app, parent=None):
        super().__init__(app, parent)
        self._cam_thread = None
        self._poll_thread = None
        self._overlay_on = True
        self._alpha = 1.0
        self._show_table = False
        self._show_axes = True
        self._current_frame = None
        self._current_angles = None
        self._renderer = None
        self._T_base_to_cam = None
        self._K = None
        self._dist = None
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)

        # Camera + overlay
        self._cam_label = QLabel('Camera')
        self._cam_label.setAlignment(Qt.AlignCenter)
        self._cam_label.setMinimumSize(400, 300)
        self._cam_label.setStyleSheet('background-color: #1a1a1a;')
        layout.addWidget(self._cam_label, stretch=3)

        # Controls
        ctrl = QWidget()
        ctrl_layout = QVBoxLayout(ctrl)

        ctrl_layout.addWidget(section_label('Overlay Controls'))
        self._overlay_btn = make_button('Overlay: ON', self._toggle_overlay)
        ctrl_layout.addWidget(self._overlay_btn)
        self._alpha_btn = make_button(f'Alpha: {self._alpha:.1f}', self._cycle_alpha)
        ctrl_layout.addWidget(self._alpha_btn)
        ctrl_layout.addWidget(make_button('Toggle Angle Table', self._toggle_table))
        ctrl_layout.addWidget(make_button('Toggle XYZ Axes', self._toggle_axes))
        ctrl_layout.addWidget(make_button('Reset', self._reset))

        # Status
        self._calib_status = QLabel('Loading calibration...')
        self._calib_status.setStyleSheet('color: #aaa; font-size: 11px;')
        ctrl_layout.addWidget(self._calib_status)

        self._angle_label = QLabel('')
        self._angle_label.setStyleSheet('color: #aaa; font-family: monospace; font-size: 10px;')
        self._angle_label.setWordWrap(True)
        ctrl_layout.addWidget(self._angle_label)

        ctrl_layout.addStretch()
        ctrl.setMaximumWidth(200)
        layout.addWidget(ctrl, stretch=0)

    def on_activate(self):
        self.app.ensure_camera()
        self.app.ensure_robot()

        # Load calibration
        import yaml
        cal_path = config_path('calibration.yaml')
        intr_path = config_path('camera_intrinsics.yaml')
        status_lines = []

        if os.path.exists(cal_path):
            with open(cal_path, 'r') as f:
                data = yaml.safe_load(f)
            T = np.array(data.get('T_camera_to_base', np.eye(4).tolist()))
            self._T_base_to_cam = np.linalg.inv(T)
            status_lines.append('Calib: OK')
        else:
            status_lines.append('Calib: MISSING')

        if os.path.exists(intr_path):
            try:
                with open(intr_path, 'r') as f:
                    intr = yaml.safe_load(f) or {}
                if 'fx' in intr:
                    self._K = np.array([
                        [intr['fx'], 0, intr['ppx']],
                        [0, intr['fy'], intr['ppy']],
                        [0, 0, 1]])
                    self._dist = np.array(intr.get('dist', [0, 0, 0, 0, 0]))
                    status_lines.append('Intrinsics: OK')
                else:
                    status_lines.append('Intrinsics: INVALID')
            except Exception as e:
                status_lines.append(f'Intrinsics: ERROR ({e})')
        else:
            status_lines.append('Intrinsics: MISSING')

        self._calib_status.setText('\n'.join(status_lines))

        try:
            from gui.arm_renderer import ArmRenderer
            self._renderer = ArmRenderer()
        except Exception:
            pass

        if self.app.camera:
            self._cam_thread = CameraThread(self.app.camera)
            self._cam_thread.frame_ready.connect(self._on_frame)
            self._cam_thread.start()

        if self.app.robot:
            self._poll_thread = RobotPollThread(self.app.robot, 0.05)
            self._poll_thread.state_updated.connect(self._on_state)
            self._poll_thread.start()

    def on_deactivate(self):
        if self._cam_thread:
            self._cam_thread.stop()
            self._cam_thread = None
        if self._poll_thread:
            self._poll_thread.stop()
            self._poll_thread = None

    def _on_frame(self, frame):
        self._current_frame = frame
        self._update_display()

    def _on_state(self, pose, angles, mode):
        self._current_angles = angles
        if angles:
            self._angle_label.setText(
                '\n'.join(f'J{i+1}: {a:.1f}' for i, a in enumerate(angles)))

    def _update_display(self):
        frame = self._current_frame
        if frame is None:
            return

        display = frame.copy()

        if (self._overlay_on and self._renderer and self._current_angles
                and self._K is not None and self._T_base_to_cam is not None):
            try:
                roi = display
                self._renderer.draw_on_camera_frame(
                    roi, self._current_angles, self._K, self._dist, self._T_base_to_cam)
            except Exception:
                pass

        img = cv_to_qimage(display)
        pix = QPixmap.fromImage(img).scaled(
            self._cam_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._cam_label.setPixmap(pix)

    def _toggle_overlay(self):
        self._overlay_on = not self._overlay_on
        self._overlay_btn.setText(f'Overlay: {"ON" if self._overlay_on else "OFF"}')

    def _cycle_alpha(self):
        alphas = [1.0, 0.7, 0.4]
        idx = alphas.index(self._alpha) if self._alpha in alphas else 0
        self._alpha = alphas[(idx + 1) % len(alphas)]
        self._alpha_btn.setText(f'Alpha: {self._alpha:.1f}')

    def _toggle_table(self):
        self._show_table = not self._show_table

    def _toggle_axes(self):
        self._show_axes = not self._show_axes

    def _reset(self):
        self._current_angles = None


# ---------------------------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------------------------

class UnifiedPyQtApp(QMainWindow):
    """Main PyQt5 application for ArmRobotics."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = load_config()
        self.robot = None
        self.camera = None
        self._robot_connected = False
        self._camera_started = False
        self._rig_lock = None  # RigLock instance, acquired on first HW access

        self._views = {}  # view_id -> widget instance
        self._active_view_id = ''

        self.setWindowTitle('ArmRobotics - Unified GUI')
        self.setMinimumSize(900, 600)
        self.resize(1200, 750)

        self._apply_dark_theme()
        self._build_ui()

    def _apply_dark_theme(self):
        self.setStyleSheet('''
            QMainWindow { background-color: #1e1e22; }
            QWidget { background-color: #1e1e22; color: #ddd; }
            QScrollArea { border: none; }
            QGroupBox { border: 1px solid #444; border-radius: 4px;
                        margin-top: 8px; padding-top: 12px; }
            QStatusBar { background-color: #181820; color: #888; }
            QListWidget { background-color: #191920; color: #ccc; border: none;
                          font-size: 13px; }
            QListWidget::item { padding: 10px 12px; border-bottom: 1px solid #2a2a30; }
            QListWidget::item:selected { background-color: #46321e; color: #ffc864; }
            QListWidget::item:hover { background-color: #32323c; }
            QTextEdit { border: 1px solid #333; }
            QTableWidget { border: 1px solid #333; }
            QScrollBar:vertical { background: #252528; width: 8px; }
            QScrollBar::handle:vertical { background: #555; border-radius: 4px; min-height: 20px; }
            QScrollBar:horizontal { background: #252528; height: 8px; }
            QScrollBar::handle:horizontal { background: #555; border-radius: 4px; min-width: 20px; }
        ''')

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        sidebar = QWidget()
        sidebar.setMaximumWidth(220)
        sidebar.setMinimumWidth(180)
        sidebar.setStyleSheet('background-color: #191920;')
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(0, 0, 0, 0)
        sb_layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setStyleSheet('background-color: #282335; padding: 10px;')
        h_layout = QVBoxLayout(header)
        h_layout.setContentsMargins(12, 8, 12, 8)
        title = QLabel('ArmRobotics')
        title.setStyleSheet('color: #00a0ff; font-size: 16px; font-weight: bold;')
        h_layout.addWidget(title)
        subtitle = QLabel('Unified GUI (PyQt)')
        subtitle.setStyleSheet('color: #787878; font-size: 11px;')
        h_layout.addWidget(subtitle)
        sb_layout.addWidget(header)

        # Navigation list
        self._nav_list = QListWidget()
        self._nav_list.currentRowChanged.connect(self._on_nav_changed)
        sb_layout.addWidget(self._nav_list)

        # Global servo toggle at bottom of sidebar (visible from all views)
        self._servo_btn = QPushButton('Servos ON')
        self._servo_btn.setStyleSheet(
            'background-color: #006400; color: white; font-weight: bold; '
            'padding: 8px; margin: 6px; border-radius: 4px;')
        self._servo_btn.clicked.connect(self._toggle_servos)
        sb_layout.addWidget(self._servo_btn)

        # Status bar at bottom of sidebar
        self._sb_status = QLabel('Robot: ? | Camera: off')
        self._sb_status.setStyleSheet(
            'color: #666; font-size: 10px; padding: 6px 10px; border-top: 1px solid #333;')
        sb_layout.addWidget(self._sb_status)

        main_layout.addWidget(sidebar)

        # View stack
        self._stack = QStackedWidget()
        main_layout.addWidget(self._stack, stretch=1)

        # Status bar
        self.statusBar().showMessage('Ready')

        # Register all views
        self._register_views()

        # Status refresh timer
        self._status_timer = QTimer()
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start(2000)

    def _register_views(self):
        """Create all view widgets and populate sidebar."""
        view_classes = [
            HomeView,
            ControlPanelView,
            CalibrationView,
            DatasetView,
            DemoCubeView,
            PipelineView,
            DiscoverView,
            ExtrasView,
            DigitalTwinView,
            LiveTwinView,
            CameraOverlayView,
            # Sub-views (not in sidebar)
            CheckerboardCalibView,
            ServoCalibView,
            HandEyeYellowView,
            ServoDirectionCalibView,
            VerifyCalibView,
        ]

        for cls in view_classes:
            widget = cls(self)
            self._views[cls.view_id] = widget
            self._stack.addWidget(widget)

            if cls.show_in_sidebar:
                item = QListWidgetItem(f'{cls.view_name}\n{cls.description}')
                item.setData(Qt.UserRole, cls.view_id)
                self._nav_list.addItem(item)

    def _on_nav_changed(self, row):
        if row < 0:
            return
        item = self._nav_list.item(row)
        view_id = item.data(Qt.UserRole)
        self.switch_view(view_id)

    def switch_view(self, view_id: str):
        """Switch to a view by ID."""
        if view_id == self._active_view_id:
            return
        widget = self._views.get(view_id)
        if widget is None:
            print(f'  Unknown view: {view_id}')
            return

        # Deactivate old view
        if self._active_view_id:
            old = self._views.get(self._active_view_id)
            if old:
                try:
                    old.on_deactivate()
                except Exception as e:
                    print(f'  Warning: deactivate error: {e}')

        # Activate new view
        self._active_view_id = view_id
        self._stack.setCurrentWidget(widget)

        try:
            widget.on_activate()
        except Exception as e:
            print(f'  Error activating {view_id}: {e}')
            import traceback
            traceback.print_exc()

        # Update nav selection (for sub-views, don't change sidebar selection)
        if widget.show_in_sidebar:
            for i in range(self._nav_list.count()):
                item = self._nav_list.item(i)
                if item.data(Qt.UserRole) == view_id:
                    self._nav_list.blockSignals(True)
                    self._nav_list.setCurrentRow(i)
                    self._nav_list.blockSignals(False)
                    break

        self.statusBar().showMessage(f'View: {widget.view_name}')
        print(f'  Switched to: {widget.view_name}')

    def _ensure_rig_lock(self) -> bool:
        """Acquire the rig lock if not already held. Returns True on success."""
        if self._rig_lock is not None:
            return True
        try:
            from rig_lock import RigLock, RigLockError
            lock = RigLock(holder='gui')
            lock.acquire()
            self._rig_lock = lock
            return True
        except RigLockError as e:
            print(f'  WARNING: Cannot acquire rig lock: {e}')
            return False
        except Exception as e:
            print(f'  WARNING: Rig lock error: {e}')
            return False

    def ensure_camera(self) -> bool:
        if self._camera_started:
            return self.camera is not None
        if getattr(self.args, 'no_camera', False):
            self._camera_started = True
            return False
        if not self._ensure_rig_lock():
            print('  WARNING: Camera skipped — rig is locked by another process.')
            self._camera_started = True
            return False
        try:
            from vision import create_camera
            cam_cfg = self.config.get('camera', {})
            cam_config = dict(self.config)
            sd = getattr(self.args, 'sd', False)
            w, h = (640, 480) if sd else (cam_cfg.get('width', 640), cam_cfg.get('height', 480))
            cam_config['camera'] = dict(cam_cfg, width=w, height=h)
            print(f'Starting camera ({w}x{h})...')
            self.camera = create_camera(cam_config)
            self.camera.start()
            print(f'  Camera started ({self.camera.width}x{self.camera.height}).')
            self._camera_started = True
            return True
        except Exception as e:
            print(f'  WARNING: Camera not available: {e}')
            self._camera_started = True
            return False

    def ensure_robot(self) -> bool:
        if self._robot_connected:
            return self.robot is not None
        if getattr(self.args, 'no_robot', False):
            self._robot_connected = True
            return False
        if not self._ensure_rig_lock():
            print('  WARNING: Robot skipped — rig is locked by another process.')
            self._robot_connected = True
            return False
        try:
            from config_loader import connect_robot
            safe = getattr(self.args, 'safe', False)
            self.robot = connect_robot(self.config, safe_mode=safe)
            print('  Robot connected.')
            self._robot_connected = True
            return True
        except Exception as e:
            print(f'  WARNING: Robot not available: {e}')
            self._robot_connected = True
            return False

    def _toggle_servos(self):
        if self.robot is None:
            return
        if getattr(self.robot, 'robot_type', None) == 'arm101':
            if getattr(self.robot, '_enabled', False):
                self.robot.disable_torque()
            else:
                self.robot.enable_torque()
        else:
            # Nova5 toggle
            self.robot.send('DisableRobot()')
            import time; time.sleep(1)
            self.robot.send('ClearError()')
            self.robot.send('EnableRobot()')
        self._update_servo_btn()

    def _update_servo_btn(self):
        if self.robot is None:
            self._servo_btn.setText('No Robot')
            self._servo_btn.setStyleSheet(
                'background-color: #333; color: #888; font-weight: bold; '
                'padding: 8px; margin: 6px; border-radius: 4px;')
            return
        if getattr(self.robot, 'robot_type', None) == 'arm101':
            enabled = getattr(self.robot, '_enabled', False)
        else:
            mode = self.robot.get_mode()
            enabled = mode == 5
        label = 'Servos OFF' if enabled else 'Servos ON'
        color = '#640000' if enabled else '#006400'
        self._servo_btn.setText(label)
        self._servo_btn.setStyleSheet(
            f'background-color: {color}; color: white; font-weight: bold; '
            f'padding: 8px; margin: 6px; border-radius: 4px;')

    def _update_status(self):
        robot_type = self.config.get('robot_type', '?')
        r = 'connected' if self.robot else 'off'
        c = 'on' if self.camera else 'off'
        self._sb_status.setText(f'Robot: {robot_type} ({r})  |  Camera: {c}')
        self._update_servo_btn()

    def closeEvent(self, event):
        """Clean up on close."""
        # Deactivate current view
        if self._active_view_id:
            widget = self._views.get(self._active_view_id)
            if widget:
                try:
                    widget.on_deactivate()
                except Exception:
                    pass

        # Clean up resources
        if self.camera:
            try:
                self.camera.stop()
            except Exception:
                pass
        if self.robot:
            try:
                if hasattr(self.robot, 'close'):
                    self.robot.close()
            except Exception:
                pass

        # Release rig lock
        if self._rig_lock:
            try:
                self._rig_lock.release()
            except Exception:
                pass

        print('  Shutdown complete.')
        event.accept()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='ArmRobotics Unified GUI (PyQt5)')
    parser.add_argument('--view', type=str, default=None,
                        help='Start with a specific view')
    parser.add_argument('--list', action='store_true',
                        help='List available views and exit')
    parser.add_argument('--no-camera', action='store_true')
    parser.add_argument('--no-robot', action='store_true')
    parser.add_argument('--safe', action='store_true',
                        help='Safe mode (arm101)')
    parser.add_argument('--sd', action='store_true',
                        help='640x480 camera')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry-run mode')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        views = [
            ('home', 'Home', 'Config editor & utilities'),
            ('control', 'Control Panel', 'Camera + robot jog controls'),
            ('calibration', 'Calibration', 'Servo & hand-eye calibration'),
            ('dataset', 'Collect Dataset', 'Capture detection dataset'),
            ('demo_cube', 'Demo Cube', 'Random poses / cube trace'),
            ('pipeline', 'Pipeline', 'Full pick-and-stand'),
            ('discover', 'Discover Cameras', 'Find & configure cameras'),
            ('extras', 'Extra Scripts', 'Utility scripts'),
            ('digital_twin', 'Digital Twin', 'Isaac Sim launcher'),
            ('live_twin', 'Live Twin', 'Real-time 3D FK skeleton'),
            ('camera_overlay', 'Camera Overlay', 'AR skeleton on camera feed'),
        ]
        print('Available views:')
        print(f'  {"ID":<20s} {"Name":<20s}  Description')
        print(f'  {"-"*20} {"-"*20}  {"-"*30}')
        for vid, name, desc in views:
            print(f'  {vid:<20s} {name:<20s}  {desc}')
        return 0

    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName('ArmRobotics')

    window = UnifiedPyQtApp(args)
    window.show()

    # Switch to requested view
    start = args.view or 'home'
    window.switch_view(start)

    return qt_app.exec_()


if __name__ == '__main__':
    sys.exit(main())
