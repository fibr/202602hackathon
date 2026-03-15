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
        self._enable_btn = make_button('Enable', self._do_enable, color='#006400')
        eh_row.addWidget(self._enable_btn)
        eh_row.addWidget(make_button('Home', self._do_home, color='#644800'))
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
        # Update enable button label
        if self._is_arm101():
            enabled = getattr(self.app.robot, '_enabled', False)
            self._enable_btn.setText('Relax' if enabled else 'Torque')
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
                self.app.robot.move_joints([0] * 6, speed=100)
            else:
                self.app.robot.send('SpeedFactor(10)')
                self.app.robot.send('MovJ(joint={43.5,-13.9,-85.4,196.3,-90.0,43.5})')
                time.sleep(3)
                self.app.robot.send(f'SpeedFactor({self._speed})')
        threading.Thread(target=_home, daemon=True).start()

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
        self._he_points = []  # hand-eye correspondences
        self._intr_frames = []
        self._ground_samples = []
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)

        # Camera feed
        self._cam_label = QLabel('Camera')
        self._cam_label.setAlignment(Qt.AlignCenter)
        self._cam_label.setMinimumSize(320, 240)
        self._cam_label.setStyleSheet('background-color: #1a1a1a;')
        layout.addWidget(self._cam_label, stretch=3)

        # Controls
        ctrl = QWidget()
        ctrl_layout = QVBoxLayout(ctrl)
        ctrl_layout.setSpacing(4)

        ctrl_layout.addWidget(section_label('Intrinsics'))
        ctrl_layout.addWidget(make_button('Capture Intrinsics Frame', self._capture_intr,
                                          'Capture checkerboard frame for intrinsics', '#3c5a70'))
        ctrl_layout.addWidget(make_button('Calibrate Intrinsics', self._calibrate_intr,
                                          'Run camera calibration', '#3c5a70'))
        ctrl_layout.addWidget(make_button('Visualize Intrinsics', self._visualize_intr,
                                          '', '#3c5a70'))

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

        ctrl_layout.addWidget(make_button('Enable/Torque', self._enable_robot, color='#006400'))

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

    def _on_frame(self, frame):
        img = cv_to_qimage(frame)
        pix = QPixmap.fromImage(img).scaled(
            self._cam_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._cam_label.setPixmap(pix)

    def _capture_intr(self):
        if self.app.camera is None:
            return
        color, _, _ = self.app.camera.get_frames()
        if color is not None:
            self._intr_frames.append(color.copy())
            print(f'  Captured intrinsics frame #{len(self._intr_frames)}')

    def _calibrate_intr(self):
        print('  Running intrinsics calibration...')
        # Delegate to the existing calibration logic
        threading.Thread(target=self._run_intr_calib, daemon=True).start()

    def _run_intr_calib(self):
        try:
            from calibration.board_detector import BoardDetector
            detector = BoardDetector()
            all_obj_pts, all_img_pts = [], []
            for frame in self._intr_frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found, corners = cv2.findChessboardCorners(gray, (9, 6), None)
                if found:
                    objp = np.zeros((54, 3), np.float32)
                    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 25.0
                    all_obj_pts.append(objp)
                    all_img_pts.append(corners)
            if len(all_obj_pts) < 3:
                print(f'  Need >=3 valid frames, got {len(all_obj_pts)}')
                return
            h, w = self._intr_frames[0].shape[:2]
            ret, K, dist, _, _ = cv2.calibrateCamera(all_obj_pts, all_img_pts, (w, h), None, None)
            print(f'  Intrinsics RMS error: {ret:.3f}')
            import yaml
            intr_path = config_path('camera_intrinsics.yaml')
            data = {
                'fx': float(K[0, 0]), 'fy': float(K[1, 1]),
                'ppx': float(K[0, 2]), 'ppy': float(K[1, 2]),
                'dist': [float(d) for d in dist[0]],
                'width': w, 'height': h,
            }
            with open(intr_path, 'w') as f:
                yaml.dump(data, f)
            print(f'  Saved intrinsics to {intr_path}')
        except Exception as e:
            print(f'  Intrinsics calibration error: {e}')

    def _visualize_intr(self):
        print('  Visualize intrinsics (not yet implemented in PyQt)')

    def _capture_ground(self):
        if self.app.camera is None:
            return
        color, _, _ = self.app.camera.get_frames()
        if color is not None:
            self._ground_samples.append(color.copy())
            print(f'  Captured ground sample #{len(self._ground_samples)}')

    def _save_ground(self):
        print('  Save ground plane (not yet reimplemented)')

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

    def __init__(self, app, parent=None):
        super().__init__(app, parent)
        self._cam_thread = None
        self._captures = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel('Hand-Eye Calibration (Yellow Tape)')
        title.setStyleSheet('font-size: 16px; font-weight: bold; color: #ffc864;')
        layout.addWidget(title)
        layout.addWidget(QLabel(
            'Move arm to diverse poses with yellow tape visible.\n'
            'Capture FK+pixel correspondences, then solve.'))

        self._status = QLabel('Captures: 0 (need >= 6)')
        self._status.setStyleSheet('color: #aaa; font-family: monospace;')
        layout.addWidget(self._status)

        self._cam_label = QLabel('Camera')
        self._cam_label.setAlignment(Qt.AlignCenter)
        self._cam_label.setMinimumSize(320, 240)
        self._cam_label.setStyleSheet('background-color: #1a1a1a;')
        layout.addWidget(self._cam_label)

        btn_row = QHBoxLayout()
        btn_row.addWidget(make_button('Capture Pose', self._capture,
                                      'Capture FK + yellow tape position', '#506430'))
        btn_row.addWidget(make_button('Solve', self._solve,
                                      'Run joint solve (>= 6 captures)', '#3c705a'))
        btn_row.addWidget(make_button('Undo', self._undo, 'Remove last capture', '#644832'))
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
            except Exception:
                pass
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

    def _capture(self):
        self._captures.append(time.time())  # placeholder
        self._status.setText(f'Captures: {len(self._captures)} (need >= 6)')
        print(f'  Captured pose #{len(self._captures)}')

    def _solve(self):
        if len(self._captures) < 6:
            self._status.setText(f'Need >= 6 captures, have {len(self._captures)}')
            return
        print('  Running joint solve...')
        self._status.setText('Solving...')

    def _undo(self):
        if self._captures:
            self._captures.pop()
        self._status.setText(f'Captures: {len(self._captures)} (need >= 6)')


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

    def __init__(self, app, parent=None):
        super().__init__(app, parent)
        self._cam_thread = None
        self._state = 'detect'  # detect, review, moving, at_corner, done
        self._dry_run = getattr(app.args, 'dry_run', False)
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

        self._state_label = QLabel('State: DETECT')
        self._state_label.setStyleSheet('color: #aaa; font-family: monospace;')
        ctrl_layout.addWidget(self._state_label)

        ctrl_layout.addWidget(make_button('Capture Board', self._capture_board,
                                          'Detect checkerboard in current frame', '#3c5a70'))
        ctrl_layout.addWidget(make_button('Start / Next', self._advance,
                                          'Advance to next step', '#506430'))
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

    def _on_frame(self, frame):
        img = cv_to_qimage(frame)
        pix = QPixmap.fromImage(img).scaled(
            self._cam_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._cam_label.setPixmap(pix)

    def _capture_board(self):
        print('  Capturing board...')
        self._state = 'review'
        self._state_label.setText('State: REVIEW')

    def _advance(self):
        transitions = {'review': 'moving', 'at_corner': 'moving', 'moving': 'at_corner'}
        if self._state in transitions:
            self._state = transitions[self._state]
            self._state_label.setText(f'State: {self._state.upper()}')

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

    def ensure_camera(self) -> bool:
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

    def _update_status(self):
        robot_type = self.config.get('robot_type', '?')
        r = 'connected' if self.robot else 'off'
        c = 'on' if self.camera else 'off'
        self._sb_status.setText(f'Robot: {robot_type} ({r})  |  Camera: {c}')

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
