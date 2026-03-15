"""Shared OpenCV GUI panel for robot arm control.

Draws interactive controls (XY pad, Z buttons, gripper, speed, enable/home,
live status) onto an OpenCV canvas. Used by detect_checkerboard.py,
control_panel.py, and collect_dataset.py.

The panel is drawn on the right side of the frame. Mouse events in the panel
area are routed to handle_mouse(); events outside are left to the app.

Supports two robot types via duck-typing:
  - Dobot Nova5: send(cmd), get_pose(), get_angles()
    XY pad = Cartesian MovL, Z = Cartesian Z, jog = continuous MoveJog
  - LeRobot arm101: robot_type='arm101', get_angles(), move_joints(),
    jog_joint(), gripper_open/close(), enable/disable_torque()
    XY pad = Cartesian IK step, Z = Cartesian IK step, jog = step-based
    J4/J5/J6 wrist buttons = direct joint steps (5 deg each)

Robot connection is duck-typed. Arm101 detected via robot.robot_type == 'arm101'.
"""

import time
import cv2
import numpy as np
from robot.motion_utils import move_to_pose


# Layout constants
PANEL_WIDTH = 250
PANEL_BG = (30, 30, 30)
PAD_SIZE = 150          # XY pad size in pixels
PAD_DEADZONE = 15       # center deadzone radius
BTN_H = 36              # button height
BTN_GAP = 6             # gap between buttons
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Cartesian step sizes (Dobot)
CART_STEP_XY_MIN = 5.0    # mm at edge of deadzone
CART_STEP_XY_MAX = 100.0  # mm at edge of pad
CART_STEP_Z = 10.0        # mm per Z button click
CART_COOLDOWN = 0.3       # seconds between commands (prevent queue buildup)

# Joint jog step sizes (arm101 keyboard jog)
JOG_STEP_MIN = 1.0       # degrees at edge of deadzone
JOG_STEP_MAX = 30.0      # degrees at edge of pad
JOG_STEP_Z = 5.0         # degrees per Z button click (J3)
WRIST_STEP_DEG = 5.0     # degrees per J4/J5/J6 wrist button click

# Cartesian step sizes for arm101 IK-based control
ARM101_CART_STEP_MIN = 2.0   # mm at edge of deadzone
ARM101_CART_STEP_MAX = 30.0  # mm at edge of pad
ARM101_CART_STEP_Z = 5.0     # mm per Z button click

# Colors
COL_PAD_BG = (50, 50, 50)
COL_PAD_CROSS = (80, 80, 80)
COL_PAD_DOT = (0, 200, 255)
COL_BTN = (60, 60, 60)
COL_BTN_HOVER = (80, 80, 80)
COL_BTN_ACTIVE = (0, 140, 200)
COL_BTN_TEXT = (220, 220, 220)
COL_LABEL = (160, 160, 160)
COL_VALUE = (255, 200, 0)
COL_GREEN = (0, 200, 0)
COL_RED = (0, 0, 200)

# Mode descriptions (Dobot)
MODE_NAMES = {
    1: 'init', 2: 'brake', 3: 'init', 4: 'disabled', 5: 'enabled',
    6: 'backdrive', 7: 'running', 9: 'error', 10: 'pause', 11: 'jog',
}


class RobotControlPanel:
    """Interactive OpenCV GUI panel for robot arm control.

    Args:
        robot: Object with send(cmd), get_pose(), get_angles() methods (Dobot),
               or robot_type='arm101' with native arm101 interface.
        panel_x: X offset where the panel starts (= camera frame width).
        panel_height: Total panel height (= canvas height).
    """

    def __init__(self, robot, panel_x, panel_height=480, config=None):
        self.robot = robot
        self.panel_x = panel_x
        self.panel_width = PANEL_WIDTH
        self.panel_height = panel_height
        self._config = config or {}

        # Detect robot type
        self._arm101 = getattr(robot, 'robot_type', None) == 'arm101'
        self._ik_solver = None  # lazy-loaded for arm101 Cartesian control

        # State
        self.speed = 30
        self.jogging = False
        self.jog_axis = None  # current jog axis string e.g. 'J1+'
        self._mouse_down = False
        self._mouse_pos = None  # (x, y) relative to panel
        self._last_cmd_time = 0.0  # cooldown for Cartesian steps

        # Cached robot state (throttled queries)
        self._cached_pose = None
        self._cached_angles = None
        self._cached_mode = None
        self._last_query = 0.0
        self._query_interval = 0.5  # seconds (arm101 can be faster)
        if self._arm101:
            self._query_interval = 0.1

        self.status_msg = ""

        # Custom buttons: list of (label, callback, color) added by callers
        self._custom_buttons = []  # [(label, callback, color)]
        self._custom_rects = []    # computed in _layout()

        # Pre-compute layout positions (relative to panel_x)
        self._layout()

    def _layout(self):
        """Compute positions of all GUI elements."""
        px = 0  # all coords relative to panel; offset by panel_x when drawing
        margin = 10
        y = margin

        # XY Pad
        pad_x = (self.panel_width - PAD_SIZE) // 2
        self.pad_rect = (pad_x, y, pad_x + PAD_SIZE, y + PAD_SIZE)
        self.pad_center = (pad_x + PAD_SIZE // 2, y + PAD_SIZE // 2)
        y += PAD_SIZE + BTN_GAP

        # Z buttons row
        btn_w = (PAD_SIZE - BTN_GAP) // 2
        z_x = pad_x
        self.z_up_rect = (z_x, y, z_x + btn_w, y + BTN_H)
        self.z_dn_rect = (z_x + btn_w + BTN_GAP, y, z_x + PAD_SIZE, y + BTN_H)
        y += BTN_H + BTN_GAP * 2

        # Wrist buttons (arm101 only): J4, J5, J6  [-] [+] rows
        if self._arm101:
            self.j4_dn_rect = (z_x, y, z_x + btn_w, y + BTN_H)
            self.j4_up_rect = (z_x + btn_w + BTN_GAP, y, z_x + PAD_SIZE, y + BTN_H)
            y += BTN_H + BTN_GAP
            self.j5_dn_rect = (z_x, y, z_x + btn_w, y + BTN_H)
            self.j5_up_rect = (z_x + btn_w + BTN_GAP, y, z_x + PAD_SIZE, y + BTN_H)
            y += BTN_H + BTN_GAP
            self.j6_dn_rect = (z_x, y, z_x + btn_w, y + BTN_H)
            self.j6_up_rect = (z_x + btn_w + BTN_GAP, y, z_x + PAD_SIZE, y + BTN_H)
            y += BTN_H + BTN_GAP * 2
        else:
            self.j4_dn_rect = self.j4_up_rect = None
            self.j5_dn_rect = self.j5_up_rect = None
            self.j6_dn_rect = self.j6_up_rect = None

        # Gripper buttons
        self.grip_open_rect = (z_x, y, z_x + btn_w, y + BTN_H)
        self.grip_close_rect = (z_x + btn_w + BTN_GAP, y, z_x + PAD_SIZE, y + BTN_H)
        y += BTN_H + BTN_GAP * 2

        # Speed row: [-] [value] [+]
        third = (PAD_SIZE - 2 * BTN_GAP) // 3
        self.spd_dn_rect = (z_x, y, z_x + third, y + BTN_H)
        self.spd_label_rect = (z_x + third + BTN_GAP, y,
                               z_x + 2 * third + BTN_GAP, y + BTN_H)
        self.spd_up_rect = (z_x + 2 * third + 2 * BTN_GAP, y,
                            z_x + PAD_SIZE, y + BTN_H)
        y += BTN_H + BTN_GAP * 2

        # Enable / Home buttons
        self.enable_rect = (z_x, y, z_x + btn_w, y + BTN_H)
        self.home_rect = (z_x + btn_w + BTN_GAP, y, z_x + PAD_SIZE, y + BTN_H)
        y += BTN_H + BTN_GAP * 2

        # Safe mode toggle (arm101 only; drawn in same slot for both)
        if self._arm101:
            self.safe_rect = (z_x, y, z_x + PAD_SIZE, y + BTN_H)
            y += BTN_H + BTN_GAP * 2
        else:
            self.safe_rect = None

        # J1 rotate button (for calibration: rotate yaw in 30° steps)
        self.j1_rotate_rect = (z_x, y, z_x + PAD_SIZE, y + BTN_H)
        y += BTN_H + BTN_GAP * 2

        # Custom buttons (added by callers via add_button)
        self._custom_rects = []
        for _ in self._custom_buttons:
            rect = (z_x, y, z_x + PAD_SIZE, y + BTN_H)
            self._custom_rects.append(rect)
            y += BTN_H + BTN_GAP * 2

        # Status area starts here
        self.status_y = y

        # Estimate total height needed (status lines + margin)
        # Status area: TCP (2 lines) + Joints (2 lines) + Mode + Jog + Msg ≈ 7×18 + margin
        self._min_height = y + 7 * 18 + 20

    @property
    def min_height(self) -> int:
        """Minimum canvas height needed to display all panel elements."""
        return self._min_height

    def add_button(self, label, callback, color=(80, 60, 100)):
        """Add a custom button to the panel. Call before first draw.

        Args:
            label: Button text (string or callable returning string).
            callback: Called on click (no args).
            color: BGR tuple for button background.
        """
        self._custom_buttons.append((label, callback, color))
        self._layout()  # recompute positions

    def _draw_btn(self, canvas, rect, label, active=False, color=None):
        """Draw a rounded-ish button on the canvas."""
        x1, y1, x2, y2 = [v + (self.panel_x if i % 2 == 0 else 0)
                           for i, v in enumerate(rect)]
        bg = color or (COL_BTN_ACTIVE if active else COL_BTN)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (100, 100, 100), 1)
        # Center text
        sz = cv2.getTextSize(label, FONT, 0.45, 1)[0]
        tx = x1 + (x2 - x1 - sz[0]) // 2
        ty = y1 + (y2 - y1 + sz[1]) // 2
        cv2.putText(canvas, label, (tx, ty), FONT, 0.45, COL_BTN_TEXT, 1)

    def _in_rect(self, x, y, rect):
        """Check if (x, y) in panel-relative coords is inside rect."""
        return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]

    def draw(self, canvas):
        """Draw all GUI elements onto canvas. Called each frame."""
        px = self.panel_x

        # Panel background
        canvas[0:self.panel_height, px:px + self.panel_width] = PANEL_BG

        # --- XY Pad ---
        pr = self.pad_rect
        cv2.rectangle(canvas, (pr[0] + px, pr[1]), (pr[2] + px, pr[3]), COL_PAD_BG, -1)
        cv2.rectangle(canvas, (pr[0] + px, pr[1]), (pr[2] + px, pr[3]), (80, 80, 80), 1)
        # Crosshair
        cx, cy = self.pad_center[0] + px, self.pad_center[1]
        cv2.line(canvas, (pr[0] + px, cy), (pr[2] + px, cy), COL_PAD_CROSS, 1)
        cv2.line(canvas, (cx, pr[1]), (cx, pr[3]), COL_PAD_CROSS, 1)
        # Deadzone circle
        cv2.circle(canvas, (cx, cy), PAD_DEADZONE, COL_PAD_CROSS, 1)
        # Axis labels
        cv2.putText(canvas, "X+", (pr[2] + px - 22, cy + 4), FONT, 0.35, COL_LABEL, 1)
        cv2.putText(canvas, "X-", (pr[0] + px + 4, cy + 4), FONT, 0.35, COL_LABEL, 1)
        cv2.putText(canvas, "Y+", (cx - 8, pr[1] + 14), FONT, 0.35, COL_LABEL, 1)
        cv2.putText(canvas, "Y-", (cx - 8, pr[3] - 5), FONT, 0.35, COL_LABEL, 1)

        # Current drag indicator with step size
        if self._mouse_down and self._mouse_pos:
            mx, my = self._mouse_pos
            if self._in_rect(mx, my, self.pad_rect):
                cv2.circle(canvas, (mx + px, my), 6, COL_PAD_DOT, -1)
                cv2.line(canvas, (cx, cy), (mx + px, my), COL_PAD_DOT, 2)
                # Show step size
                pcx, pcy = self.pad_center
                dist = ((mx - pcx) ** 2 + (my - pcy) ** 2) ** 0.5
                if dist >= PAD_DEADZONE:
                    max_dist = PAD_SIZE / 2.0
                    t = min((dist - PAD_DEADZONE) / (max_dist - PAD_DEADZONE), 1.0)
                    if self._arm101:
                        step = JOG_STEP_MIN + t * (JOG_STEP_MAX - JOG_STEP_MIN)
                        cv2.putText(canvas, f"{step:.0f}deg",
                                    (mx + px + 8, my - 8), FONT, 0.35, COL_PAD_DOT, 1)
                    else:
                        step = CART_STEP_XY_MIN + t * (CART_STEP_XY_MAX - CART_STEP_XY_MIN)
                        cv2.putText(canvas, f"{step:.0f}mm",
                                    (mx + px + 8, my - 8), FONT, 0.35, COL_PAD_DOT, 1)

        # --- Z buttons ---
        self._draw_btn(canvas, self.z_up_rect, "Z +")
        self._draw_btn(canvas, self.z_dn_rect, "Z -")

        # --- Wrist buttons (arm101 only): J4 / J5 / J6 ---
        if self._arm101 and self.j4_dn_rect:
            wrist_col = (90, 55, 110)  # purple-ish to distinguish from motion controls
            self._draw_btn(canvas, self.j4_dn_rect, "J4 -", color=wrist_col)
            self._draw_btn(canvas, self.j4_up_rect, "J4 +", color=wrist_col)
            self._draw_btn(canvas, self.j5_dn_rect, "J5 -", color=wrist_col)
            self._draw_btn(canvas, self.j5_up_rect, "J5 +", color=wrist_col)
            self._draw_btn(canvas, self.j6_dn_rect, "J6 -", color=wrist_col)
            self._draw_btn(canvas, self.j6_up_rect, "J6 +", color=wrist_col)

        # --- Gripper ---
        self._draw_btn(canvas, self.grip_open_rect, "Open",
                       color=(0, 130, 0))
        self._draw_btn(canvas, self.grip_close_rect, "Close",
                       color=(0, 0, 160))

        # --- Speed ---
        self._draw_btn(canvas, self.spd_dn_rect, "<<")
        if self._arm101:
            self._draw_btn(canvas, self.spd_label_rect, f"spd {self.speed}")
        else:
            self._draw_btn(canvas, self.spd_label_rect, f"{self.speed}%")
        self._draw_btn(canvas, self.spd_up_rect, ">>")

        # --- Enable / Home ---
        if self._arm101:
            label = "Torque" if not getattr(self.robot, '_enabled', False) else "Relax"
            color = (0, 100, 0) if not getattr(self.robot, '_enabled', False) else (0, 0, 160)
            self._draw_btn(canvas, self.enable_rect, label, color=color)
        else:
            self._draw_btn(canvas, self.enable_rect, "Enable",
                           color=(0, 100, 0))
        self._draw_btn(canvas, self.home_rect, "Home",
                       color=(100, 80, 0))

        # --- Safe mode toggle (arm101 only) ---
        if self._arm101 and self.safe_rect:
            is_safe = getattr(self.robot, 'safe_mode', False)
            safe_label = "Safe: ON" if is_safe else "Safe: OFF"
            safe_color = (0, 120, 60) if is_safe else (80, 50, 50)
            self._draw_btn(canvas, self.safe_rect, safe_label, color=safe_color)

        # --- J1 Rotate ---
        j1_label = "J1 +30deg"
        if self._cached_angles is not None:
            j1_label = f"J1 +30  ({self._cached_angles[0]:.0f} now)"
        self._draw_btn(canvas, self.j1_rotate_rect, j1_label,
                       color=(100, 60, 0))

        # --- Custom buttons ---
        for i, (label, _cb, color) in enumerate(self._custom_buttons):
            text = label() if callable(label) else label
            self._draw_btn(canvas, self._custom_rects[i], text, color=color)

        # --- Status ---
        self._update_status_cache()
        y = self.status_y
        line_h = 18

        def put(text, col=COL_LABEL, yoff=0):
            nonlocal y
            cv2.putText(canvas, text, (px + 10, y + yoff), FONT, 0.38, col, 1)
            y += line_h

        if self.robot is None:
            put("NO ROBOT", COL_RED)
        elif self._cached_pose is not None:
            p = self._cached_pose
            put(f"TCP: {p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}", COL_VALUE)
            put(f"     {p[3]:.1f}, {p[4]:.1f}, {p[5]:.1f}", COL_VALUE)
        else:
            put("TCP: ---", COL_LABEL)
            y += line_h

        if self.robot is not None and self._cached_angles is not None:
            a = self._cached_angles
            put(f"J: {a[0]:.1f},{a[1]:.1f},{a[2]:.1f}", COL_VALUE)
            put(f"   {a[3]:.1f},{a[4]:.1f},{a[5]:.1f}", COL_VALUE)
        elif self.robot is not None:
            put("J: ---", COL_LABEL)
            y += line_h

        if self.robot is not None and self._cached_mode is not None:
            name = MODE_NAMES.get(self._cached_mode, '?')
            col = COL_GREEN if self._cached_mode == 5 else COL_RED
            put(f"Mode: {self._cached_mode} ({name})", col)
        elif self.robot is not None:
            put("Mode: ---", COL_LABEL)

        if self.jogging:
            put(f"JOG: {self.jog_axis}", (0, 200, 255))

        if self.status_msg:
            # Word-wrap status message
            msg = self.status_msg
            while msg:
                chunk = msg[:30]
                put(chunk, COL_GREEN)
                msg = msg[30:]

    def _update_status_cache(self):
        """Throttled robot state queries."""
        if self.robot is None:
            return
        now = time.time()
        if now - self._last_query < self._query_interval:
            return
        self._last_query = now
        try:
            self._cached_pose = self.robot.get_pose()
            self._cached_angles = self.robot.get_angles()
            if self._arm101:
                self._cached_mode = self.robot.get_mode()
            else:
                resp = self.robot.send('RobotMode()')
                if '{' in resp:
                    val = resp.split('{')[1].split('}')[0]
                    self._cached_mode = int(float(val))
        except Exception:
            pass

    def handle_mouse(self, event, x, y, flags):
        """Handle mouse events. x, y are in panel-relative coordinates.

        Returns True if the event was consumed (was in the panel area).
        """
        # Convert from canvas coords to panel-relative
        px = x - self.panel_x
        py = y
        if px < 0 or px >= self.panel_width:
            return False

        if event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_down = True
            self._mouse_pos = (px, py)
            self._on_press(px, py)
            return True

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._mouse_down:
                self._mouse_pos = (px, py)
                self._on_drag(px, py)
            return True

        elif event == cv2.EVENT_LBUTTONUP:
            was_down = self._mouse_down
            self._mouse_down = False
            self._mouse_pos = None
            if was_down:
                self._on_release(px, py)
            return True

        return True  # consume all events in panel area

    def _on_press(self, x, y):
        """Handle mouse button press in panel coords."""
        # XY Pad — Cartesian step (Dobot) or J1/J2 jog (arm101)
        if self._in_rect(x, y, self.pad_rect):
            self._do_xy_step(x, y)
            return

        # Z buttons — Cartesian step
        if self._in_rect(x, y, self.z_up_rect):
            if self._arm101:
                self._do_cart_step_ik(2, +1, ARM101_CART_STEP_Z)
            else:
                self._do_cart_step(2, +1, CART_STEP_Z)
            return
        if self._in_rect(x, y, self.z_dn_rect):
            if self._arm101:
                self._do_cart_step_ik(2, -1, ARM101_CART_STEP_Z)
            else:
                self._do_cart_step(2, -1, CART_STEP_Z)
            return

        # Wrist buttons — J4/J5/J6 joint steps (arm101 only)
        if self.j4_dn_rect:
            if self._in_rect(x, y, self.j4_dn_rect):
                self._do_joint_step(3, -1, WRIST_STEP_DEG)
                return
            if self._in_rect(x, y, self.j4_up_rect):
                self._do_joint_step(3, +1, WRIST_STEP_DEG)
                return
            if self._in_rect(x, y, self.j5_dn_rect):
                self._do_joint_step(4, -1, WRIST_STEP_DEG)
                return
            if self._in_rect(x, y, self.j5_up_rect):
                self._do_joint_step(4, +1, WRIST_STEP_DEG)
                return
            if self._in_rect(x, y, self.j6_dn_rect):
                self._do_joint_step(5, -1, WRIST_STEP_DEG)
                return
            if self._in_rect(x, y, self.j6_up_rect):
                self._do_joint_step(5, +1, WRIST_STEP_DEG)
                return

        # Gripper
        if self._in_rect(x, y, self.grip_open_rect):
            self._gripper_open()
            return
        if self._in_rect(x, y, self.grip_close_rect):
            self._gripper_close()
            return

        # Speed
        if self._in_rect(x, y, self.spd_dn_rect):
            self._speed_change(-10)
            return
        if self._in_rect(x, y, self.spd_up_rect):
            self._speed_change(+10)
            return

        # Enable / Home
        if self._in_rect(x, y, self.enable_rect):
            self._do_enable()
            return
        if self._in_rect(x, y, self.home_rect):
            self._do_home()
            return

        # Safe mode toggle (arm101)
        if self.safe_rect and self._in_rect(x, y, self.safe_rect):
            self._toggle_safe_mode()
            return

        # J1 rotate
        if self._in_rect(x, y, self.j1_rotate_rect):
            self._do_j1_rotate()
            return

        # Custom buttons
        for i, (_label, callback, _color) in enumerate(self._custom_buttons):
            if self._in_rect(x, y, self._custom_rects[i]):
                callback()
                return

    def _on_drag(self, x, y):
        """Handle mouse drag in panel coords (no-op for Cartesian steps)."""
        pass

    def _on_release(self, x, y):
        """Handle mouse button release."""
        # Stop any active joint jog (from keyboard)
        if self.jogging:
            self._stop_jog()

    # --- arm101: Joint step (for XY pad and Z buttons) ---

    def _do_joint_step(self, joint_idx, direction, step_deg):
        """Step a single joint by step_deg (arm101 only)."""
        if self.robot is None:
            return
        now = time.time()
        if now - self._last_cmd_time < CART_COOLDOWN:
            return
        self._last_cmd_time = now

        try:
            self.robot.jog_joint(joint_idx, direction, step_deg)
            jname = f"J{joint_idx + 1}"
            dir_ch = '+' if direction > 0 else '-'
            self.status_msg = f"{jname}{dir_ch} {step_deg:.0f}deg"
        except Exception as e:
            self.status_msg = f"Jog error: {e}"

    def _do_cart_step_ik(self, axis_idx, sign, step_mm):
        """IK-based Cartesian step for arm101: read pose, offset, solve IK, move."""
        if self.robot is None:
            return
        now = time.time()
        if now - self._last_cmd_time < CART_COOLDOWN:
            return
        self._last_cmd_time = now

        labels = {0: 'X', 1: 'Y', 2: 'Z'}
        axis_name = labels[axis_idx]
        dir_ch = '+' if sign > 0 else '-'

        # Lazy-load IK solver
        if self._ik_solver is None:
            try:
                from kinematics.arm101_ik_solver import Arm101IKSolver
                self._ik_solver = Arm101IKSolver()
            except Exception as e:
                self.status_msg = f"IK init error: {e}"
                return

        angles = self.robot.get_angles()
        if angles is None:
            self.status_msg = "ERROR: can't read angles"
            return

        # Current FK pose
        motor_deg = np.array(angles[:5], dtype=float)
        pos_mm, rpy_deg = self._ik_solver.forward_kin(motor_deg)

        # Offset target
        target_pos = pos_mm.copy()
        target_pos[axis_idx] += sign * step_mm

        # Solve position-only IK (5-DOF arm, orientation floats)
        result = self._ik_solver.solve_ik_position(
            target_pos, seed_motor_deg=motor_deg)
        if result is None:
            self.status_msg = f"IK failed for {axis_name}{dir_ch}"
            return

        # Send joint angles (first 5 joints, keep gripper unchanged)
        try:
            full_angles = list(result) + [angles[5]]
            self.robot.move_joints(full_angles)
            self.status_msg = f"{axis_name}{dir_ch} {step_mm:.0f}mm"
        except Exception as e:
            self.status_msg = f"Move error: {e}"

    def _do_xy_step(self, x, y):
        """Do a Cartesian XY step (Dobot) or Cartesian IK step (arm101)."""
        if self.robot is None:
            return
        cx, cy = self.pad_center
        dx = x - cx
        dy = y - cy
        dist = (dx * dx + dy * dy) ** 0.5

        if dist < PAD_DEADZONE:
            return

        # Scale step by distance from center
        max_dist = PAD_SIZE / 2.0
        t = min((dist - PAD_DEADZONE) / (max_dist - PAD_DEADZONE), 1.0)

        if self._arm101:
            step_mm = ARM101_CART_STEP_MIN + t * (ARM101_CART_STEP_MAX - ARM101_CART_STEP_MIN)
            # Determine dominant axis: right=X+, up=Y+
            if abs(dx) >= abs(dy):
                axis_idx = 0  # X
                sign = +1 if dx > 0 else -1
            else:
                axis_idx = 1  # Y
                sign = +1 if dy < 0 else -1  # up on screen = Y+
            self._do_cart_step_ik(axis_idx, sign, step_mm)
        else:
            step_mm = CART_STEP_XY_MIN + t * (CART_STEP_XY_MAX - CART_STEP_XY_MIN)
            # Determine dominant axis: X or Y
            # Pad layout: right = X+, left = X-, up = Y+, down = Y-
            if abs(dx) >= abs(dy):
                axis_idx = 0  # X
                sign = +1 if dx > 0 else -1
            else:
                axis_idx = 1  # Y
                sign = +1 if dy < 0 else -1  # up on screen = Y+
            self._do_cart_step(axis_idx, sign, step_mm)

    def _do_cart_step(self, axis_idx, sign, step_mm):
        """Fire-and-forget Cartesian step: read pose, offset, move to target pose (Dobot/arm101)."""
        if self.robot is None:
            return
        now = time.time()
        if now - self._last_cmd_time < CART_COOLDOWN:
            return  # too soon, skip
        self._last_cmd_time = now

        labels = {0: 'X', 1: 'Y', 2: 'Z'}
        axis_name = labels[axis_idx]
        dir_ch = '+' if sign > 0 else '-'

        pose = self.robot.get_pose()
        if not pose or len(pose) < 6:
            self.status_msg = "ERROR: can't read pose"
            return

        target = list(pose)
        target[axis_idx] += sign * step_mm

        # Use unified move_to_pose() for both Nova5 and arm101
        success = move_to_pose(
            self.robot,
            x=target[0],
            y=target[1],
            z=target[2],
            rx=target[3],
            ry=target[4],
            rz=target[5],
            speed=self.speed
        )

        if not success:
            self.status_msg = f"Move failed: {axis_name}{dir_ch}"
            return

        self.status_msg = f"{axis_name}{dir_ch} {step_mm:.0f}mm"

    def _stop_jog(self):
        """Stop any active joint jog."""
        if self.robot is None:
            return
        if not self._arm101:
            self.robot.send('MoveJog()')
        # arm101 jogs are step-based (fire-and-forget), no stop needed
        self.jogging = False
        self.jog_axis = None
        self.status_msg = "Stopped"

    def _gripper_open(self):
        if self.robot is None:
            return
        if self._arm101:
            try:
                self.robot.gripper_open()
            except Exception as e:
                self.status_msg = f"Gripper error: {e}"
                return
        else:
            self.robot.send('ToolDOInstant(1,0)')
            self.robot.send('ToolDOInstant(2,1)')
        self.status_msg = "Gripper OPEN"

    def _gripper_close(self):
        if self.robot is None:
            return
        if self._arm101:
            try:
                self.robot.gripper_close()
            except Exception as e:
                self.status_msg = f"Gripper error: {e}"
                return
        else:
            self.robot.send('ToolDOInstant(2,0)')
            self.robot.send('ToolDOInstant(1,1)')
        self.status_msg = "Gripper CLOSED"

    def _speed_change(self, delta):
        if self._arm101:
            # arm101 speed: 0-4095, step by 50
            self.speed = max(10, min(1000, self.speed + delta * 5))
            if self.robot is not None:
                self.robot.speed = self.speed
        else:
            self.speed = max(1, min(100, self.speed + delta))
            if self.robot is not None:
                self.robot.send(f'SpeedFactor({self.speed})')
        self.status_msg = f"Speed: {self.speed}"

    def _do_enable(self):
        if self.robot is None:
            return
        if self._arm101:
            if getattr(self.robot, '_enabled', False):
                self.status_msg = "Disabling torque..."
                try:
                    self.robot.disable_torque()
                    self.status_msg = "Torque OFF (freedrive)"
                except Exception as e:
                    self.status_msg = f"Disable error: {e}"
            else:
                self.status_msg = "Enabling torque..."
                try:
                    self.robot.enable_torque()
                    self.status_msg = "Torque ON"
                except Exception as e:
                    self.status_msg = f"Enable error: {e}"
        else:
            self.status_msg = "Enabling..."
            self.robot.send('DisableRobot()')
            time.sleep(1)
            self.robot.send('ClearError()')
            self.robot.send('EnableRobot()')
            time.sleep(1)
            self.status_msg = "Robot enabled"

    def _toggle_safe_mode(self):
        """Toggle safe mode on arm101 (reduced torque/speed)."""
        if self.robot is None or not self._arm101:
            return
        is_safe = getattr(self.robot, 'safe_mode', False)
        try:
            self.robot.set_safe_mode(not is_safe)
            self.speed = self.robot.speed
            if self.robot.safe_mode:
                self.status_msg = "Safe mode ON"
            else:
                self.status_msg = "Safe mode OFF"
        except Exception as e:
            self.status_msg = f"Safe mode error: {e}"

    def _do_j1_rotate(self, step_deg=30.0):
        """Rotate J1 by step_deg, keeping all other joints the same."""
        if self.robot is None:
            return
        angles = self.robot.get_angles()
        if not angles or len(angles) < 6:
            self.status_msg = "ERROR: can't read angles"
            return

        target = list(angles)
        target[0] += step_deg

        if self._arm101:
            self.status_msg = f"J1 -> {target[0]:.0f} deg..."
            try:
                self.robot.move_joints(target)
                # Brief wait for servo to reach position
                time.sleep(0.5)
                final = self.robot.get_angles()
                if final:
                    self.status_msg = f"J1 = {final[0]:.1f} deg"
                else:
                    self.status_msg = "J1 rotate done"
            except Exception as e:
                self.status_msg = f"J1 error: {e}"
        else:
            jstr = ','.join(f'{v:.2f}' for v in target)
            cmd = f'MovJ(joint={{{jstr}}})'
            self.status_msg = f"J1 -> {target[0]:.0f} deg..."
            resp = self.robot.send(cmd)
            code = resp.split(',')[0] if resp else '-1'
            if code != '0':
                self.status_msg = f"MovJ error: {resp}"
                return

            # Wait for motion to complete (poll joint stability)
            prev = self.robot.get_angles()
            for _ in range(150):  # up to 30s
                time.sleep(0.2)
                cur = self.robot.get_angles()
                if prev and cur and len(prev) >= 6 and len(cur) >= 6:
                    if max(abs(cur[i] - prev[i]) for i in range(6)) < 0.05:
                        break
                prev = cur

            final = self.robot.get_angles()
            if final:
                self.status_msg = f"J1 = {final[0]:.1f} deg — click TCP tip"
            else:
                self.status_msg = "J1 rotate done"

    def _do_home(self):
        if self.robot is None:
            return
        if self.jogging:
            self._stop_jog()
        self.status_msg = "Homing..."

        if self._arm101:
            try:
                cfg = getattr(self, '_config', None) or {}
                home = cfg.get('arm101', {}).get(
                    'home_angles', [0.0, 0.0, 90.0, 90.0, 0.0, 0.0])
                self.robot.move_joints(home, speed=100)
                time.sleep(1.0)
                self.status_msg = "Home done"
            except Exception as e:
                self.status_msg = f"Home error: {e}"
        else:
            self.robot.send('SpeedFactor(10)')
            resp = self.robot.send('MovJ(joint={43.5,-13.9,-85.4,196.3,-90.0,43.5})')
            code = resp.split(',')[0] if resp else '-1'
            if code != '0':
                self.status_msg = f"Home failed: {resp}"
                self.robot.send(f'SpeedFactor({self.speed})')
                return
            # Wait for motion (poll stability)
            prev = self.robot.get_angles()
            for _ in range(150):
                time.sleep(0.2)
                cur = self.robot.get_angles()
                if prev and cur and len(prev) >= 6 and len(cur) >= 6:
                    if max(abs(cur[i] - prev[i]) for i in range(6)) < 0.05:
                        break
                prev = cur
            self.robot.send(f'SpeedFactor({self.speed})')
            self.status_msg = "Home done"

    def handle_key(self, key):
        """Handle keyboard shortcuts for arm control.

        Returns True if the key was consumed.
        """
        # Joint jog keys
        jog_pos = {ord('1'): 'J1+', ord('2'): 'J2+', ord('3'): 'J3+',
                   ord('4'): 'J4+', ord('5'): 'J5+', ord('6'): 'J6+'}
        jog_neg = {ord('!'): 'J1-', ord('@'): 'J2-', ord('#'): 'J3-',
                   ord('$'): 'J4-', ord('%'): 'J5-', ord('^'): 'J6-'}
        all_jog = {**jog_pos, **jog_neg}

        if key in all_jog:
            if self.robot is None:
                return True
            axis = all_jog[key]
            if self._arm101:
                # Parse J<n><+/-> into joint_idx and direction
                joint_idx = int(axis[1]) - 1  # 0-based
                direction = +1 if axis[2] == '+' else -1
                try:
                    self.robot.jog_joint(joint_idx, direction)
                    self.status_msg = f"JOG {axis}"
                except Exception as e:
                    self.status_msg = f"Jog error: {e}"
            else:
                self.robot.send(f'MoveJog({axis})')
                self.jogging = True
                self.jog_axis = axis
                self.status_msg = f"JOG {axis}"
            return True

        if key == ord(' '):
            self._stop_jog()
            return True

        # Gripper
        if key == ord('c'):
            self._gripper_close()
            return True
        if key == ord('o'):
            self._gripper_open()
            return True

        # Speed
        if key == ord('['):
            self._speed_change(-10)
            return True
        if key == ord(']'):
            self._speed_change(+10)
            return True

        # Enable
        if key == ord('v'):
            self._do_enable()
            return True

        # Safe mode toggle (arm101 only)
        if key == ord('s') and self._arm101:
            self._toggle_safe_mode()
            return True

        return False
