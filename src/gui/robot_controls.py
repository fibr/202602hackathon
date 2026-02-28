"""Shared OpenCV GUI panel for robot arm control.

Draws interactive controls (XY jog pad, Z buttons, gripper, speed, enable/home,
live status) onto an OpenCV canvas. Used by detect_checkerboard.py and
control_panel.py.

The panel is drawn on the right side of the frame. Mouse events in the panel
area are routed to handle_mouse(); events outside are left to the app.

Robot connection is duck-typed: needs send(), get_pose(), get_angles() methods.
"""

import time
import cv2
import numpy as np


# Layout constants
PANEL_WIDTH = 250
PANEL_BG = (30, 30, 30)
PAD_SIZE = 150          # XY pad size in pixels
PAD_DEADZONE = 15       # center deadzone radius
BTN_H = 36              # button height
BTN_GAP = 6             # gap between buttons
FONT = cv2.FONT_HERSHEY_SIMPLEX

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

# Mode descriptions
MODE_NAMES = {
    1: 'init', 2: 'brake', 3: 'init', 4: 'disabled', 5: 'enabled',
    6: 'backdrive', 7: 'running', 9: 'error', 10: 'pause', 11: 'jog',
}


class RobotControlPanel:
    """Interactive OpenCV GUI panel for robot arm control.

    Args:
        robot: Object with send(cmd), get_pose(), get_angles() methods.
        panel_x: X offset where the panel starts (= camera frame width).
        panel_height: Total panel height (= canvas height).
    """

    def __init__(self, robot, panel_x, panel_height=480):
        self.robot = robot
        self.panel_x = panel_x
        self.panel_width = PANEL_WIDTH
        self.panel_height = panel_height

        # State
        self.speed = 30
        self.jogging = False
        self.jog_axis = None  # current jog axis string e.g. 'J1+'
        self._mouse_down = False
        self._mouse_pos = None  # (x, y) relative to panel
        self._z_jog_dir = None  # 'up' or 'down' while held

        # Cached robot state (throttled queries)
        self._cached_pose = None
        self._cached_angles = None
        self._cached_mode = None
        self._last_query = 0.0
        self._query_interval = 0.5  # seconds

        self.status_msg = ""

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

        # Status area starts here
        self.status_y = y

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

        # Current drag indicator
        if self._mouse_down and self._mouse_pos:
            mx, my = self._mouse_pos
            if self._in_rect(mx, my, self.pad_rect):
                cv2.circle(canvas, (mx + px, my), 6, COL_PAD_DOT, -1)
                cv2.line(canvas, (cx, cy), (mx + px, my), COL_PAD_DOT, 2)

        # --- Z buttons ---
        self._draw_btn(canvas, self.z_up_rect, "Z +",
                       active=(self._z_jog_dir == 'up'))
        self._draw_btn(canvas, self.z_dn_rect, "Z -",
                       active=(self._z_jog_dir == 'down'))

        # --- Gripper ---
        self._draw_btn(canvas, self.grip_open_rect, "Open",
                       color=(0, 130, 0))
        self._draw_btn(canvas, self.grip_close_rect, "Close",
                       color=(0, 0, 160))

        # --- Speed ---
        self._draw_btn(canvas, self.spd_dn_rect, "<<")
        self._draw_btn(canvas, self.spd_label_rect, f"{self.speed}%")
        self._draw_btn(canvas, self.spd_up_rect, ">>")

        # --- Enable / Home ---
        self._draw_btn(canvas, self.enable_rect, "Enable",
                       color=(0, 100, 0))
        self._draw_btn(canvas, self.home_rect, "Home",
                       color=(100, 80, 0))

        # --- Status ---
        self._update_status_cache()
        y = self.status_y
        line_h = 18

        def put(text, col=COL_LABEL, yoff=0):
            nonlocal y
            cv2.putText(canvas, text, (px + 10, y + yoff), FONT, 0.38, col, 1)
            y += line_h

        if self._cached_pose is not None:
            p = self._cached_pose
            put(f"TCP: {p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}", COL_VALUE)
            put(f"     {p[3]:.1f}, {p[4]:.1f}, {p[5]:.1f}", COL_VALUE)
        else:
            put("TCP: ---", COL_LABEL)
            y += line_h

        if self._cached_angles is not None:
            a = self._cached_angles
            put(f"J: {a[0]:.1f},{a[1]:.1f},{a[2]:.1f}", COL_VALUE)
            put(f"   {a[3]:.1f},{a[4]:.1f},{a[5]:.1f}", COL_VALUE)
        else:
            put("J: ---", COL_LABEL)
            y += line_h

        if self._cached_mode is not None:
            name = MODE_NAMES.get(self._cached_mode, '?')
            col = COL_GREEN if self._cached_mode == 5 else COL_RED
            put(f"Mode: {self._cached_mode} ({name})", col)
        else:
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
        now = time.time()
        if now - self._last_query < self._query_interval:
            return
        self._last_query = now
        try:
            self._cached_pose = self.robot.get_pose()
            self._cached_angles = self.robot.get_angles()
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
        # XY Pad
        if self._in_rect(x, y, self.pad_rect):
            self._do_xy_jog(x, y)
            return

        # Z buttons
        if self._in_rect(x, y, self.z_up_rect):
            self._start_z_jog('up')
            return
        if self._in_rect(x, y, self.z_dn_rect):
            self._start_z_jog('down')
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

    def _on_drag(self, x, y):
        """Handle mouse drag in panel coords."""
        if self._in_rect(x, y, self.pad_rect):
            self._do_xy_jog(x, y)
        # Z buttons: if dragged off, stop
        elif self._z_jog_dir is not None:
            if self._z_jog_dir == 'up' and not self._in_rect(x, y, self.z_up_rect):
                self._stop_jog()
            elif self._z_jog_dir == 'down' and not self._in_rect(x, y, self.z_dn_rect):
                self._stop_jog()

    def _on_release(self, x, y):
        """Handle mouse button release."""
        if self.jogging or self._z_jog_dir is not None:
            self._stop_jog()

    def _do_xy_jog(self, x, y):
        """Start or update XY jog based on pad position."""
        cx, cy = self.pad_center
        dx = x - cx
        dy = y - cy
        dist = (dx * dx + dy * dy) ** 0.5

        if dist < PAD_DEADZONE:
            if self.jogging:
                self._stop_jog()
            return

        # Determine dominant axis: X or Y
        # In robot frame for the calibration app: WASD mapping
        # Right = X+, Left = X-, Up = Y+, Down = Y-
        # On the pad: right = +dx, up = -dy
        if abs(dx) >= abs(dy):
            # Horizontal: X axis
            axis = 'X+' if dx > 0 else 'X-'
        else:
            # Vertical: Y axis (up on screen = Y+)
            axis = 'Y+' if dy < 0 else 'Y-'

        # Map to MoveJog joint axes (approximate via J1/J2)
        # For better UX, use actual Cartesian jog isn't supported,
        # so we use joint jog axes:
        # X ~ J1 rotation, Y ~ J2 tilt — this is a rough mapping
        # Actually, the plan says to use MoveJog() which only supports joint axes.
        # Let's map XY pad to J1 (rotation around base = roughly X/Y in workspace)
        # and J2 (shoulder = roughly forward/back).
        # But this is confusing. Better approach: map to the same jog axes that
        # the keyboard used, but those were Cartesian steps (MovL), not jog.
        #
        # The plan says: "Click and hold: continuous MoveJog in the direction"
        # MoveJog only works with J1+ through J6+. No Cartesian jog.
        # Map: X+/- -> J1+/-, Y+/- -> J2+/-, this is the best we can do with jog.
        jog_map = {'X+': 'J1+', 'X-': 'J1-', 'Y+': 'J2+', 'Y-': 'J2-'}
        jog_axis = jog_map[axis]

        if self.jog_axis != jog_axis:
            if self.jogging:
                self.robot.send('MoveJog()')
            self.robot.send(f'MoveJog({jog_axis})')
            self.jog_axis = jog_axis
            self.jogging = True
            self.status_msg = f"JOG {jog_axis}"

    def _start_z_jog(self, direction):
        """Start Z-axis jog (up or down)."""
        if self.jogging:
            self.robot.send('MoveJog()')
        axis = 'J3-' if direction == 'up' else 'J3+'  # J3- raises Z, J3+ lowers
        self.robot.send(f'MoveJog({axis})')
        self._z_jog_dir = direction
        self.jog_axis = axis
        self.jogging = True
        self.status_msg = f"JOG Z {'up' if direction == 'up' else 'down'}"

    def _stop_jog(self):
        """Stop any active jog."""
        self.robot.send('MoveJog()')
        self.jogging = False
        self.jog_axis = None
        self._z_jog_dir = None
        self.status_msg = "Stopped"

    def _gripper_open(self):
        self.robot.send('ToolDOInstant(1,0)')
        self.robot.send('ToolDOInstant(2,1)')
        self.status_msg = "Gripper OPEN"

    def _gripper_close(self):
        self.robot.send('ToolDOInstant(2,0)')
        self.robot.send('ToolDOInstant(1,1)')
        self.status_msg = "Gripper CLOSED"

    def _speed_change(self, delta):
        self.speed = max(1, min(100, self.speed + delta))
        self.robot.send(f'SpeedFactor({self.speed})')
        self.status_msg = f"Speed: {self.speed}%"

    def _do_enable(self):
        self.status_msg = "Enabling..."
        self.robot.send('DisableRobot()')
        time.sleep(1)
        self.robot.send('ClearError()')
        self.robot.send('EnableRobot()')
        time.sleep(1)
        self.status_msg = "Robot enabled"

    def _do_home(self):
        if self.jogging:
            self._stop_jog()
        self.status_msg = "Homing..."
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
            axis = all_jog[key]
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

        return False
