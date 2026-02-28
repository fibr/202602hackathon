"""Collapsible overlay panel for calibration offset/rotation adjustment.

Draws semi-transparent controls on top of the camera image (bottom-left).
Provides clickable -/+ buttons for X, Y, Z, Roll, Pitch, Yaw and a Save button.
Toggle expanded/collapsed by clicking the header or pressing 't'.
"""

import cv2
import numpy as np


# Layout
_PANEL_W = 300
_ROW_H = 30
_HEADER_H = 28
_BTN_W = 30
_MARGIN = 10
_SAVE_BTN_H = 30

# Colors (BGR) — matches robot_controls.py scheme
_BG = (30, 30, 30)
_BTN = (60, 60, 60)
_BTN_HOVER = (80, 80, 80)
_TEXT = (220, 220, 220)
_VALUE = (255, 200, 0)       # cyan-ish
_HEADER_BG = (50, 50, 50)
_SAVE_BG = (0, 140, 80)
_SAVE_BG_HOVER = (0, 180, 100)
_ALPHA = 0.75

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Row definitions: (label, unit, is_rotation)
_ROWS = [
    ('X', 'mm', False),
    ('Y', 'mm', False),
    ('Z', 'mm', False),
    ('R', 'deg', True),
    ('P', 'deg', True),
    ('W', 'deg', True),
]


class OverlayCalibPanel:
    """Collapsible calibration adjustment panel drawn on the camera image.

    Args:
        robot_overlay: RobotOverlay instance (has base_offset_m, base_rpy_deg,
                       nudge_base, nudge_base_rpy).
        transform: CoordinateTransform instance (has base_offset_mm, base_rpy_deg, save).
        calibration_path: Path to calibration.yaml for saving.
    """

    def __init__(self, robot_overlay, transform, calibration_path):
        self.overlay = robot_overlay
        self.transform = transform
        self.calibration_path = calibration_path
        self.expanded = False
        self._nudge_xyz_mm = 10.0
        self._nudge_rpy_deg = 2.0
        # Cached geometry for hit testing (set during draw)
        self._panel_rect = None   # (x, y, w, h)
        self._header_rect = None  # (x, y, w, h)
        self._btn_rects = []      # list of (x, y, w, h, action)
        self._save_rect = None    # (x, y, w, h)

    def _panel_height(self):
        if not self.expanded:
            return _HEADER_H
        return _HEADER_H + len(_ROWS) * _ROW_H + _SAVE_BTN_H + _MARGIN

    def draw(self, image):
        """Draw the panel on the image (bottom-left). Modifies image in place."""
        h_img, w_img = image.shape[:2]
        pw = _PANEL_W
        ph = self._panel_height()
        px = 0
        py = h_img - ph

        self._panel_rect = (px, py, pw, ph)
        self._btn_rects = []

        # Semi-transparent background
        roi = image[py:py + ph, px:px + pw]
        bg = np.full_like(roi, _BG)
        cv2.addWeighted(bg, _ALPHA, roi, 1 - _ALPHA, 0, roi)

        # Header bar
        hx, hy = px, py
        cv2.rectangle(image, (hx, hy), (hx + pw, hy + _HEADER_H), _HEADER_BG, -1)
        arrow = '\u25BC' if self.expanded else '\u25B6'  # ▼ or ▶
        cv2.putText(image, "Overlay Calibration", (hx + 8, hy + 19),
                    _FONT, 0.45, _TEXT, 1)
        # Draw toggle indicator (simple text since unicode may not render)
        indicator = "[-]" if self.expanded else "[+]"
        cv2.putText(image, indicator, (hx + pw - 40, hy + 19),
                    _FONT, 0.42, _VALUE, 1)
        self._header_rect = (hx, hy, pw, _HEADER_H)

        if not self.expanded:
            return

        # Get current values
        offset_mm = self.overlay.base_offset_m * 1000.0
        rpy_deg = self.overlay.base_rpy_deg
        values = [offset_mm[0], offset_mm[1], offset_mm[2],
                  rpy_deg[0], rpy_deg[1], rpy_deg[2]]

        # Draw rows
        row_y = py + _HEADER_H
        for i, ((label, unit, _is_rot), val) in enumerate(zip(_ROWS, values)):
            ry = row_y + i * _ROW_H
            # Label
            cv2.putText(image, label, (px + 10, ry + 21),
                        _FONT, 0.5, _TEXT, 1)
            # Minus button
            bx_minus = px + 40
            by = ry + 3
            bw, bh = _BTN_W, _ROW_H - 6
            cv2.rectangle(image, (bx_minus, by), (bx_minus + bw, by + bh), _BTN, -1)
            cv2.putText(image, "-", (bx_minus + 10, by + 18),
                        _FONT, 0.5, _TEXT, 1)
            self._btn_rects.append((bx_minus, by, bw, bh, ('minus', i)))

            # Value
            val_str = f"{val:+.1f} {unit}"
            cv2.putText(image, val_str, (bx_minus + bw + 10, ry + 21),
                        _FONT, 0.45, _VALUE, 1)

            # Plus button
            bx_plus = px + 210
            cv2.rectangle(image, (bx_plus, by), (bx_plus + bw, by + bh), _BTN, -1)
            cv2.putText(image, "+", (bx_plus + 9, by + 18),
                        _FONT, 0.5, _TEXT, 1)
            self._btn_rects.append((bx_plus, by, bw, bh, ('plus', i)))

        # Save button
        save_y = row_y + len(_ROWS) * _ROW_H + 4
        save_x = px + pw // 2 - 40
        save_w, save_h = 80, _SAVE_BTN_H - 6
        cv2.rectangle(image, (save_x, save_y), (save_x + save_w, save_y + save_h),
                      _SAVE_BG, -1)
        cv2.putText(image, "Save", (save_x + 18, save_y + 19),
                    _FONT, 0.5, _TEXT, 1)
        self._save_rect = (save_x, save_y, save_w, save_h)

        # Border
        cv2.rectangle(image, (px, py), (px + pw, py + ph), (80, 80, 80), 1)

    def handle_mouse(self, event, x, y) -> bool:
        """Handle mouse click. Returns True if the event was consumed."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return False

        # Check header click (toggle)
        if self._header_rect:
            hx, hy, hw, hh = self._header_rect
            if hx <= x < hx + hw and hy <= y < hy + hh:
                self.expanded = not self.expanded
                return True

        if not self.expanded:
            return False

        # Check +/- buttons
        for bx, by, bw, bh, action in self._btn_rects:
            if bx <= x < bx + bw and by <= y < by + bh:
                self._do_nudge(action)
                return True

        # Check Save button
        if self._save_rect:
            sx, sy, sw, sh = self._save_rect
            if sx <= x < sx + sw and sy <= y < sy + sh:
                self._do_save()
                return True

        # Click inside panel area but not on a button — consume to prevent passthrough
        if self._panel_rect:
            px, py, pw, ph = self._panel_rect
            if px <= x < px + pw and py <= y < py + ph:
                return True

        return False

    def _do_nudge(self, action):
        """Apply a single nudge step."""
        direction, idx = action
        sign = 1.0 if direction == 'plus' else -1.0
        if idx < 3:
            # XYZ
            delta = [0.0, 0.0, 0.0]
            delta[idx] = sign * self._nudge_xyz_mm
            self.overlay.nudge_base(*delta)
        else:
            # RPY (idx 3=roll, 4=pitch, 5=yaw)
            delta = [0.0, 0.0, 0.0]
            delta[idx - 3] = sign * self._nudge_rpy_deg
            self.overlay.nudge_base_rpy(*delta)

    def _do_save(self):
        """Save current overlay offsets to calibration file."""
        self.transform.base_offset_mm = self.overlay.base_offset_m * 1000.0
        self.transform.base_rpy_deg = self.overlay.base_rpy_deg
        self.transform.save(self.calibration_path)
        print(f"  Saved to {self.calibration_path}:")
        print(f"    offset: {self.transform.base_offset_mm} mm")
        print(f"    rpy:    {self.transform.base_rpy_deg} deg")

    def toggle(self):
        """Toggle expanded/collapsed state."""
        self.expanded = not self.expanded
