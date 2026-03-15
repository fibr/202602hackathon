"""Home view: config display, editing, and utility script launcher."""

import os
import subprocess
import sys
import threading

import cv2
import numpy as np
import yaml

from config_loader import config_path
from gui.views.base import BaseView, ViewRegistry

FONT = cv2.FONT_HERSHEY_SIMPLEX
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))


@ViewRegistry.register
class HomeView(BaseView):
    view_id = 'home'
    view_name = 'Home'
    description = 'Config & utilities'
    needs_camera = False
    needs_robot = False
    headless_ok = False

    def __init__(self, app):
        super().__init__(app)
        self._config_lines = []
        self._scroll_offset = 0
        self._selected_field = -1
        self._editing = False
        self._edit_buffer = ''
        self._edit_key = ''
        self._edit_section = ''

        # Utility scripts that can be launched
        self._utilities = [
            ('Discover Cameras', 'scripts/discover_cameras.py', []),
            ('Discover Cameras (dry-run)', 'scripts/discover_cameras.py', ['--dry-run']),
        ]
        self._utility_running = False
        self._utility_output = ''
        self._utility_thread = None

        # Flat config entries for display/edit: list of (section, key, value, line_y)
        self._config_entries = []

        # Button areas: list of (x1, y1, x2, y2, action_name)
        self._buttons = []

        # Mode: 'config' or 'output'
        self._mode = 'config'

    def setup(self):
        self._load_config_display()

    def _load_config_display(self):
        """Load config into displayable flat list."""
        config = self.app.config
        self._config_entries = []
        for section, values in config.items():
            if isinstance(values, dict):
                for key, val in values.items():
                    self._config_entries.append((section, key, val))
            else:
                self._config_entries.append(('', section, values))

    def update(self, canvas):
        vw = self.app.view_width
        vh = self.app.canvas_height
        self._buttons = []

        # Background
        canvas[:vh, :vw] = (30, 30, 35)

        # Title area
        cv2.putText(canvas, 'ArmRobotics Configuration', (20, 35),
                    FONT, 0.65, (255, 200, 100), 1)

        robot_type = self.app.config.get('robot_type', '?')
        cam_type = self.app.config.get('camera', {}).get('type', '?')
        cv2.putText(canvas, f'Robot: {robot_type}  |  Camera: {cam_type}',
                    (20, 58), FONT, 0.4, (150, 150, 150), 1)

        # Separator
        cv2.line(canvas, (10, 68), (vw - 10, 68), (60, 60, 70), 1)

        if self._mode == 'config':
            self._draw_config(canvas, vw, vh)
        else:
            self._draw_output(canvas, vw, vh)

    def _draw_config(self, canvas, vw, vh):
        """Draw config entries and utility buttons."""
        y = 90
        line_h = 22
        max_lines = (vh - 180) // line_h

        # Config section
        cv2.putText(canvas, 'Configuration (from robot_config.yaml + settings.yaml):',
                    (20, y), FONT, 0.38, (160, 160, 160), 1)
        y += 10

        prev_section = None
        visible = 0
        for i, (section, key, val) in enumerate(self._config_entries):
            if i < self._scroll_offset:
                continue
            if visible >= max_lines:
                break

            y += line_h
            visible += 1

            # Section header
            if section and section != prev_section:
                cv2.putText(canvas, f'[{section}]', (25, y),
                            FONT, 0.36, (100, 180, 255), 1)
                y += line_h
                visible += 1
                prev_section = section

            # Key = value
            if self._editing and self._selected_field == i:
                # Edit mode
                display = f'  {key}: {self._edit_buffer}_'
                cv2.putText(canvas, display, (30, y),
                            FONT, 0.36, (255, 255, 100), 1)
            elif i == self._selected_field:
                display = f'  {key}: {val}'
                cv2.putText(canvas, display, (30, y),
                            FONT, 0.36, (255, 220, 150), 1)
                # Edit hint
                cv2.putText(canvas, '[Enter to edit]', (vw - 140, y),
                            FONT, 0.3, (100, 100, 100), 1)
            else:
                # Truncate long values
                val_str = str(val)
                if len(val_str) > 50:
                    val_str = val_str[:47] + '...'
                display = f'  {key}: {val_str}'
                cv2.putText(canvas, display, (30, y),
                            FONT, 0.36, (200, 200, 200), 1)

        # Scroll indicators
        if self._scroll_offset > 0:
            cv2.putText(canvas, '  ^ scroll up (PgUp)', (20, 100),
                        FONT, 0.3, (100, 100, 100), 1)
        if self._scroll_offset + max_lines < len(self._config_entries) + 10:
            cv2.putText(canvas, '  v scroll down (PgDn)',
                        (20, vh - 100), FONT, 0.3, (100, 100, 100), 1)

        # Utility buttons at bottom
        btn_y = vh - 80
        cv2.line(canvas, (10, btn_y - 10), (vw - 10, btn_y - 10),
                 (60, 60, 70), 1)
        cv2.putText(canvas, 'Utilities:', (20, btn_y),
                    FONT, 0.4, (160, 160, 160), 1)
        btn_y += 8

        bx = 20
        for i, (name, _script, _args) in enumerate(self._utilities):
            tw = cv2.getTextSize(name, FONT, 0.38, 1)[0][0] + 20
            bx2 = bx + tw
            by2 = btn_y + 28
            cv2.rectangle(canvas, (bx, btn_y), (bx2, by2), (60, 50, 70), -1)
            cv2.rectangle(canvas, (bx, btn_y), (bx2, by2), (100, 100, 100), 1)
            cv2.putText(canvas, name, (bx + 10, by2 - 8),
                        FONT, 0.38, (220, 220, 220), 1)
            self._buttons.append((bx, btn_y, bx2, by2, f'utility_{i}'))
            bx = bx2 + 10

        # Help text
        cv2.putText(canvas, 'Up/Down=select  Enter=edit  PgUp/PgDn=scroll  Click sidebar=switch view',
                    (20, vh - 10), FONT, 0.3, (80, 80, 80), 1)

    def _draw_output(self, canvas, vw, vh):
        """Draw utility script output."""
        y = 90
        cv2.putText(canvas, 'Utility Output:', (20, y),
                    FONT, 0.4, (160, 160, 160), 1)
        y += 10

        if self._utility_running:
            cv2.putText(canvas, 'Running...', (20, y + 20),
                        FONT, 0.4, (0, 200, 255), 1)

        # Show output lines
        lines = self._utility_output.split('\n')
        line_h = 18
        max_lines = (vh - 140) // line_h
        start = max(0, len(lines) - max_lines)
        for i, line in enumerate(lines[start:start + max_lines]):
            y += line_h
            text = line[:80]  # truncate long lines
            cv2.putText(canvas, text, (25, y),
                        FONT, 0.33, (200, 200, 200), 1)

        # Back button
        bx, by = 20, vh - 50
        bx2, by2 = 160, by + 30
        cv2.rectangle(canvas, (bx, by), (bx2, by2), (60, 50, 70), -1)
        cv2.putText(canvas, 'Back (Esc)', (bx + 10, by2 - 8),
                    FONT, 0.38, (220, 220, 220), 1)
        self._buttons.append((bx, by, bx2, by2, 'back'))

    def handle_key(self, key):
        if self._mode == 'output':
            if key == 27:  # ESC
                self._mode = 'config'
                return True
            return False

        if self._editing:
            return self._handle_edit_key(key)

        # Navigation
        if key == 82 or key == ord('k'):  # Up
            self._selected_field = max(0, self._selected_field - 1)
            return True
        if key == 84 or key == ord('j'):  # Down
            self._selected_field = min(
                len(self._config_entries) - 1, self._selected_field + 1)
            return True
        if key == 85:  # PgUp
            self._scroll_offset = max(0, self._scroll_offset - 10)
            return True
        if key == 86:  # PgDn
            self._scroll_offset += 10
            return True
        if key == 13:  # Enter - start editing
            if 0 <= self._selected_field < len(self._config_entries):
                section, k, v = self._config_entries[self._selected_field]
                self._editing = True
                self._edit_buffer = str(v)
                self._edit_key = k
                self._edit_section = section
            return True

        return False

    def _handle_edit_key(self, key):
        """Handle keys while editing a config value."""
        if key == 27:  # ESC - cancel
            self._editing = False
            return True
        if key == 13:  # Enter - save
            self._save_edit()
            self._editing = False
            return True
        if key == 8:  # Backspace
            self._edit_buffer = self._edit_buffer[:-1]
            return True
        if 32 <= key < 127:
            self._edit_buffer += chr(key)
            return True
        return True

    def _save_edit(self):
        """Save edited value to settings.yaml (local overrides)."""
        settings_path = config_path('settings.yaml')

        # Load existing settings
        settings = {}
        if os.path.exists(settings_path):
            with open(settings_path, 'r') as f:
                settings = yaml.safe_load(f) or {}

        # Parse value
        val = self._edit_buffer.strip()
        try:
            # Try numeric
            if '.' in val:
                parsed = float(val)
            else:
                parsed = int(val)
        except ValueError:
            if val.lower() in ('true', 'false'):
                parsed = val.lower() == 'true'
            else:
                parsed = val

        # Set in settings
        if self._edit_section:
            if self._edit_section not in settings:
                settings[self._edit_section] = {}
            settings[self._edit_section][self._edit_key] = parsed
        else:
            settings[self._edit_key] = parsed

        # Write
        with open(settings_path, 'w') as f:
            yaml.dump(settings, f, default_flow_style=False)

        # Reload config
        from config_loader import load_config
        self.app.config = load_config()
        self._load_config_display()

        # Update the entry in our list
        idx = self._selected_field
        if 0 <= idx < len(self._config_entries):
            s, k, _ = self._config_entries[idx]
            self._config_entries[idx] = (s, k, parsed)

        print(f"  Saved: {self._edit_section}.{self._edit_key} = {parsed}")

    def handle_mouse(self, event, x, y, flags):
        if event != cv2.EVENT_LBUTTONDOWN:
            return False

        for bx1, by1, bx2, by2, action in self._buttons:
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                if action == 'back':
                    self._mode = 'config'
                    return True
                if action.startswith('utility_'):
                    idx = int(action.split('_')[1])
                    self._run_utility(idx)
                    return True
        return False

    def _run_utility(self, idx):
        """Run a utility script in a background thread."""
        if self._utility_running:
            return
        name, script, extra_args = self._utilities[idx]
        script_path = os.path.join(_PROJECT_ROOT, script)
        self._utility_output = f"Running: {name}\n$ python3 {script} {' '.join(extra_args)}\n\n"
        self._utility_running = True
        self._mode = 'output'

        def _run():
            try:
                result = subprocess.run(
                    [sys.executable, script_path] + extra_args,
                    capture_output=True, text=True, timeout=30,
                    cwd=_PROJECT_ROOT)
                self._utility_output += result.stdout
                if result.stderr:
                    self._utility_output += '\n--- stderr ---\n' + result.stderr
                self._utility_output += f'\n\nExit code: {result.returncode}'
            except subprocess.TimeoutExpired:
                self._utility_output += '\n\nERROR: Timeout (30s)'
            except Exception as e:
                self._utility_output += f'\n\nERROR: {e}'
            self._utility_running = False

        self._utility_thread = threading.Thread(target=_run, daemon=True)
        self._utility_thread.start()
