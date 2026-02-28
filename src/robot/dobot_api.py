"""TCP/IP driver for Dobot Nova5 robot arm (firmware 4.6.2).

Dashboard-only driver. All commands go through port 29999.
Motion via MoveJog (joint-space jog with uppercase axis names).
Gripper via ToolDO (dual-solenoid pneumatic).

Tested protocol behavior on Nova5 4.6.2:
  - MoveJog(J1+) .. MoveJog(J6-): joint jog, uppercase only
  - MoveJog(): stop jog
  - MovJ/MovL: return -30001 (need port 30003 / ROS2 driver)
  - Cartesian jog (z+/Z+): silently ignored or error -6
  - ToolDO(index, status): works for gripper
  - DO(port, val): accepted but no visible effect on this gripper
  - Response format: "code,{value},CommandName();"
"""

import socket
import time
import numpy as np
from dataclasses import dataclass, field


# Response code meanings (observed on 4.6.2)
RC_OK = 0
RC_UNKNOWN_CMD = -10000
RC_NO_MOTION_PORT = -30001
RC_INVALID_PARAM = -20000
RC_JOG_INVALID = -6


@dataclass
class RobotState:
    """Current robot state."""
    joints: np.ndarray = None       # 6 joint angles in degrees
    tcp_pose: np.ndarray = None     # [x, y, z, rx, ry, rz] mm/degrees
    enabled: bool = False
    mode: int = 0


class DobotNova5:
    """Driver for Dobot Nova5 6-axis robot arm over TCP/IP.

    Uses only the dashboard port (29999). Motion is via joint-space jog
    commands, which is the only motion method that works without the
    ROS2 driver providing port 30003.
    """

    # Valid jog axis names (must be uppercase)
    JOG_AXES = ["J1", "J2", "J3", "J4", "J5", "J6"]

    def __init__(self, ip: str = "192.168.5.1",
                 dashboard_port: int = 29999):
        self.ip = ip
        self.dashboard_port = dashboard_port
        self._sock = None
        self.state = RobotState()

    def connect(self):
        """Connect to the dashboard port."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(5.0)
        self._sock.connect((self.ip, self.dashboard_port))
        # Read welcome banner
        try:
            self._sock.recv(1024)
        except socket.timeout:
            pass

    def _send(self, cmd: str) -> str:
        """Send a command and return raw response string."""
        self._sock.send(f"{cmd}\n".encode())
        time.sleep(0.1)
        try:
            return self._sock.recv(4096).decode().strip()
        except socket.timeout:
            return ""

    def _parse_response(self, resp: str) -> tuple[int, str]:
        """Parse 'code,{value},cmd;' -> (code, value_string)."""
        try:
            code = int(resp.split(",")[0])
            inner = resp.split("{")[1].split("}")[0] if "{" in resp else ""
            return code, inner
        except (ValueError, IndexError):
            return -1, resp

    def _parse_numbers(self, resp: str) -> np.ndarray:
        """Extract numeric values from response '{n1,n2,...}'."""
        _, inner = self._parse_response(resp)
        try:
            return np.array([float(x) for x in inner.split(",")])
        except (ValueError, IndexError):
            return np.zeros(6)

    # --- Lifecycle ---

    def enable(self):
        """Enable the robot. Includes disable/enable cycle for reliable startup."""
        self._send("DisableRobot()")
        time.sleep(1)
        self._send("ClearError()")
        resp = self._send("EnableRobot()")
        time.sleep(1)
        code, _ = self._parse_response(resp)
        self.state.enabled = (code == RC_OK)
        return code == RC_OK

    def disable(self):
        """Disable the robot."""
        resp = self._send("DisableRobot()")
        self.state.enabled = False
        return resp

    def clear_error(self):
        """Clear any alarm/error state."""
        return self._send("ClearError()")

    def get_mode(self) -> int:
        """Get robot mode. 5=enabled, 6=backdrive, 9=error, etc."""
        resp = self._send("RobotMode()")
        _, val = self._parse_response(resp)
        try:
            self.state.mode = int(val)
        except ValueError:
            pass
        return self.state.mode

    def get_errors(self) -> str:
        """Get active error IDs."""
        resp = self._send("GetErrorID()")
        _, val = self._parse_response(resp)
        return val

    def set_speed(self, speed_percent: int):
        """Set robot speed as percentage (1-100)."""
        return self._send(f"SpeedFactor({speed_percent})")

    # --- State queries ---

    def get_pose(self) -> np.ndarray:
        """Get current TCP pose [x, y, z, rx, ry, rz] in mm/degrees."""
        resp = self._send("GetPose()")
        pose = self._parse_numbers(resp)
        if len(pose) >= 6:
            self.state.tcp_pose = pose[:6]
        return self.state.tcp_pose if self.state.tcp_pose is not None else np.zeros(6)

    def get_joint_angles(self) -> np.ndarray:
        """Get current joint angles [j1..j6] in degrees."""
        resp = self._send("GetAngle()")
        angles = self._parse_numbers(resp)
        if len(angles) >= 6:
            self.state.joints = angles[:6]
        return self.state.joints if self.state.joints is not None else np.zeros(6)

    # --- Jog motion ---

    def jog_start(self, axis: str):
        """Start jogging an axis. axis must be e.g. 'J1+', 'J3-' (uppercase).

        Valid axes: J1+, J1-, J2+, J2-, ..., J6+, J6-
        """
        return self._send(f"MoveJog({axis})")

    def jog_stop(self):
        """Stop any active jog motion."""
        return self._send("MoveJog()")

    def jog_joint(self, joint: int, direction: str, duration: float):
        """Jog a single joint for a given duration.

        Args:
            joint: Joint number 1-6
            direction: '+' or '-'
            duration: Seconds to jog
        """
        axis = f"J{joint}{direction}"
        self.jog_start(axis)
        time.sleep(duration)
        self.jog_stop()
        time.sleep(0.3)  # settling time

    def move_to_joints(self, target: np.ndarray, speed_percent: int = 30,
                       tolerance_deg: float = 1.0, timeout: float = 30.0):
        """Move to target joint angles by jogging each joint.

        Jogs each joint individually toward the target. Not a coordinated
        move — joints move one at a time.

        Args:
            target: Array of 6 target joint angles in degrees
            speed_percent: Jog speed (1-100)
            tolerance_deg: Acceptable error per joint in degrees
            timeout: Max time for the entire move
        """
        self.set_speed(speed_percent)
        start_time = time.time()

        for joint_idx in range(6):
            joint_num = joint_idx + 1

            while time.time() - start_time < timeout:
                current = self.get_joint_angles()
                error = target[joint_idx] - current[joint_idx]

                if abs(error) <= tolerance_deg:
                    break

                direction = "+" if error > 0 else "-"
                # Short jog pulses for precision
                jog_time = min(0.5, abs(error) / 20.0)  # scale with distance
                jog_time = max(0.1, jog_time)

                self.jog_start(f"J{joint_num}{direction}")
                time.sleep(jog_time)
                self.jog_stop()
                time.sleep(0.2)

        return self.get_joint_angles()

    # --- ToolDO (gripper) ---

    def tool_do(self, index: int, status: int):
        """Set a tool digital output (ToolDO).

        For dual-solenoid pneumatic gripper:
            ToolDO(1, 1) = close gripper
            ToolDO(2, 1) = open gripper

        Args:
            index: ToolDO index (1 or 2)
            status: 0 or 1
        """
        resp = self._send(f"ToolDO({index},{status})")
        code, _ = self._parse_response(resp)
        return code == RC_OK

    # --- Connection lifecycle ---

    def disconnect(self):
        """Close the dashboard connection."""
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
