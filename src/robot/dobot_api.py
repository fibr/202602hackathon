"""TCP/IP driver for Dobot Nova5 robot arm (firmware 4.6.2).

Dual-port architecture:
  - Port 29999 (dashboard): state queries, enable/disable, jog, IK/FK, gripper
  - Port 30003 (motion):    MovJ, MovL coordinated motion commands

Port 30003 requires the ROS2 driver to be running:
    docker compose --profile dobot up -d

MovJ/MovL on port 29999 return error -7. This is a firmware limitation,
not a syntax issue — neither V3 nor V4 command syntax works on dashboard.

Motion commands are fire-and-forget: they return immediately, completion
is detected by polling joint stability via dashboard.
"""

import socket
import time
import numpy as np
from dataclasses import dataclass


# Response code meanings (observed on 4.6.2)
RC_OK = 0
RC_UNKNOWN_CMD = -10000
RC_INVALID_PARAM = -20000
RC_JOG_INVALID = -6
RC_NOT_ON_DASHBOARD = -7     # MovJ/MovL sent to port 29999 instead of 30003


@dataclass
class RobotState:
    """Current robot state."""
    joints: np.ndarray = None       # 6 joint angles in degrees
    tcp_pose: np.ndarray = None     # [x, y, z, rx, ry, rz] mm/degrees
    enabled: bool = False
    mode: int = 0


class DobotNova5:
    """Driver for Dobot Nova5 6-axis robot arm over TCP/IP.

    Dashboard port 29999 handles state queries, enable/disable, jog, IK/FK.
    Motion port 30003 handles MovJ/MovL (requires ROS2 driver).
    """

    JOG_AXES = ["J1", "J2", "J3", "J4", "J5", "J6"]

    def __init__(self, ip: str = "192.168.5.1",
                 dashboard_port: int = 29999,
                 motion_port: int = 30003):
        self.ip = ip
        self.dashboard_port = dashboard_port
        self.motion_port = motion_port
        self._dash_sock = None
        self._motion_sock = None
        self.state = RobotState()

    def connect(self):
        """Connect to dashboard and motion ports.

        Raises ConnectionError if the motion port (30003) is not available.
        The ROS2 driver must be running: docker compose --profile dobot up -d
        """
        # Dashboard port (always available)
        self._dash_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._dash_sock.settimeout(5.0)
        self._dash_sock.connect((self.ip, self.dashboard_port))
        try:
            self._dash_sock.recv(1024)
        except socket.timeout:
            pass

        # Motion port (requires ROS2 driver)
        try:
            self._motion_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._motion_sock.settimeout(5.0)
            self._motion_sock.connect((self.ip, self.motion_port))
            try:
                self._motion_sock.recv(1024)
            except socket.timeout:
                pass
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            if self._motion_sock:
                self._motion_sock.close()
                self._motion_sock = None
            raise ConnectionError(
                f"Cannot connect to motion port {self.motion_port}. "
                f"Is the ROS2 driver running? "
                f"Start it with: docker compose --profile dobot up -d"
            ) from e

    def _send_dash(self, cmd: str) -> str:
        """Send a command on the dashboard port and return raw response."""
        self._dash_sock.send(f"{cmd}\n".encode())
        time.sleep(0.1)
        try:
            return self._dash_sock.recv(4096).decode().strip()
        except socket.timeout:
            return ""

    def _send_motion(self, cmd: str) -> str:
        """Send a command on the motion port and return raw response."""
        self._motion_sock.send(f"{cmd}\n".encode())
        time.sleep(0.1)
        try:
            return self._motion_sock.recv(4096).decode().strip()
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
        self._send_dash("DisableRobot()")
        time.sleep(1)
        self._send_dash("ClearError()")
        resp = self._send_dash("EnableRobot()")
        time.sleep(1)
        code, _ = self._parse_response(resp)
        self.state.enabled = (code == RC_OK)
        return code == RC_OK

    def disable(self):
        """Disable the robot."""
        resp = self._send_dash("DisableRobot()")
        self.state.enabled = False
        return resp

    def clear_error(self):
        """Clear any alarm/error state."""
        return self._send_dash("ClearError()")

    def get_mode(self) -> int:
        """Get robot mode. 5=enabled, 6=backdrive, 9=error, etc."""
        resp = self._send_dash("RobotMode()")
        _, val = self._parse_response(resp)
        try:
            self.state.mode = int(val)
        except ValueError:
            pass
        return self.state.mode

    def get_errors(self) -> str:
        """Get active error IDs."""
        resp = self._send_dash("GetErrorID()")
        _, val = self._parse_response(resp)
        return val

    def set_speed(self, speed_percent: int):
        """Set robot speed as percentage (1-100)."""
        return self._send_dash(f"SpeedFactor({speed_percent})")

    # --- State queries ---

    def get_pose(self) -> np.ndarray:
        """Get current TCP pose [x, y, z, rx, ry, rz] in mm/degrees."""
        resp = self._send_dash("GetPose()")
        pose = self._parse_numbers(resp)
        if len(pose) >= 6:
            self.state.tcp_pose = pose[:6]
        return self.state.tcp_pose if self.state.tcp_pose is not None else np.zeros(6)

    def get_joint_angles(self) -> np.ndarray:
        """Get current joint angles [j1..j6] in degrees."""
        resp = self._send_dash("GetAngle()")
        angles = self._parse_numbers(resp)
        if len(angles) >= 6:
            self.state.joints = angles[:6]
        return self.state.joints if self.state.joints is not None else np.zeros(6)

    # --- Kinematics ---

    def inverse_kin(self, x: float, y: float, z: float,
                    rx: float, ry: float, rz: float) -> np.ndarray:
        """Inverse kinematics: Cartesian pose -> joint angles.

        Args:
            x, y, z: Position in mm
            rx, ry, rz: Orientation in degrees

        Returns:
            Array of 6 joint angles in degrees, or zeros on failure.
        """
        resp = self._send_dash(f"InverseKin({x},{y},{z},{rx},{ry},{rz})")
        return self._parse_numbers(resp)

    def forward_kin(self, j1: float, j2: float, j3: float,
                    j4: float, j5: float, j6: float) -> np.ndarray:
        """Forward kinematics: joint angles -> Cartesian pose.

        Args:
            j1..j6: Joint angles in degrees

        Returns:
            Array [x, y, z, rx, ry, rz] in mm/degrees, or zeros on failure.
        """
        resp = self._send_dash(f"PositiveKin({j1},{j2},{j3},{j4},{j5},{j6})")
        return self._parse_numbers(resp)

    # --- Motion completion ---

    def _wait_motion_complete(self, timeout: float = 30.0,
                              stable_threshold: int = 3,
                              poll_interval: float = 0.2) -> bool:
        """Wait for motion to complete by polling joint angle stability.

        Considers motion complete when joint angles change by less than
        0.05 degrees for stable_threshold consecutive polls.
        """
        elapsed = 0.0
        prev_joints = None
        stable_count = 0

        while elapsed < timeout:
            time.sleep(poll_interval)
            elapsed += poll_interval

            current = self.get_joint_angles()
            if current is None:
                continue

            if prev_joints is not None:
                max_diff = np.max(np.abs(current - prev_joints))
                if max_diff < 0.05:
                    stable_count += 1
                    if stable_count >= stable_threshold:
                        return True
                else:
                    stable_count = 0

            prev_joints = current.copy()

        return False

    # --- Jog motion (dashboard port) ---

    def jog_start(self, axis: str):
        """Start jogging an axis. axis must be e.g. 'J1+', 'J3-' (uppercase).

        Valid axes: J1+, J1-, J2+, J2-, ..., J6+, J6-
        """
        return self._send_dash(f"MoveJog({axis})")

    def jog_stop(self):
        """Stop any active jog motion."""
        return self._send_dash("MoveJog()")

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

    # --- Coordinated motion (motion port 30003) ---

    def movl(self, x: float, y: float, z: float,
             rx: float, ry: float, rz: float,
             timeout: float = 30.0) -> bool:
        """Cartesian linear move.

        Args:
            x, y, z: Target position in mm
            rx, ry, rz: Target orientation in degrees
            timeout: Max wait for completion

        Returns:
            True if motion completed, False on failure.
        """
        cmd = f"MovL({x},{y},{z},{rx},{ry},{rz})"
        resp = self._send_motion(cmd)
        code, _ = self._parse_response(resp)
        if code != RC_OK:
            return False
        return self._wait_motion_complete(timeout)

    def movj(self, x: float, y: float, z: float,
             rx: float, ry: float, rz: float,
             timeout: float = 30.0) -> bool:
        """Joint-space move to Cartesian pose.

        Args:
            x, y, z: Target position in mm
            rx, ry, rz: Target orientation in degrees
            timeout: Max wait for completion

        Returns:
            True if motion completed, False on failure.
        """
        cmd = f"MovJ({x},{y},{z},{rx},{ry},{rz})"
        resp = self._send_motion(cmd)
        code, _ = self._parse_response(resp)
        if code != RC_OK:
            return False
        return self._wait_motion_complete(timeout)

    def movj_joints(self, j1: float, j2: float, j3: float,
                    j4: float, j5: float, j6: float,
                    timeout: float = 30.0) -> bool:
        """Joint-space move to joint angles.

        Args:
            j1..j6: Target joint angles in degrees
            timeout: Max wait for completion

        Returns:
            True if motion completed, False on failure.
        """
        cmd = f"MovJ({j1},{j2},{j3},{j4},{j5},{j6})"
        resp = self._send_motion(cmd)
        code, _ = self._parse_response(resp)
        if code != RC_OK:
            return False
        return self._wait_motion_complete(timeout)

    def move_linear(self, x: float, y: float, z: float,
                    rx: float, ry: float, rz: float,
                    speed_percent: int = 30,
                    timeout: float = 30.0) -> bool:
        """Move TCP in a straight line to a Cartesian pose.

        Args:
            x, y, z: Target position in mm
            rx, ry, rz: Target orientation in degrees
            speed_percent: Speed (1-100)
            timeout: Max time for the move

        Returns:
            True if motion completed, False on failure.
        """
        self.set_speed(speed_percent)
        return self.movl(x, y, z, rx, ry, rz, timeout=timeout)

    def move_joint(self, x: float, y: float, z: float,
                   rx: float, ry: float, rz: float,
                   speed_percent: int = 30,
                   timeout: float = 30.0) -> bool:
        """Move to a Cartesian pose via joint-space motion.

        Args:
            x, y, z: Target position in mm
            rx, ry, rz: Target orientation in degrees
            speed_percent: Speed (1-100)
            timeout: Max time for the move

        Returns:
            True if motion completed, False on failure.
        """
        self.set_speed(speed_percent)
        return self.movj(x, y, z, rx, ry, rz, timeout=timeout)

    # --- ToolDO (gripper, dashboard port) ---

    def tool_do(self, index: int, status: int):
        """Set a tool digital output (ToolDO). May require running mode."""
        resp = self._send_dash(f"ToolDO({index},{status})")
        code, _ = self._parse_response(resp)
        return code == RC_OK

    def tool_do_instant(self, index: int, status: int):
        """Set a tool digital output immediately (ToolDOInstant).

        Works in enabled-idle mode (no motion queue needed).

        Electric gripper dual-channel control:
            Close: tool_do_instant(2, 0) then tool_do_instant(1, 1)
            Open:  tool_do_instant(1, 0) then tool_do_instant(2, 1)

        Args:
            index: ToolDO index (1 or 2)
            status: 0 or 1
        """
        resp = self._send_dash(f"ToolDOInstant({index},{status})")
        code, _ = self._parse_response(resp)
        return code == RC_OK

    # --- Connection lifecycle ---

    def disconnect(self):
        """Close all connections."""
        for sock in [self._dash_sock, self._motion_sock]:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass
        self._dash_sock = None
        self._motion_sock = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
