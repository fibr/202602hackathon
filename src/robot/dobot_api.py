"""TCP/IP driver for Dobot Nova5 robot arm (firmware 4.6.2).

Supports three motion modes (auto-detected on connect):
  1. "dashboard" — MovJ/MovL work on dashboard port 29999 (ROS2 driver running)
  2. "motion_port" — MovJ/MovL via separate motion port 30003
  3. "jog" — fallback to MoveJog pulses + InverseKin (no ROS2 driver)

Dashboard port (29999) is always used for state queries, gripper, enable/disable.
Motion commands are routed to the best available port.

Tested protocol behavior on Nova5 4.6.2:
  - MoveJog(J1+) .. MoveJog(J6-): joint jog, uppercase only
  - MoveJog(): stop jog
  - MovJ/MovL: return -30001 without ROS2 driver, work with it
  - ToolDOInstant(index, status): works for electric gripper
  - InverseKin / PositiveKin: work on dashboard
  - Response format: "code,{value},CommandName();"
"""

import socket
import time
import numpy as np
from dataclasses import dataclass


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

    Uses dashboard port 29999 for all commands. When the ROS2 driver is
    running (docker compose --profile dobot up), real MovJ/MovL become
    available for precise coordinated motion. Without it, falls back to
    jog-based motion.
    """

    # Valid jog axis names (must be uppercase)
    JOG_AXES = ["J1", "J2", "J3", "J4", "J5", "J6"]

    def __init__(self, ip: str = "192.168.5.1",
                 dashboard_port: int = 29999,
                 motion_port: int = 30003):
        self.ip = ip
        self.dashboard_port = dashboard_port
        self.motion_port = motion_port
        self._sock = None
        self._motion_sock = None
        self._motion_mode = "jog"  # "dashboard", "motion_port", or "jog"
        self.state = RobotState()

    @property
    def motion_mode(self) -> str:
        """Current motion mode: 'dashboard', 'motion_port', or 'jog'."""
        return self._motion_mode

    def connect(self):
        """Connect to dashboard port and probe for motion support."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(5.0)
        self._sock.connect((self.ip, self.dashboard_port))
        # Read welcome banner
        try:
            self._sock.recv(1024)
        except socket.timeout:
            pass
        self._probe_motion_support()

    def _probe_motion_support(self):
        """Test whether MovJ/MovL are available and set motion mode."""
        # Strategy 1: Try MovJ on dashboard with current joint angles (no-op move)
        current = self.get_joint_angles()
        if current is not None and not np.allclose(current, 0):
            j = current
            resp = self._send(
                f"MovJ({j[0]:.2f},{j[1]:.2f},{j[2]:.2f},"
                f"{j[3]:.2f},{j[4]:.2f},{j[5]:.2f})")
            code, _ = self._parse_response(resp)
            if code == RC_OK:
                self._motion_mode = "dashboard"
                return

        # Strategy 2: Try connecting to motion port directly
        try:
            self._motion_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._motion_sock.settimeout(2.0)
            self._motion_sock.connect((self.ip, self.motion_port))
            try:
                self._motion_sock.recv(1024)  # Read banner
            except socket.timeout:
                pass
            self._motion_mode = "motion_port"
            return
        except (ConnectionRefusedError, socket.timeout, OSError):
            if self._motion_sock:
                self._motion_sock.close()
                self._motion_sock = None

        # Fallback: jog-based motion
        self._motion_mode = "jog"

    def _send(self, cmd: str) -> str:
        """Send a command on the dashboard port and return raw response."""
        self._sock.send(f"{cmd}\n".encode())
        time.sleep(0.1)
        try:
            return self._sock.recv(4096).decode().strip()
        except socket.timeout:
            return ""

    def _send_motion(self, cmd: str) -> str:
        """Send a motion command to the best available port."""
        if self._motion_mode == "motion_port" and self._motion_sock:
            self._motion_sock.send(f"{cmd}\n".encode())
            time.sleep(0.1)
            try:
                return self._motion_sock.recv(4096).decode().strip()
            except socket.timeout:
                return ""
        else:
            return self._send(cmd)

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
        resp = self._send(f"InverseKin({x},{y},{z},{rx},{ry},{rz})")
        return self._parse_numbers(resp)

    def forward_kin(self, j1: float, j2: float, j3: float,
                    j4: float, j5: float, j6: float) -> np.ndarray:
        """Forward kinematics: joint angles -> Cartesian pose.

        Args:
            j1..j6: Joint angles in degrees

        Returns:
            Array [x, y, z, rx, ry, rz] in mm/degrees, or zeros on failure.
        """
        resp = self._send(f"PositiveKin({j1},{j2},{j3},{j4},{j5},{j6})")
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

    # --- Cartesian motion ---

    def movl(self, x: float, y: float, z: float,
             rx: float, ry: float, rz: float,
             timeout: float = 30.0) -> bool:
        """Cartesian linear move via firmware MovL.

        Requires ROS2 driver. Returns False if unavailable or command fails.
        Units: mm for xyz, degrees for rx/ry/rz.
        """
        if self._motion_mode == "jog":
            return False
        resp = self._send_motion(f"MovL({x},{y},{z},{rx},{ry},{rz})")
        code, _ = self._parse_response(resp)
        if code != RC_OK:
            return False
        return self._wait_motion_complete(timeout)

    def movj(self, x: float, y: float, z: float,
             rx: float, ry: float, rz: float,
             timeout: float = 30.0) -> bool:
        """Joint-space move to Cartesian pose via firmware MovJ.

        Requires ROS2 driver. Returns False if unavailable or command fails.
        Units: mm for xyz, degrees for rx/ry/rz.
        """
        if self._motion_mode == "jog":
            return False
        resp = self._send_motion(f"MovJ({x},{y},{z},{rx},{ry},{rz})")
        code, _ = self._parse_response(resp)
        if code != RC_OK:
            return False
        return self._wait_motion_complete(timeout)

    def move_linear(self, x: float, y: float, z: float,
                    rx: float, ry: float, rz: float,
                    speed_percent: int = 30,
                    tolerance_deg: float = 1.0,
                    timeout: float = 30.0) -> bool:
        """Move TCP in a straight line to a Cartesian pose.

        Uses MovL if ROS2 driver is running, otherwise falls back to
        InverseKin + jog-based motion.

        Args:
            x, y, z: Target position in mm
            rx, ry, rz: Target orientation in degrees
            speed_percent: Jog speed (1-100)
            tolerance_deg: Acceptable joint error in degrees
            timeout: Max time for the move

        Returns:
            True if motion executed, False on failure.
        """
        self.set_speed(speed_percent)

        # Try real MovL first
        if self._motion_mode != "jog":
            if self.movl(x, y, z, rx, ry, rz, timeout):
                return True

        # Fallback: IK + jog
        target_joints = self.inverse_kin(x, y, z, rx, ry, rz)
        if target_joints is None or np.allclose(target_joints, 0):
            return False
        self.move_to_joints(target_joints, speed_percent, tolerance_deg, timeout)
        return True

    def move_joint(self, x: float, y: float, z: float,
                   rx: float, ry: float, rz: float,
                   speed_percent: int = 30,
                   tolerance_deg: float = 1.0,
                   timeout: float = 30.0) -> bool:
        """Move to a Cartesian pose via joint-space motion.

        Uses MovJ if ROS2 driver is running, otherwise falls back to
        InverseKin + jog-based motion.

        Args:
            x, y, z: Target position in mm
            rx, ry, rz: Target orientation in degrees
            speed_percent: Jog speed (1-100)
            tolerance_deg: Acceptable joint error in degrees
            timeout: Max time for the move

        Returns:
            True if motion executed, False on failure.
        """
        self.set_speed(speed_percent)

        # Try real MovJ first
        if self._motion_mode != "jog":
            if self.movj(x, y, z, rx, ry, rz, timeout):
                return True

        # Fallback: IK + jog
        target_joints = self.inverse_kin(x, y, z, rx, ry, rz)
        if target_joints is None or np.allclose(target_joints, 0):
            return False
        self.move_to_joints(target_joints, speed_percent, tolerance_deg, timeout)
        return True

    # --- ToolDO (gripper) ---

    def tool_do(self, index: int, status: int):
        """Set a tool digital output (ToolDO). May require running mode."""
        resp = self._send(f"ToolDO({index},{status})")
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
        resp = self._send(f"ToolDOInstant({index},{status})")
        code, _ = self._parse_response(resp)
        return code == RC_OK

    # --- Connection lifecycle ---

    def disconnect(self):
        """Close all connections."""
        for sock in [self._sock, self._motion_sock]:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass
        self._sock = None
        self._motion_sock = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
