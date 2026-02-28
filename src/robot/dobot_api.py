"""TCP/IP driver for Dobot Nova5 robot arm (firmware V4, e.g. 4.6.2).

All commands go through dashboard port 29999 using V4 named-parameter syntax.

V4 motion command syntax (different from V3!):
  MovJ(pose={x,y,z,rx,ry,rz})        — joint-space move to Cartesian pose
  MovL(pose={x,y,z,rx,ry,rz})        — linear move to Cartesian pose
  MovJ(joint={j1,j2,j3,j4,j5,j6})   — joint-space move to joint angles

V3 syntax MovL(x,y,z,...) returns -30001 on this firmware.

Motion commands are fire-and-forget: they return immediately, completion
is detected by polling joint stability.
"""

import socket
import time
import numpy as np
from dataclasses import dataclass
from logger import get_logger

log = get_logger('robot')

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

    Uses dashboard port 29999 with V4 command syntax for all operations
    including MovJ/MovL motion commands.
    """

    JOG_AXES = ["J1", "J2", "J3", "J4", "J5", "J6"]

    def __init__(self, ip: str = "192.168.5.1",
                 dashboard_port: int = 29999):
        self.ip = ip
        self.dashboard_port = dashboard_port
        self._sock = None
        self.state = RobotState()

    def connect(self):
        """Connect to the dashboard port."""
        log.info(f"Connecting to {self.ip}:{self.dashboard_port}")
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(5.0)
        self._sock.connect((self.ip, self.dashboard_port))
        try:
            self._sock.recv(1024)
        except socket.timeout:
            pass
        log.debug("Connected")

    def _send(self, cmd: str) -> str:
        """Send a command and return raw response string."""
        log.debug(f"TX> {cmd}")
        self._sock.send(f"{cmd}\n".encode())
        time.sleep(0.1)
        try:
            resp = self._sock.recv(4096).decode().strip()
            log.debug(f"RX< {resp}")
            return resp
        except socket.timeout:
            log.warning(f"Timeout: {cmd}")
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

    # --- Kinematics ---

    def inverse_kin(self, x: float, y: float, z: float,
                    rx: float, ry: float, rz: float) -> np.ndarray:
        """Inverse kinematics: Cartesian pose -> joint angles."""
        resp = self._send(f"InverseKin({x},{y},{z},{rx},{ry},{rz})")
        return self._parse_numbers(resp)

    def forward_kin(self, j1: float, j2: float, j3: float,
                    j4: float, j5: float, j6: float) -> np.ndarray:
        """Forward kinematics: joint angles -> Cartesian pose."""
        resp = self._send(f"PositiveKin({j1},{j2},{j3},{j4},{j5},{j6})")
        return self._parse_numbers(resp)

    # --- Motion completion ---

    def _wait_motion_complete(self, timeout: float = 30.0,
                              stable_threshold: int = 3,
                              poll_interval: float = 0.2) -> bool:
        """Wait for motion to complete by polling joint angle stability."""
        elapsed = 0.0
        prev_joints = None
        stable_count = 0

        while elapsed < timeout:
            time.sleep(poll_interval)
            elapsed += poll_interval

            current = self.get_joint_angles()
            if current is None or np.allclose(current, 0):
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
        """Start jogging an axis. axis must be e.g. 'J1+', 'J3-' (uppercase)."""
        return self._send(f"MoveJog({axis})")

    def jog_stop(self):
        """Stop any active jog motion."""
        return self._send("MoveJog()")

    def jog_joint(self, joint: int, direction: str, duration: float):
        """Jog a single joint for a given duration."""
        axis = f"J{joint}{direction}"
        self.jog_start(axis)
        time.sleep(duration)
        self.jog_stop()
        time.sleep(0.3)

    # --- Cartesian motion (V4 syntax) ---

    def movl(self, x: float, y: float, z: float,
             rx: float, ry: float, rz: float,
             timeout: float = 30.0) -> bool:
        """Cartesian linear move via V4 MovL."""
        cmd = f"MovL(pose={{{x},{y},{z},{rx},{ry},{rz}}})"
        resp = self._send(cmd)
        code, _ = self._parse_response(resp)
        if code != RC_OK:
            log.error(f"MovL failed (code={code}): {resp}")
            return False
        ok = self._wait_motion_complete(timeout)
        if not ok:
            log.warning(f"MovL timeout after {timeout}s")
        return ok

    def movj(self, x: float, y: float, z: float,
             rx: float, ry: float, rz: float,
             timeout: float = 30.0) -> bool:
        """Joint-space move to Cartesian pose via V4 MovJ."""
        cmd = f"MovJ(pose={{{x},{y},{z},{rx},{ry},{rz}}})"
        resp = self._send(cmd)
        code, _ = self._parse_response(resp)
        if code != RC_OK:
            log.error(f"MovJ failed (code={code}): {resp}")
            return False
        ok = self._wait_motion_complete(timeout)
        if not ok:
            log.warning(f"MovJ timeout after {timeout}s")
        return ok

    def movj_joints(self, j1: float, j2: float, j3: float,
                    j4: float, j5: float, j6: float,
                    timeout: float = 30.0) -> bool:
        """Joint-space move to joint angles via V4 MovJ."""
        cmd = f"MovJ(joint={{{j1},{j2},{j3},{j4},{j5},{j6}}})"
        resp = self._send(cmd)
        code, _ = self._parse_response(resp)
        if code != RC_OK:
            log.error(f"MovJ(joint) failed (code={code}): {resp}")
            return False
        ok = self._wait_motion_complete(timeout)
        if not ok:
            log.warning(f"MovJ(joint) timeout after {timeout}s")
        return ok

    def move_linear(self, x: float, y: float, z: float,
                    rx: float, ry: float, rz: float,
                    speed_percent: int = 30,
                    timeout: float = 30.0) -> bool:
        """Move TCP in a straight line to a Cartesian pose."""
        self.set_speed(speed_percent)
        return self.movl(x, y, z, rx, ry, rz, timeout=timeout)

    def move_joint(self, x: float, y: float, z: float,
                   rx: float, ry: float, rz: float,
                   speed_percent: int = 30,
                   timeout: float = 30.0) -> bool:
        """Move to a Cartesian pose via joint-space motion."""
        self.set_speed(speed_percent)
        return self.movj(x, y, z, rx, ry, rz, timeout=timeout)

    # --- ToolDO (gripper) ---

    def tool_do(self, index: int, status: int):
        """Set a tool digital output (ToolDO). May require running mode."""
        resp = self._send(f"ToolDO({index},{status})")
        code, _ = self._parse_response(resp)
        return code == RC_OK

    def tool_do_instant(self, index: int, status: int):
        """Set a tool digital output immediately (ToolDOInstant).

        Electric gripper dual-channel control:
            Close: tool_do_instant(2, 0) then tool_do_instant(1, 1)
            Open:  tool_do_instant(1, 0) then tool_do_instant(2, 1)
        """
        resp = self._send(f"ToolDOInstant({index},{status})")
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
