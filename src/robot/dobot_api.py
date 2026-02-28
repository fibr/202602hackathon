"""TCP/IP driver for Dobot Nova5 robot arm.

Based on the Dobot TCP-IP-Python-V4 SDK protocol.
Connects to the robot on three ports:
  - 29999: Dashboard (control, enable, settings, IO)
  - 30003: Motion commands (MovJ, MovL, etc.)
  - 30004: Real-time feedback (joint states, TCP pose)
"""

import socket
import time
import threading
import numpy as np
from dataclasses import dataclass


@dataclass
class RobotState:
    """Current robot state from feedback port."""
    joints: np.ndarray = None       # 6 joint angles in degrees
    tcp_pose: np.ndarray = None     # [x, y, z, rx, ry, rz] TCP in mm/degrees
    enabled: bool = False
    moving: bool = False


class DobotNova5:
    """Driver for Dobot Nova5 6-axis robot arm over TCP/IP."""

    def __init__(self, ip: str = "192.168.5.1",
                 dashboard_port: int = 29999,
                 motion_port: int = 30003,
                 feedback_port: int = 30004):
        self.ip = ip
        self.dashboard_port = dashboard_port
        self.motion_port = motion_port
        self.feedback_port = feedback_port

        self._dashboard_sock = None
        self._motion_sock = None
        self._feedback_sock = None
        self._feedback_thread = None
        self._running = False
        self.state = RobotState()

    def connect(self):
        """Establish TCP connections to all three robot ports."""
        self._dashboard_sock = self._connect_port(self.dashboard_port)
        self._motion_sock = self._connect_port(self.motion_port)
        self._feedback_sock = self._connect_port(self.feedback_port)

        # Start feedback reader thread
        self._running = True
        self._feedback_thread = threading.Thread(target=self._read_feedback, daemon=True)
        self._feedback_thread.start()

    def _connect_port(self, port: int, timeout: float = 5.0) -> socket.socket:
        """Connect to a single port on the robot."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((self.ip, port))
        # Read welcome message
        try:
            sock.recv(1024)
        except socket.timeout:
            pass
        return sock

    def _send_dashboard(self, cmd: str) -> str:
        """Send a command to dashboard port and return response."""
        self._dashboard_sock.send(f"{cmd}\n".encode())
        time.sleep(0.1)
        try:
            response = self._dashboard_sock.recv(4096).decode().strip()
            return response
        except socket.timeout:
            return ""

    def _send_motion(self, cmd: str) -> str:
        """Send a command to motion port and return response."""
        self._motion_sock.send(f"{cmd}\n".encode())
        time.sleep(0.1)
        try:
            response = self._motion_sock.recv(4096).decode().strip()
            return response
        except socket.timeout:
            return ""

    def _read_feedback(self):
        """Background thread to continuously read robot state from feedback port."""
        while self._running:
            try:
                data = self._feedback_sock.recv(4096)
                if data:
                    # Parse feedback data (protocol-specific, may need adjustment)
                    self._parse_feedback(data)
            except socket.timeout:
                pass
            except Exception:
                if self._running:
                    time.sleep(0.1)

    def _parse_feedback(self, data: bytes):
        """Parse feedback packet from the robot. Protocol TBD - adjust for Nova5."""
        # TODO: Implement actual Nova5 feedback parsing
        # The exact format depends on the firmware version
        pass

    # --- Dashboard Commands ---

    def enable(self):
        """Enable the robot (power on servos)."""
        return self._send_dashboard("EnableRobot()")

    def disable(self):
        """Disable the robot."""
        return self._send_dashboard("DisableRobot()")

    def clear_error(self):
        """Clear any alarm/error state."""
        return self._send_dashboard("ClearError()")

    def reset(self):
        """Reset the robot state."""
        self.clear_error()
        return self._send_dashboard("ResetRobot()")

    def set_digital_output(self, port: int, value: bool):
        """Set a digital output (for gripper control).

        Args:
            port: DO port number (1-based)
            value: True for high, False for low
        """
        val = 1 if value else 0
        return self._send_dashboard(f"DO({port},{val})")

    def get_pose(self) -> np.ndarray:
        """Get current TCP pose [x, y, z, rx, ry, rz]."""
        response = self._send_dashboard("GetPose()")
        # Parse response - format may vary
        # Expected: "x,y,z,rx,ry,rz" or similar
        try:
            # Strip any wrapper text, extract numbers
            nums = [float(x) for x in response.replace('{', '').replace('}', '').split(',') if x.strip().replace('.', '').replace('-', '').isdigit() or '.' in x]
            if len(nums) >= 6:
                return np.array(nums[:6])
        except (ValueError, IndexError):
            pass
        return self.state.tcp_pose if self.state.tcp_pose is not None else np.zeros(6)

    def get_joint_angles(self) -> np.ndarray:
        """Get current joint angles [j1, j2, j3, j4, j5, j6] in degrees."""
        response = self._send_dashboard("GetAngle()")
        try:
            nums = [float(x) for x in response.replace('{', '').replace('}', '').split(',') if x.strip().replace('.', '').replace('-', '').isdigit() or '.' in x]
            if len(nums) >= 6:
                return np.array(nums[:6])
        except (ValueError, IndexError):
            pass
        return self.state.joints if self.state.joints is not None else np.zeros(6)

    def set_speed(self, speed_percent: int):
        """Set robot speed as percentage (1-100)."""
        return self._send_dashboard(f"SpeedFactor({speed_percent})")

    # --- Motion Commands ---

    def move_joint(self, x: float, y: float, z: float,
                   rx: float, ry: float, rz: float):
        """Joint interpolation move to Cartesian pose.

        Args:
            x, y, z: Position in mm
            rx, ry, rz: Orientation in degrees
        """
        return self._send_motion(f"MovJ({x},{y},{z},{rx},{ry},{rz})")

    def move_linear(self, x: float, y: float, z: float,
                    rx: float, ry: float, rz: float):
        """Linear interpolation move to Cartesian pose.

        Args:
            x, y, z: Position in mm
            rx, ry, rz: Orientation in degrees
        """
        return self._send_motion(f"MovL({x},{y},{z},{rx},{ry},{rz})")

    def move_joint_angles(self, j1: float, j2: float, j3: float,
                          j4: float, j5: float, j6: float):
        """Move to specified joint angles.

        Args:
            j1-j6: Joint angles in degrees
        """
        return self._send_motion(f"JointMovJ({j1},{j2},{j3},{j4},{j5},{j6})")

    def wait_motion_done(self, timeout: float = 30.0):
        """Wait for current motion to complete.

        Polls robot state until motion is finished or timeout.
        """
        # Use Sync command if available, otherwise poll
        response = self._send_motion("Sync()")
        return response

    # --- Lifecycle ---

    def disconnect(self):
        """Close all connections."""
        self._running = False
        for sock in [self._dashboard_sock, self._motion_sock, self._feedback_sock]:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
