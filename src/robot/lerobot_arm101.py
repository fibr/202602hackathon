"""Driver for LeRobot SO-ARM101 follower arm (Feetech STS3215 servos).

Uses the Feetech servo SDK (scservo_sdk) to communicate with 6 STS3215 servos
on a serial bus. Provides the same duck-typed interface that RobotControlPanel
expects: get_pose(), get_angles(), send(), plus native methods for direct control.

Motor mapping (SO-101 follower):
    ID 1: shoulder_pan   (base rotation)
    ID 2: shoulder_lift
    ID 3: elbow_flex
    ID 4: wrist_flex
    ID 5: wrist_roll
    ID 6: gripper

Position range: 0-4095 (12-bit), 2048 = center (~180°).
Resolution: 360° / 4096 ≈ 0.088° per step.
"""

import time
from typing import Optional

# Try importing scservo_sdk; provide helpful error if missing
try:
    from scservo_sdk import PortHandler, PacketHandler, COMM_SUCCESS
    HAS_SCSERVO = True
except ImportError:
    HAS_SCSERVO = False

# Motor configuration
MOTOR_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
               'wrist_flex', 'wrist_roll', 'gripper']
DEFAULT_MOTOR_IDS = [1, 2, 3, 4, 5, 6]
DEFAULT_BAUDRATE = 1_000_000

# STS3215 register addresses (protocol 0 / SCS)
ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42
ADDR_GOAL_SPEED = 46
ADDR_PRESENT_POSITION = 56
ADDR_PRESENT_SPEED = 58
ADDR_PRESENT_LOAD = 60

# Position conversion constants
POS_CENTER = 2048       # Center position (0° = center for our purposes)
POS_PER_DEG = 4096.0 / 360.0   # ~11.378 steps per degree
DEG_PER_POS = 360.0 / 4096.0   # ~0.0879 degrees per step

# Gripper positions (motor 6)
GRIPPER_OPEN_POS = 1800
GRIPPER_CLOSE_POS = 2600

# Default speed for moves (0-4095, ~0 means max speed for STS)
DEFAULT_MOVE_SPEED = 200


class LeRobotArm101:
    """Driver for SO-ARM101 follower arm with Feetech STS3215 servos.

    Provides both a native interface and a duck-typed interface compatible
    with RobotControlPanel (get_pose, get_angles, send).

    Args:
        port: Serial port path (e.g., '/dev/ttyACM0').
        baudrate: Serial baudrate (default 1000000).
        motor_ids: List of 6 motor IDs (default [1,2,3,4,5,6]).
        speed: Default movement speed (0-4095, default 200).
    """

    # Robot type identifier (used by control panel for mode-specific behavior)
    robot_type = 'arm101'

    def __init__(self, port: str, baudrate: int = DEFAULT_BAUDRATE,
                 motor_ids: Optional[list] = None, speed: int = DEFAULT_MOVE_SPEED):
        if not HAS_SCSERVO:
            raise ImportError(
                "scservo_sdk not installed. Run: pip install feetech-servo-sdk"
            )

        self.port_path = port
        self.baudrate = baudrate
        self.motor_ids = motor_ids or list(DEFAULT_MOTOR_IDS)
        self.speed = speed
        self._enabled = False

        # Initialize SDK
        self.port_handler = PortHandler(port)
        self.packet_handler = PacketHandler(0)  # Protocol 0 for STS/SCS

        # State cache
        self._cached_positions = None  # raw positions (0-4095)
        self._last_query_time = 0.0
        self._query_interval = 0.05  # 50ms between queries

    def connect(self):
        """Open serial port and verify motor communication."""
        if not self.port_handler.openPort():
            raise ConnectionError(f"Failed to open port {self.port_path}")

        if not self.port_handler.setBaudRate(self.baudrate):
            raise ConnectionError(f"Failed to set baudrate {self.baudrate}")

        print(f"  Arm101: connected to {self.port_path} @ {self.baudrate}")

        # Ping each motor to verify connection
        for motor_id in self.motor_ids:
            model, result, error = self.packet_handler.ping(
                self.port_handler, motor_id
            )
            if result != COMM_SUCCESS:
                print(f"  WARNING: Motor {motor_id} ({MOTOR_NAMES[motor_id-1]}) "
                      f"not responding: {self.packet_handler.getTxRxResult(result)}")
            else:
                print(f"  Motor {motor_id} ({MOTOR_NAMES[motor_id-1]}): OK "
                      f"(model {model})")

    def disconnect(self):
        """Disable torque and close serial port."""
        try:
            self.disable_torque()
        except Exception:
            pass
        self.port_handler.closePort()
        print("  Arm101: disconnected.")

    def close(self):
        """Alias for disconnect (compatibility)."""
        self.disconnect()

    # --- Torque control ---

    def enable_torque(self, motor_ids: Optional[list] = None):
        """Enable torque on specified motors (default: all)."""
        ids = motor_ids or self.motor_ids
        for mid in ids:
            self._write1(mid, ADDR_TORQUE_ENABLE, 1)
        self._enabled = True

    def disable_torque(self, motor_ids: Optional[list] = None):
        """Disable torque on specified motors (default: all)."""
        ids = motor_ids or self.motor_ids
        for mid in ids:
            self._write1(mid, ADDR_TORQUE_ENABLE, 0)
        self._enabled = False

    # --- Position reading ---

    def read_position(self, motor_id: int) -> int:
        """Read raw position (0-4095) of a single motor."""
        pos, result, error = self.packet_handler.read2ByteTxRx(
            self.port_handler, motor_id, ADDR_PRESENT_POSITION
        )
        if result != COMM_SUCCESS:
            return -1
        return pos

    def read_all_positions(self) -> list:
        """Read raw positions (0-4095) for all motors.

        Returns:
            List of 6 position values.
        """
        positions = []
        for mid in self.motor_ids:
            positions.append(self.read_position(mid))
        self._cached_positions = positions
        self._last_query_time = time.time()
        return positions

    def read_all_angles(self) -> list:
        """Read joint angles in degrees for all motors.

        Center position (2048) maps to 0°. Range is approximately ±180°.

        Returns:
            List of 6 angle values in degrees.
        """
        positions = self.read_all_positions()
        return [self._pos_to_deg(p) for p in positions]

    # --- Position writing ---

    def write_position(self, motor_id: int, position: int, speed: int = None):
        """Write goal position to a single motor.

        Args:
            motor_id: Motor ID (1-6).
            position: Target position (0-4095).
            speed: Movement speed (0-4095). None = use default.
        """
        position = max(0, min(4095, position))
        spd = speed if speed is not None else self.speed
        self._write2(motor_id, ADDR_GOAL_SPEED, spd)
        self._write2(motor_id, ADDR_GOAL_POSITION, position)

    def write_all_positions(self, positions: list, speed: int = None):
        """Write goal positions to all motors.

        Args:
            positions: List of 6 target positions (0-4095).
            speed: Movement speed. None = use default.
        """
        for i, mid in enumerate(self.motor_ids):
            self.write_position(mid, positions[i], speed)

    def write_angle(self, motor_id: int, angle_deg: float, speed: int = None):
        """Write goal angle in degrees to a single motor.

        Args:
            motor_id: Motor ID (1-6).
            angle_deg: Target angle in degrees (0° = center/2048).
            speed: Movement speed. None = use default.
        """
        pos = self._deg_to_pos(angle_deg)
        self.write_position(motor_id, pos, speed)

    def write_all_angles(self, angles: list, speed: int = None):
        """Write goal angles in degrees to all motors.

        Args:
            angles: List of 6 target angles in degrees.
            speed: Movement speed. None = use default.
        """
        positions = [self._deg_to_pos(a) for a in angles]
        self.write_all_positions(positions, speed)

    # --- Gripper ---

    def gripper_open(self, speed: int = None):
        """Open the gripper (motor 6)."""
        self.write_position(self.motor_ids[5], GRIPPER_OPEN_POS,
                            speed or self.speed)

    def gripper_close(self, speed: int = None):
        """Close the gripper (motor 6)."""
        self.write_position(self.motor_ids[5], GRIPPER_CLOSE_POS,
                            speed or self.speed)

    # --- Joint jog ---

    def jog_joint(self, joint_idx: int, direction: int, step_deg: float = 5.0,
                  speed: int = None):
        """Jog a single joint by step_deg in the given direction.

        Args:
            joint_idx: Joint index (0-5).
            direction: +1 or -1.
            step_deg: Step size in degrees.
            speed: Movement speed. None = use default.
        """
        current = self.read_all_angles()
        if current[joint_idx] < -180:
            return  # read error
        target = current[joint_idx] + direction * step_deg
        self.write_angle(self.motor_ids[joint_idx], target, speed)

    # --- Duck-typed interface for RobotControlPanel ---

    def get_pose(self) -> Optional[list]:
        """Get Cartesian pose. Returns None (no FK available for arm101)."""
        return None

    def get_angles(self) -> Optional[list]:
        """Get joint angles in degrees. Compatible with RobotControlPanel.

        Returns:
            List of 6 joint angles in degrees, or None on error.
        """
        try:
            angles = self.read_all_angles()
            if any(a < -180 for a in angles):
                return None
            return angles
        except Exception:
            return None

    def get_mode(self) -> int:
        """Get robot mode. Returns 5 (enabled) if torque is on, 4 otherwise."""
        return 5 if self._enabled else 4

    def move_joints(self, angles: list, speed: int = None) -> bool:
        """Move all joints to target angles.

        Args:
            angles: List of 6 target angles in degrees.
            speed: Movement speed. None = use default.

        Returns:
            True if command was sent successfully.
        """
        try:
            self.write_all_angles(angles, speed)
            return True
        except Exception:
            return False

    # --- Low-level helpers ---

    def _write1(self, motor_id: int, address: int, value: int):
        """Write 1-byte value to motor register."""
        result, error = self.packet_handler.write1ByteTxRx(
            self.port_handler, motor_id, address, value
        )
        if result != COMM_SUCCESS:
            raise IOError(
                f"Write1 failed motor {motor_id} addr {address}: "
                f"{self.packet_handler.getTxRxResult(result)}"
            )

    def _write2(self, motor_id: int, address: int, value: int):
        """Write 2-byte value to motor register."""
        result, error = self.packet_handler.write2ByteTxRx(
            self.port_handler, motor_id, address, value
        )
        if result != COMM_SUCCESS:
            raise IOError(
                f"Write2 failed motor {motor_id} addr {address}: "
                f"{self.packet_handler.getTxRxResult(result)}"
            )

    @staticmethod
    def _pos_to_deg(pos: int) -> float:
        """Convert raw position (0-4095) to degrees. 2048 = 0°."""
        if pos < 0:
            return -999.0  # error sentinel
        return (pos - POS_CENTER) * DEG_PER_POS

    @staticmethod
    def _deg_to_pos(deg: float) -> int:
        """Convert degrees to raw position (0-4095). 0° = 2048."""
        pos = int(round(deg * POS_PER_DEG + POS_CENTER))
        return max(0, min(4095, pos))

    # --- Context manager ---

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()

    # --- Port discovery ---

    @staticmethod
    def find_port() -> str:
        """Auto-detect the serial port for the arm101 servo bus.

        Searches /dev/ttyACM* and /dev/ttyUSB* for a Feetech controller.

        Returns:
            Port path string.

        Raises:
            FileNotFoundError: If no suitable port found.
        """
        import glob
        candidates = sorted(
            glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*')
        )
        if not candidates:
            raise FileNotFoundError(
                "No serial ports found (/dev/ttyACM* or /dev/ttyUSB*). "
                "Is the arm101 USB cable connected?"
            )
        # Return first candidate; user can override via config
        return candidates[0]
