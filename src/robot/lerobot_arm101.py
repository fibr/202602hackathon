"""Driver for LeRobot SO-ARM101 follower arm (Feetech STS3215 servos).

Uses the Feetech servo SDK (scservo_sdk) to communicate with 6 STS3215 servos
on a serial bus. Provides native methods for direct control of joint angles and
positions, plus FK/IK-based Cartesian interface.

Motor mapping (SO-101 follower):
    ID 1: shoulder_pan   (base rotation)
    ID 2: shoulder_lift
    ID 3: elbow_flex
    ID 4: wrist_flex
    ID 5: wrist_roll
    ID 6: gripper

Position range: 0-4095 (12-bit).
Resolution: 360° / 4096 ≈ 0.088° per step.

Zero-offset calibration:
    Each motor's 0° may NOT be at raw position 2048. Servo horns are
    installed at different physical angles, so per-motor offsets are loaded
    from config/servo_offsets.yaml. Without offsets, defaults to 2048.
"""

import os
import threading
import time
import numpy as np
import yaml
from typing import Optional

from logger import get_logger

log = get_logger('arm101')

# Try importing scservo_sdk; provide helpful error if missing
try:
    from scservo_sdk import PortHandler, PacketHandler, COMM_SUCCESS
    HAS_SCSERVO = True
except ImportError:
    HAS_SCSERVO = False

# Lazy-loaded FK solver (avoid import cost if not needed)
_fk_solver = None


def _get_fk_solver():
    """Get or create the shared Arm101IKSolver instance."""
    global _fk_solver
    if _fk_solver is None:
        from kinematics.arm101_ik_solver import Arm101IKSolver
        _fk_solver = Arm101IKSolver()
    return _fk_solver

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
POS_CENTER = 2048       # Default center (used if no calibration offset)
POS_PER_DEG = 4096.0 / 360.0   # ~11.378 steps per degree
DEG_PER_POS = 360.0 / 4096.0   # ~0.0879 degrees per step

# Per-motor zero offsets file
from config_loader import config_path as _config_path
_SERVO_OFFSETS_PATH = _config_path('servo_offsets.yaml')


def _load_servo_offsets() -> tuple:
    """Load per-motor zero offsets and signs from config/servo_offsets.yaml.

    Returns:
        (offsets, signs) where:
        - offsets: Dict mapping motor_id (int) -> zero_raw (int).
        - signs: Dict mapping motor_id (int) -> sign (float, +1 or -1).
    """
    path = os.path.normpath(_SERVO_OFFSETS_PATH)
    offsets = {}
    signs = {}
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        for name, info in data.get('zero_offsets', {}).items():
            if isinstance(info, dict) and 'motor_id' in info and 'zero_raw' in info:
                offsets[info['motor_id']] = info['zero_raw']
        # Load joint signs
        motor_name_to_id = {
            'shoulder_pan': 1, 'shoulder_lift': 2, 'elbow_flex': 3,
            'wrist_flex': 4, 'wrist_roll': 5,
        }
        for name, sign_val in data.get('joint_signs', {}).items():
            mid = motor_name_to_id.get(name)
            if mid is not None:
                signs[mid] = float(sign_val)
    return offsets, signs

# Gripper positions (motor 6)
GRIPPER_OPEN_POS = 2600
GRIPPER_CLOSE_POS = 1800

# Default speed for moves (0-4095, ~0 means max speed for STS)
DEFAULT_MOVE_SPEED = 200

# STS3215 EEPROM registers for angle limits (2-byte each)
ADDR_MIN_ANGLE_LIMIT = 9       # Minimum angle limit (0-4095), EEPROM
ADDR_MAX_ANGLE_LIMIT = 11      # Maximum angle limit (0-4095), EEPROM
ADDR_LOCK = 55                 # EEPROM write lock (0=unlocked, 1=locked)

# Safe mode defaults (reduced torque / speed for cautious operation)
SAFE_MODE_SPEED = 80            # Slow movement speed
SAFE_MODE_MAX_TORQUE = 300      # Reduced max torque (0-1023 register range)
ADDR_MAX_TORQUE = 48            # STS3215 max torque register (2-byte)


class LeRobotArm101:
    """Driver for SO-ARM101 follower arm with Feetech STS3215 servos.

    Provides native methods for joint control, position reading, and FK-based
    Cartesian interface.

    Args:
        port: Serial port path (e.g., '/dev/ttyACM0').
        baudrate: Serial baudrate (default 1000000).
        motor_ids: List of 6 motor IDs (default [1,2,3,4,5,6]).
        speed: Default movement speed (0-4095, default 200).
    """

    # Robot type identifier (used by control panel for mode-specific behavior)
    robot_type = 'arm101'

    def __init__(self, port: str, baudrate: int = DEFAULT_BAUDRATE,
                 motor_ids: Optional[list] = None, speed: int = DEFAULT_MOVE_SPEED,
                 safe_mode: bool = False):
        if not HAS_SCSERVO:
            raise ImportError(
                "scservo_sdk not installed. Run: pip install feetech-servo-sdk"
            )

        self.port_path = port
        self.baudrate = baudrate
        self.motor_ids = motor_ids or list(DEFAULT_MOTOR_IDS)
        self.safe_mode = safe_mode
        self.speed = SAFE_MODE_SPEED if safe_mode else speed
        self._enabled = False

        # Initialize SDK
        self._lock = threading.Lock()
        self.port_handler = PortHandler(port)
        self.packet_handler = PacketHandler(0)  # Protocol 0 for STS/SCS

        # Per-motor zero offsets and signs
        self._zero_offsets, self._joint_signs = _load_servo_offsets()
        if self._zero_offsets:
            non_default = {mid: off for mid, off in self._zero_offsets.items()
                          if off != POS_CENTER}
            if non_default:
                print(f"  Servo offsets loaded: "
                      f"{', '.join(f'M{m}={o}' for m, o in sorted(non_default.items()))}")

        # FK-based Z safety: prevent table collisions
        # Set min_safe_z_mm to the table height + margin (mm above arm base)
        self._z_safety_enabled = True
        self._min_safe_z_mm = 30.0  # Minimum safe Z in mm (above arm base origin)

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

        # Servos are enabled by default after power-on
        self._enabled = True

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
        """Enable torque on specified motors (default: all).

        In safe mode, applies reduced max torque before enabling.
        """
        ids = motor_ids or self.motor_ids
        if self.safe_mode:
            for mid in ids:
                self._write2(mid, ADDR_MAX_TORQUE, SAFE_MODE_MAX_TORQUE)
        for mid in ids:
            self._write1(mid, ADDR_TORQUE_ENABLE, 1)
        self._enabled = True

    def disable_torque(self, motor_ids: Optional[list] = None):
        """Disable torque on specified motors (default: all).

        Also sets _enabled = False which blocks subsequent write commands
        until enable_torque() is called again.
        """
        self._enabled = False  # block writes immediately
        ids = motor_ids or self.motor_ids
        for mid in ids:
            self._write1(mid, ADDR_TORQUE_ENABLE, 0)

    def set_safe_mode(self, enabled: bool):
        """Toggle safe mode (reduced torque and speed).

        Args:
            enabled: True to enable safe mode, False for normal operation.
        """
        self.safe_mode = enabled
        if enabled:
            self.speed = SAFE_MODE_SPEED
            # Apply torque limit if already enabled
            if self._enabled:
                for mid in self.motor_ids:
                    self._write2(mid, ADDR_MAX_TORQUE, SAFE_MODE_MAX_TORQUE)
            print(f"  Safe mode ON (speed={SAFE_MODE_SPEED}, "
                  f"torque_limit={SAFE_MODE_MAX_TORQUE})")
        else:
            self.speed = DEFAULT_MOVE_SPEED
            if self._enabled:
                for mid in self.motor_ids:
                    self._write2(mid, ADDR_MAX_TORQUE, 1023)  # full torque
            print(f"  Safe mode OFF (speed={DEFAULT_MOVE_SPEED}, full torque)")

    def set_z_safety(self, enabled: bool = True, min_z_mm: float = 30.0):
        """Configure FK-based Z-height safety to prevent table collisions.

        When enabled, write_all_angles() checks the FK position of the
        commanded joint angles and rejects commands that would place the
        end-effector below min_z_mm.

        Args:
            enabled: True to enable safety checks.
            min_z_mm: Minimum allowed Z height in mm (above arm base origin).
        """
        self._z_safety_enabled = enabled
        self._min_safe_z_mm = min_z_mm
        print(f"  Z safety {'ON' if enabled else 'OFF'}"
              f"{f' (min_z={min_z_mm:.0f}mm)' if enabled else ''}")

    # --- Servo angle limit configuration (EEPROM) ---

    def read_angle_limits(self, motor_id: int) -> tuple:
        """Read the min/max angle limits from servo EEPROM.

        These are raw position values (0-4095) stored in the servo's EEPROM
        that restrict the servo's movement range.  When both are 0, the
        servo operates in continuous rotation mode (no limits).

        Args:
            motor_id: Motor ID (1-6).

        Returns:
            (min_raw, max_raw) tuple of raw position values, or (-1, -1) on error.
        """
        with self._lock:
            min_val, res1, _ = self.packet_handler.read2ByteTxRx(
                self.port_handler, motor_id, ADDR_MIN_ANGLE_LIMIT
            )
            max_val, res2, _ = self.packet_handler.read2ByteTxRx(
                self.port_handler, motor_id, ADDR_MAX_ANGLE_LIMIT
            )
        if res1 != COMM_SUCCESS or res2 != COMM_SUCCESS:
            return (-1, -1)
        return (min_val, max_val)

    def read_all_angle_limits(self) -> list:
        """Read angle limits for all motors.

        Returns:
            List of (min_raw, max_raw) tuples, one per motor.
        """
        return [self.read_angle_limits(mid) for mid in self.motor_ids]

    def write_angle_limits(self, motor_id: int, min_raw: int, max_raw: int):
        """Write min/max angle limits to servo EEPROM.

        IMPORTANT: Torque must be disabled on the target servo before writing
        EEPROM registers.  This method handles the EEPROM unlock/lock sequence
        automatically.  Setting both to 0 removes limits (continuous rotation).

        Args:
            motor_id: Motor ID (1-6).
            min_raw: Minimum position limit (0-4095).
            max_raw: Maximum position limit (0-4095).

        Raises:
            ValueError: If min_raw >= max_raw (unless both are 0).
            IOError: If servo communication fails.
        """
        min_raw = max(0, min(4095, int(min_raw)))
        max_raw = max(0, min(4095, int(max_raw)))
        if min_raw > 0 or max_raw > 0:
            if min_raw >= max_raw:
                raise ValueError(
                    f"min_raw ({min_raw}) must be less than max_raw ({max_raw})"
                )

        # Disable torque on this motor (required for EEPROM writes)
        self._write1(motor_id, ADDR_TORQUE_ENABLE, 0)
        time.sleep(0.02)

        # Unlock EEPROM
        self._write1(motor_id, ADDR_LOCK, 0)
        time.sleep(0.02)

        # Write limits
        self._write2(motor_id, ADDR_MIN_ANGLE_LIMIT, min_raw)
        time.sleep(0.01)
        self._write2(motor_id, ADDR_MAX_ANGLE_LIMIT, max_raw)
        time.sleep(0.01)

        # Lock EEPROM
        self._write1(motor_id, ADDR_LOCK, 1)
        time.sleep(0.02)

        # Re-enable torque if arm was previously enabled
        if self._enabled:
            if self.safe_mode:
                self._write2(motor_id, ADDR_MAX_TORQUE, SAFE_MODE_MAX_TORQUE)
            self._write1(motor_id, ADDR_TORQUE_ENABLE, 1)

    def angle_limits_to_deg(self, motor_id: int, min_raw: int,
                            max_raw: int) -> tuple:
        """Convert raw angle limits to degrees using the motor's zero offset.

        Args:
            motor_id: Motor ID for offset lookup.
            min_raw: Minimum raw position.
            max_raw: Maximum raw position.

        Returns:
            (min_deg, max_deg) tuple in degrees.
        """
        return (self._pos_to_deg_motor(min_raw, motor_id),
                self._pos_to_deg_motor(max_raw, motor_id))

    def deg_to_angle_limits(self, motor_id: int, min_deg: float,
                            max_deg: float) -> tuple:
        """Convert degree limits to raw positions using the motor's zero offset.

        Args:
            motor_id: Motor ID for offset lookup.
            min_deg: Minimum angle in degrees.
            max_deg: Maximum angle in degrees.

        Returns:
            (min_raw, max_raw) tuple of raw positions.
        """
        return (self._deg_to_pos_motor(min_deg, motor_id),
                self._deg_to_pos_motor(max_deg, motor_id))

    # --- Position reading ---

    def read_position(self, motor_id: int) -> int:
        """Read raw position (0-4095) of a single motor."""
        with self._lock:
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
        with self._lock:
            positions = []
            for mid in self.motor_ids:
                pos, result, error = self.packet_handler.read2ByteTxRx(
                    self.port_handler, mid, ADDR_PRESENT_POSITION
                )
                positions.append(pos if result == COMM_SUCCESS else -1)
        self._cached_positions = positions
        self._last_query_time = time.time()
        return positions

    def read_all_angles(self) -> list:
        """Read joint angles in degrees for all motors.

        Uses calibrated per-motor zero offsets from servo_offsets.yaml.

        Returns:
            List of 6 angle values in degrees.
        """
        positions = self.read_all_positions()
        return [self._pos_to_deg_motor(p, mid)
                for p, mid in zip(positions, self.motor_ids)]

    # --- Position writing ---

    def write_position(self, motor_id: int, position: int, speed: int = None):
        """Write goal position to a single motor.

        Silently ignored if servos are disabled (torque off).

        Args:
            motor_id: Motor ID (1-6).
            position: Target position (0-4095).
            speed: Movement speed (0-4095). None = use default.
        """
        if not self._enabled:
            return
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
            angle_deg: Target angle in degrees (0° = calibrated zero).
            speed: Movement speed. None = use default.
        """
        pos = self._deg_to_pos(angle_deg)
        self.write_position(motor_id, pos, speed)

    def write_all_angles(self, angles: list, speed: int = None):
        """Write goal angles in degrees to all motors.

        Includes FK-based safety check: rejects commands that would move
        the end-effector below MIN_SAFE_Z_MM to prevent table collisions.

        Args:
            angles: List of 6 target angles in degrees.
            speed: Movement speed. None = use default.

        Raises:
            ValueError: If the target angles would place the arm below safe Z.
        """
        # FK safety check (first 5 joints determine TCP position)
        if len(angles) >= 5 and self._z_safety_enabled:
            try:
                solver = _get_fk_solver()
                pos_mm, _ = solver.forward_kin(np.array(angles[:5], dtype=float))
                if pos_mm[2] < self._min_safe_z_mm:
                    raise ValueError(
                        f"SAFETY: Target Z={pos_mm[2]:.0f}mm is below minimum "
                        f"{self._min_safe_z_mm}mm. FK=({pos_mm[0]:.0f},"
                        f"{pos_mm[1]:.0f},{pos_mm[2]:.0f}). "
                        f"Command rejected to prevent table collision."
                    )
            except ImportError:
                pass  # FK solver not available, skip check
            except ValueError:
                raise  # Re-raise our safety exception
            except Exception:
                pass  # Other errors (e.g., bad angles), skip check

        positions = [self._deg_to_pos_motor(a, mid)
                     for a, mid in zip(angles, self.motor_ids)]
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

    # --- Cartesian interface (FK/IK) ---

    def get_pose(self) -> Optional[list]:
        """Get Cartesian TCP pose [x,y,z,rx,ry,rz] in mm/deg via FK.

        Returns:
            List of 6 floats [x,y,z,rx,ry,rz], or None on error.
        """
        angles = self.get_angles()
        if angles is None:
            return None
        try:
            solver = _get_fk_solver()
            pos_mm, rpy_deg = solver.forward_kin(np.array(angles[:5]))
            return [pos_mm[0], pos_mm[1], pos_mm[2],
                    rpy_deg[0], rpy_deg[1], rpy_deg[2]]
        except Exception:
            return None

    def get_angles(self) -> Optional[list]:
        """Get joint angles in degrees.

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
        except Exception as e:
            log.warning(f"move_joints failed: {e}")
            return False

    # --- Low-level helpers ---

    def _write1(self, motor_id: int, address: int, value: int):
        """Write 1-byte value to motor register."""
        with self._lock:
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
        with self._lock:
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
        """Convert raw position (0-4095) to degrees relative to center (2048).

        Args:
            pos: Raw servo position (0-4095). Negative values return error sentinel.

        Returns:
            Angle in degrees (0° = POS_CENTER = 2048), or -999.0 on error.
        """
        if pos < 0:
            return -999.0  # error sentinel
        return (pos - POS_CENTER) * DEG_PER_POS

    @staticmethod
    def _deg_to_pos(deg: float) -> int:
        """Convert degrees to raw position (0-4095) relative to center (2048).

        Args:
            deg: Angle in degrees (0° = POS_CENTER = 2048).

        Returns:
            Clamped raw position in [0, 4095].
        """
        pos = int(round(deg * POS_PER_DEG + POS_CENTER))
        return max(0, min(4095, pos))

    def _deg_to_pos_motor(self, deg: float, motor_id: int) -> int:
        """Convert motor angle (degrees) to raw position using per-motor offset.

        Args:
            deg: Motor angle in degrees (same convention as get_angles()).
            motor_id: Motor ID for offset lookup.

        Returns:
            Clamped raw position in [0, 4095].
        """
        center = self._zero_offsets.get(motor_id, POS_CENTER)
        pos = int(round(deg * POS_PER_DEG + center))
        return max(0, min(4095, pos))

    def _pos_to_deg_motor(self, pos: int, motor_id: int) -> float:
        """Convert raw position to degrees using per-motor zero offset.

        Returns the motor angle without sign correction. The IK solver
        applies joint signs when converting motor angles to URDF angles.

        Args:
            pos: Raw servo position (0-4095).
            motor_id: Motor ID for offset lookup.

        Returns:
            Motor angle in degrees, or -999.0 on error.
        """
        if pos < 0:
            return -999.0
        center = self._zero_offsets.get(motor_id, POS_CENTER)
        return (pos - center) * DEG_PER_POS

    def _deg_to_pos_motor(self, deg: float, motor_id: int) -> int:
        """Convert degrees to raw position using per-motor zero offset.

        Uses the calibrated zero_raw for motor_id (from servo_offsets.yaml)
        as the baseline instead of POS_CENTER.  Falls back to _deg_to_pos()
        if no offset is loaded for the motor.

        Args:
            deg: Angle in degrees (0° = calibrated zero position).
            motor_id: Motor ID for offset lookup.

        Returns:
            Clamped raw position in [0, 4095].
        """
        center = self._zero_offsets.get(motor_id, POS_CENTER)
        pos = int(round(deg * POS_PER_DEG + center))
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
