"""Unit tests for LeRobotArm101 with mock scservo_sdk.

Tests position conversion, jog logic, gripper control, torque/safe-mode,
connection management, and the duck-typed panel integration interface.
No physical hardware or serial port required.
"""

import sys
import os
import math
import pytest
from unittest import mock

# ─────────────────────────────────────────────────────────────────────────────
# Mock scservo_sdk BEFORE importing lerobot_arm101.
#
# lerobot_arm101 does `from scservo_sdk import PortHandler, PacketHandler,
# COMM_SUCCESS` at module level.  If the real SDK is absent, HAS_SCSERVO
# becomes False and the constructor raises ImportError.  We prevent that by
# injecting a mock module into sys.modules first.
# ─────────────────────────────────────────────────────────────────────────────
_COMM_SUCCESS = 0

if 'scservo_sdk' not in sys.modules:
    _mock_scservo = mock.MagicMock()
    _mock_scservo.COMM_SUCCESS = _COMM_SUCCESS
    sys.modules['scservo_sdk'] = _mock_scservo
else:
    # Already present — make sure COMM_SUCCESS is the sentinel we expect.
    sys.modules['scservo_sdk'].COMM_SUCCESS = _COMM_SUCCESS

# Remove any cached copy of lerobot_arm101 so it re-imports with our mock.
for _k in list(sys.modules.keys()):
    if 'lerobot_arm101' in _k:
        del sys.modules[_k]

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Now import — will bind PortHandler / PacketHandler / COMM_SUCCESS from our mock.
from robot.lerobot_arm101 import (  # noqa: E402
    LeRobotArm101,
    POS_CENTER, POS_PER_DEG, DEG_PER_POS,
    GRIPPER_OPEN_POS, GRIPPER_CLOSE_POS,
    DEFAULT_MOVE_SPEED, SAFE_MODE_SPEED, SAFE_MODE_MAX_TORQUE,
    ADDR_TORQUE_ENABLE, ADDR_GOAL_POSITION, ADDR_GOAL_SPEED,
    ADDR_PRESENT_POSITION, ADDR_MAX_TORQUE,
    DEFAULT_MOTOR_IDS,
)

# Grab the COMM_SUCCESS value as seen by lerobot_arm101 (from our mock).
import robot.lerobot_arm101 as _arm101_mod
COMM_SUCCESS = _arm101_mod.COMM_SUCCESS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _ph(open_ok=True, baud_ok=True):
    """Create a PortHandler mock."""
    ph = mock.MagicMock()
    ph.openPort.return_value = open_ok
    ph.setBaudRate.return_value = baud_ok
    ph.closePort.return_value = True
    return ph


def _pkt(read_pos=POS_CENTER):
    """Create a PacketHandler mock that returns read_pos for all reads.

    read_pos  – raw position returned by read2ByteTxRx.
                Pass a negative value to simulate a communication error.
    """
    pkt = mock.MagicMock()
    if read_pos < 0:
        # Simulate comm failure: result != COMM_SUCCESS
        pkt.read2ByteTxRx.return_value = (0, 1, 0)
    else:
        pkt.read2ByteTxRx.return_value = (read_pos, COMM_SUCCESS, 0)
    pkt.ping.return_value = (100, COMM_SUCCESS, 0)
    pkt.write1ByteTxRx.return_value = (COMM_SUCCESS, 0)
    pkt.write2ByteTxRx.return_value = (COMM_SUCCESS, 0)
    pkt.getTxRxResult.return_value = "COMM_SUCCESS"
    return pkt


def _goal_pos_calls(pkt):
    """Return all write2ByteTxRx calls that target ADDR_GOAL_POSITION."""
    return [c for c in pkt.write2ByteTxRx.call_args_list
            if c[0][2] == ADDR_GOAL_POSITION]


def _goal_speed_calls(pkt):
    """Return all write2ByteTxRx calls that target ADDR_GOAL_SPEED."""
    return [c for c in pkt.write2ByteTxRx.call_args_list
            if c[0][2] == ADDR_GOAL_SPEED]


def _torque_enable_calls(pkt, value):
    """Return write1ByteTxRx calls for ADDR_TORQUE_ENABLE with the given value."""
    return [c for c in pkt.write1ByteTxRx.call_args_list
            if c[0][2] == ADDR_TORQUE_ENABLE and c[0][3] == value]


def _max_torque_calls(pkt, value):
    """Return write2ByteTxRx calls for ADDR_MAX_TORQUE with the given value."""
    return [c for c in pkt.write2ByteTxRx.call_args_list
            if c[0][2] == ADDR_MAX_TORQUE and c[0][3] == value]


@pytest.fixture
def arm():
    """LeRobotArm101 with mock serial port; all motors at center (0°)."""
    robot = LeRobotArm101('/dev/ttyACM0')
    robot.port_handler = _ph()
    robot.packet_handler = _pkt(read_pos=POS_CENTER)
    return robot


@pytest.fixture
def arm_enabled(arm):
    """arm101 with torque enabled (simulates post-enable state)."""
    arm._enabled = True
    return arm


# ─────────────────────────────────────────────────────────────────────────────
# 1. Position conversion — pure static methods, no mocking required
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionConversion:
    """Tests for _pos_to_deg and _deg_to_pos static methods."""

    # ── POS_TO_DEG ────────────────────────────────────────────────────────────

    def test_center_pos_to_zero_deg(self):
        assert LeRobotArm101._pos_to_deg(POS_CENTER) == pytest.approx(0.0, abs=1e-6)

    def test_pos_to_deg_positive(self):
        """2048 + 90*POS_PER_DEG → ~90°."""
        pos = POS_CENTER + int(90.0 * POS_PER_DEG)
        assert LeRobotArm101._pos_to_deg(pos) == pytest.approx(90.0, abs=0.1)

    def test_pos_to_deg_negative(self):
        """2048 - 90*POS_PER_DEG → ~-90°."""
        pos = POS_CENTER - int(90.0 * POS_PER_DEG)
        assert LeRobotArm101._pos_to_deg(pos) == pytest.approx(-90.0, abs=0.1)

    def test_pos_to_deg_zero_position(self):
        """Position 0 → approximately -180°."""
        assert LeRobotArm101._pos_to_deg(0) == pytest.approx(-180.0, abs=0.1)

    def test_pos_to_deg_max_position(self):
        """Position 4095 → approximately +180°."""
        assert LeRobotArm101._pos_to_deg(4095) == pytest.approx(180.0, abs=0.1)

    def test_pos_to_deg_error_sentinel(self):
        """Negative position → -999.0 (error sentinel)."""
        assert LeRobotArm101._pos_to_deg(-1) == -999.0
        assert LeRobotArm101._pos_to_deg(-100) == -999.0

    # ── DEG_TO_POS ────────────────────────────────────────────────────────────

    def test_zero_deg_to_center_pos(self):
        assert LeRobotArm101._deg_to_pos(0.0) == POS_CENTER

    def test_deg_to_pos_positive(self):
        """90° → int(round(90 * POS_PER_DEG + 2048))."""
        expected = int(round(90.0 * POS_PER_DEG + POS_CENTER))
        assert LeRobotArm101._deg_to_pos(90.0) == expected

    def test_deg_to_pos_negative(self):
        """-90° → int(round(-90 * POS_PER_DEG + 2048))."""
        expected = int(round(-90.0 * POS_PER_DEG + POS_CENTER))
        assert LeRobotArm101._deg_to_pos(-90.0) == expected

    def test_deg_to_pos_clamp_high(self):
        """Very large angle clamps to 4095."""
        assert LeRobotArm101._deg_to_pos(1000.0) == 4095

    def test_deg_to_pos_clamp_low(self):
        """Very negative angle clamps to 0."""
        assert LeRobotArm101._deg_to_pos(-1000.0) == 0

    def test_deg_to_pos_boundary_positive_180(self):
        """180° should be within [0, 4095]."""
        pos = LeRobotArm101._deg_to_pos(180.0)
        assert 0 <= pos <= 4095

    def test_deg_to_pos_boundary_negative_180(self):
        """-180° should be within [0, 4095]."""
        pos = LeRobotArm101._deg_to_pos(-180.0)
        assert 0 <= pos <= 4095

    # ── Constants ─────────────────────────────────────────────────────────────

    def test_pos_per_deg_value(self):
        assert POS_PER_DEG == pytest.approx(4096.0 / 360.0, rel=1e-6)

    def test_deg_per_pos_value(self):
        assert DEG_PER_POS == pytest.approx(360.0 / 4096.0, rel=1e-6)

    def test_constants_are_reciprocal(self):
        assert POS_PER_DEG * DEG_PER_POS == pytest.approx(1.0, rel=1e-6)

    # ── Roundtrips ────────────────────────────────────────────────────────────

    def test_roundtrip_deg_to_pos_to_deg(self):
        """deg → pos → deg should be close (within ±0.1°) for typical angles."""
        for angle in [-135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0]:
            pos = LeRobotArm101._deg_to_pos(angle)
            deg = LeRobotArm101._pos_to_deg(pos)
            assert abs(deg - angle) < 0.1, (
                f"Roundtrip error for {angle}°: got {deg:.4f}°"
            )

    def test_roundtrip_pos_to_deg_to_pos(self):
        """pos → deg → pos should differ by at most 1 (quantisation)."""
        for pos in [0, 512, 1024, 2048, 2560, 3072, 3584, 4095]:
            deg = LeRobotArm101._pos_to_deg(pos)
            pos2 = LeRobotArm101._deg_to_pos(deg)
            assert abs(pos2 - pos) <= 1, (
                f"Roundtrip error for pos={pos}: got {pos2}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Connection management
# ─────────────────────────────────────────────────────────────────────────────

class TestConnection:
    """Test connect / disconnect lifecycle with mock serial."""

    def test_connect_opens_port(self, arm):
        arm.connect()
        arm.port_handler.openPort.assert_called_once()

    def test_connect_sets_baudrate(self, arm):
        arm.connect()
        arm.port_handler.setBaudRate.assert_called_once_with(arm.baudrate)

    def test_connect_pings_all_motors(self, arm):
        arm.connect()
        assert arm.packet_handler.ping.call_count == len(arm.motor_ids)

    def test_connect_raises_on_port_open_failure(self, arm):
        arm.port_handler.openPort.return_value = False
        with pytest.raises(ConnectionError, match="Failed to open port"):
            arm.connect()

    def test_connect_raises_on_baudrate_failure(self, arm):
        arm.port_handler.setBaudRate.return_value = False
        with pytest.raises(ConnectionError, match="Failed to set baudrate"):
            arm.connect()

    def test_connect_continues_if_motor_ping_fails(self, arm):
        """A failed ping (motor not responding) should warn but not raise."""
        arm.packet_handler.ping.return_value = (0, 1, 0)  # result != COMM_SUCCESS
        arm.connect()  # should not raise

    def test_disconnect_closes_port(self, arm):
        arm.disconnect()
        arm.port_handler.closePort.assert_called_once()

    def test_close_is_alias_for_disconnect(self, arm):
        with mock.patch.object(arm, 'disconnect') as mock_disc:
            arm.close()
        mock_disc.assert_called_once()

    def test_context_manager_calls_connect_and_disconnect(self, arm):
        with mock.patch.object(arm, 'connect') as mc, \
             mock.patch.object(arm, 'disconnect') as md:
            with arm:
                mc.assert_called_once()
            md.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Torque control and safe mode
# ─────────────────────────────────────────────────────────────────────────────

class TestTorqueControl:
    """Test enable/disable torque and safe mode toggling."""

    def test_enable_torque_sets_enabled_flag(self, arm):
        arm.enable_torque()
        assert arm._enabled is True

    def test_enable_torque_writes_1_to_all_motors(self, arm):
        arm.enable_torque()
        assert len(_torque_enable_calls(arm.packet_handler, 1)) == len(arm.motor_ids)

    def test_disable_torque_clears_enabled_flag(self, arm_enabled):
        arm_enabled.disable_torque()
        assert arm_enabled._enabled is False

    def test_disable_torque_writes_0_to_all_motors(self, arm_enabled):
        arm_enabled.disable_torque()
        assert len(_torque_enable_calls(arm_enabled.packet_handler, 0)) == len(arm_enabled.motor_ids)

    def test_enable_torque_subset_of_motors(self, arm):
        arm.enable_torque(motor_ids=[1, 3])
        assert len(_torque_enable_calls(arm.packet_handler, 1)) == 2

    def test_disable_torque_subset_of_motors(self, arm_enabled):
        arm_enabled.disable_torque(motor_ids=[2, 4, 6])
        assert len(_torque_enable_calls(arm_enabled.packet_handler, 0)) == 3

    def test_enable_torque_safe_mode_applies_torque_limit(self, arm):
        """In safe mode, enable_torque writes max-torque register before enabling."""
        arm.safe_mode = True
        arm.enable_torque()
        assert len(_max_torque_calls(arm.packet_handler, SAFE_MODE_MAX_TORQUE)) == len(arm.motor_ids)

    def test_enable_torque_normal_mode_no_max_torque_write(self, arm):
        """In normal mode, enable_torque does NOT touch the max-torque register."""
        arm.safe_mode = False
        arm.enable_torque()
        assert len([c for c in arm.packet_handler.write2ByteTxRx.call_args_list
                    if c[0][2] == ADDR_MAX_TORQUE]) == 0

    def test_set_safe_mode_on_reduces_speed(self, arm):
        arm.set_safe_mode(True)
        assert arm.speed == SAFE_MODE_SPEED
        assert arm.safe_mode is True

    def test_set_safe_mode_off_restores_speed(self, arm):
        arm.safe_mode = True
        arm.speed = SAFE_MODE_SPEED
        arm.set_safe_mode(False)
        assert arm.speed == DEFAULT_MOVE_SPEED
        assert arm.safe_mode is False

    def test_set_safe_mode_on_while_enabled_writes_torque_limit(self, arm_enabled):
        """set_safe_mode(True) while enabled → writes reduced max-torque to all motors."""
        arm_enabled.packet_handler.write2ByteTxRx.reset_mock()
        arm_enabled.set_safe_mode(True)
        assert len(_max_torque_calls(arm_enabled.packet_handler, SAFE_MODE_MAX_TORQUE)) == len(arm_enabled.motor_ids)

    def test_set_safe_mode_off_while_enabled_restores_full_torque(self, arm_enabled):
        """set_safe_mode(False) while enabled → writes 1023 (full torque) to all motors."""
        arm_enabled.safe_mode = True
        arm_enabled.packet_handler.write2ByteTxRx.reset_mock()
        arm_enabled.set_safe_mode(False)
        assert len(_max_torque_calls(arm_enabled.packet_handler, 1023)) == len(arm_enabled.motor_ids)

    def test_set_safe_mode_on_while_disabled_no_torque_write(self, arm):
        """set_safe_mode(True) while disabled should NOT write max-torque register."""
        assert arm._enabled is False
        arm.packet_handler.write2ByteTxRx.reset_mock()
        arm.set_safe_mode(True)
        assert len([c for c in arm.packet_handler.write2ByteTxRx.call_args_list
                    if c[0][2] == ADDR_MAX_TORQUE]) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Position reading and writing
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionReadWrite:
    """Test read_position, read_all_positions, write_position, write_angle."""

    def test_read_position_returns_value_on_success(self, arm):
        arm.packet_handler.read2ByteTxRx.return_value = (1500, COMM_SUCCESS, 0)
        assert arm.read_position(1) == 1500

    def test_read_position_returns_minus_one_on_comm_error(self, arm):
        arm.packet_handler.read2ByteTxRx.return_value = (0, 1, 0)  # result != COMM_SUCCESS
        assert arm.read_position(1) == -1

    def test_read_all_positions_returns_six_values(self, arm):
        arm.packet_handler.read2ByteTxRx.return_value = (POS_CENTER, COMM_SUCCESS, 0)
        positions = arm.read_all_positions()
        assert len(positions) == 6

    def test_read_all_positions_updates_cache(self, arm):
        arm.packet_handler.read2ByteTxRx.return_value = (2100, COMM_SUCCESS, 0)
        positions = arm.read_all_positions()
        assert arm._cached_positions == positions

    def test_read_all_angles_at_center_are_zero(self, arm):
        arm.packet_handler.read2ByteTxRx.return_value = (POS_CENTER, COMM_SUCCESS, 0)
        angles = arm.read_all_angles()
        assert len(angles) == 6
        assert all(abs(a) < 1e-3 for a in angles)

    def test_read_all_angles_non_center(self, arm):
        """3072 = 2048 + 1024 ≈ 2048 + 90*POS_PER_DEG → ~90°."""
        arm.packet_handler.read2ByteTxRx.return_value = (3072, COMM_SUCCESS, 0)
        angles = arm.read_all_angles()
        for angle in angles:
            assert angle == pytest.approx(90.0, abs=0.5)

    def test_write_position_clamps_high(self, arm):
        arm.write_position(1, 99999)
        gp = _goal_pos_calls(arm.packet_handler)
        assert gp[-1][0][3] == 4095

    def test_write_position_clamps_low(self, arm):
        arm.write_position(1, -500)
        gp = _goal_pos_calls(arm.packet_handler)
        assert gp[-1][0][3] == 0

    def test_write_position_sends_speed_before_position(self, arm):
        """write_position should write speed first, then goal position."""
        arm.write_position(2, 2048, speed=150)
        calls = arm.packet_handler.write2ByteTxRx.call_args_list
        addrs = [c[0][2] for c in calls]
        # ADDR_GOAL_SPEED must appear before ADDR_GOAL_POSITION
        idx_spd = next(i for i, a in enumerate(addrs) if a == ADDR_GOAL_SPEED)
        idx_pos = next(i for i, a in enumerate(addrs) if a == ADDR_GOAL_POSITION)
        assert idx_spd < idx_pos

    def test_write_position_uses_custom_speed(self, arm):
        arm.write_position(1, 2048, speed=300)
        gs = _goal_speed_calls(arm.packet_handler)
        assert gs[-1][0][3] == 300

    def test_write_position_uses_default_speed_when_none(self, arm):
        arm.speed = 123
        arm.write_position(1, 2048, speed=None)
        gs = _goal_speed_calls(arm.packet_handler)
        assert gs[-1][0][3] == 123

    def test_write_angle_zero_writes_center(self, arm):
        arm.write_angle(1, 0.0)
        gp = _goal_pos_calls(arm.packet_handler)
        assert gp[-1][0][3] == POS_CENTER

    def test_write_angle_90_degrees(self, arm):
        expected = LeRobotArm101._deg_to_pos(90.0)
        arm.write_angle(3, 90.0)
        gp = _goal_pos_calls(arm.packet_handler)
        assert gp[-1][0][3] == expected

    def test_write_angle_negative(self, arm):
        expected = LeRobotArm101._deg_to_pos(-45.0)
        arm.write_angle(2, -45.0)
        gp = _goal_pos_calls(arm.packet_handler)
        assert gp[-1][0][3] == expected

    def test_write_all_angles_sends_to_all_motors(self, arm):
        angles = [0.0, 10.0, -20.0, 30.0, -45.0, 5.0]
        arm.packet_handler.write2ByteTxRx.reset_mock()
        arm.write_all_angles(angles)
        gp = _goal_pos_calls(arm.packet_handler)
        assert len(gp) == len(arm.motor_ids)

    def test_write_all_angles_correct_positions(self, arm):
        angles = [0.0, 90.0, -90.0, 45.0, -45.0, 0.0]
        expected_positions = [LeRobotArm101._deg_to_pos(a) for a in angles]
        arm.packet_handler.write2ByteTxRx.reset_mock()
        arm.write_all_angles(angles)
        gp = _goal_pos_calls(arm.packet_handler)
        written = [c[0][3] for c in gp]
        for i, (exp, got) in enumerate(zip(expected_positions, written)):
            assert abs(exp - got) <= 1, f"Motor {i}: expected pos {exp}, got {got}"

    def test_write2_raises_ioerror_on_comm_failure(self, arm):
        arm.packet_handler.write2ByteTxRx.return_value = (1, 0)  # result != COMM_SUCCESS
        arm.packet_handler.getTxRxResult.return_value = "COMM_TX_FAIL"
        with pytest.raises(IOError, match="Write2 failed"):
            arm._write2(1, ADDR_GOAL_POSITION, 2048)

    def test_write1_raises_ioerror_on_comm_failure(self, arm):
        arm.packet_handler.write1ByteTxRx.return_value = (1, 0)  # result != COMM_SUCCESS
        arm.packet_handler.getTxRxResult.return_value = "COMM_TX_FAIL"
        with pytest.raises(IOError, match="Write1 failed"):
            arm._write1(1, ADDR_TORQUE_ENABLE, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Gripper control
# ─────────────────────────────────────────────────────────────────────────────

class TestGripperControl:
    """Test gripper_open / gripper_close."""

    def test_gripper_open_writes_open_position(self, arm):
        arm.gripper_open()
        gp = _goal_pos_calls(arm.packet_handler)
        assert gp[-1][0][3] == GRIPPER_OPEN_POS

    def test_gripper_close_writes_close_position(self, arm):
        arm.gripper_close()
        gp = _goal_pos_calls(arm.packet_handler)
        assert gp[-1][0][3] == GRIPPER_CLOSE_POS

    def test_gripper_open_targets_motor_6(self, arm):
        arm.gripper_open()
        gp = _goal_pos_calls(arm.packet_handler)
        assert gp[-1][0][1] == arm.motor_ids[5]  # motor_ids[5] == 6

    def test_gripper_close_targets_motor_6(self, arm):
        arm.gripper_close()
        gp = _goal_pos_calls(arm.packet_handler)
        assert gp[-1][0][1] == arm.motor_ids[5]

    def test_gripper_open_with_custom_speed(self, arm):
        arm.gripper_open(speed=400)
        gs = _goal_speed_calls(arm.packet_handler)
        assert gs[-1][0][3] == 400

    def test_gripper_close_with_custom_speed(self, arm):
        arm.gripper_close(speed=150)
        gs = _goal_speed_calls(arm.packet_handler)
        assert gs[-1][0][3] == 150

    def test_gripper_open_uses_arm_speed_when_none(self, arm):
        arm.speed = 77
        arm.gripper_open(speed=None)
        gs = _goal_speed_calls(arm.packet_handler)
        assert gs[-1][0][3] == 77

    def test_gripper_close_uses_arm_speed_when_none(self, arm):
        arm.speed = 88
        arm.gripper_close(speed=None)
        gs = _goal_speed_calls(arm.packet_handler)
        assert gs[-1][0][3] == 88

    def test_open_and_close_positions_differ(self):
        assert GRIPPER_OPEN_POS != GRIPPER_CLOSE_POS

    def test_gripper_positions_in_servo_range(self):
        assert 0 <= GRIPPER_OPEN_POS <= 4095
        assert 0 <= GRIPPER_CLOSE_POS <= 4095


# ─────────────────────────────────────────────────────────────────────────────
# 6. Jog logic
# ─────────────────────────────────────────────────────────────────────────────

class TestJogJoint:
    """Test jog_joint: reads current angles → steps → writes target motor."""

    def _reset_write(self, arm):
        arm.packet_handler.write2ByteTxRx.reset_mock()

    def test_jog_positive_direction_from_center(self, arm):
        """Jog joint 0 by +5° from 0° → target 5°."""
        arm.packet_handler.read2ByteTxRx.return_value = (POS_CENTER, COMM_SUCCESS, 0)
        self._reset_write(arm)

        arm.jog_joint(0, direction=+1, step_deg=5.0)

        expected = LeRobotArm101._deg_to_pos(5.0)
        gp = _goal_pos_calls(arm.packet_handler)
        assert len(gp) == 1
        assert abs(gp[0][0][3] - expected) <= 1

    def test_jog_negative_direction_from_center(self, arm):
        """Jog joint 0 by -5° from 0° → target -5°."""
        arm.packet_handler.read2ByteTxRx.return_value = (POS_CENTER, COMM_SUCCESS, 0)
        self._reset_write(arm)

        arm.jog_joint(0, direction=-1, step_deg=5.0)

        expected = LeRobotArm101._deg_to_pos(-5.0)
        gp = _goal_pos_calls(arm.packet_handler)
        assert len(gp) == 1
        assert abs(gp[0][0][3] - expected) <= 1

    def test_jog_custom_step_size(self, arm):
        arm.packet_handler.read2ByteTxRx.return_value = (POS_CENTER, COMM_SUCCESS, 0)
        self._reset_write(arm)

        arm.jog_joint(1, direction=+1, step_deg=10.0)

        expected = LeRobotArm101._deg_to_pos(10.0)
        gp = _goal_pos_calls(arm.packet_handler)
        assert abs(gp[0][0][3] - expected) <= 1

    @pytest.mark.parametrize("joint_idx", range(6))
    def test_jog_targets_correct_motor(self, arm, joint_idx):
        """jog_joint(joint_idx) must write to motor_ids[joint_idx]."""
        arm.packet_handler.read2ByteTxRx.return_value = (POS_CENTER, COMM_SUCCESS, 0)
        self._reset_write(arm)

        arm.jog_joint(joint_idx, direction=+1, step_deg=5.0)

        gp = _goal_pos_calls(arm.packet_handler)
        assert len(gp) == 1
        assert gp[0][0][1] == arm.motor_ids[joint_idx]

    def test_jog_read_error_returns_early_no_write(self, arm):
        """If read fails (sentinel < -180), jog_joint must not write."""
        arm.packet_handler.read2ByteTxRx.return_value = (0, 1, 0)  # comm error → -1
        self._reset_write(arm)

        arm.jog_joint(0, direction=+1)

        gp = _goal_pos_calls(arm.packet_handler)
        assert len(gp) == 0

    def test_jog_accumulates_from_non_center_position(self, arm):
        """Jog from 45° by +5° should produce target 50°."""
        pos_45 = LeRobotArm101._deg_to_pos(45.0)
        arm.packet_handler.read2ByteTxRx.return_value = (pos_45, COMM_SUCCESS, 0)
        self._reset_write(arm)

        arm.jog_joint(0, direction=+1, step_deg=5.0)

        expected = LeRobotArm101._deg_to_pos(50.0)
        gp = _goal_pos_calls(arm.packet_handler)
        assert abs(gp[0][0][3] - expected) <= 1

    def test_jog_reads_all_six_motors_before_writing(self, arm):
        arm.packet_handler.read2ByteTxRx.return_value = (POS_CENTER, COMM_SUCCESS, 0)
        arm.packet_handler.read2ByteTxRx.reset_mock()
        self._reset_write(arm)

        arm.jog_joint(2, direction=+1)

        assert arm.packet_handler.read2ByteTxRx.call_count == 6

    def test_jog_default_step_is_5_degrees(self, arm):
        """jog_joint without step_deg uses default 5.0°."""
        arm.packet_handler.read2ByteTxRx.return_value = (POS_CENTER, COMM_SUCCESS, 0)
        self._reset_write(arm)

        arm.jog_joint(0, direction=+1)  # step_deg omitted → 5.0

        expected = LeRobotArm101._deg_to_pos(5.0)
        gp = _goal_pos_calls(arm.packet_handler)
        assert abs(gp[0][0][3] - expected) <= 1

    def test_jog_with_custom_speed(self, arm):
        arm.packet_handler.read2ByteTxRx.return_value = (POS_CENTER, COMM_SUCCESS, 0)
        self._reset_write(arm)

        arm.jog_joint(0, direction=+1, step_deg=5.0, speed=600)

        gs = _goal_speed_calls(arm.packet_handler)
        assert gs[-1][0][3] == 600

    def test_jog_uses_default_speed_when_none(self, arm):
        arm.speed = 250
        arm.packet_handler.read2ByteTxRx.return_value = (POS_CENTER, COMM_SUCCESS, 0)
        self._reset_write(arm)

        arm.jog_joint(0, direction=+1, speed=None)

        gs = _goal_speed_calls(arm.packet_handler)
        assert gs[-1][0][3] == 250


# ─────────────────────────────────────────────────────────────────────────────
# 7. Duck-typed panel integration interface
# ─────────────────────────────────────────────────────────────────────────────

class TestPanelIntegration:
    """Test the duck-typed interface consumed by RobotControlPanel.

    LeRobotArm101 must provide:
      robot_type  – 'arm101'
      get_mode()  – 4 (disabled) or 5 (enabled)
      get_angles()  – list[6 float] or None
      get_pose()    – list[6 float] or None
      move_joints() – bool
      gripper_open / gripper_close
      speed (read/write attribute)
      _enabled (read/write for panel enable-toggle logic)
    """

    # ── Type detection ────────────────────────────────────────────────────────

    def test_robot_type_attribute(self, arm):
        """Panel detects arm101 via getattr(robot, 'robot_type') == 'arm101'."""
        assert arm.robot_type == 'arm101'

    def test_robot_type_detected_by_panel_pattern(self, arm):
        """Replicate RobotControlPanel's detection: getattr(...) == 'arm101'."""
        detected = getattr(arm, 'robot_type', None) == 'arm101'
        assert detected is True

    # ── get_mode ─────────────────────────────────────────────────────────────

    def test_get_mode_disabled(self, arm):
        arm._enabled = False
        assert arm.get_mode() == 4

    def test_get_mode_enabled(self, arm):
        arm._enabled = True
        assert arm.get_mode() == 5

    def test_get_mode_reflects_enable_torque(self, arm):
        arm.enable_torque()
        assert arm.get_mode() == 5

    def test_get_mode_reflects_disable_torque(self, arm_enabled):
        arm_enabled.disable_torque()
        assert arm_enabled.get_mode() == 4

    # ── get_angles ────────────────────────────────────────────────────────────

    def test_get_angles_returns_six_floats_on_success(self, arm):
        arm.packet_handler.read2ByteTxRx.return_value = (POS_CENTER, COMM_SUCCESS, 0)
        angles = arm.get_angles()
        assert angles is not None
        assert len(angles) == 6
        assert all(isinstance(a, float) for a in angles)

    def test_get_angles_zero_at_center(self, arm):
        arm.packet_handler.read2ByteTxRx.return_value = (POS_CENTER, COMM_SUCCESS, 0)
        angles = arm.get_angles()
        assert all(abs(a) < 1e-3 for a in angles)

    def test_get_angles_returns_none_on_comm_error(self, arm):
        """Any read returning error sentinel → get_angles returns None."""
        arm.packet_handler.read2ByteTxRx.return_value = (0, 1, 0)  # comm error
        assert arm.get_angles() is None

    def test_get_angles_returns_none_on_exception(self, arm):
        arm.packet_handler.read2ByteTxRx.side_effect = Exception("serial error")
        assert arm.get_angles() is None

    # ── get_pose ─────────────────────────────────────────────────────────────

    def test_get_pose_returns_none_when_fk_unavailable(self, arm):
        """get_pose() must return None (not raise) when FK import fails."""
        arm.packet_handler.read2ByteTxRx.return_value = (POS_CENTER, COMM_SUCCESS, 0)
        with mock.patch('robot.lerobot_arm101._get_fk_solver',
                        side_effect=ImportError("no pinocchio")):
            pose = arm.get_pose()
        assert pose is None

    def test_get_pose_returns_six_element_list_with_mock_fk(self, arm):
        """get_pose() returns [x,y,z,rx,ry,rz] when FK succeeds."""
        import numpy as np
        arm.packet_handler.read2ByteTxRx.return_value = (POS_CENTER, COMM_SUCCESS, 0)
        mock_solver = mock.MagicMock()
        mock_solver.forward_kin.return_value = (
            np.array([100.0, 50.0, 300.0]),
            np.array([180.0, 0.0, 30.0]),
        )
        with mock.patch('robot.lerobot_arm101._get_fk_solver',
                        return_value=mock_solver):
            pose = arm.get_pose()

        assert pose is not None
        assert len(pose) == 6
        assert pose[0] == pytest.approx(100.0)
        assert pose[1] == pytest.approx(50.0)
        assert pose[2] == pytest.approx(300.0)
        assert pose[3] == pytest.approx(180.0)
        assert pose[5] == pytest.approx(30.0)

    def test_get_pose_returns_none_when_angles_fail(self, arm):
        """get_pose() returns None if get_angles() returns None."""
        arm.packet_handler.read2ByteTxRx.return_value = (0, 1, 0)  # comm error
        assert arm.get_pose() is None

    def test_get_pose_passes_first_five_angles_to_fk(self, arm):
        """FK solver receives only the first 5 motor angles (gripper excluded)."""
        import numpy as np
        arm.packet_handler.read2ByteTxRx.return_value = (POS_CENTER, COMM_SUCCESS, 0)
        mock_solver = mock.MagicMock()
        mock_solver.forward_kin.return_value = (
            np.zeros(3), np.zeros(3)
        )
        with mock.patch('robot.lerobot_arm101._get_fk_solver',
                        return_value=mock_solver):
            arm.get_pose()

        args = mock_solver.forward_kin.call_args[0][0]
        assert len(args) == 5

    # ── move_joints ───────────────────────────────────────────────────────────

    def test_move_joints_returns_true_on_success(self, arm):
        assert arm.move_joints([0.0] * 6) is True

    def test_move_joints_sends_to_all_motors(self, arm):
        arm.packet_handler.write2ByteTxRx.reset_mock()
        arm.move_joints([0.0, 10.0, -20.0, 30.0, -45.0, 5.0])
        gp = _goal_pos_calls(arm.packet_handler)
        assert len(gp) == 6

    def test_move_joints_returns_false_on_exception(self, arm):
        arm.packet_handler.write2ByteTxRx.return_value = (1, 0)  # comm error
        arm.packet_handler.getTxRxResult.return_value = "COMM_TX_FAIL"
        assert arm.move_joints([0.0] * 6) is False

    def test_move_joints_uses_custom_speed(self, arm):
        arm.packet_handler.write2ByteTxRx.reset_mock()
        arm.move_joints([0.0] * 6, speed=500)
        gs = _goal_speed_calls(arm.packet_handler)
        assert all(c[0][3] == 500 for c in gs)

    # ── speed attribute ───────────────────────────────────────────────────────

    def test_speed_attribute_readable_and_writable(self, arm):
        arm.speed = 350
        assert arm.speed == 350

    def test_default_speed_matches_constant(self, arm):
        assert arm.speed == DEFAULT_MOVE_SPEED

    def test_safe_mode_init_sets_reduced_speed(self):
        robot = LeRobotArm101('/dev/ttyACM0', safe_mode=True)
        assert robot.speed == SAFE_MODE_SPEED


# ─────────────────────────────────────────────────────────────────────────────
# 8. Initialisation / constructor
# ─────────────────────────────────────────────────────────────────────────────

class TestInitialisation:
    """Test constructor parameters and defaults."""

    def test_default_motor_ids(self, arm):
        assert arm.motor_ids == list(DEFAULT_MOTOR_IDS)

    def test_custom_motor_ids(self):
        robot = LeRobotArm101('/dev/ttyACM0', motor_ids=[1, 2, 3])
        assert robot.motor_ids == [1, 2, 3]

    def test_default_speed(self, arm):
        assert arm.speed == DEFAULT_MOVE_SPEED

    def test_custom_speed(self):
        robot = LeRobotArm101('/dev/ttyACM0', speed=300)
        assert robot.speed == 300

    def test_safe_mode_on_sets_speed_and_flag(self):
        robot = LeRobotArm101('/dev/ttyACM0', safe_mode=True)
        assert robot.safe_mode is True
        assert robot.speed == SAFE_MODE_SPEED

    def test_safe_mode_off_by_default(self, arm):
        assert arm.safe_mode is False

    def test_not_enabled_by_default(self, arm):
        assert arm._enabled is False

    def test_custom_baudrate(self):
        robot = LeRobotArm101('/dev/ttyACM0', baudrate=115200)
        assert robot.baudrate == 115200

    def test_port_path_stored(self, arm):
        assert arm.port_path == '/dev/ttyACM0'
