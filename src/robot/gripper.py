"""Gripper control abstraction for Dobot Nova5.

The gripper is controlled via digital output (DO) ports on the robot.
Adjust DO_PORT and logic levels based on the actual gripper hardware.
"""

import time


class Gripper:
    """Controls a gripper attached to the Dobot Nova5 via digital I/O."""

    def __init__(self, robot, do_port: int = 1, close_is_high: bool = True,
                 actuate_delay: float = 0.5):
        """
        Args:
            robot: DobotNova5 instance
            do_port: Digital output port number controlling the gripper
            close_is_high: If True, DO high = gripper closed; False = inverted
            actuate_delay: Time in seconds to wait after sending command
        """
        self.robot = robot
        self.do_port = do_port
        self.close_is_high = close_is_high
        self.actuate_delay = actuate_delay
        self.is_closed = False

    def open(self):
        """Open the gripper."""
        value = not self.close_is_high
        self.robot.set_digital_output(self.do_port, value)
        time.sleep(self.actuate_delay)
        self.is_closed = False

    def close(self):
        """Close the gripper."""
        value = self.close_is_high
        self.robot.set_digital_output(self.do_port, value)
        time.sleep(self.actuate_delay)
        self.is_closed = True
