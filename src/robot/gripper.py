"""Gripper control for Dobot Nova5 via ToolDO.

Uses the dual-solenoid pneumatic gripper convention:
  ToolDO(1, 1) = close gripper
  ToolDO(2, 1) = open gripper
"""

import time


class Gripper:
    """Controls a dual-solenoid pneumatic gripper via ToolDO."""

    def __init__(self, robot, actuate_delay: float = 0.5):
        """
        Args:
            robot: DobotNova5 instance
            actuate_delay: Time in seconds to wait after sending command
        """
        self.robot = robot
        self.actuate_delay = actuate_delay
        self.is_closed = False

    def open(self):
        """Open the gripper."""
        self.robot.tool_do(2, 1)
        time.sleep(self.actuate_delay)
        self.is_closed = False

    def close(self):
        """Close the gripper."""
        self.robot.tool_do(1, 1)
        time.sleep(self.actuate_delay)
        self.is_closed = True
