"""Gripper control for Dobot Nova5 via ToolDOInstant.

Electric motor gripper with dual-channel control:
  Close: ToolDOInstant(2,0) then ToolDOInstant(1,1)
  Open:  ToolDOInstant(1,0) then ToolDOInstant(2,1)

Channel 1 = close signal, Channel 2 = open signal.
Must turn off the opposing channel before activating.
"""

import time


class Gripper:
    """Controls an electric gripper via ToolDOInstant."""

    def __init__(self, robot, actuate_delay: float = 1.0):
        """
        Args:
            robot: DobotNova5 instance
            actuate_delay: Time in seconds to wait for gripper to finish moving
        """
        self.robot = robot
        self.actuate_delay = actuate_delay
        self.is_closed = False

    def open(self):
        """Open the gripper: release close channel, then activate open channel."""
        self.robot.tool_do_instant(1, 0)
        self.robot.tool_do_instant(2, 1)
        time.sleep(self.actuate_delay)
        self.is_closed = False

    def close(self):
        """Close the gripper: release open channel, then activate close channel."""
        self.robot.tool_do_instant(2, 0)
        self.robot.tool_do_instant(1, 1)
        time.sleep(self.actuate_delay)
        self.is_closed = True
