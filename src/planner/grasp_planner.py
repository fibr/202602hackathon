"""Grasp planning for picking up a lying rod and standing it upright."""

import numpy as np
from dataclasses import dataclass
from enum import Enum


class MotionType(Enum):
    JOINT = "MovJ"    # Fast joint-space move (for large repositioning)
    LINEAR = "MovL"   # Smooth linear move (for precise approach/place)


class GripperAction(Enum):
    OPEN = "open"
    CLOSE = "close"
    NONE = "none"


@dataclass
class Waypoint:
    """A single motion waypoint."""
    x: float          # mm
    y: float          # mm
    z: float          # mm
    rx: float         # degrees
    ry: float         # degrees
    rz: float         # degrees
    motion: MotionType
    gripper: GripperAction = GripperAction.NONE
    label: str = ""


class GraspPlanner:
    """Plans waypoints for picking up a horizontal rod and standing it vertically."""

    def __init__(self, safe_z: float = 300.0, approach_offset_z: float = 100.0,
                 place_position: tuple = (300.0, 0.0, 50.0),
                 gripper_down_rx: float = 180.0):
        """
        Args:
            safe_z: Safe height above table in mm
            approach_offset_z: Height above rod for pre-grasp in mm
            place_position: (x, y, z) where to place the standing rod in mm
            gripper_down_rx: rx angle when gripper points straight down (degrees)
        """
        self.safe_z = safe_z
        self.approach_offset_z = approach_offset_z
        self.place_x, self.place_y, self.place_z = place_position
        self.gripper_down_rx = gripper_down_rx

    def plan(self, rod_center_base: np.ndarray, rod_axis_base: np.ndarray,
             home_joints: np.ndarray = None) -> list[Waypoint]:
        """Generate waypoints for the full pick-and-stand sequence.

        Args:
            rod_center_base: [x, y, z] rod center in robot base frame (mm)
            rod_axis_base: unit vector along rod axis in robot base frame
            home_joints: Optional home joint angles to return to

        Returns:
            List of Waypoint objects to execute in order
        """
        rod_x, rod_y, rod_z = rod_center_base

        # Compute gripper orientation to align with rod
        # The gripper should approach perpendicular to the rod axis
        grasp_rz = np.degrees(np.arctan2(rod_axis_base[1], rod_axis_base[0]))

        waypoints = []

        # 1. Pre-grasp: above the rod, gripper pointing down
        waypoints.append(Waypoint(
            x=rod_x, y=rod_y, z=rod_z + self.approach_offset_z,
            rx=self.gripper_down_rx, ry=0.0, rz=grasp_rz,
            motion=MotionType.JOINT,
            gripper=GripperAction.OPEN,
            label="PRE_GRASP"
        ))

        # 2. Descend to grasp position
        waypoints.append(Waypoint(
            x=rod_x, y=rod_y, z=rod_z,
            rx=self.gripper_down_rx, ry=0.0, rz=grasp_rz,
            motion=MotionType.LINEAR,
            gripper=GripperAction.NONE,
            label="GRASP_DESCEND"
        ))

        # 3. Close gripper
        waypoints.append(Waypoint(
            x=rod_x, y=rod_y, z=rod_z,
            rx=self.gripper_down_rx, ry=0.0, rz=grasp_rz,
            motion=MotionType.LINEAR,
            gripper=GripperAction.CLOSE,
            label="GRASP_CLOSE"
        ))

        # 4. Lift to safe height
        waypoints.append(Waypoint(
            x=rod_x, y=rod_y, z=self.safe_z,
            rx=self.gripper_down_rx, ry=0.0, rz=grasp_rz,
            motion=MotionType.LINEAR,
            label="LIFT"
        ))

        # 5. Move above placement location (still horizontal)
        waypoints.append(Waypoint(
            x=self.place_x, y=self.place_y, z=self.safe_z,
            rx=self.gripper_down_rx, ry=0.0, rz=0.0,
            motion=MotionType.JOINT,
            label="MOVE_TO_PLACE"
        ))

        # 6. Reorient: rotate wrist so rod is vertical
        # Tilt the gripper 90 degrees so the rod points down
        waypoints.append(Waypoint(
            x=self.place_x, y=self.place_y, z=self.safe_z,
            rx=self.gripper_down_rx - 90.0, ry=0.0, rz=0.0,
            motion=MotionType.JOINT,
            label="REORIENT"
        ))

        # 7. Lower to table surface
        waypoints.append(Waypoint(
            x=self.place_x, y=self.place_y, z=self.place_z,
            rx=self.gripper_down_rx - 90.0, ry=0.0, rz=0.0,
            motion=MotionType.LINEAR,
            label="PLACE_DESCEND"
        ))

        # 8. Open gripper to release rod standing up
        waypoints.append(Waypoint(
            x=self.place_x, y=self.place_y, z=self.place_z,
            rx=self.gripper_down_rx - 90.0, ry=0.0, rz=0.0,
            motion=MotionType.LINEAR,
            gripper=GripperAction.OPEN,
            label="RELEASE"
        ))

        # 9. Retract upward
        waypoints.append(Waypoint(
            x=self.place_x, y=self.place_y, z=self.safe_z,
            rx=self.gripper_down_rx, ry=0.0, rz=0.0,
            motion=MotionType.LINEAR,
            label="RETRACT"
        ))

        return waypoints
