"""Cube face alignment — align gripper jaws with cube edges during top-down approach.

When grasping a cube from above, the gripper's jaw orientation must match one
of the cube's four edge directions so the jaws close cleanly on opposite faces.
This module computes the optimal wrist_roll (J5) angle to achieve that alignment.

Key concepts:
  - A square cube has 4-fold rotational symmetry: edges repeat every 90 degrees.
  - The gripper has 2-fold symmetry: jaws repeat every 180 degrees.
  - Combined, we need to align within ±45 degrees of any cube edge.
  - The cube yaw detector (minAreaRect) already normalizes to [-45, +45).
  - We pick the J5 adjustment that requires minimum rotation from current pose.

Algorithm:
  1. Detect the cube's yaw angle in the camera image (via green_cube_detector).
  2. Transform the image-space yaw to robot-space yaw (accounting for camera
     mount angle and any camera-to-base rotation).
  3. Compute the 4 candidate J5 angles (one per cube face, 90 degrees apart).
  4. Pick the candidate closest to the current J5 to minimize movement.
  5. Move J5 to the selected angle.

The module also provides a ``FaceAlignmentPlan`` that bundles: the detected
cube yaw, all candidate J5 angles, the selected angle, and quality metrics.

Usage::

    from cube_face_aligner import CubeFaceAligner

    aligner = CubeFaceAligner.from_config(config)
    plan = aligner.compute_alignment(
        detected_yaw_deg=15.3,
        current_j5_deg=-10.0,
    )
    if plan.valid:
        new_j5 = plan.selected_j5_deg
        # Apply: set J5 = new_j5 while keeping other joints unchanged
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from logger import get_logger

log = get_logger(__name__)


@dataclass
class FaceAlignmentPlan:
    """Result of computing gripper-to-cube face alignment."""

    # Detected cube yaw in image space (degrees, [-45, +45) from detector).
    detected_yaw_deg: float = 0.0

    # Cube yaw transformed to robot/gripper space (degrees).
    robot_yaw_deg: float = 0.0

    # All 4 candidate J5 angles (one per cube face, 90 degrees apart).
    candidate_j5_degs: list[float] = field(default_factory=list)

    # Current J5 angle (degrees).
    current_j5_deg: float = 0.0

    # Selected J5 angle (minimum rotation from current).
    selected_j5_deg: float = 0.0

    # Rotation delta to apply (degrees, signed).
    delta_deg: float = 0.0

    # Which face index was selected (0-3).
    face_index: int = 0

    # Whether the plan is valid and actionable.
    valid: bool = False

    # Human-readable status.
    status: str = 'not computed'

    def summary(self) -> str:
        """One-line summary of the alignment plan."""
        if not self.valid:
            return f'Invalid: {self.status}'
        return (f'Face {self.face_index}: J5 {self.current_j5_deg:.1f}'
                + chr(176) + f' -> {self.selected_j5_deg:.1f}' + chr(176)
                + f' (delta={self.delta_deg:+.1f}' + chr(176)
                + f', cube_yaw={self.detected_yaw_deg:.1f}' + chr(176) + ')')


class CubeFaceAligner:
    """Compute optimal wrist_roll (J5) for gripper-to-cube edge alignment.

    Args:
        mount_angle_deg: Rotation of the gripper camera relative to the
            gripper body (degrees).  0 = camera X-axis aligned with gripper.
        cam_flip: If True, negate the detected yaw (for mirrored cameras).
        j5_min_deg: Minimum allowed J5 angle (degrees).
        j5_max_deg: Maximum allowed J5 angle (degrees).
        deadband_deg: If the required rotation is smaller than this, skip
            the move (already well-aligned).
    """

    def __init__(self,
                 mount_angle_deg: float = 0.0,
                 cam_flip: bool = False,
                 j5_min_deg: float = -150.0,
                 j5_max_deg: float = 150.0,
                 deadband_deg: float = 2.0):
        self.mount_angle_deg = mount_angle_deg
        self.cam_flip = cam_flip
        self.j5_min_deg = j5_min_deg
        self.j5_max_deg = j5_max_deg
        self.deadband_deg = deadband_deg

    @classmethod
    def from_config(cls, config: dict) -> 'CubeFaceAligner':
        """Create from robot_config.yaml dict.

        Reads ``visual_servoing.mount_angle_deg`` and camera flip settings.

        Args:
            config: Loaded robot_config dict.

        Returns:
            Configured CubeFaceAligner instance.
        """
        vs_cfg = config.get('visual_servoing', config.get('visual_servo', {}))
        mount_angle = float(vs_cfg.get('mount_angle_deg', 0.0))
        cam_flip = bool(vs_cfg.get('cam_flip_x', False))
        return cls(mount_angle_deg=mount_angle, cam_flip=cam_flip)

    def compute_alignment(self,
                          detected_yaw_deg: float,
                          current_j5_deg: float,
                          gripper_rz_deg: float = 0.0) -> FaceAlignmentPlan:
        """Compute the optimal J5 angle to align gripper with cube face.

        Args:
            detected_yaw_deg: Cube yaw from the green_cube_detector, in image
                coordinates (degrees, typically [-45, +45) from minAreaRect).
            current_j5_deg: Current wrist_roll (J5) angle in degrees.
            gripper_rz_deg: Current gripper Rz in the base frame (degrees).
                For top-down approach this is approximately equal to J1
                (shoulder_pan) + J5 offsets.  Only needed if the camera is
                the overhead (non-gripper) camera; for gripper-cam this is 0.

        Returns:
            FaceAlignmentPlan with the selected J5 angle and diagnostics.
        """
        plan = FaceAlignmentPlan(
            detected_yaw_deg=detected_yaw_deg,
            current_j5_deg=current_j5_deg,
        )

        # Step 1: Transform image yaw to robot/gripper space
        yaw = detected_yaw_deg
        if self.cam_flip:
            yaw = -yaw
        # Account for camera mount rotation
        robot_yaw = yaw + self.mount_angle_deg + gripper_rz_deg
        # Normalize to [-45, +45) (cube 4-fold symmetry)
        robot_yaw = _normalize_45(robot_yaw)
        plan.robot_yaw_deg = robot_yaw

        # Step 2: Generate 4 candidate J5 angles (one per cube face)
        # The base alignment: current_j5 + robot_yaw aligns with one face.
        # The other 3 faces are at +90, +180, +270 from there.
        base_j5 = current_j5_deg + robot_yaw
        candidates = []
        for k in range(4):
            c = base_j5 + k * 90.0
            # Normalize to range centered around current J5
            c = _wrap_nearest(c, current_j5_deg)
            candidates.append(c)
        plan.candidate_j5_degs = candidates

        # Step 3: Filter candidates within joint limits
        valid_candidates = []
        for i, c in enumerate(candidates):
            if self.j5_min_deg <= c <= self.j5_max_deg:
                valid_candidates.append((i, c))

        if not valid_candidates:
            plan.status = 'no candidate within J5 limits'
            plan.valid = False
            log.warning(f'Face alignment: no valid J5 candidate. '
                        f'Candidates={candidates}, limits=[{self.j5_min_deg}, {self.j5_max_deg}]')
            return plan

        # Step 4: Pick candidate with minimum rotation from current J5
        best_idx, best_j5 = min(valid_candidates, key=lambda x: abs(x[1] - current_j5_deg))
        delta = best_j5 - current_j5_deg

        plan.selected_j5_deg = best_j5
        plan.delta_deg = delta
        plan.face_index = best_idx
        plan.valid = True

        if abs(delta) < self.deadband_deg:
            plan.status = f'already aligned (delta={delta:.1f}' + chr(176) + ')'
        else:
            plan.status = f'align face {best_idx}: J5 {current_j5_deg:.1f} -> {best_j5:.1f}'

        log.debug(f'Face alignment: {plan.summary()}')
        return plan

    def align_robot(self, robot, detected_yaw_deg: float,
                    gripper_rz_deg: float = 0.0,
                    speed: int = 100) -> FaceAlignmentPlan:
        """Compute alignment and move the robot's J5.

        Convenience method that reads the current J5 from the robot, computes
        the plan, and executes the move if needed.

        Args:
            robot: Connected robot object with get_angles() and move_joints().
            detected_yaw_deg: Cube yaw from the detector (degrees).
            gripper_rz_deg: Gripper Rz in base frame (degrees), for overhead cam.
            speed: Servo speed for the alignment move.

        Returns:
            FaceAlignmentPlan (with valid=True if move was made or not needed).
        """
        angles = robot.get_angles()
        if not angles or len(angles) < 6:
            plan = FaceAlignmentPlan(status='cannot read joint angles')
            return plan

        current_j5 = angles[4]  # wrist_roll is motor index 4
        plan = self.compute_alignment(detected_yaw_deg, current_j5, gripper_rz_deg)

        if not plan.valid:
            return plan

        if abs(plan.delta_deg) < self.deadband_deg:
            log.info(f'Face alignment: already aligned ({plan.delta_deg:.1f} deg)')
            return plan

        # Execute the move — only change J5, keep everything else
        cmd = list(angles)
        cmd[4] = plan.selected_j5_deg
        log.info(f'Face alignment: J5 {current_j5:.1f} -> {plan.selected_j5_deg:.1f} '
                 f'(delta={plan.delta_deg:+.1f} deg)')
        ok = robot.move_joints(cmd, speed=speed)
        if not ok:
            plan.status = 'move_joints failed'
            plan.valid = False

        return plan


def _normalize_45(angle_deg: float) -> float:
    """Normalize angle to [-45, +45) range (4-fold symmetry)."""
    a = angle_deg % 360.0  # [0, 360)
    a = ((a + 45.0) % 90.0) - 45.0
    return a


def _wrap_nearest(target: float, reference: float) -> float:
    """Wrap target angle to be within ±180 degrees of reference.

    Used to pick the candidate J5 angle nearest to the current J5 in
    the circular angle space.
    """
    delta = target - reference
    delta = (delta + 180.0) % 360.0 - 180.0
    return reference + delta


def batch_alignment_test(aligner: CubeFaceAligner,
                         yaw_angles: list[float],
                         current_j5: float = 0.0) -> str:
    """Run alignment computation for a batch of yaw angles and format results.

    Useful for validating the algorithm across many orientations.

    Args:
        aligner: Configured CubeFaceAligner.
        yaw_angles: List of detected yaw angles to test (degrees).
        current_j5: Starting J5 angle (degrees).

    Returns:
        Formatted multi-line result string.
    """
    lines = []
    lines.append(f'{"Yaw":>8} {"RobotYaw":>9} {"J5 curr":>8} {"J5 new":>8} '
                 f'{"Delta":>8} {"Face":>5} {"Status"}')
    lines.append('-' * 70)

    for yaw in yaw_angles:
        plan = aligner.compute_alignment(yaw, current_j5)
        d = chr(176)  # degree symbol
        lines.append(
            f'{yaw:>7.1f}{d} {plan.robot_yaw_deg:>8.1f}{d} '
            f'{current_j5:>7.1f}{d} {plan.selected_j5_deg:>7.1f}{d} '
            f'{plan.delta_deg:>+7.1f}{d} {plan.face_index:>5} '
            f'{plan.status}'
        )

    return '\n'.join(lines)
