"""Local inverse kinematics solver for Dobot Nova5 using Pinocchio.

Uses the official Nova5 URDF with an added tool_tip frame for the gripper.
All external interfaces use mm/degrees (matching the Dobot protocol).
Pinocchio internally uses m/radians.
"""

import os
import numpy as np
import pinocchio as pin
from logger import get_logger

log = get_logger('ik')

# Path to the URDF (relative to project root)
_URDF_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'nova5_robot.urdf')


class IKSolver:
    """Inverse/forward kinematics for the Nova5 + gripper.

    All public methods use the Dobot convention:
      - Positions in mm
      - Angles in degrees
      - RPY orientation (rx, ry, rz) in degrees

    Internally converts to Pinocchio's m/radians.
    """

    def __init__(self, tool_length_mm: float = 100.0, urdf_path: str = None):
        """
        Args:
            tool_length_mm: Distance from flange to gripper fingertip in mm.
                            The URDF ships with 100mm default; this adjusts it.
            urdf_path: Override path to URDF file.
        """
        urdf = urdf_path or _URDF_PATH
        self.model = pin.buildModelFromUrdf(urdf)
        self.data = self.model.createData()

        # Find the tool_tip frame
        self.ee_frame_id = self.model.getFrameId('tool_tip')
        if self.ee_frame_id >= self.model.nframes:
            raise RuntimeError("tool_tip frame not found in URDF")

        # Adjust tool length if different from the URDF default (100mm)
        self._adjust_tool_length(tool_length_mm)

        # Joint limits from URDF (in radians)
        self.q_min = self.model.lowerPositionLimit.copy()
        self.q_max = self.model.upperPositionLimit.copy()

        log.info(f"IK solver loaded: {self.model.nq} joints, "
                 f"tool_length={tool_length_mm}mm, ee_frame=tool_tip")

    def _adjust_tool_length(self, desired_mm: float):
        """Adjust the tool_tip frame placement to match actual gripper length."""
        frame = self.model.frames[self.ee_frame_id]
        # The URDF has tool_joint at z=-0.100 (100mm in local -Z direction)
        # Replace with the desired length
        current_offset = frame.placement.translation.copy()
        # Tool extends along local -Z of Link6
        current_offset[2] = -desired_mm / 1000.0
        frame.placement.translation = current_offset

    def forward_kin(self, joints_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics: joint angles -> TCP pose.

        Args:
            joints_deg: 6 joint angles in degrees

        Returns:
            (position_mm, rpy_deg): position [x,y,z] in mm, orientation [rx,ry,rz] in degrees
        """
        q = np.radians(joints_deg)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        placement = self.data.oMf[self.ee_frame_id]
        pos_mm = placement.translation * 1000.0
        rpy_deg = np.degrees(pin.rpy.matrixToRpy(placement.rotation))

        return pos_mm, rpy_deg

    def solve_ik(self, target_pos_mm: np.ndarray, target_rpy_deg: np.ndarray,
                 seed_joints_deg: np.ndarray = None,
                 max_iter: int = 200, eps: float = 1e-4,
                 dt: float = 0.1, damp: float = 1e-6) -> np.ndarray | None:
        """Inverse kinematics: Cartesian pose -> joint angles.

        Uses damped least-squares (Levenberg-Marquardt) iterative IK.

        Args:
            target_pos_mm: [x, y, z] target position in mm
            target_rpy_deg: [rx, ry, rz] target orientation in degrees
            seed_joints_deg: Initial joint angles in degrees (uses zeros if None)
            max_iter: Maximum iterations
            eps: Convergence threshold (combined position m + rotation rad error)
            dt: Step size for integration
            damp: Damping factor for singularity robustness

        Returns:
            6 joint angles in degrees, or None if no solution found.
        """
        # Convert to Pinocchio units (m, rad)
        target_pos = target_pos_mm / 1000.0
        target_rpy = np.radians(target_rpy_deg)
        target_rotation = pin.rpy.rpyToMatrix(target_rpy[0], target_rpy[1], target_rpy[2])
        oMdes = pin.SE3(target_rotation, target_pos)

        # Seed
        if seed_joints_deg is not None:
            q = np.radians(seed_joints_deg).copy()
        else:
            q = pin.neutral(self.model)

        for i in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            current = self.data.oMf[self.ee_frame_id]
            error_se3 = pin.log(current.actInv(oMdes))
            err = error_se3.vector

            if np.linalg.norm(err) < eps:
                # Normalize to [-pi, pi]
                q_result = np.degrees(self._normalize_angles(q))
                log.debug(f"IK converged in {i+1} iterations")
                return q_result

            # Compute frame Jacobian
            J = pin.computeFrameJacobian(self.model, self.data, q,
                                          self.ee_frame_id,
                                          pin.ReferenceFrame.LOCAL)

            # Damped pseudo-inverse
            JtJ = J.T @ J + damp * np.eye(self.model.nv)
            dq = np.linalg.solve(JtJ, J.T @ err)

            q = pin.integrate(self.model, q, dt * dq)

            # Clamp to joint limits
            q = np.clip(q, self.q_min, self.q_max)

        log.warning(f"IK failed to converge after {max_iter} iterations "
                    f"(final error={np.linalg.norm(err):.6f})")
        return None

    def interpolate_linear(self, start_pos_mm: np.ndarray, start_rpy_deg: np.ndarray,
                           end_pos_mm: np.ndarray, end_rpy_deg: np.ndarray,
                           seed_joints_deg: np.ndarray,
                           step_mm: float = 5.0) -> list[np.ndarray] | None:
        """Interpolate a Cartesian linear path and solve IK for each waypoint.

        Args:
            start_pos_mm, start_rpy_deg: Start pose
            end_pos_mm, end_rpy_deg: End pose
            seed_joints_deg: Starting joint angles for first IK solve
            step_mm: Maximum Cartesian step size in mm

        Returns:
            List of joint angle arrays (degrees), or None if any IK fails.
        """
        dist = np.linalg.norm(end_pos_mm - start_pos_mm)
        n_steps = max(2, int(np.ceil(dist / step_mm)))

        joint_path = []
        current_seed = seed_joints_deg.copy()

        for i in range(n_steps):
            t = i / (n_steps - 1)
            pos = start_pos_mm + t * (end_pos_mm - start_pos_mm)
            rpy = start_rpy_deg + t * (end_rpy_deg - start_rpy_deg)

            joints = self.solve_ik(pos, rpy, seed_joints_deg=current_seed)
            if joints is None:
                log.error(f"Linear interpolation IK failed at step {i}/{n_steps} "
                          f"pos={pos}")
                return None
            joint_path.append(joints)
            current_seed = joints

        return joint_path

    @staticmethod
    def _normalize_angles(q: np.ndarray) -> np.ndarray:
        """Normalize joint angles to [-pi, pi]."""
        return (q + np.pi) % (2 * np.pi) - np.pi
