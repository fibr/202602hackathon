"""FK/IK solver for SO-ARM101 using Pinocchio + official URDF.

Uses the SO-101 "new calibration" URDF where joint zero corresponds to
the middle of each joint's range — matching our servo convention where
raw position 2048 = 0 degrees.

The URDF kinematic chain (5 DOF + gripper):
  base_link -> shoulder_pan -> shoulder_lift -> elbow_flex
            -> wrist_flex -> wrist_roll -> gripper

The gripper joint is excluded from IK (always fixed at 0).  IK solves
for the first 5 joints only.

All public interfaces use mm and degrees.  Internally uses m and radians.

Per-joint sign corrections (JOINT_SIGNS) account for differences between
the motor's positive rotation direction and the URDF convention.  These
are determined empirically — jog each joint with the control panel and
verify FK moves the same way.
"""

import os
import numpy as np
import pinocchio as pin

# URDF path relative to this file
_URDF_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'assets', 'so101', 'so101_new_calib.urdf')

# Per-joint sign correction: +1 = motor and URDF agree, -1 = inverted.
# Order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
# These may need empirical tuning — start with all +1.
JOINT_SIGNS = np.array([+1, +1, +1, +1, +1], dtype=float)

# Per-joint offset in degrees (motor_deg * sign + offset = urdf_deg)
JOINT_OFFSETS_DEG = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# Number of IK-controlled joints (exclude gripper)
N_IK_JOINTS = 5

# End-effector frame name in the URDF
EE_FRAME = 'gripper_frame_link'


class Arm101IKSolver:
    """FK/IK for SO-ARM101 using Pinocchio.

    All public methods use mm/degrees (matching the servo driver).

    Args:
        urdf_path: Override path to URDF file.
        joint_signs: Per-joint sign array (5,). None = use defaults.
        joint_offsets_deg: Per-joint offset array (5,) in degrees. None = use defaults.
    """

    def __init__(self, urdf_path: str = None,
                 joint_signs: np.ndarray = None,
                 joint_offsets_deg: np.ndarray = None):
        urdf = urdf_path or _URDF_PATH
        if not os.path.exists(urdf):
            raise FileNotFoundError(f"URDF not found: {urdf}")

        self.model = pin.buildModelFromUrdf(urdf)
        self.data = self.model.createData()

        # Find end-effector frame
        self.ee_frame_id = self.model.getFrameId(EE_FRAME)
        if self.ee_frame_id >= self.model.nframes:
            raise RuntimeError(f"Frame '{EE_FRAME}' not found in URDF")

        self.signs = joint_signs if joint_signs is not None else JOINT_SIGNS.copy()
        self.offsets_deg = (joint_offsets_deg if joint_offsets_deg is not None
                           else JOINT_OFFSETS_DEG.copy())

        # Joint limits from URDF (all 6 joints including gripper)
        self.q_min = self.model.lowerPositionLimit.copy()
        self.q_max = self.model.upperPositionLimit.copy()

    def _motor_to_urdf(self, motor_deg: np.ndarray) -> np.ndarray:
        """Convert motor angles (deg) to URDF joint angles (rad).

        Args:
            motor_deg: 5 or 6 motor angles in degrees.

        Returns:
            6 joint angles in radians (gripper set to 0 if only 5 given).
        """
        q_rad = np.zeros(self.model.nq)
        n = min(len(motor_deg), N_IK_JOINTS)
        for i in range(n):
            q_rad[i] = np.radians(motor_deg[i] * self.signs[i] + self.offsets_deg[i])
        # If 6 angles given, pass gripper through (motor 6, URDF joint 6)
        if len(motor_deg) > N_IK_JOINTS:
            q_rad[N_IK_JOINTS] = np.radians(motor_deg[N_IK_JOINTS])
        return q_rad

    def _urdf_to_motor(self, q_rad: np.ndarray) -> np.ndarray:
        """Convert URDF joint angles (rad) to motor angles (deg).

        Returns:
            5 motor angles in degrees (excludes gripper).
        """
        motor_deg = np.zeros(N_IK_JOINTS)
        for i in range(N_IK_JOINTS):
            urdf_deg = np.degrees(q_rad[i])
            motor_deg[i] = (urdf_deg - self.offsets_deg[i]) / self.signs[i]
        return motor_deg

    def forward_kin(self, motor_angles_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics: motor angles -> TCP pose.

        Args:
            motor_angles_deg: 5 or 6 motor angles in degrees
                              (from get_angles(), 0° = servo center).

        Returns:
            (position_mm, rpy_deg): [x,y,z] in mm, [rx,ry,rz] in degrees.
        """
        q = self._motor_to_urdf(motor_angles_deg)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        placement = self.data.oMf[self.ee_frame_id]
        pos_mm = placement.translation * 1000.0
        rpy_deg = np.degrees(pin.rpy.matrixToRpy(placement.rotation))
        return pos_mm, rpy_deg

    def solve_ik(self, target_pos_mm: np.ndarray, target_rpy_deg: np.ndarray,
                 seed_motor_deg: np.ndarray = None,
                 max_iter: int = 200, eps: float = 1e-4,
                 dt: float = 0.1, damp: float = 1e-6) -> np.ndarray | None:
        """Inverse kinematics: Cartesian pose -> motor angles.

        Uses damped least-squares on the first 5 joints (gripper fixed).

        Args:
            target_pos_mm: [x, y, z] in mm.
            target_rpy_deg: [rx, ry, rz] in degrees.
            seed_motor_deg: 5 motor angles in degrees as starting point.
            max_iter: Maximum iterations.
            eps: Convergence threshold.
            dt: Step size.
            damp: Damping factor.

        Returns:
            5 motor angles in degrees, or None if no solution.
        """
        target_pos = target_pos_mm / 1000.0
        target_rpy = np.radians(target_rpy_deg)
        target_rot = pin.rpy.rpyToMatrix(target_rpy[0], target_rpy[1], target_rpy[2])
        oMdes = pin.SE3(target_rot, target_pos)

        # Build full q (6 joints) from seed
        if seed_motor_deg is not None:
            q = self._motor_to_urdf(seed_motor_deg)
        else:
            q = pin.neutral(self.model)

        # Mask: only solve for first 5 joints
        active = np.array([True] * N_IK_JOINTS + [False], dtype=bool)

        for i in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            current = self.data.oMf[self.ee_frame_id]
            err = pin.log(current.actInv(oMdes)).vector

            if np.linalg.norm(err) < eps:
                return self._urdf_to_motor(q)

            J_full = pin.computeFrameJacobian(
                self.model, self.data, q, self.ee_frame_id,
                pin.ReferenceFrame.LOCAL)

            # Use only columns for active joints
            J = J_full[:, active]
            JtJ = J.T @ J + damp * np.eye(J.shape[1])
            dq_active = np.linalg.solve(JtJ, J.T @ err)

            dq = np.zeros(self.model.nv)
            dq[active] = dq_active
            q = pin.integrate(self.model, q, dt * dq)
            q = np.clip(q, self.q_min, self.q_max)

        return None

    def solve_ik_position(self, target_pos_mm: np.ndarray,
                          seed_motor_deg: np.ndarray = None,
                          max_iter: int = 200, eps: float = 1e-3,
                          dt: float = 0.2, damp: float = 1e-4) -> np.ndarray | None:
        """Position-only IK: move TCP to target position, let orientation float.

        Better suited for 5-DOF arms where full 6-DOF pose is over-constrained.

        Args:
            target_pos_mm: [x, y, z] in mm.
            seed_motor_deg: 5 motor angles in degrees as starting point.
            max_iter: Maximum iterations.
            eps: Position convergence threshold in meters.
            dt: Step size.
            damp: Damping factor.

        Returns:
            5 motor angles in degrees, or None if no solution.
        """
        target_pos = target_pos_mm / 1000.0

        if seed_motor_deg is not None:
            q = self._motor_to_urdf(seed_motor_deg)
        else:
            q = pin.neutral(self.model)

        active = np.array([True] * N_IK_JOINTS + [False], dtype=bool)

        for i in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            current_pos = self.data.oMf[self.ee_frame_id].translation
            err = target_pos - current_pos  # 3D position error

            if np.linalg.norm(err) < eps:
                return self._urdf_to_motor(q)

            # Position-only Jacobian (top 3 rows of full 6x6 Jacobian)
            J_full = pin.computeFrameJacobian(
                self.model, self.data, q, self.ee_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J_pos = J_full[:3, :][:, active]  # 3 x n_active

            JtJ = J_pos.T @ J_pos + damp * np.eye(J_pos.shape[1])
            dq_active = np.linalg.solve(JtJ, J_pos.T @ err)

            dq = np.zeros(self.model.nv)
            dq[active] = dq_active
            q = pin.integrate(self.model, q, dt * dq)
            q = np.clip(q, self.q_min, self.q_max)

        return None

    def get_joint_info(self) -> list[dict]:
        """Return joint name, limits (deg), and current sign/offset config."""
        info = []
        for i in range(N_IK_JOINTS):
            info.append({
                'name': self.model.names[i + 1],  # skip universe
                'min_deg': np.degrees(self.q_min[i]),
                'max_deg': np.degrees(self.q_max[i]),
                'sign': self.signs[i],
                'offset_deg': self.offsets_deg[i],
            })
        return info
