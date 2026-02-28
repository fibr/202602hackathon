"""Draw projected robot joint positions on camera images.

Uses the Nova5 URDF kinematic chain + calibration transform to project
the robot's joint positions from base frame into camera pixel coordinates.

Works without pinocchio — implements FK directly from URDF joint data.
"""

import numpy as np
import cv2


def _rpy_to_matrix(roll, pitch, yaw):
    """Convert RPY angles (radians) to a 3x3 rotation matrix."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])
    return R


def _rot_z(angle):
    """Rotation matrix about Z axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _make_transform(xyz, rpy):
    """Create a 4x4 homogeneous transform from xyz translation and rpy rotation."""
    T = np.eye(4)
    T[:3, :3] = _rpy_to_matrix(*rpy)
    T[:3, 3] = xyz
    return T


# Nova5 URDF kinematic chain: (joint_name, xyz, rpy, axis)
# All joints rotate about their local Z axis.
_NOVA5_JOINTS = [
    # joint1: base_link -> Link1
    ('joint1', [0, 0, 0.240], [0, 0, 0]),
    # joint2: Link1 -> Link2
    ('joint2', [0, 0, 0], [-1.57080287682252, 1.53586622836832, 3.14159265358979]),
    # joint3: Link2 -> Link3
    ('joint3', [-0.399756009268664, -0.0139690033141953, 0]),
    # joint4: Link3 -> Link4
    ('joint4', [-0.329798707647103, -0.0115244277211674, 0.134999532858734],
     [0, 0, -1.53586622840712]),
    # joint5: Link4 -> Link5
    ('joint5', [0, -0.12, 0], [1.5708, 0, 0]),
    # joint6: Link5 -> Link6
    ('joint6', [0, 0.088328, 0], [-1.5708, 0, 0]),
]

# Tool joint: Link6 -> tool_tip (fixed)
_TOOL_OFFSET = [0, 0, -0.100]

# Joint labels for display
_JOINT_LABELS = ['base', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'TCP']

# Colors for each joint (BGR)
_JOINT_COLORS = [
    (0, 200, 255),    # base - gold/yellow
    (0, 255, 0),      # J1 - green
    (0, 255, 0),      # J2 - green
    (0, 255, 0),      # J3 - green
    (0, 255, 0),      # J4 - green
    (0, 255, 0),      # J5 - green
    (0, 255, 0),      # J6 - green
    (0, 0, 255),      # TCP - red
]

_BASE_COLOR = (0, 200, 255)   # gold/yellow for base marker
_LINK_COLOR = (200, 200, 200) # gray for links
_TCP_COLOR = (0, 0, 255)      # red for TCP


class RobotOverlay:
    """Projects robot joint positions onto camera images.

    Uses the calibration transform (T_camera_to_base) to map from
    robot base frame to camera frame, then projects to pixels using
    camera intrinsics.
    """

    def __init__(self, T_camera_to_base: np.ndarray, tool_length_mm: float = 100.0,
                 base_offset_mm: np.ndarray = None, base_rpy_deg: np.ndarray = None):
        """
        Args:
            T_camera_to_base: 4x4 homogeneous transform (meters)
            tool_length_mm: Gripper length in mm
            base_offset_mm: Optional [dx, dy, dz] offset to the robot base in mm.
                           Use this to correct calibration errors.
            base_rpy_deg: Optional [roll, pitch, yaw] rotation correction in degrees.
                         Applied at the base before the FK chain.
        """
        self.T_cam_to_base = T_camera_to_base.copy()
        self.T_base_to_cam = np.linalg.inv(T_camera_to_base)
        self.tool_offset = np.array([0, 0, -tool_length_mm / 1000.0])
        # Base offset in meters (applied to all joint positions)
        self.base_offset_m = np.zeros(3)
        if base_offset_mm is not None:
            self.base_offset_m = np.array(base_offset_mm, dtype=float) / 1000.0
        # Base rotation correction in radians (roll, pitch, yaw)
        self.base_rpy_rad = np.zeros(3)
        if base_rpy_deg is not None:
            self.base_rpy_rad = np.radians(np.array(base_rpy_deg, dtype=float))

    def compute_joint_positions(self, joint_angles_deg: np.ndarray) -> list[np.ndarray]:
        """Compute 3D positions of all joints in robot base frame (meters).

        Args:
            joint_angles_deg: 6 joint angles in degrees

        Returns:
            List of 8 positions: [base_origin, J1, J2, J3, J4, J5, J6, TCP]
            Each is [x, y, z] in meters in the robot base frame.
        """
        q = np.radians(joint_angles_deg)
        positions = []

        # Base origin (at 0,0,0 + offset in base frame, with rotation correction)
        T_current = np.eye(4)
        T_current[:3, 3] = self.base_offset_m
        if np.any(self.base_rpy_rad != 0):
            T_current[:3, :3] = _rpy_to_matrix(*self.base_rpy_rad)
        positions.append(T_current[:3, 3].copy())

        for i, (name, xyz, *rest) in enumerate(_NOVA5_JOINTS):
            rpy = rest[0] if rest else [0, 0, 0]
            # Apply the fixed joint transform
            T_joint = _make_transform(xyz, rpy)
            T_current = T_current @ T_joint
            # Apply the joint rotation (about local Z)
            T_rot = np.eye(4)
            T_rot[:3, :3] = _rot_z(q[i])
            T_current = T_current @ T_rot
            positions.append(T_current[:3, 3].copy())

        # Tool tip
        T_tool = np.eye(4)
        T_tool[:3, 3] = self.tool_offset
        T_current = T_current @ T_tool
        positions.append(T_current[:3, 3].copy())

        return positions

    def base_position_m(self) -> np.ndarray:
        """Return the robot base origin in base frame (with offset applied)."""
        return self.base_offset_m.copy()

    def nudge_base(self, dx_mm: float = 0, dy_mm: float = 0, dz_mm: float = 0):
        """Nudge the base offset incrementally (for interactive adjustment)."""
        self.base_offset_m += np.array([dx_mm, dy_mm, dz_mm]) / 1000.0

    @property
    def base_rpy_deg(self) -> np.ndarray:
        """Current base rotation correction in degrees [roll, pitch, yaw]."""
        return np.degrees(self.base_rpy_rad)

    def nudge_base_rpy(self, droll_deg: float = 0, dpitch_deg: float = 0, dyaw_deg: float = 0):
        """Nudge the base rotation incrementally (for interactive adjustment)."""
        self.base_rpy_rad += np.radians(np.array([droll_deg, dpitch_deg, dyaw_deg]))

    def project_to_pixels(self, points_base_m: list[np.ndarray],
                          intrinsics) -> list[tuple[int, int] | None]:
        """Project 3D base-frame points to camera pixel coordinates.

        Args:
            points_base_m: List of [x, y, z] in robot base frame (meters)
            intrinsics: RealSense-style intrinsics with fx, fy, ppx, ppy

        Returns:
            List of (u, v) pixel coordinates, or None if behind camera
        """
        pixels = []
        for p_base in points_base_m:
            p_hom = np.append(p_base, 1.0)
            p_cam = self.T_base_to_cam @ p_hom
            p_cam = p_cam[:3]

            # Point must be in front of camera (positive Z in camera frame)
            if p_cam[2] <= 0:
                pixels.append(None)
                continue

            u = int(intrinsics.fx * p_cam[0] / p_cam[2] + intrinsics.ppx)
            v = int(intrinsics.fy * p_cam[1] / p_cam[2] + intrinsics.ppy)
            pixels.append((u, v))

        return pixels

    def draw_base_marker(self, image: np.ndarray, intrinsics) -> np.ndarray:
        """Draw just the robot base position on the image (no joint angles needed).

        Draws a distinctive diamond marker at the projected base position.
        """
        vis = image.copy()
        base_pos = [self.base_position_m()]
        pixels = self.project_to_pixels(base_pos, intrinsics)
        if pixels[0] is not None:
            u, v = pixels[0]
            h, w = vis.shape[:2]
            if 0 <= u < w and 0 <= v < h:
                # Diamond marker for base
                size = 12
                pts = np.array([
                    [u, v - size], [u + size, v],
                    [u, v + size], [u - size, v],
                ], dtype=np.int32)
                cv2.fillPoly(vis, [pts], _BASE_COLOR)
                cv2.polylines(vis, [pts], True, (0, 0, 0), 2)
                cv2.putText(vis, "BASE", (u + 14, v + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, _BASE_COLOR, 1)
        return vis

    def draw_joints(self, image: np.ndarray, joint_angles_deg: np.ndarray,
                    intrinsics) -> np.ndarray:
        """Draw robot joints and links on the image.

        Args:
            image: BGR image to draw on
            joint_angles_deg: 6 joint angles in degrees
            intrinsics: Camera intrinsics (fx, fy, ppx, ppy)

        Returns:
            Image with overlay drawn
        """
        vis = image.copy()
        positions = self.compute_joint_positions(joint_angles_deg)
        pixels = self.project_to_pixels(positions, intrinsics)

        h, w = vis.shape[:2]

        # Draw links (lines between consecutive joints)
        for i in range(len(pixels) - 1):
            if pixels[i] is not None and pixels[i + 1] is not None:
                p1 = pixels[i]
                p2 = pixels[i + 1]
                if (0 <= p1[0] < w and 0 <= p1[1] < h and
                        0 <= p2[0] < w and 0 <= p2[1] < h):
                    cv2.line(vis, p1, p2, _LINK_COLOR, 2, cv2.LINE_AA)

        # Draw joints
        for i, (px, label) in enumerate(zip(pixels, _JOINT_LABELS)):
            if px is None:
                continue
            u, v = px
            if not (0 <= u < w and 0 <= v < h):
                continue

            color = _JOINT_COLORS[i]
            if i == 0:
                # Base: diamond marker
                size = 12
                pts = np.array([
                    [u, v - size], [u + size, v],
                    [u, v + size], [u - size, v],
                ], dtype=np.int32)
                cv2.fillPoly(vis, [pts], color)
                cv2.polylines(vis, [pts], True, (0, 0, 0), 2)
            elif i == len(pixels) - 1:
                # TCP: larger circle with crosshair
                cv2.circle(vis, (u, v), 8, _TCP_COLOR, -1)
                cv2.circle(vis, (u, v), 8, (0, 0, 0), 2)
                cv2.line(vis, (u - 12, v), (u + 12, v), _TCP_COLOR, 1)
                cv2.line(vis, (u, v - 12), (u, v + 12), _TCP_COLOR, 1)
            else:
                # Regular joints: filled circle
                cv2.circle(vis, (u, v), 5, color, -1)
                cv2.circle(vis, (u, v), 5, (0, 0, 0), 1)

            # Label
            cv2.putText(vis, label, (u + 8, v - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        return vis
