"""Lightweight 3D arm skeleton renderer using OpenCV + Pinocchio FK.

Computes all joint positions via forward kinematics and projects them
to 2D for visualization.  Supports multiple view angles and overlaying
two arm configurations (commanded vs actual) for misconfiguration detection.

All rendering is pure OpenCV — no Isaac Sim, matplotlib, or other heavy deps.
"""

import math
import numpy as np
import cv2

try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False


# Joint chain: list of (frame_name, color_bgr) pairs in kinematic order.
# We draw line segments between consecutive frames.
_JOINT_CHAIN = [
    'base_link',
    'shoulder_link',
    'upper_arm_link',
    'lower_arm_link',
    'wrist_link',
    'gripper_link',
    'gripper_frame_link',
]

# Colors for each link segment (BGR)
_LINK_COLORS = [
    (180, 180, 180),   # base -> shoulder  (gray)
    (255, 160, 50),    # shoulder -> upper  (blue-ish)
    (50, 200, 255),    # upper -> lower     (yellow-ish)
    (50, 255, 100),    # lower -> wrist     (green)
    (200, 100, 255),   # wrist -> gripper   (purple)
    (100, 200, 255),   # gripper -> frame   (light orange)
]

# Per-link thickness
_LINK_THICKNESS = 3


class ArmRenderer:
    """Renders SO-ARM101 arm skeleton using FK and 3D->2D projection.

    Args:
        urdf_path: Path to the URDF file.
        joint_signs: Per-joint sign array (5,).
        joint_offsets_deg: Per-joint offset array (5,) in degrees.
        width: Render width in pixels.
        height: Render height in pixels.
    """

    def __init__(self, urdf_path: str = None,
                 joint_signs: np.ndarray = None,
                 joint_offsets_deg: np.ndarray = None,
                 width: int = 400, height: int = 400):
        if not HAS_PINOCCHIO:
            raise ImportError("pinocchio not installed")

        import os
        from kinematics.arm101_ik_solver import (
            _URDF_PATH, JOINT_SIGNS, JOINT_OFFSETS_DEG, N_IK_JOINTS)

        urdf = urdf_path or _URDF_PATH
        self.model = pin.buildModelFromUrdf(urdf)
        self.data = self.model.createData()

        self.signs = joint_signs if joint_signs is not None else JOINT_SIGNS.copy()
        self.offsets_deg = (joint_offsets_deg if joint_offsets_deg is not None
                           else JOINT_OFFSETS_DEG.copy())
        self.n_ik = N_IK_JOINTS

        # Resolve frame IDs
        self.frame_ids = []
        for name in _JOINT_CHAIN:
            fid = self.model.getFrameId(name)
            if fid < self.model.nframes:
                self.frame_ids.append(fid)

        self.width = width
        self.height = height

        # Camera parameters for 3D->2D projection
        # Viewpoint: azimuth and elevation in degrees
        self.azimuth = -45.0   # degrees around Y axis
        self.elevation = 25.0  # degrees above horizontal
        self.zoom = 1200.0     # scale factor (pixels per meter)
        self.center_offset = np.array([0.0, 0.08, 0.0])  # world offset to center arm

    def _motor_to_urdf(self, motor_deg: np.ndarray) -> np.ndarray:
        """Convert motor angles (deg) to URDF joint angles (rad)."""
        q_rad = np.zeros(self.model.nq)
        n = min(len(motor_deg), self.n_ik)
        for i in range(n):
            q_rad[i] = np.radians(motor_deg[i] * self.signs[i] + self.offsets_deg[i])
        if len(motor_deg) > self.n_ik:
            q_rad[self.n_ik] = np.radians(motor_deg[self.n_ik])
        return q_rad

    def get_joint_positions(self, motor_angles_deg: np.ndarray) -> list[np.ndarray]:
        """Compute 3D positions of all chain frames given motor angles.

        Args:
            motor_angles_deg: 5 or 6 motor angles in degrees.

        Returns:
            List of 3D positions (meters) for each frame in the chain.
        """
        q = self._motor_to_urdf(motor_angles_deg)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        positions = []
        for fid in self.frame_ids:
            pos = self.data.oMf[fid].translation.copy()
            positions.append(pos)
        return positions

    def project_3d_to_2d(self, points_3d: list[np.ndarray]) -> list[tuple[int, int]]:
        """Project 3D points to 2D screen coordinates.

        Uses an isometric-style projection with configurable viewpoint.

        Args:
            points_3d: List of 3D positions in meters.

        Returns:
            List of (x, y) pixel coordinates.
        """
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)

        # Rotation: first around Y (azimuth), then around X (elevation)
        cos_az, sin_az = math.cos(az), math.sin(az)
        cos_el, sin_el = math.cos(el), math.sin(el)

        points_2d = []
        cx = self.width // 2
        cy = int(self.height * 0.75)  # base near bottom

        for p in points_3d:
            # Shift to center
            x, y, z = p - self.center_offset

            # Rotate around Z (vertical) axis by azimuth
            x2 = x * cos_az - y * sin_az
            y2 = x * sin_az + y * cos_az

            # Rotate around X axis by elevation
            y3 = y2 * cos_el - z * sin_el
            z3 = y2 * sin_el + z * cos_el

            # Project: x2 -> screen x, z3 -> screen y (inverted)
            sx = int(cx + x2 * self.zoom)
            sy = int(cy - z3 * self.zoom)
            points_2d.append((sx, sy))

        return points_2d

    def render(self, canvas: np.ndarray,
               motor_angles_deg: np.ndarray,
               color_override: tuple = None,
               alpha: float = 1.0,
               label: str = None,
               thickness: int = None,
               draw_joints: bool = True,
               draw_grid: bool = True,
               offset_x: int = 0, offset_y: int = 0) -> np.ndarray:
        """Render the arm skeleton onto a canvas.

        Args:
            canvas: BGR image to draw on.
            motor_angles_deg: 5 or 6 motor angles in degrees.
            color_override: Optional single color for all links (BGR).
            alpha: Opacity (1.0 = opaque, used for ghost overlay).
            label: Optional text label to draw near base.
            thickness: Override link thickness.
            draw_joints: Whether to draw joint circles.
            draw_grid: Whether to draw ground grid.
            offset_x, offset_y: Pixel offset for the rendering area.

        Returns:
            The modified canvas.
        """
        positions_3d = self.get_joint_positions(motor_angles_deg)
        points_2d = self.project_3d_to_2d(positions_3d)

        # Offset
        points_2d = [(x + offset_x, y + offset_y) for x, y in points_2d]

        lw = thickness or _LINK_THICKNESS

        if alpha < 1.0:
            # Draw on overlay for alpha blending
            overlay = canvas.copy()
            self._draw_skeleton(overlay, points_2d, color_override, lw, draw_joints)
            cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)
        else:
            self._draw_skeleton(canvas, points_2d, color_override, lw, draw_joints)

        # Draw base marker
        if len(points_2d) > 0:
            bx, by = points_2d[0]
            cv2.rectangle(canvas, (bx - 15, by - 3), (bx + 15, by + 3),
                          (100, 100, 100), -1)

        # Label
        if label and len(points_2d) > 0:
            bx, by = points_2d[0]
            cv2.putText(canvas, label, (bx - 30, by + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        color_override or (200, 200, 200), 1)

        return canvas

    def _draw_skeleton(self, canvas, points_2d, color_override, thickness, draw_joints):
        """Draw line segments and joint circles."""
        for i in range(len(points_2d) - 1):
            color = color_override or _LINK_COLORS[min(i, len(_LINK_COLORS) - 1)]
            cv2.line(canvas, points_2d[i], points_2d[i + 1], color, thickness)

        if draw_joints:
            for i, pt in enumerate(points_2d):
                if i == 0:
                    # Base: square (drawn separately)
                    continue
                elif i == len(points_2d) - 1:
                    # End-effector: diamond
                    size = 5
                    pts = np.array([
                        [pt[0], pt[1] - size],
                        [pt[0] + size, pt[1]],
                        [pt[0], pt[1] + size],
                        [pt[0] - size, pt[1]],
                    ], dtype=np.int32)
                    cv2.fillPoly(canvas, [pts],
                                 color_override or (0, 200, 255))
                else:
                    # Regular joint: circle
                    cv2.circle(canvas, pt, 4,
                               color_override or (255, 255, 255), -1)
                    cv2.circle(canvas, pt, 4, (80, 80, 80), 1)

    def render_comparison(self, canvas: np.ndarray,
                          actual_angles: np.ndarray,
                          commanded_angles: np.ndarray = None,
                          offset_x: int = 0, offset_y: int = 0) -> np.ndarray:
        """Render actual arm state with optional commanded overlay.

        The actual arm is drawn solid; the commanded pose is drawn as
        a translucent green ghost for comparison.

        Args:
            canvas: BGR image to draw on.
            actual_angles: Current motor angles (degrees) read from arm.
            commanded_angles: Target motor angles (degrees) sent to arm.
            offset_x, offset_y: Pixel offset.

        Returns:
            The modified canvas.
        """
        # Draw commanded first (ghost, behind)
        if commanded_angles is not None:
            self.render(canvas, commanded_angles,
                        color_override=(0, 200, 100),
                        alpha=0.4,
                        label='Commanded',
                        thickness=2,
                        draw_joints=False,
                        offset_x=offset_x, offset_y=offset_y)

        # Draw actual on top
        self.render(canvas, actual_angles,
                    label='Actual' if commanded_angles is not None else None,
                    offset_x=offset_x, offset_y=offset_y)

        return canvas

    def render_angle_table(self, canvas: np.ndarray,
                           actual_angles: np.ndarray,
                           commanded_angles: np.ndarray = None,
                           x: int = 10, y: int = 20) -> np.ndarray:
        """Draw a compact angle comparison table.

        Args:
            canvas: Image to draw on.
            actual_angles: Current angles (5 or 6 values).
            commanded_angles: Target angles (5 or 6 values), or None.
            x, y: Top-left position.

        Returns:
            Modified canvas.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        names = ['J1:Pan', 'J2:Lift', 'J3:Elbow', 'J4:WrFl', 'J5:WrRo', 'J6:Grip']

        # Header
        header = "Joint     Actual"
        if commanded_angles is not None:
            header += "   Cmd    Delta"
        cv2.putText(canvas, header, (x, y), font, 0.32, (150, 150, 150), 1)
        y += 16

        n = min(len(actual_angles), 6)
        for i in range(n):
            name = names[i] if i < len(names) else f'J{i+1}'
            actual = actual_angles[i]
            line = f"{name:9s} {actual:7.1f}"

            color = (200, 200, 200)
            if commanded_angles is not None and i < len(commanded_angles):
                cmd = commanded_angles[i]
                delta = actual - cmd
                line += f"  {cmd:7.1f}  {delta:+6.1f}"
                # Color based on delta magnitude
                if abs(delta) > 10:
                    color = (0, 50, 255)   # red = big mismatch
                elif abs(delta) > 3:
                    color = (0, 180, 255)  # orange = moderate
                else:
                    color = (0, 255, 100)  # green = close

            cv2.putText(canvas, line, (x, y), font, 0.32, color, 1)
            y += 14

        return canvas
