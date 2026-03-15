"""Lightweight 3D arm skeleton renderer using OpenCV + Pinocchio FK.

Computes all joint positions via forward kinematics and projects them
to 2D for visualization.  Supports multiple view angles, mouse-drag
rotation, shadow projection, and overlaying two arm configurations
(commanded vs actual) for misconfiguration detection.

Now also supports rendering the actual SO101 STL meshes from the URDF,
with depth-sorted triangle rasterization for a solid 3D appearance.

All rendering is pure OpenCV — no Isaac Sim, matplotlib, or other heavy deps.
"""

import math
import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2

try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


# Joint chain: frame names in kinematic order.
_JOINT_CHAIN = [
    'base_link',
    'shoulder_link',
    'upper_arm_link',
    'lower_arm_link',
    'wrist_link',
    'gripper_link',
    'gripper_frame_link',
]

# Short labels for each frame in the chain.
# Note: the label marks the joint AT that frame, i.e. the rotation
# point between the incoming link and the outgoing link.
_JOINT_LABELS = [
    'Base',        # base_link (fixed)
    'J1',          # shoulder_link (pan rotation point)
    'J2',          # upper_arm_link (lift rotation point)
    'J3',          # lower_arm_link (elbow rotation point)
    'J4',          # wrist_link (wrist flex rotation point)
    'J5',          # gripper_link (wrist roll rotation point)
    'TCP',         # gripper_frame_link (tool center point)
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

# Material colors (BGR) — from URDF material definitions
_MAT_3D_PRINTED = (30, 209, 255)   # yellow-gold (rgba 1.0, 0.82, 0.12)
_MAT_STS3215 = (30, 30, 30)        # dark (rgba 0.1, 0.1, 0.1)
_MAT_DEFAULT = (160, 160, 160)     # gray fallback

# Target face count per mesh after decimation (for real-time rendering).
# With ~17 mesh parts, this gives ~13k total faces at ~8ms per frame.
_MESH_TARGET_FACES = 800


def _parse_urdf_visuals(urdf_path: str) -> dict[str, list[dict]]:
    """Parse URDF to extract visual mesh info for each link.

    Returns:
        Dict mapping link_name -> list of dicts, each with keys:
            'mesh': relative filename (e.g. 'assets/base_so101_v2.stl')
            'origin_xyz': (x, y, z) translation
            'origin_rpy': (r, p, y) rotation
            'material': material name string
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    result = {}
    for link_el in root.findall('link'):
        name = link_el.get('name')
        visuals = []
        for vis in link_el.findall('visual'):
            geom = vis.find('geometry')
            if geom is None:
                continue
            mesh_el = geom.find('mesh')
            if mesh_el is None:
                continue
            entry = {
                'mesh': mesh_el.get('filename', ''),
                'origin_xyz': (0.0, 0.0, 0.0),
                'origin_rpy': (0.0, 0.0, 0.0),
                'material': '',
            }
            origin = vis.find('origin')
            if origin is not None:
                xyz_str = origin.get('xyz', '0 0 0')
                rpy_str = origin.get('rpy', '0 0 0')
                entry['origin_xyz'] = tuple(float(v) for v in xyz_str.split())
                entry['origin_rpy'] = tuple(float(v) for v in rpy_str.split())
            mat = vis.find('material')
            if mat is not None:
                entry['material'] = mat.get('name', '')
            visuals.append(entry)
        if visuals:
            result[name] = visuals
    return result


def _rpy_to_matrix(rpy: tuple) -> np.ndarray:
    """Convert roll-pitch-yaw to a 3x3 rotation matrix."""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    # ZYX convention (yaw * pitch * roll)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr],
    ])


def _load_and_decimate_mesh(filepath: str, target_faces: int) -> trimesh.Trimesh:
    """Load an STL mesh and decimate it for fast rendering.

    Tries quadric decimation first; falls back to vertex clustering
    if the fast_simplification extension isn't available.
    """
    mesh = trimesh.load(filepath)
    if len(mesh.faces) <= target_faces:
        return mesh

    try:
        mesh = mesh.simplify_quadric_decimation(target_faces)
        return mesh
    except Exception:
        pass

    # Fallback: vertex clustering — merge nearby vertices, which
    # naturally reduces face count while preserving mesh topology.
    # Determine grid cell size from target reduction ratio.
    ratio = (len(mesh.faces) / target_faces) ** (1.0 / 3.0)
    extents = mesh.bounding_box.extents
    cell_size = float(np.mean(extents)) / max(8, 30 / ratio)

    # Quantize vertices to grid
    quantized = np.round(mesh.vertices / cell_size) * cell_size
    # Map each vertex to its quantized representative
    _, inverse, counts = np.unique(
        quantized, axis=0, return_inverse=True, return_counts=True)
    # Remap faces
    new_faces = inverse[mesh.faces]
    # Remove degenerate faces (where two or more vertices collapsed)
    valid = ((new_faces[:, 0] != new_faces[:, 1]) &
             (new_faces[:, 1] != new_faces[:, 2]) &
             (new_faces[:, 0] != new_faces[:, 2]))
    new_faces = new_faces[valid]
    # Build new mesh with quantized vertices
    unique_verts = np.zeros((inverse.max() + 1, 3))
    np.add.at(unique_verts, inverse, mesh.vertices)
    vert_counts = np.zeros(inverse.max() + 1)
    np.add.at(vert_counts, inverse, 1.0)
    unique_verts /= vert_counts[:, None].clip(1)

    result = trimesh.Trimesh(vertices=unique_verts, faces=new_faces,
                             process=True)
    return result


class _LinkMeshData:
    """Pre-loaded mesh data for one visual element of a link."""
    __slots__ = ('vertices', 'faces', 'color_bgr', 'origin_T')

    def __init__(self, vertices: np.ndarray, faces: np.ndarray,
                 color_bgr: tuple, origin_T: np.ndarray):
        self.vertices = vertices      # (N, 3) float32
        self.faces = faces            # (M, 3) int32 — triangle indices
        self.color_bgr = color_bgr    # (B, G, R)
        self.origin_T = origin_T      # 4x4 visual origin transform


class ArmRenderer:
    """Renders SO-ARM101 arm skeleton using FK and 3D->2D projection.

    Supports mouse-drag rotation, shadow projection on the floor plane,
    and visual depth cues (thickness/shading variation, drop-lines).

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

        # Table height relative to arm base (negative = below base)
        # Arm base is typically ~100mm above the table surface.
        self.table_z_m = 0.0  # table at arm base level

        # Camera parameters for 3D->2D projection
        # Default: 3/4 side view — shows depth much better than front view
        self.azimuth = 60.0    # degrees around Z axis (side-ish view)
        self.elevation = 25.0  # degrees above horizontal
        self.zoom = 1200.0     # scale factor (pixels per meter)
        self.center_offset = np.array([0.0, 0.08, 0.0])  # world offset to center arm
        self.depth_shading = True   # darken links further from camera
        self.draw_shadows = True    # project shadow on floor plane
        self.draw_drop_lines = True  # vertical drop lines from joints to floor
        self.draw_axes = True       # XYZ axis indicator in corner

        # Mesh rendering
        self.render_mesh_mode = False   # False = skeleton (default), True = mesh
        self._link_meshes = {}          # link_name -> list[_LinkMeshData]
        self._urdf_path = urdf
        self._mesh_loaded = False
        self._mesh_load_error = None

        # Mouse drag state (managed externally by the view widget)
        self._drag_start = None
        self._drag_az_start = 0.0
        self._drag_el_start = 0.0

    def _ensure_meshes_loaded(self):
        """Lazy-load and decimate STL meshes on first use."""
        if self._mesh_loaded:
            return self._mesh_load_error is None
        self._mesh_loaded = True

        if not HAS_TRIMESH:
            self._mesh_load_error = "trimesh not installed"
            return False

        try:
            urdf_dir = os.path.dirname(self._urdf_path)
            visuals = _parse_urdf_visuals(self._urdf_path)

            mat_colors = {
                '3d_printed': _MAT_3D_PRINTED,
                'sts3215': _MAT_STS3215,
            }

            for link_name, vis_list in visuals.items():
                mesh_data_list = []
                for vis in vis_list:
                    mesh_file = os.path.join(urdf_dir, vis['mesh'])
                    if not os.path.isfile(mesh_file):
                        continue

                    mesh = _load_and_decimate_mesh(mesh_file, _MESH_TARGET_FACES)

                    # Build 4x4 visual origin transform
                    R = _rpy_to_matrix(vis['origin_rpy'])
                    t = np.array(vis['origin_xyz'])
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = t

                    color = mat_colors.get(vis['material'], _MAT_DEFAULT)

                    mesh_data_list.append(_LinkMeshData(
                        vertices=np.array(mesh.vertices, dtype=np.float32),
                        faces=np.array(mesh.faces, dtype=np.int32),
                        color_bgr=color,
                        origin_T=T,
                    ))
                if mesh_data_list:
                    self._link_meshes[link_name] = mesh_data_list

        except Exception as e:
            self._mesh_load_error = str(e)
            return False

        return True

    def start_drag(self, x: int, y: int):
        """Begin a mouse-drag rotation. Call on mouse-press."""
        self._drag_start = (x, y)
        self._drag_az_start = self.azimuth
        self._drag_el_start = self.elevation

    def update_drag(self, x: int, y: int):
        """Update view rotation during mouse-drag. Call on mouse-move."""
        if self._drag_start is None:
            return
        dx = x - self._drag_start[0]
        dy = y - self._drag_start[1]
        # 0.5 deg per pixel of drag
        self.azimuth = self._drag_az_start + dx * 0.5
        self.elevation = np.clip(self._drag_el_start - dy * 0.5, -89, 89)

    def end_drag(self):
        """End mouse-drag rotation. Call on mouse-release."""
        self._drag_start = None

    def handle_scroll(self, delta: float):
        """Zoom in/out via scroll wheel. delta>0 = zoom in."""
        factor = 1.1 if delta > 0 else 0.9
        self.zoom = np.clip(self.zoom * factor, 400, 4000)

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

    def project_3d_to_2d(self, points_3d: list[np.ndarray]):
        """Project 3D points to 2D screen coordinates with depth info.

        Uses a mild perspective projection with configurable viewpoint.

        Args:
            points_3d: List of 3D positions in meters.

        Returns:
            List of (x, y, depth) tuples. depth is the camera-space Y
            coordinate (larger = further from camera).
        """
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)

        cos_az, sin_az = math.cos(az), math.sin(az)
        cos_el, sin_el = math.cos(el), math.sin(el)

        results = []
        cx = self.width // 2
        cy = int(self.height * 0.75)  # base near bottom

        for p in points_3d:
            x, y, z = p - self.center_offset

            # Rotate around Z (vertical) axis by azimuth
            x2 = x * cos_az - y * sin_az
            y2 = x * sin_az + y * cos_az

            # Rotate around X axis by elevation
            y3 = y2 * cos_el - z * sin_el
            z3 = y2 * sin_el + z * cos_el

            # Mild perspective: scale by distance from virtual camera
            persp = 1.0 + y3 * 0.8  # subtle foreshortening
            persp = max(persp, 0.3)

            sx = int(cx + x2 * self.zoom * persp)
            sy = int(cy - z3 * self.zoom * persp)
            results.append((sx, sy, y3))

        return results

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
        projected = self.project_3d_to_2d(positions_3d)

        # Split into 2D coords and depths
        points_2d = [(x + offset_x, y + offset_y) for x, y, _ in projected]
        depths = [d for _, _, d in projected]

        # Draw table surface
        if draw_grid and self.table_z_m is not None:
            self._draw_table(canvas, offset_x, offset_y)

        # Draw shadow on floor plane (before the skeleton so it's behind)
        if self.draw_shadows and not color_override:
            self._draw_shadow(canvas, positions_3d, offset_x, offset_y)

        # Draw vertical drop-lines from joints to floor (depth cue)
        if self.draw_drop_lines and not color_override:
            self._draw_drop_lines(canvas, positions_3d, offset_x, offset_y)

        lw = thickness or _LINK_THICKNESS

        if alpha < 1.0:
            overlay = canvas.copy()
            self._draw_skeleton(overlay, points_2d, depths, color_override, lw, draw_joints)
            cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)
        else:
            self._draw_skeleton(canvas, points_2d, depths, color_override, lw, draw_joints)

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

    def _project_points_batch(self, points_3d: np.ndarray) -> np.ndarray:
        """Project an (N, 3) array of 3D points to (N, 3) screen coords.

        Returns:
            (N, 3) array with columns [screen_x, screen_y, depth].
        """
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        cos_az, sin_az = math.cos(az), math.sin(az)
        cos_el, sin_el = math.cos(el), math.sin(el)

        cx = self.width // 2
        cy = int(self.height * 0.75)

        # Shift by center offset
        pts = points_3d - self.center_offset
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

        # Azimuth rotation around Z
        x2 = x * cos_az - y * sin_az
        y2 = x * sin_az + y * cos_az

        # Elevation rotation around X
        y3 = y2 * cos_el - z * sin_el
        z3 = y2 * sin_el + z * cos_el

        # Perspective
        persp = np.clip(1.0 + y3 * 0.8, 0.3, None)

        sx = cx + x2 * self.zoom * persp
        sy = cy - z3 * self.zoom * persp

        return np.column_stack([sx, sy, y3])

    def render_mesh(self, canvas: np.ndarray,
                    motor_angles_deg: np.ndarray,
                    draw_grid: bool = True,
                    offset_x: int = 0, offset_y: int = 0) -> np.ndarray:
        """Render the arm with actual STL meshes using depth-sorted triangles.

        Falls back to skeleton rendering if meshes aren't available.

        Args:
            canvas: BGR image to draw on.
            motor_angles_deg: 5 or 6 motor angles in degrees.
            draw_grid: Whether to draw ground grid.
            offset_x, offset_y: Pixel offset.

        Returns:
            The modified canvas.
        """
        if not self._ensure_meshes_loaded():
            # Fall back to skeleton
            return self.render(canvas, motor_angles_deg,
                               draw_grid=draw_grid,
                               offset_x=offset_x, offset_y=offset_y)

        # Run FK to get frame transforms
        q = self._motor_to_urdf(motor_angles_deg)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # Draw table surface first
        if draw_grid and self.table_z_m is not None:
            self._draw_table(canvas, offset_x, offset_y)

        # Light direction for diffuse shading (normalized)
        light_dir = np.array([0.3, -0.4, 0.8])
        light_dir /= np.linalg.norm(light_dir)

        # Collect all triangles across all links for depth sorting
        # Arrays: depths (N,), screen_pts (N, 3, 2), colors (N, 3)
        all_depths = []
        all_screen_pts = []
        all_colors = []

        for link_name, mesh_list in self._link_meshes.items():
            # Get the frame transform for this link
            fid = self.model.getFrameId(link_name)
            if fid >= self.model.nframes:
                continue
            frame_T = self.data.oMf[fid].homogeneous  # 4x4

            for md in mesh_list:
                # Full transform: frame_T * visual_origin_T
                world_T = frame_T @ md.origin_T
                R = world_T[:3, :3]
                t = world_T[:3, 3]

                # Transform vertices to world frame
                world_verts = (R @ md.vertices.T).T + t  # (N, 3)

                # Project all vertices to screen in one batch
                proj = self._project_points_batch(world_verts)  # (N, 3)
                screen_xy = proj[:, :2] + np.array([offset_x, offset_y])
                depths = proj[:, 2]

                faces = md.faces
                nf = len(faces)

                # Face centroid depth for sorting
                tri_depths = (depths[faces[:, 0]] + depths[faces[:, 1]] +
                              depths[faces[:, 2]]) / 3.0

                # Screen-space backface culling (skip faces whose screen
                # winding is CW → back-facing in our projection)
                sv0 = screen_xy[faces[:, 0]]
                sv1 = screen_xy[faces[:, 1]]
                sv2 = screen_xy[faces[:, 2]]
                cross_z = ((sv1[:, 0] - sv0[:, 0]) * (sv2[:, 1] - sv0[:, 1]) -
                           (sv1[:, 1] - sv0[:, 1]) * (sv2[:, 0] - sv0[:, 0]))
                # Keep front-facing faces (negative cross = CW in screen)
                # STL winding varies, so keep both large-magnitude faces and
                # cull only degenerate (zero-area) triangles
                keep = np.abs(cross_z) > 0.5

                if not np.any(keep):
                    continue

                faces_k = faces[keep]
                tri_depths_k = tri_depths[keep]

                # World-space face normals for lighting
                wv0 = world_verts[faces_k[:, 0]]
                wv1 = world_verts[faces_k[:, 1]]
                wv2 = world_verts[faces_k[:, 2]]
                normals = np.cross(wv1 - wv0, wv2 - wv0)
                norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
                normals = normals / np.maximum(norm_len, 1e-10)

                # Diffuse lighting (vectorized)
                ndotl = np.abs(normals @ light_dir)  # abs for double-sided
                brightness = 0.35 + 0.65 * ndotl     # ambient + diffuse

                # Per-face colors
                base = np.array(md.color_bgr, dtype=np.float32)
                face_colors = np.clip(
                    np.outer(brightness, base), 0, 255
                ).astype(np.int32)  # (nf, 3)

                # Screen triangle vertices: (nf, 3, 2)
                tri_screen = np.stack([
                    screen_xy[faces_k[:, 0]],
                    screen_xy[faces_k[:, 1]],
                    screen_xy[faces_k[:, 2]],
                ], axis=1).astype(np.int32)

                all_depths.append(tri_depths_k)
                all_screen_pts.append(tri_screen)
                all_colors.append(face_colors)

        if not all_depths:
            return canvas

        # Concatenate and sort by depth (painter's algorithm: far first)
        depths_cat = np.concatenate(all_depths)
        screen_cat = np.concatenate(all_screen_pts)
        colors_cat = np.concatenate(all_colors)

        order = np.argsort(-depths_cat)

        # Render all triangles — must iterate but fillPoly is fast
        for idx in order:
            tri = screen_cat[idx]  # (3, 2)
            c = colors_cat[idx]    # (3,)
            cv2.fillPoly(canvas, [tri], (int(c[0]), int(c[1]), int(c[2])))

        return canvas

    def render_mesh_comparison(self, canvas: np.ndarray,
                               motor_angles_deg: np.ndarray,
                               draw_grid: bool = True,
                               offset_x: int = 0,
                               offset_y: int = 0) -> np.ndarray:
        """Render mesh view with HUD elements (axis indicator, view info).

        Args:
            canvas: BGR image to draw on.
            motor_angles_deg: Motor angles in degrees.
            draw_grid: Whether to draw the ground grid.
            offset_x, offset_y: Pixel offset.

        Returns:
            Modified canvas.
        """
        self.render_mesh(canvas, motor_angles_deg,
                         draw_grid=draw_grid,
                         offset_x=offset_x, offset_y=offset_y)

        # Draw axis indicator and view HUD
        if self.draw_axes:
            self.draw_axis_indicator(canvas, x=offset_x + 40, y=offset_y + 40)
            self.draw_view_hud(canvas, x=offset_x + 8)

        return canvas

    def _draw_shadow(self, canvas, positions_3d, offset_x=0, offset_y=0):
        """Project arm skeleton shadow onto the floor (table) plane.

        Simulates a light source directly above, projecting each joint
        position down to table_z_m and drawing the resulting silhouette.
        """
        floor_z = self.table_z_m if self.table_z_m is not None else -0.10
        # Project each joint straight down to floor plane
        shadow_3d = [np.array([p[0], p[1], floor_z]) for p in positions_3d]
        projected = self.project_3d_to_2d(shadow_3d)
        shadow_2d = [(x + offset_x, y + offset_y) for x, y, _ in projected]

        # Draw shadow links as thin dark lines
        shadow_color = (35, 30, 25)
        for i in range(len(shadow_2d) - 1):
            cv2.line(canvas, shadow_2d[i], shadow_2d[i + 1], shadow_color, 2)
        # Shadow joint dots
        for pt in shadow_2d[1:]:
            cv2.circle(canvas, pt, 2, shadow_color, -1)

    def _draw_drop_lines(self, canvas, positions_3d, offset_x=0, offset_y=0):
        """Draw dashed vertical lines from each joint down to the floor.

        These provide a strong visual cue for height and help disambiguate
        the shoulder offset that makes the upper arm look sideways.
        """
        floor_z = self.table_z_m if self.table_z_m is not None else -0.10
        drop_color = (60, 55, 50)

        for i, p in enumerate(positions_3d):
            if i == 0:
                continue  # skip base
            # Only draw if joint is above the floor
            if p[2] <= floor_z + 0.005:
                continue
            top_pt = self.project_3d_to_2d([p])[0]
            bot_3d = np.array([p[0], p[1], floor_z])
            bot_pt = self.project_3d_to_2d([bot_3d])[0]

            x1, y1 = top_pt[0] + offset_x, top_pt[1] + offset_y
            x2, y2 = bot_pt[0] + offset_x, bot_pt[1] + offset_y

            # Draw dashed line
            self._draw_dashed_line(canvas, (x1, y1), (x2, y2),
                                   drop_color, thickness=1, dash_len=4, gap_len=4)

    @staticmethod
    def _draw_dashed_line(canvas, pt1, pt2, color, thickness=1,
                          dash_len=6, gap_len=4):
        """Draw a dashed line between two points."""
        x1, y1 = pt1
        x2, y2 = pt2
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 1:
            return
        ux, uy = dx / length, dy / length
        pos = 0.0
        while pos < length:
            end = min(pos + dash_len, length)
            p1 = (int(x1 + ux * pos), int(y1 + uy * pos))
            p2 = (int(x1 + ux * end), int(y1 + uy * end))
            cv2.line(canvas, p1, p2, color, thickness)
            pos = end + gap_len

    def _draw_table(self, canvas, offset_x=0, offset_y=0):
        """Draw a semi-transparent table surface at table_z_m with grid."""
        half = 0.30  # 300mm half-width
        corners_3d = [
            np.array([-half, -half, self.table_z_m]),
            np.array([+half, -half, self.table_z_m]),
            np.array([+half, +half, self.table_z_m]),
            np.array([-half, +half, self.table_z_m]),
        ]
        projected = self.project_3d_to_2d(corners_3d)
        pts = np.array([(x + offset_x, y + offset_y)
                        for x, y, _ in projected], dtype=np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], (40, 35, 30))
        cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)
        cv2.polylines(canvas, [pts], True, (80, 70, 60), 1)

        # Draw grid lines on the table for spatial reference
        grid_color = (55, 50, 45)
        n_lines = 6  # lines per axis
        step = 2 * half / n_lines
        for i in range(1, n_lines):
            t = -half + i * step
            # Lines along X axis (constant Y)
            p1 = self.project_3d_to_2d([np.array([-half, t, self.table_z_m])])[0]
            p2 = self.project_3d_to_2d([np.array([+half, t, self.table_z_m])])[0]
            cv2.line(canvas,
                     (p1[0] + offset_x, p1[1] + offset_y),
                     (p2[0] + offset_x, p2[1] + offset_y),
                     grid_color, 1)
            # Lines along Y axis (constant X)
            p1 = self.project_3d_to_2d([np.array([t, -half, self.table_z_m])])[0]
            p2 = self.project_3d_to_2d([np.array([t, +half, self.table_z_m])])[0]
            cv2.line(canvas,
                     (p1[0] + offset_x, p1[1] + offset_y),
                     (p2[0] + offset_x, p2[1] + offset_y),
                     grid_color, 1)

        # Label
        mid = pts.mean(axis=0).astype(int)
        cv2.putText(canvas, 'table', (mid[0] - 15, mid[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 90, 80), 1)

    def draw_axis_indicator(self, canvas, x=40, y=40, length=30):
        """Draw a small XYZ axis indicator showing current view orientation.

        Args:
            canvas: BGR image to draw on.
            x, y: Center position of the axis indicator.
            length: Length of each axis arrow in pixels.
        """
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        cos_az, sin_az = math.cos(az), math.sin(az)
        cos_el, sin_el = math.cos(el), math.sin(el)

        # World axes -> screen space (same transform as project_3d_to_2d)
        axes = {
            'X': np.array([1, 0, 0]),
            'Y': np.array([0, 1, 0]),
            'Z': np.array([0, 0, 1]),
        }
        colors = {
            'X': (0, 0, 220),    # red
            'Y': (0, 180, 0),    # green
            'Z': (220, 100, 0),  # blue
        }

        for label, axis in axes.items():
            ax, ay, az_v = axis
            # Rotate by azimuth around Z
            x2 = ax * cos_az - ay * sin_az
            y2 = ax * sin_az + ay * cos_az
            # Rotate by elevation around X
            z3 = y2 * sin_el + az_v * cos_el

            sx = int(x + x2 * length)
            sy = int(y - z3 * length)

            cv2.arrowedLine(canvas, (x, y), (sx, sy), colors[label], 2,
                            tipLength=0.25)
            cv2.putText(canvas, label, (sx + 3, sy + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, colors[label], 1)

    def draw_view_hud(self, canvas, x=None, y=None):
        """Draw a small HUD showing current azimuth/elevation angles.

        Args:
            canvas: BGR image to draw on.
            x, y: Position. Defaults to bottom-left corner.
        """
        h, w = canvas.shape[:2]
        if x is None:
            x = 8
        if y is None:
            y = h - 12
        text = f"Az:{self.azimuth:.0f} El:{self.elevation:.0f}"
        cv2.putText(canvas, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1)

    def _draw_skeleton(self, canvas, points_2d, depths, color_override, thickness, draw_joints):
        """Draw line segments and joint circles with depth-based shading."""
        # Depth range for shading
        if depths:
            d_min = min(depths)
            d_range = max(depths) - d_min
            if d_range < 0.001:
                d_range = 1.0
        else:
            d_min, d_range = 0, 1.0

        def _shade(color, depth):
            """Darken color based on depth (further = darker)."""
            if not self.depth_shading:
                return color
            t = (depth - d_min) / d_range  # 0=near, 1=far
            factor = 1.0 - 0.4 * t  # near=100%, far=60%
            return tuple(int(c * factor) for c in color)

        def _thick(depth):
            """Thicker for near links, thinner for far."""
            t = (depth - d_min) / d_range
            return max(2, int(thickness + 2 * (1 - t)))

        for i in range(len(points_2d) - 1):
            color = color_override or _LINK_COLORS[min(i, len(_LINK_COLORS) - 1)]
            avg_depth = (depths[i] + depths[i + 1]) / 2 if depths else 0
            cv2.line(canvas, points_2d[i], points_2d[i + 1],
                     _shade(color, avg_depth), _thick(avg_depth))

        if draw_joints:
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i, pt in enumerate(points_2d):
                d = depths[i] if depths else 0
                r = max(3, int(6 * (1 - (d - d_min) / d_range)))  # near=6px, far=3px
                if i == 0:
                    continue  # base drawn separately
                elif i == len(points_2d) - 1:
                    size = r + 1
                    pts = np.array([
                        [pt[0], pt[1] - size],
                        [pt[0] + size, pt[1]],
                        [pt[0], pt[1] + size],
                        [pt[0] - size, pt[1]],
                    ], dtype=np.int32)
                    cv2.fillPoly(canvas, [pts],
                                 _shade(color_override or (0, 200, 255), d))
                else:
                    cv2.circle(canvas, pt, r,
                               _shade(color_override or (255, 255, 255), d), -1)
                    cv2.circle(canvas, pt, r, (60, 60, 60), 1)

                # Joint label
                if i < len(_JOINT_LABELS) and not color_override:
                    label = _JOINT_LABELS[i]
                    lx, ly = pt[0] + r + 4, pt[1] - 2
                    cv2.putText(canvas, label, (lx, ly), font, 0.3,
                                _shade((180, 180, 180), d), 1)

    def render_comparison(self, canvas: np.ndarray,
                          actual_angles: np.ndarray,
                          commanded_angles: np.ndarray = None,
                          offset_x: int = 0, offset_y: int = 0) -> np.ndarray:
        """Render actual arm state with optional commanded overlay.

        When render_mesh_mode is True, renders the full 3D mesh.
        Otherwise renders the skeleton (line/wireframe) view.

        The actual arm is drawn solid; the commanded pose is drawn as
        a translucent green ghost for comparison (skeleton mode only).

        Args:
            canvas: BGR image to draw on.
            actual_angles: Current motor angles (degrees) read from arm.
            commanded_angles: Target motor angles (degrees) sent to arm.
            offset_x, offset_y: Pixel offset.

        Returns:
            The modified canvas.
        """
        if self.render_mesh_mode:
            return self.render_mesh_comparison(
                canvas, actual_angles,
                draw_grid=True,
                offset_x=offset_x, offset_y=offset_y)

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

        # Draw axis indicator and view HUD
        if self.draw_axes:
            self.draw_axis_indicator(canvas, x=offset_x + 40, y=offset_y + 40)
            self.draw_view_hud(canvas, x=offset_x + 8)

        return canvas

    def draw_on_camera_frame(self, canvas: np.ndarray,
                             motor_angles_deg: np.ndarray,
                             K: np.ndarray,
                             dist_coeffs: np.ndarray,
                             T_base_to_camera: np.ndarray,
                             color_override: tuple = None,
                             alpha: float = 1.0,
                             thickness: int = None,
                             draw_joints: bool = True) -> np.ndarray:
        """Draw the FK skeleton onto a camera frame using the calibration transform.

        Projects 3D joint positions (robot base frame) into the camera image
        using the hand-eye calibration transform and camera intrinsics.

        Args:
            canvas: BGR camera frame to draw on.
            motor_angles_deg: 5 or 6 motor angles in degrees.
            K: 3x3 camera intrinsic matrix.
            dist_coeffs: Distortion coefficients (5-element array).
            T_base_to_camera: 4x4 homogeneous transform from robot base frame
                to camera frame (i.e. the inverse of T_camera_to_base).
            color_override: Optional single color for all links (BGR).
            alpha: Opacity (1.0 = opaque).
            thickness: Override link thickness.
            draw_joints: Whether to draw joint circles.

        Returns:
            The modified canvas.
        """
        positions_3d = self.get_joint_positions(motor_angles_deg)
        if not positions_3d:
            return canvas

        # Stack positions into (N, 3) array (meters, in base frame)
        pts_base = np.array(positions_3d, dtype=np.float64)

        # Extract rotation and translation from T_base_to_camera
        R = T_base_to_camera[:3, :3]
        t = T_base_to_camera[:3, 3]
        rvec, _ = cv2.Rodrigues(R)
        tvec = t.reshape(3, 1)

        # Project 3D points into camera image
        pts_img, _ = cv2.projectPoints(
            pts_base.reshape(-1, 1, 3),
            rvec, tvec,
            K.astype(np.float64),
            dist_coeffs.astype(np.float64),
        )
        # pts_img shape: (N, 1, 2)
        points_2d = [(int(round(p[0][0])), int(round(p[0][1]))) for p in pts_img]

        # Filter out-of-frame points (mark as None)
        h, w = canvas.shape[:2]
        valid = [0 <= px < w and 0 <= py < h for px, py in points_2d]

        lw = thickness or (_LINK_THICKNESS + 1)

        if alpha < 1.0:
            overlay = canvas.copy()
            self._draw_skeleton_with_validity(
                overlay, points_2d, valid, color_override, lw, draw_joints)
            cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)
        else:
            self._draw_skeleton_with_validity(
                canvas, points_2d, valid, color_override, lw, draw_joints)

        return canvas

    def _draw_skeleton_with_validity(self, canvas, points_2d, valid,
                                     color_override, thickness, draw_joints):
        """Draw skeleton segments, skipping invalid (off-frame) points."""
        for i in range(len(points_2d) - 1):
            if not valid[i] or not valid[i + 1]:
                continue
            color = color_override or _LINK_COLORS[min(i, len(_LINK_COLORS) - 1)]
            cv2.line(canvas, points_2d[i], points_2d[i + 1], color, thickness)

        if draw_joints:
            for i, (pt, is_valid) in enumerate(zip(points_2d, valid)):
                if not is_valid:
                    continue
                if i == 0:
                    # Base: small square marker
                    cv2.rectangle(canvas, (pt[0] - 6, pt[1] - 6),
                                  (pt[0] + 6, pt[1] + 6),
                                  color_override or (180, 180, 180), 2)
                elif i == len(points_2d) - 1:
                    # End-effector: diamond
                    size = 7
                    pts = np.array([
                        [pt[0], pt[1] - size],
                        [pt[0] + size, pt[1]],
                        [pt[0], pt[1] + size],
                        [pt[0] - size, pt[1]],
                    ], dtype=np.int32)
                    cv2.fillPoly(canvas, [pts],
                                 color_override or (0, 200, 255))
                    cv2.polylines(canvas, [pts], True, (0, 0, 0), 1)
                else:
                    # Regular joint: circle
                    cv2.circle(canvas, pt, 5,
                               color_override or (255, 255, 255), -1)
                    cv2.circle(canvas, pt, 5, (0, 0, 0), 1)

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
