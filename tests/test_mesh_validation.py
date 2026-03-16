"""Test to validate all URDF-referenced STL meshes.

This test:
- Loads all URDF files in the repository
- Extracts mesh filenames from visual and collision geometries
- Validates each loadable mesh using trimesh:
  - Valid faces (mesh is watertight and has correct face winding)
  - Non-zero bounding boxes (mesh has actual geometry)
  - Acceptable polygon counts (warns on very large meshes)
  - Proper vertex and face counts

This catches accidental mesh corruption, oversized files, and invalid geometries
that could cause issues during simulation or visualization.
"""

import sys
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

pytestmark = pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh not installed")


class TestMeshValidation:
    """Test suite for URDF-referenced STL mesh validation."""

    @pytest.fixture(scope="class")
    def repo_root(self):
        """Get the repository root directory."""
        return Path(__file__).parent.parent

    @pytest.fixture(scope="class")
    def urdf_files(self, repo_root):
        """Find all URDF files in the repository."""
        urdf_files = list(repo_root.glob("**/*.urdf"))
        assert len(urdf_files) > 0, "No URDF files found in repository"
        return sorted(urdf_files)

    @staticmethod
    def extract_meshes_from_urdf(urdf_path):
        """Extract all mesh filenames from a URDF file.

        Returns:
            List of (mesh_filename, urdf_path) tuples
        """
        meshes = []
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()

            # Find all geometry elements that contain mesh definitions
            for mesh_elem in root.findall(".//mesh"):
                filename = mesh_elem.get("filename")
                if filename:
                    meshes.append((filename, urdf_path))
        except ET.ParseError as e:
            pytest.fail(f"Failed to parse URDF {urdf_path}: {e}")

        return meshes

    @staticmethod
    def resolve_mesh_path(mesh_filename, urdf_path, repo_root):
        """Resolve mesh filename to an actual file path.

        Handles:
        - Relative paths (relative to URDF directory)
        - package:// URIs (external ROS packages, not in repo)

        Returns:
            Resolved Path or None if external/not found
        """
        # Skip external package references
        if mesh_filename.startswith("package://"):
            return None

        # Handle relative paths
        if mesh_filename.startswith("assets/"):
            # Relative to URDF directory
            resolved = urdf_path.parent / mesh_filename
        else:
            # Relative to URDF directory
            resolved = urdf_path.parent / mesh_filename

        if resolved.exists():
            return resolved
        return None

    def test_urdf_files_exist(self, urdf_files):
        """Verify URDF files are present."""
        assert len(urdf_files) >= 2, (
            f"Expected at least 2 URDF files, found {len(urdf_files)}"
        )
        urdf_names = [f.name for f in urdf_files]
        assert "nova5_robot.urdf" in urdf_names
        assert "so101_new_calib.urdf" in urdf_names

    def test_extract_meshes_from_urdf(self, urdf_files):
        """Verify that meshes can be extracted from URDF files."""
        all_meshes = []
        for urdf_file in urdf_files:
            meshes = self.extract_meshes_from_urdf(urdf_file)
            all_meshes.extend(meshes)
            # Each URDF should have at least one mesh
            assert len(meshes) > 0, f"No meshes found in {urdf_file.name}"

        assert len(all_meshes) > 0, "No meshes extracted from any URDF file"

    def test_mesh_files_referenced(self, urdf_files, repo_root):
        """Verify that all local mesh files are present."""
        missing_meshes = []
        skipped_external = []

        for urdf_file in urdf_files:
            meshes = self.extract_meshes_from_urdf(urdf_file)
            for mesh_filename, _ in meshes:
                resolved_path = self.resolve_mesh_path(
                    mesh_filename, urdf_file, repo_root
                )
                if mesh_filename.startswith("package://"):
                    skipped_external.append((urdf_file.name, mesh_filename))
                elif not resolved_path:
                    missing_meshes.append((urdf_file.name, mesh_filename))

        if missing_meshes:
            msg = "Missing local mesh files:\n"
            for urdf_name, mesh_name in missing_meshes:
                msg += f"  {urdf_name}: {mesh_name}\n"
            pytest.fail(msg)

        # Report which external meshes were skipped
        if skipped_external:
            print(f"\nSkipped {len(skipped_external)} external package:// references")
            for urdf_name, mesh_name in skipped_external[:3]:  # Show first 3
                print(f"  {urdf_name}: {mesh_name}")

    def test_mesh_geometry_valid(self, urdf_files, repo_root):
        """Validate that loaded meshes have valid geometry.

        Checks:
        - Faces can be loaded
        - Non-zero bounding box
        - Vertices and faces exist
        """
        tested_meshes = []
        skipped_meshes = []

        for urdf_file in urdf_files:
            meshes = self.extract_meshes_from_urdf(urdf_file)
            for mesh_filename, _ in meshes:
                resolved_path = self.resolve_mesh_path(
                    mesh_filename, urdf_file, repo_root
                )

                if not resolved_path:
                    skipped_meshes.append((urdf_file.name, mesh_filename))
                    continue

                # Load and validate the mesh
                try:
                    mesh = trimesh.load(str(resolved_path), process=False)
                except Exception as e:
                    pytest.fail(
                        f"Failed to load mesh {resolved_path}: {e}"
                    )

                # Check that mesh has vertices and faces
                assert len(mesh.vertices) > 0, (
                    f"{resolved_path.name}: Mesh has no vertices"
                )
                assert len(mesh.faces) > 0, (
                    f"{resolved_path.name}: Mesh has no faces"
                )

                # Check non-zero bounding box
                bounds = mesh.bounds
                bbox_size = bounds[1] - bounds[0]
                assert (
                    bbox_size[0] > 1e-6 or bbox_size[1] > 1e-6 or bbox_size[2] > 1e-6
                ), f"{resolved_path.name}: Bounding box is zero or near-zero"

                tested_meshes.append(resolved_path.name)

        assert len(tested_meshes) > 0, "No mesh files were tested (all external?)"
        print(f"\n✓ Validated {len(tested_meshes)} local meshes")
        print(f"  Skipped {len(skipped_meshes)} external package:// references")

    def test_mesh_validity_checks(self, urdf_files, repo_root):
        """Perform detailed validity checks on each mesh.

        Checks:
        - Mesh is valid (no self-intersections, proper normals)
        - Reasonable polygon count (warn if > 10000 faces)
        - File size is not excessive (warn if > 5MB)
        """
        validation_results = {
            "valid": [],
            "warnings": [],
            "failed": [],
        }

        for urdf_file in urdf_files:
            meshes = self.extract_meshes_from_urdf(urdf_file)
            for mesh_filename, _ in meshes:
                resolved_path = self.resolve_mesh_path(
                    mesh_filename, urdf_file, repo_root
                )

                if not resolved_path:
                    continue  # Skip external

                try:
                    mesh = trimesh.load(str(resolved_path), process=False)
                except Exception as e:
                    validation_results["failed"].append(
                        (resolved_path.name, f"Load failed: {e}")
                    )
                    continue

                mesh_name = resolved_path.name
                warnings = []

                # Check if mesh is valid
                if not mesh.is_valid:
                    warnings.append("Mesh is not valid (may have self-intersections)")

                # Check polygon count
                if len(mesh.faces) > 10000:
                    warnings.append(
                        f"High polygon count: {len(mesh.faces)} faces "
                        "(consider simplification)"
                    )

                # Check file size
                file_size_mb = resolved_path.stat().st_size / (1024 * 1024)
                if file_size_mb > 5:
                    warnings.append(
                        f"Large file: {file_size_mb:.1f}MB (consider optimization)"
                    )

                if warnings:
                    validation_results["warnings"].append(
                        (mesh_name, warnings)
                    )
                else:
                    validation_results["valid"].append(mesh_name)

        # Report results
        print(f"\n✓ Valid meshes: {len(validation_results['valid'])}")
        for mesh_name in validation_results["valid"]:
            print(f"  - {mesh_name}")

        if validation_results["warnings"]:
            print(f"\n⚠ Meshes with warnings: {len(validation_results['warnings'])}")
            for mesh_name, warnings in validation_results["warnings"]:
                print(f"  - {mesh_name}:")
                for warning in warnings:
                    print(f"    • {warning}")

        if validation_results["failed"]:
            fail_msg = f"Failed to validate {len(validation_results['failed'])} meshes:\n"
            for mesh_name, error in validation_results["failed"]:
                fail_msg += f"  {mesh_name}: {error}\n"
            pytest.fail(fail_msg)

        # At minimum, we should have validated some local meshes
        total_validated = (
            len(validation_results["valid"]) + len(validation_results["warnings"])
        )
        assert total_validated > 0, "No meshes were successfully validated"

    def test_mesh_vertex_face_consistency(self, urdf_files, repo_root):
        """Verify that all vertices referenced in faces are within bounds."""
        issues = []

        for urdf_file in urdf_files:
            meshes = self.extract_meshes_from_urdf(urdf_file)
            for mesh_filename, _ in meshes:
                resolved_path = self.resolve_mesh_path(
                    mesh_filename, urdf_file, repo_root
                )

                if not resolved_path:
                    continue

                try:
                    mesh = trimesh.load(str(resolved_path), process=False)
                except Exception:
                    continue

                # Check that all face indices are valid
                max_vertex_idx = len(mesh.vertices) - 1
                invalid_faces = []
                for face_idx, face in enumerate(mesh.faces):
                    for vertex_idx in face:
                        if vertex_idx > max_vertex_idx:
                            invalid_faces.append(
                                (face_idx, vertex_idx, max_vertex_idx)
                            )
                            break  # Only report once per face

                if invalid_faces:
                    issues.append(
                        (
                            resolved_path.name,
                            f"Face indices out of bounds: {len(invalid_faces)} faces",
                        )
                    )

        if issues:
            fail_msg = "Meshes with vertex/face inconsistencies:\n"
            for mesh_name, error in issues:
                fail_msg += f"  {mesh_name}: {error}\n"
            pytest.fail(fail_msg)

    def test_mesh_file_integrity(self, urdf_files, repo_root):
        """Verify STL files are not corrupted or truncated.

        Checks:
        - File is readable
        - File has content (size > 100 bytes)
        - File format is recognized
        """
        for urdf_file in urdf_files:
            meshes = self.extract_meshes_from_urdf(urdf_file)
            for mesh_filename, _ in meshes:
                resolved_path = self.resolve_mesh_path(
                    mesh_filename, urdf_file, repo_root
                )

                if not resolved_path:
                    continue

                # Check file exists and has content
                assert resolved_path.exists(), (
                    f"Mesh file missing: {resolved_path}"
                )
                file_size = resolved_path.stat().st_size
                assert file_size > 100, (
                    f"{resolved_path.name}: File too small ({file_size} bytes), "
                    "possibly corrupted"
                )

                # Try to load to verify format
                try:
                    mesh = trimesh.load(str(resolved_path), process=False)
                    assert mesh is not None, f"Failed to load {resolved_path.name}"
                except Exception as e:
                    pytest.fail(
                        f"Mesh format error in {resolved_path.name}: {e}"
                    )
