#!/usr/bin/env python3
"""
Analyze STL meshes and extract exact axis-aligned bounding boxes (AABBs).

This tool computes the exact AABB for each STL mesh and compares against
the current hand-estimated collision primitives in URDF files. It generates
recommendations for tighter, more accurate collision geometries.

Usage:
    ./run.sh scripts/analyze_stl_aabb.py                    # Analyze SO101
    ./run.sh scripts/analyze_stl_aabb.py --robot nova5      # Analyze Nova5
    ./run.sh scripts/analyze_stl_aabb.py --detailed          # Show detailed metrics
    ./run.sh scripts/analyze_stl_aabb.py --generate-urdf     # Generate updated URDF
"""

import sys
import os
import json
import argparse
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET

try:
    import trimesh
except ImportError:
    print("ERROR: trimesh not found. Install with: pip install trimesh")
    sys.exit(1)


@dataclass
class AABB:
    """Axis-Aligned Bounding Box with center and half-extents."""
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float

    @property
    def center(self) -> Tuple[float, float, float]:
        """Return center (x, y, z)."""
        return (
            (self.min_x + self.max_x) / 2,
            (self.min_y + self.max_y) / 2,
            (self.min_z + self.max_z) / 2,
        )

    @property
    def extents(self) -> Tuple[float, float, float]:
        """Return half-extents (dx, dy, dz)."""
        return (
            (self.max_x - self.min_x),
            (self.max_y - self.min_y),
            (self.max_z - self.min_z),
        )

    @property
    def volume(self) -> float:
        """Return bounding box volume."""
        dx, dy, dz = self.extents
        return dx * dy * dz

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "min": [self.min_x, self.min_y, self.min_z],
            "max": [self.max_x, self.max_y, self.max_z],
            "center": list(self.center),
            "extents": list(self.extents),
            "volume": self.volume,
        }


@dataclass
class MeshAnalysis:
    """Analysis result for a single mesh file."""
    filename: str
    filepath: Path
    exists: bool
    error: Optional[str] = None
    vertex_count: int = 0
    face_count: int = 0
    aabb: Optional[AABB] = None
    volume: float = 0.0
    is_watertight: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "filename": self.filename,
            "exists": self.exists,
            "error": self.error,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "aabb": self.aabb.to_dict() if self.aabb else None,
            "volume": self.volume,
            "is_watertight": self.is_watertight,
        }


def analyze_mesh(filepath: Path) -> MeshAnalysis:
    """
    Load and analyze a single STL mesh.

    Returns:
        MeshAnalysis with computed AABB, vertex/face counts, etc.
    """
    result = MeshAnalysis(filename=filepath.name, filepath=filepath, exists=filepath.exists())

    if not result.exists:
        result.error = "File not found"
        return result

    try:
        mesh = trimesh.load(filepath, process=False)

        result.vertex_count = len(mesh.vertices)
        result.face_count = len(mesh.faces)
        result.volume = mesh.volume

        # Compute AABB from vertices
        bounds = mesh.bounds
        result.aabb = AABB(
            min_x=float(bounds[0][0]),
            max_x=float(bounds[1][0]),
            min_y=float(bounds[0][1]),
            max_y=float(bounds[1][1]),
            min_z=float(bounds[0][2]),
            max_z=float(bounds[1][2]),
        )

        # Check if mesh is watertight (closed manifold)
        result.is_watertight = mesh.is_watertight

    except Exception as e:
        result.error = str(e)

    return result


def analyze_so101_meshes(asset_dir: Path) -> Dict[str, MeshAnalysis]:
    """
    Analyze all STL meshes for SO101.

    Args:
        asset_dir: Path to assets/so101/assets directory

    Returns:
        Dictionary mapping filename to MeshAnalysis
    """
    results = {}

    # List of SO101 meshes used in URDF
    mesh_files = [
        "base_motor_holder_so101_v1.stl",
        "base_so101_v2.stl",
        "motor_holder_so101_base_v1.stl",
        "motor_holder_so101_wrist_v1.stl",
        "moving_jaw_so101_v1.stl",
        "rotation_pitch_so101_v1.stl",
        "sts3215_03a_v1.stl",
        "sts3215_03a_no_horn_v1.stl",
        "under_arm_so101_v1.stl",
        "upper_arm_so101_v1.stl",
        "waveshare_mounting_plate_so101_v2.stl",
        "wrist_roll_follower_so101_v1.stl",
        "wrist_roll_pitch_so101_v2.stl",
        "wrist_camera_so101_v1.stl",
    ]

    for mesh_file in mesh_files:
        filepath = asset_dir / mesh_file
        print(f"Analyzing {mesh_file}...", end=" ", flush=True)
        result = analyze_mesh(filepath)
        if result.error:
            print(f"ERROR: {result.error}")
        else:
            print(f"OK ({result.vertex_count} vertices, {result.face_count} faces)")
        results[mesh_file] = result

    return results


def load_urdf_collisions(urdf_path: Path) -> Dict[str, Dict]:
    """
    Parse URDF and extract current collision geometries.

    Returns:
        Dictionary mapping link_name to collision geometry info
    """
    collisions = {}

    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        # Namespace handling for URDF
        ns = {"": "http://www.ros.org/urdf"}

        for link in root.findall("link"):
            link_name = link.get("name")
            collision = link.find("collision")

            if collision is not None:
                origin = collision.find("origin")
                geometry = collision.find("geometry")

                collision_info = {
                    "name": link_name,
                    "origin": {},
                    "geometry": {},
                }

                if origin is not None:
                    collision_info["origin"] = {
                        "xyz": origin.get("xyz", "0 0 0"),
                        "rpy": origin.get("rpy", "0 0 0"),
                    }

                if geometry is not None:
                    # Check for box
                    box = geometry.find("box")
                    if box is not None:
                        collision_info["geometry"]["type"] = "box"
                        collision_info["geometry"]["size"] = box.get("size", "0 0 0")

                    # Check for cylinder
                    cylinder = geometry.find("cylinder")
                    if cylinder is not None:
                        collision_info["geometry"]["type"] = "cylinder"
                        collision_info["geometry"]["radius"] = cylinder.get("radius", "0")
                        collision_info["geometry"]["length"] = cylinder.get("length", "0")

                    # Check for sphere
                    sphere = geometry.find("sphere")
                    if sphere is not None:
                        collision_info["geometry"]["type"] = "sphere"
                        collision_info["geometry"]["radius"] = sphere.get("radius", "0")

                collisions[link_name] = collision_info

    except Exception as e:
        print(f"ERROR parsing URDF {urdf_path}: {e}")

    return collisions


def compare_collision_primitives(
    mesh_analyses: Dict[str, MeshAnalysis],
    link_meshes: Dict[str, List[str]],
    current_collisions: Dict[str, Dict],
) -> Dict[str, Dict]:
    """
    Compare mesh AABBs with current collision primitives.

    Args:
        mesh_analyses: Dictionary of mesh analysis results
        link_meshes: Mapping of link_name to list of mesh filenames
        current_collisions: Current URDF collision definitions

    Returns:
        Dictionary with comparison results and recommendations
    """
    recommendations = {}

    for link_name, mesh_files in link_meshes.items():
        print(f"\n=== Link: {link_name} ===")

        # Combine AABBs of all meshes in this link
        combined_aabb = None
        all_valid = True

        for mesh_file in mesh_files:
            if mesh_file not in mesh_analyses:
                print(f"  WARNING: {mesh_file} not analyzed")
                all_valid = False
                continue

            analysis = mesh_analyses[mesh_file]
            if analysis.error or analysis.aabb is None:
                print(f"  WARNING: {mesh_file} - {analysis.error}")
                all_valid = False
                continue

            if combined_aabb is None:
                combined_aabb = analysis.aabb
            else:
                # Merge AABBs
                combined_aabb = AABB(
                    min_x=min(combined_aabb.min_x, analysis.aabb.min_x),
                    max_x=max(combined_aabb.max_x, analysis.aabb.max_x),
                    min_y=min(combined_aabb.min_y, analysis.aabb.min_y),
                    max_y=max(combined_aabb.max_y, analysis.aabb.max_y),
                    min_z=min(combined_aabb.min_z, analysis.aabb.min_z),
                    max_z=max(combined_aabb.max_z, analysis.aabb.max_z),
                )

        if combined_aabb is None or not all_valid:
            print(f"  SKIP: Could not compute complete AABB")
            continue

        # Get current collision definition
        current = current_collisions.get(link_name)
        if current is None:
            print(f"  NOTE: No collision geometry in current URDF")
            current = {}

        center, extents = combined_aabb.center, combined_aabb.extents

        recommendation = {
            "link_name": link_name,
            "meshes": mesh_files,
            "current": current,
            "aabb": combined_aabb.to_dict(),
            "recommended": {
                "type": "box",
                "origin": {
                    "xyz": f"{center[0]:.6f} {center[1]:.6f} {center[2]:.6f}",
                    "rpy": "0 0 0",
                },
                "size": f"{extents[0]:.6f} {extents[1]:.6f} {extents[2]:.6f}",
            },
        }

        recommendations[link_name] = recommendation

        # Print comparison
        print(f"  Meshes: {', '.join(mesh_files)}")
        print(f"  AABB: [{combined_aabb.min_x:.6f}, {combined_aabb.min_y:.6f}, {combined_aabb.min_z:.6f}]")
        print(f"       to [{combined_aabb.max_x:.6f}, {combined_aabb.max_y:.6f}, {combined_aabb.max_z:.6f}]")
        print(f"  Center: {center}")
        print(f"  Extents (size): {extents}")
        print(f"  Volume: {combined_aabb.volume:.6e} m³")

        if current:
            print(f"  Current: {current['geometry'].get('type', 'N/A')} - {current['geometry'].get('size', current['geometry'].get('radius', 'N/A'))}")

    return recommendations


def print_mesh_analysis_report(analyses: Dict[str, MeshAnalysis], detailed: bool = False):
    """Print detailed mesh analysis report."""
    print("\n" + "=" * 80)
    print("MESH ANALYSIS REPORT")
    print("=" * 80)

    for filename in sorted(analyses.keys()):
        analysis = analyses[filename]
        print(f"\n{filename}")
        print(f"  Path: {analysis.filepath}")
        print(f"  Exists: {analysis.exists}")

        if analysis.error:
            print(f"  ERROR: {analysis.error}")
            continue

        print(f"  Vertices: {analysis.vertex_count}")
        print(f"  Faces: {analysis.face_count}")
        print(f"  Volume: {analysis.volume:.6e} m³")
        print(f"  Watertight: {analysis.is_watertight}")

        if analysis.aabb:
            cx, cy, cz = analysis.aabb.center
            dx, dy, dz = analysis.aabb.extents
            print(f"  AABB Center: ({cx:.6f}, {cy:.6f}, {cz:.6f})")
            print(f"  AABB Extents: ({dx:.6f}, {dy:.6f}, {dz:.6f})")
            print(f"  AABB Min: ({analysis.aabb.min_x:.6f}, {analysis.aabb.min_y:.6f}, {analysis.aabb.min_z:.6f})")
            print(f"  AABB Max: ({analysis.aabb.max_x:.6f}, {analysis.aabb.max_y:.6f}, {analysis.aabb.max_z:.6f})")


def generate_updated_urdf(
    original_urdf: Path,
    recommendations: Dict[str, Dict],
    output_path: Path,
):
    """
    Generate an updated URDF file with exact collision dimensions.

    Args:
        original_urdf: Path to original URDF file
        recommendations: Dictionary of recommended collision updates
        output_path: Path to write updated URDF
    """
    print(f"\nGenerating updated URDF: {output_path}")

    try:
        tree = ET.parse(original_urdf)
        root = tree.getroot()

        # Register namespace to preserve prefixes
        namespaces = {
            "": "http://www.ros.org/urdf",
        }
        for prefix, uri in namespaces.items():
            ET.register_namespace(prefix if prefix else "", uri)

        updated_count = 0

        for link in root.findall("link"):
            link_name = link.get("name")

            if link_name not in recommendations:
                continue

            rec = recommendations[link_name]
            collision = link.find("collision")

            if collision is None:
                # Create new collision element
                collision = ET.Element("collision")
                link.append(collision)

            # Update origin
            origin = collision.find("origin")
            if origin is None:
                origin = ET.SubElement(collision, "origin")

            xyz_parts = rec["recommended"]["origin"]["xyz"].split()
            origin.set("xyz", rec["recommended"]["origin"]["xyz"])
            origin.set("rpy", rec["recommended"]["origin"]["rpy"])

            # Update geometry
            geometry = collision.find("geometry")
            if geometry is None:
                geometry = ET.SubElement(collision, "geometry")
            else:
                # Clear existing geometry
                for child in list(geometry):
                    geometry.remove(child)

            # Add box with exact dimensions
            box = ET.SubElement(geometry, "box")
            box.set("size", rec["recommended"]["size"])

            # Add comment with the recommendation details
            link_index = list(root).index(link)
            comment = ET.Comment(
                f" Updated collision for {link_name}: {' '.join(rec['meshes'])} "
            )
            root.insert(link_index, comment)

            updated_count += 1

        # Write updated URDF
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        print(f"  Updated {updated_count} collision geometries")
        print(f"  Saved to: {output_path}")

    except Exception as e:
        print(f"ERROR generating URDF: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze STL meshes and extract collision primitive dimensions"
    )
    parser.add_argument(
        "--robot",
        choices=["so101", "nova5"],
        default="so101",
        help="Robot to analyze (default: so101)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed mesh metrics",
    )
    parser.add_argument(
        "--generate-urdf",
        action="store_true",
        help="Generate updated URDF file with exact collision dimensions",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for updated URDF (default: assets/so101/so101_new_calib_updated.urdf)",
    )

    args = parser.parse_args()

    # Get project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    if args.robot == "so101":
        # SO101 analysis
        asset_dir = project_root / "assets" / "so101" / "assets"
        urdf_path = project_root / "assets" / "so101" / "so101_new_calib.urdf"

        print(f"Analyzing SO101 meshes in {asset_dir}")
        print("=" * 80)

        # Analyze all meshes
        analyses = analyze_so101_meshes(asset_dir)

        # Print mesh analysis
        if args.detailed:
            print_mesh_analysis_report(analyses, detailed=True)

        # Load current URDF collisions
        current_collisions = load_urdf_collisions(urdf_path)

        # Map link names to their component meshes
        link_meshes = {
            "base_link": [
                "base_motor_holder_so101_v1.stl",
                "base_so101_v2.stl",
                "sts3215_03a_v1.stl",
                "waveshare_mounting_plate_so101_v2.stl",
            ],
            "shoulder_link": [
                "sts3215_03a_v1.stl",  # servo
                "motor_holder_so101_base_v1.stl",
                "rotation_pitch_so101_v1.stl",
            ],
            "upper_arm_link": [
                "sts3215_03a_v1.stl",  # servo
                "upper_arm_so101_v1.stl",
            ],
            "lower_arm_link": [
                "under_arm_so101_v1.stl",
                "motor_holder_so101_wrist_v1.stl",
                "sts3215_03a_v1.stl",  # servo
            ],
            "wrist_link": [
                "sts3215_03a_no_horn_v1.stl",  # servo (no horn)
                "wrist_roll_pitch_so101_v2.stl",
            ],
            "gripper_link": [
                "sts3215_03a_v1.stl",  # servo
                "wrist_roll_follower_so101_v1.stl",
                "wrist_camera_so101_v1.stl",
            ],
            "moving_jaw_so101_v1_link": [
                "moving_jaw_so101_v1.stl",
            ],
        }

        # Generate comparison and recommendations
        recommendations = compare_collision_primitives(analyses, link_meshes, current_collisions)

        # Save analysis to JSON
        json_path = project_root / "analysis_stl_aabb.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    "robot": "so101",
                    "timestamp": str(Path(__file__).stat().st_mtime),
                    "meshes": {k: v.to_dict() for k, v in analyses.items()},
                    "recommendations": recommendations,
                },
                f,
                indent=2,
            )
        print(f"\nAnalysis saved to: {json_path}")

        # Generate updated URDF if requested
        if args.generate_urdf:
            output_urdf = args.output or (project_root / "assets" / "so101" / "so101_new_calib_updated.urdf")
            generate_updated_urdf(urdf_path, recommendations, Path(output_urdf))

    else:
        print("Nova5 analysis not yet implemented (meshes are in external ROS package)")
        print("To implement:")
        print("  1. Set ROS_PACKAGE_PATH to include dobot_description package")
        print("  2. Or download Nova5 URDF with meshes from http://www.dobot.cc")
        sys.exit(1)


if __name__ == "__main__":
    main()
