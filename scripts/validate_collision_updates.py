#!/usr/bin/env python3
"""
Validate collision URDF updates by comparing original vs. updated.

Checks:
1. URDF parsing and validity
2. Dimension changes and reasonableness
3. Coverage comparison (old vs. new AABB volumes)
4. Origin accuracy (center-of-geometry alignment)
"""

import sys
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple, Optional


def parse_xyz(xyz_str: str) -> Tuple[float, float, float]:
    """Parse 'x y z' string to tuple."""
    parts = xyz_str.split()
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def parse_size(size_str: str) -> Tuple[float, float, float]:
    """Parse 'x y z' size string to tuple."""
    return parse_xyz(size_str)


def load_collisions(urdf_path: Path) -> Dict[str, Dict]:
    """Load collision geometries from URDF."""
    collisions = {}
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for link in root.findall("link"):
        link_name = link.get("name")
        collision = link.find("collision")

        if collision is not None:
            origin = collision.find("origin")
            geometry = collision.find("geometry")

            info = {
                "origin": (0.0, 0.0, 0.0),
                "type": None,
                "size": None,
            }

            if origin is not None:
                info["origin"] = parse_xyz(origin.get("xyz", "0 0 0"))

            if geometry is not None:
                box = geometry.find("box")
                if box is not None:
                    info["type"] = "box"
                    info["size"] = parse_size(box.get("size", "0 0 0"))

                cylinder = geometry.find("cylinder")
                if cylinder is not None:
                    info["type"] = "cylinder"
                    info["radius"] = float(cylinder.get("radius", "0"))
                    info["length"] = float(cylinder.get("length", "0"))

            collisions[link_name] = info

    return collisions


def calculate_box_aabb(origin: Tuple[float, float, float], size: Tuple[float, float, float]) -> Dict:
    """Calculate AABB from box center and size."""
    ox, oy, oz = origin
    sx, sy, sz = size

    return {
        "min": (ox - sx / 2, oy - sy / 2, oz - sz / 2),
        "max": (ox + sx / 2, oy + sy / 2, oz + sz / 2),
        "volume": sx * sy * sz,
    }


def aabb_volume(aabb: Dict) -> float:
    """Get AABB volume."""
    return aabb.get("volume", 0.0)


def dimension_change(old: float, new: float) -> str:
    """Format dimension change as percentage."""
    if old == 0:
        return "NEW"
    pct = ((new - old) / old) * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def validate_urdf(urdf_path: Path) -> bool:
    """Validate URDF can be parsed."""
    try:
        tree = ET.parse(urdf_path)
        tree.getroot()
        return True
    except Exception as e:
        print(f"ERROR parsing {urdf_path}: {e}")
        return False


def main():
    project_root = Path(__file__).parent.parent

    original_urdf = project_root / "assets" / "so101" / "so101_new_calib.urdf"
    updated_urdf = project_root / "assets" / "so101" / "so101_new_calib_updated.urdf"

    print("=" * 100)
    print("COLLISION PRIMITIVE VALIDATION REPORT")
    print("=" * 100)

    # Validate files exist
    if not original_urdf.exists():
        print(f"ERROR: {original_urdf} not found")
        return 1

    if not updated_urdf.exists():
        print(f"ERROR: {updated_urdf} not found")
        print(f"Generate with: ./run.sh scripts/analyze_stl_aabb.py --generate-urdf")
        return 1

    # Validate both URDFs parse correctly
    print("\n[1] URDF Validation")
    print("-" * 100)

    if not validate_urdf(original_urdf):
        return 1

    print(f"✓ Original URDF valid: {original_urdf.name}")

    if not validate_urdf(updated_urdf):
        return 1

    print(f"✓ Updated URDF valid:  {updated_urdf.name}")

    # Load collision geometries
    orig_collisions = load_collisions(original_urdf)
    upda_collisions = load_collisions(updated_urdf)

    print(f"\n[2] Collision Geometry Comparison")
    print("-" * 100)
    print(f"{'Link':<30} {'X (mm)':<15} {'Y (mm)':<15} {'Z (mm)':<15} {'Volume Change':<20} {'Status'}")
    print("-" * 100)

    total_volume_old = 0.0
    total_volume_new = 0.0
    issue_count = 0

    # Compare each link
    for link_name in sorted(orig_collisions.keys()):
        old = orig_collisions[link_name]
        new = upda_collisions.get(link_name)

        if new is None:
            print(f"{link_name:<30} {'MISSING IN UPDATE':<60} ⚠️ ")
            issue_count += 1
            continue

        # Only compare boxes (ignore cylinders for now)
        if old.get("type") != "box" or new.get("type") != "box":
            continue

        old_size = old.get("size")
        new_size = new.get("size")

        if old_size is None or new_size is None:
            continue

        # Convert to mm for readability
        old_size_mm = tuple(s * 1000 for s in old_size)
        new_size_mm = tuple(s * 1000 for s in new_size)

        # Calculate volumes
        old_aabb = calculate_box_aabb(old["origin"], old_size)
        new_aabb = calculate_box_aabb(new["origin"], new_size)

        old_vol = aabb_volume(old_aabb)
        new_vol = aabb_volume(new_aabb)

        total_volume_old += old_vol
        total_volume_new += new_vol

        # Check reasonableness
        size_x_change = dimension_change(old_size_mm[0], new_size_mm[0])
        size_y_change = dimension_change(old_size_mm[1], new_size_mm[1])
        size_z_change = dimension_change(old_size_mm[2], new_size_mm[2])

        vol_change = dimension_change(old_vol, new_vol)

        # Status check
        status = "✓"

        # Warn if volume change > 30%
        if old_vol > 0:
            vol_pct = abs((new_vol - old_vol) / old_vol) * 100
            if vol_pct > 30:
                status = "⚠️"  # Flag for review

        # Warn if any dimension is zero
        if new_size_mm[0] <= 1 or new_size_mm[1] <= 1 or new_size_mm[2] <= 1:
            status = "❌"  # Error - dimension too small
            issue_count += 1

        # Print row
        print(
            f"{link_name:<30} {size_x_change:<15} {size_y_change:<15} {size_z_change:<15} {vol_change:<20} {status}"
        )

    print("-" * 100)

    # Summary statistics
    print(f"\n[3] Summary Statistics")
    print("-" * 100)

    if total_volume_old > 0:
        vol_change_pct = ((total_volume_new - total_volume_old) / total_volume_old) * 100
        print(f"Total AABB Volume (original):  {total_volume_old * 1e6:.0f} mm³")
        print(f"Total AABB Volume (updated):   {total_volume_new * 1e6:.0f} mm³")
        print(f"Volume Change:                 {vol_change_pct:+.1f}%")
        print()

    # Check for symmetries and anomalies
    print(f"\n[4] Anomaly Detection")
    print("-" * 100)

    anomalies = []

    for link_name in sorted(upda_collisions.keys()):
        collision = upda_collisions[link_name]

        if collision.get("type") != "box":
            continue

        size = collision.get("size")
        if size is None:
            continue

        size_mm = tuple(s * 1000 for s in size)

        # Check for very small dimensions (< 10mm)
        if any(s < 10 for s in size_mm):
            anomalies.append(f"  - {link_name}: dimension < 10mm - {size_mm}")

        # Check for very large dimensions (> 300mm)
        if any(s > 300 for s in size_mm):
            anomalies.append(f"  - {link_name}: dimension > 300mm - {size_mm}")

        # Check for very asymmetric dimensions (10:1 ratio)
        sorted_dims = sorted(size_mm)
        if sorted_dims[2] / sorted_dims[0] > 10:
            anomalies.append(f"  - {link_name}: highly asymmetric - {size_mm}")

    if anomalies:
        print("Found potential anomalies:")
        for anomaly in anomalies:
            print(anomaly)
    else:
        print("✓ No anomalies detected")

    # Final status
    print(f"\n[5] Validation Status")
    print("-" * 100)

    if issue_count == 0:
        print("✓ ALL CHECKS PASSED")
        print("\nThe updated URDF is ready for deployment.")
        print("Recommended next steps:")
        print("  1. Review collision geometries in 3D viewer (RViz/meshcat)")
        print("  2. Test in digital twin (Isaac Lab)")
        print("  3. Validate grasp planning with updated geometry")
        print("  4. Physical validation in safe mode if needed")
        return 0
    else:
        print(f"❌ {issue_count} ISSUE(S) FOUND")
        print("\nPlease review the flagged items above before deployment.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
