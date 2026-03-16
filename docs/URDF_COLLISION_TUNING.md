# URDF Collision Primitive Tuning Guide

## Overview

This document describes how to use STL mesh analysis to tune collision primitive sizes for robot URDFs. The process extracts exact axis-aligned bounding boxes (AABBs) from 3D mesh files and updates URDF collision geometry dimensions to match actual CAD geometry.

## Why Tuning Matters

- **Collision Accuracy**: Hand-estimated collision boxes can miss geometry or create false gaps
- **Grasp Planning**: Tighter bounds improve obstacle avoidance and reachability analysis
- **Simulation Fidelity**: Accurate collision geometry improves simulation accuracy
- **Performance**: No overhead—geometric primitives are equally fast whether loose or tight

## Quick Start

### 1. Analyze Meshes and Generate Recommendations

```bash
./run.sh scripts/analyze_stl_aabb.py
```

This command:
- Loads all STL files in `assets/so101/assets/`
- Computes AABB for each mesh
- Aggregates meshes by robot link
- Generates `analysis_stl_aabb.json` with recommendations

### 2. Review Analysis Output

```bash
cat analysis_stl_aabb.json | jq '.recommendations'
```

See JSON file for:
- **Current**: Hand-estimated collision dimensions
- **Recommended**: Exact AABB dimensions from meshes
- **AABB**: Computed bounds (min, max, center, extents)

### 3. Generate Updated URDF

```bash
./run.sh scripts/analyze_stl_aabb.py --generate-urdf
```

Creates: `assets/so101/so101_new_calib_updated.urdf`

### 4. Validate Updates

```bash
./run.sh scripts/validate_collision_updates.py
```

Checks:
- ✓ URDF parses correctly
- ✓ All dimensions are non-zero and reasonable
- ✓ No anomalies detected
- ✓ Volume changes are sensible

### 5. Deploy Updated URDF

Once validated:
```bash
# Backup original
cp assets/so101/so101_new_calib.urdf assets/so101/so101_new_calib.urdf.bak

# Deploy updated version
mv assets/so101/so101_new_calib_updated.urdf assets/so101/so101_new_calib.urdf
```

## Understanding the Analysis

### What is an AABB?

An Axis-Aligned Bounding Box (AABB) is the smallest box aligned with coordinate axes that fully contains a 3D object.

```
    max (x_max, y_max, z_max)
        +-------+
       /|      /|
      / |     / |
     +-------+  |
     |  |    |  |
     |  +----|-++
     | /     | /
     |/      |/
     +-------+
min (x_min, y_min, z_min)
```

### How Mesh Aggregation Works

For each robot link with multiple visual meshes:

1. **Identify components**: Servo, 3D-printed parts, brackets, etc.
2. **Compute individual AABBs**: For each STL file
3. **Merge AABBs**: Find bounds encompassing all components
4. **Calculate center**: Geometric center of merged AABB
5. **Calculate extents**: Full width, height, depth from bounds

Example (shoulder_link):
```
servo (sts3215_03a_v1.stl):
  AABB: [-0.0227, -0.0124, -0.0194] to [0.0241, 0.0124, 0.0206]

motor_holder (motor_holder_so101_base_v1.stl):
  AABB: [-0.0540, -0.0551, -0.0289] to [-0.0254, -0.0170, 0.0262]

rotation_pitch (rotation_pitch_so101_v1.stl):
  AABB: [-0.0598, 0.0001, -0.0230] to [0.0006, 0.0846, 0.0230]

MERGED (shoulder_link):
  AABB: [-0.0598, -0.0551, -0.0289] to [0.0241, 0.0846, 0.0262]
  Center: (-0.0178, 0.0148, -0.0013)
  Extents: (0.0839, 0.1397, 0.0551)
```

## File Structure

### Input Files
```
assets/so101/assets/
├── base_motor_holder_so101_v1.stl
├── base_so101_v2.stl
├── motor_holder_so101_base_v1.stl
├── motor_holder_so101_wrist_v1.stl
├── moving_jaw_so101_v1.stl
├── rotation_pitch_so101_v1.stl
├── sts3215_03a_v1.stl              # Servo with horn
├── sts3215_03a_no_horn_v1.stl      # Servo without horn
├── under_arm_so101_v1.stl
├── upper_arm_so101_v1.stl
├── waveshare_mounting_plate_so101_v2.stl
├── wrist_roll_follower_so101_v1.stl
├── wrist_roll_pitch_so101_v2.stl
└── wrist_camera_so101_v1.stl
```

### Output Files
```
analysis_stl_aabb.json                          # Complete analysis data
assets/so101/so101_new_calib_updated.urdf       # Updated URDF
COLLISION_TUNING_REPORT.md                      # Detailed report
```

### Scripts
```
scripts/analyze_stl_aabb.py                     # Main analysis tool
scripts/validate_collision_updates.py           # Validation tool
```

## Interpreting Results

### Dimension Changes

**base_link**: -2.2% X, +3.4% Y, -6.5% Z (Total: -5.4% volume)
- **Interpretation**: Hand estimate was slightly oversized. New bounds are tighter.

**shoulder_link**: -37.8% X, +154% Y, -49.9% Z (Total: -20.9% volume)
- **Interpretation**: Hand estimate missed vertical extent of servo stack. Y is 2.5× larger in reality.

**gripper_link**: +59% X, +15.6% Y, +14.3% Z (Total: +110% volume)
- **Interpretation**: Wrist camera not included in hand estimate. Gripper envelope should be larger.

### Origin Shifts

Origin (center point) can shift significantly when hand-estimated positions are off:

- **lower_arm_link**: X origin +76mm - indicates misalignment
- **gripper_link**: Z origin +91.5mm - camera offset was ignored
- **shoulder_link**: Y origin +14.7mm - incorrect vertical centering

### Volume Changes

Overall:
- **-8.8% total volume** across all links
- **Most links tighter** (better accuracy, fewer false positives)
- **Gripper larger** (accounts for wrist camera)

## Validation Checks

The `validate_collision_updates.py` script checks:

1. **URDF Parsing**
   - Both original and updated URDFs parse correctly
   - No XML syntax errors

2. **Dimension Validity**
   - All box dimensions > 0
   - No dimensions > 300mm (robot arms < 30cm)
   - No dimensions < 10mm (manufacturing limit)

3. **Asymmetry Detection**
   - Warns if any dimension ratio > 10:1
   - Flags highly asymmetric shapes for review

4. **Volume Sanity Check**
   - Flags volume changes > 30% for manual review
   - Detects suspiciously large increases/decreases

## Advanced Usage

### Custom Output Path

```bash
./run.sh scripts/analyze_stl_aabb.py --generate-urdf --output /tmp/custom_urdf.urdf
```

### Detailed Mesh Metrics

```bash
./run.sh scripts/analyze_stl_aabb.py --detailed
```

Shows:
- Vertex and face counts
- Mesh volume
- Watertightness status
- Detailed AABB bounds for each mesh

## Troubleshooting

### "File not found" for STL mesh

**Issue**: Analysis reports missing STL file

**Solution**: Verify file exists in `assets/so101/assets/` and check spelling

### Dimension seems wrong

**Options**:
1. Verify mesh file is up-to-date (not outdated/backup version)
2. Check link mesh composition in `analyze_stl_aabb.py` (line ~400)
3. Manually inspect STL in CAD tool to verify bounds
4. Consider if mesh origin is offset from visual appearance

### URDF doesn't load in RViz

**Checklist**:
- Run `./run.sh scripts/validate_collision_updates.py` first
- Verify URD is valid XML (try parsing online)
- Check file paths in URDF (relative vs. absolute)
- Ensure asset directory structure unchanged

## Performance Impact

**Load Time**: <1ms per mesh (trimesh is highly optimized)

**Collision Checking**: No change
- Uses same geometric primitives (boxes/cylinders)
- Computational complexity identical
- Only benefit: more accurate collision results

**Simulation**: Marginal improvement
- Tighter bounds = fewer false positives
- Better grasp planning = fewer retry cycles

## Future Enhancements

### Oriented Bounding Boxes (OBB)

Current implementation uses axis-aligned boxes (AABB). Could use rotated boxes for better fit:

```bash
./run.sh scripts/analyze_stl_aabb.py --obb  # Future feature
```

Tradeoff: Better fit (fewer false positives) but more complex collision checking

### Convex Hulls

For non-convex shapes, convex hull provides tighter approximation:

```bash
./run.sh scripts/analyze_stl_aabb.py --convex-hull  # Future feature
```

Tradeoff: More accurate but slower collision checks

### Multi-Box Decomposition

Split complex geometry into multiple simpler boxes:

```bash
./run.sh scripts/analyze_stl_aabb.py --multi-box-threshold 0.5  # Future feature
```

### CAD Pipeline Integration

Automate analysis in `onshape-to-robot` export process

## References

- [Trimesh Documentation](https://trimsh.org/)
- [URDF Specification](http://wiki.ros.org/urdf/XML)
- [ROS Geometry](http://wiki.ros.org/geometry/bullet_algorithms)
- [Collision Detection](https://en.wikipedia.org/wiki/Collision_detection)

---

**Last Updated**: 2026-03-16
**Tool Version**: analyze_stl_aabb.py v1.0
**Robot**: SO-ARM101 (so101_new_calib.urdf)
