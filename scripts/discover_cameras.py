#!/usr/bin/env python3
"""Detect connected cameras and write config/cameras.yaml.

Scans V4L2 devices and Intel RealSense cameras, reads factory intrinsics
where possible, and writes a structured YAML registry that the rest of the
pipeline can consume.

Usage:
    ./run.sh scripts/discover_cameras.py            # Detect & write config
    ./run.sh scripts/discover_cameras.py --dry-run   # Print without writing
    ./run.sh scripts/discover_cameras.py --merge      # Merge into existing (keep mount/extrinsics)
"""

import argparse
import datetime
import glob
import os
import re
import subprocess
import sys
import math

import cv2
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from config_loader import config_path, _SHARED_CONFIG_DIR

# Optional RealSense support
try:
    import pyrealsense2 as rs
    HAS_RS = True
except ImportError:
    HAS_RS = False

# Project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
# Read from active config dir; write to shared (~/.config) dir.
_CAMERAS_YAML_READ = config_path('cameras.yaml')
_CALIBRATION_YAML = config_path('calibration.yaml')
_CAMERAS_YAML = os.path.join(_SHARED_CONFIG_DIR, 'cameras.yaml')


# ---------------------------------------------------------------------------
# V4L2 helpers
# ---------------------------------------------------------------------------

def _v4l2_device_info(dev_path: str) -> dict | None:
    """Read V4L2 device capabilities via v4l2-ctl.

    Returns dict with keys: driver, card, bus_info, or None on failure.
    """
    try:
        out = subprocess.check_output(
            ['v4l2-ctl', '--device', dev_path, '--all'],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode('utf-8', errors='replace')
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None

    info = {}
    for line in out.splitlines():
        line = line.strip()
        if line.startswith('Driver name'):
            info['driver'] = line.split(':', 1)[1].strip()
        elif line.startswith('Card type'):
            info['card'] = line.split(':', 1)[1].strip()
        elif line.startswith('Bus info'):
            info['bus_info'] = line.split(':', 1)[1].strip()
        elif line.startswith('Serial'):
            serial = line.split(':', 1)[1].strip()
            if serial:
                info['serial'] = serial
    return info if info else None


def _list_video_devices() -> list[str]:
    """Return sorted list of /dev/videoN paths — one per physical camera.

    Many cameras expose multiple /dev/video* nodes (capture, metadata, IR).
    We keep only the lowest-numbered node per physical USB device (bus_info)
    that supports 'Video Capture'.
    """
    devices = sorted(glob.glob('/dev/video*'),
                     key=lambda d: int(re.search(r'(\d+)$', d).group(1)))
    seen_bus = set()
    capture_devices = []
    for dev in devices:
        try:
            out = subprocess.check_output(
                ['v4l2-ctl', '--device', dev, '--all'],
                stderr=subprocess.DEVNULL, timeout=5,
            ).decode('utf-8', errors='replace')
        except (subprocess.CalledProcessError, FileNotFoundError,
                subprocess.TimeoutExpired):
            continue

        if 'Video Capture' not in out:
            continue

        # Deduplicate by bus_info (same physical camera = same bus path)
        bus_info = None
        for line in out.splitlines():
            if line.strip().startswith('Bus info'):
                bus_info = line.split(':', 1)[1].strip()
                break

        if bus_info and bus_info in seen_bus:
            continue
        if bus_info:
            seen_bus.add(bus_info)

        capture_devices.append(dev)
    return capture_devices


def _probe_webcam_intrinsics(device_index: int, width: int = 640,
                              height: int = 480) -> dict | None:
    """Open a webcam briefly to get actual resolution and estimate intrinsics."""
    try:
        cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        # Read one frame to verify
        ret, _ = cap.read()
        cap.release()

        if not ret:
            return None

        # Estimate pinhole intrinsics assuming 60-degree HFOV
        fov_h_deg = 60.0
        fx = actual_w / (2.0 * math.tan(math.radians(fov_h_deg / 2.0)))
        fy = fx
        cx = actual_w / 2.0
        cy = actual_h / 2.0

        return {
            'camera_matrix': [
                [round(fx, 1), 0, round(cx, 1)],
                [0, round(fy, 1), round(cy, 1)],
                [0, 0, 1],
            ],
            'dist_coeffs': [0.0, 0.0, 0.0, 0.0, 0.0],
            'image_size': [actual_w, actual_h],
            'source': 'estimated',
            'actual_fps': round(actual_fps, 1) if actual_fps > 0 else 30,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# RealSense helpers
# ---------------------------------------------------------------------------

def _discover_realsense() -> list[dict]:
    """Discover connected Intel RealSense cameras.

    Returns list of dicts with serial, name, intrinsics, etc.
    """
    if not HAS_RS:
        return []

    ctx = rs.context()
    devices = ctx.query_devices()
    results = []

    for dev in devices:
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        fw = dev.get_info(rs.camera_info.firmware_version) if dev.supports(
            rs.camera_info.firmware_version) else 'unknown'

        # Get intrinsics by briefly starting a pipeline
        intrinsics = None
        try:
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
            profile = pipe.start(cfg)
            color_stream = profile.get_stream(
                rs.stream.color).as_video_stream_profile()
            rs_intr = color_stream.get_intrinsics()
            intrinsics = {
                'camera_matrix': [
                    [round(rs_intr.fx, 2), 0, round(rs_intr.ppx, 2)],
                    [0, round(rs_intr.fy, 2), round(rs_intr.ppy, 2)],
                    [0, 0, 1],
                ],
                'dist_coeffs': [round(c, 6) for c in rs_intr.coeffs],
                'image_size': [rs_intr.width, rs_intr.height],
                'source': 'factory',
            }
            pipe.stop()
        except Exception as e:
            print(f"  Warning: could not get intrinsics for {serial}: {e}")

        # Check capabilities
        has_depth = False
        has_imu = False
        for sensor in dev.query_sensors():
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() == rs.stream.depth:
                    has_depth = True
                if profile.stream_type() == rs.stream.accel:
                    has_imu = True

        results.append({
            'serial': serial,
            'name': name,
            'firmware': fw,
            'intrinsics': intrinsics,
            'has_depth': has_depth,
            'has_imu': has_imu,
        })

    return results


# ---------------------------------------------------------------------------
# Logical name assignment
# ---------------------------------------------------------------------------

def _make_camera_name(card: str, dev_path: str, index: int,
                      existing_names: set) -> str:
    """Generate a unique logical name for a camera."""
    # Normalize card name to a slug
    slug = re.sub(r'[^a-z0-9]+', '_', card.lower()).strip('_') if card else 'cam'
    # Shorten common prefixes
    slug = slug.replace('integrated_', '').replace('usb_', '')
    if len(slug) > 30:
        slug = slug[:30]

    # Extract device index
    match = re.search(r'video(\d+)', dev_path)
    dev_idx = match.group(1) if match else str(index)

    name = f"{slug}_{dev_idx}"
    # Ensure uniqueness
    base = name
    counter = 2
    while name in existing_names:
        name = f"{base}_{counter}"
        counter += 1
    return name


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

_IDENTITY_4x4 = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
]


def _default_mount() -> dict:
    """Return default mount info for an unclassified camera."""
    return {
        'type': 'other',
        'parent_link': None,
        'T_cam_to_link': None,
        'notes': '',
    }


def _load_existing_cameras() -> dict:
    """Load existing cameras.yaml if present (from active config dir)."""
    # Check both the write target and the read path (may differ on first run)
    for path in (_CAMERAS_YAML, _CAMERAS_YAML_READ):
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = yaml.safe_load(f)
                return data.get('cameras', {}) or {}
            except Exception:
                continue
    return {}


def _load_calibration_extrinsics() -> dict | None:
    """Load T_camera_to_base from config/calibration.yaml if present."""
    if not os.path.exists(_CALIBRATION_YAML):
        return None
    try:
        with open(_CALIBRATION_YAML, 'r') as f:
            data = yaml.safe_load(f)
        return data.get('T_camera_to_base')
    except Exception:
        return None


def _merge_camera(new_entry: dict, old_entry: dict | None) -> dict:
    """Merge a newly-detected camera with an existing entry.

    Preserves hand-tuned fields (mount, extrinsics) from old_entry while
    updating detection fields (device_path, intrinsics, capabilities, etc.).
    """
    if old_entry is None:
        return new_entry

    result = dict(new_entry)

    # Preserve mount config if it was manually set
    old_mount = old_entry.get('mount', {})
    if old_mount.get('type') and old_mount['type'] != 'other':
        result['mount'] = old_mount

    # Preserve calibrated extrinsics
    old_ext = old_entry.get('extrinsics', {})
    if old_ext.get('source') == 'calibrated':
        result['extrinsics'] = old_ext

    # Preserve calibrated intrinsics over factory/estimated
    old_intr = old_entry.get('intrinsics', {})
    if old_intr.get('source') == 'calibrated':
        result['intrinsics'] = old_intr

    return result


# ---------------------------------------------------------------------------
# Main discovery
# ---------------------------------------------------------------------------

def discover_all() -> dict:
    """Discover all connected cameras and return a cameras dict.

    Returns:
        Dict mapping logical camera names to camera info dicts.
    """
    now = datetime.datetime.now().isoformat(timespec='seconds')
    cameras = {}
    names_used = set()

    # 1. Intel RealSense cameras
    rs_devices = _discover_realsense()
    for rs_dev in rs_devices:
        serial = rs_dev['serial']
        model = rs_dev['name']

        # Determine logical name
        slug = re.sub(r'[^a-z0-9]+', '_', model.lower()).strip('_')
        name = f"{slug}_{serial[-4:]}" if len(serial) >= 4 else slug
        while name in names_used:
            name += '_2'
        names_used.add(name)

        # Load calibration extrinsics if available
        ext_matrix = _load_calibration_extrinsics()
        ext_source = 'calibrated' if ext_matrix else 'none'

        cameras[name] = {
            'device_path': f"realsense:{serial}",
            'serial': serial,
            'driver': model,
            'type': 'realsense',
            'resolution': rs_dev['intrinsics']['image_size'] if rs_dev['intrinsics'] else [640, 480],
            'fps': 15,
            'mount': {
                'type': 'fixed',
                'parent_link': None,
                'T_cam_to_link': None,
                'notes': 'Eye-to-hand fixed mount (default assumption)',
            },
            'intrinsics': rs_dev['intrinsics'] or {
                'camera_matrix': None,
                'dist_coeffs': None,
                'image_size': [640, 480],
                'source': 'unknown',
            },
            'extrinsics': {
                'T_cam_to_base': ext_matrix,
                'source': ext_source,
                'calibration_date': now[:10] if ext_matrix else None,
            },
            'capabilities': {
                'depth': rs_dev['has_depth'],
                'color': True,
                'imu': rs_dev['has_imu'],
            },
            'detected_at': now,
        }
        print(f"  Found RealSense: {model} (serial {serial})")

    # 2. V4L2 webcams (skip RealSense V4L2 nodes)
    rs_serials = {d['serial'] for d in rs_devices}
    v4l_devices = _list_video_devices()

    for idx, dev_path in enumerate(v4l_devices):
        info = _v4l2_device_info(dev_path)
        if info is None:
            continue

        card = info.get('card', '')
        driver = info.get('driver', '')

        # Skip RealSense V4L2 nodes (already handled above)
        if 'realsense' in driver.lower() or 'realsense' in card.lower():
            continue

        # Extract device index from path
        match = re.search(r'video(\d+)', dev_path)
        dev_idx = int(match.group(1)) if match else idx

        name = _make_camera_name(card, dev_path, idx, names_used)
        names_used.add(name)

        # Probe intrinsics
        intrinsics_data = _probe_webcam_intrinsics(dev_idx)
        resolution = intrinsics_data['image_size'] if intrinsics_data else [640, 480]
        fps = intrinsics_data.get('actual_fps', 30) if intrinsics_data else 30

        intrinsics = {
            'camera_matrix': intrinsics_data['camera_matrix'] if intrinsics_data else None,
            'dist_coeffs': intrinsics_data['dist_coeffs'] if intrinsics_data else None,
            'image_size': resolution,
            'source': intrinsics_data['source'] if intrinsics_data else 'unknown',
        }

        cameras[name] = {
            'device_path': dev_path,
            'device_index': dev_idx,
            'serial': info.get('serial', 'unknown'),
            'driver': driver,
            'card': card,
            'type': 'webcam',
            'resolution': resolution,
            'fps': fps,
            'mount': _default_mount(),
            'intrinsics': intrinsics,
            'extrinsics': {
                'T_cam_to_base': None,
                'source': 'none',
                'calibration_date': None,
            },
            'capabilities': {
                'depth': False,
                'color': True,
                'imu': False,
            },
            'detected_at': now,
        }
        print(f"  Found webcam: {card} at {dev_path} (index {dev_idx})")

    return cameras


# ---------------------------------------------------------------------------
# ARM101 gripper camera mount defaults
# ---------------------------------------------------------------------------

# Approximate camera-to-gripper_link transform for the SO-ARM101 32x32 UVC
# wrist camera mount.  The camera optical frame has Z forward (into scene),
# X right, Y down.  When mounted on the gripper looking downward:
#
#   Camera optical Z  ≈  -gripper Z  (camera looks down)
#   Camera optical X  ≈  +gripper X  (same horizontal direction)
#   Camera optical Y  ≈  -gripper Y  (image Y down = robot Y backward)
#
# Translation offset from gripper_link origin to camera lens center,
# expressed in the gripper_link frame (meters):
#   X ≈ +0.010 m  (camera slightly right of center)
#   Y ≈ -0.020 m  (behind the gripper pivot)
#   Z ≈ -0.035 m  (below the gripper flange)
#
# These are rough estimates from the CAD geometry.  For precision, run
# hand-eye calibration (scripts/detect_checkerboard.py).

ARM101_GRIPPER_CAM_MOUNT = {
    'type': 'gripper',
    'parent_link': 'gripper_link',
    'T_cam_to_link': [
        [1.0,  0.0,  0.0, 0.010],
        [0.0, -1.0,  0.0, -0.020],
        [0.0,  0.0, -1.0, -0.035],
        [0.0,  0.0,  0.0, 1.0],
    ],
    'notes': (
        'SO-ARM101 32x32 UVC wrist camera mount (snap-on). '
        'Camera faces downward (-Z in gripper frame). '
        'Translation is approximate from CAD — calibrate for precision.'
    ),
}


OVERVIEW_CAM_MOUNT = {
    'type': 'fixed',
    'parent_link': None,
    'T_cam_to_link': None,
    'notes': 'Overview camera (eye-to-hand, fixed mount)',
}

# Patterns for cameras that should be skipped (built-in laptop cameras)
_BUILTIN_PATTERNS = re.compile(
    r'integrated|built.?in|laptop|notebook|ir.?camera|dell\s+m',
    re.IGNORECASE)

# Patterns for known overview / external cameras (not gripper-mounted)
_OVERVIEW_PATTERNS = re.compile(
    r'logitech|brio|c920|c922|c930|c270|streamcam|razer|elgato',
    re.IGNORECASE)


def _apply_arm101_heuristic(cameras: dict) -> dict:
    """Classify webcams into built-in (skip), overview, and gripper.

    Heuristics:
      - Built-in laptop cameras (Integrated, Dell, IR) → type 'builtin', skipped
      - Known brands (Logitech, Razer, Elgato, ...) → overview (fixed mount)
      - Generic 'USB Camera' → likely the arm101 gripper camera
    """
    for name, cam in cameras.items():
        if cam['type'] != 'webcam':
            continue

        card = cam.get('card', '') or ''

        # Built-in laptop cameras — mark and skip
        if _BUILTIN_PATTERNS.search(card):
            cam['type'] = 'builtin'
            cam['mount'] = _default_mount()
            cam['mount']['notes'] = 'Built-in laptop camera (ignored)'
            print(f"  Skipped '{name}': built-in camera")
            continue

        # Known overview camera brands
        if _OVERVIEW_PATTERNS.search(card):
            cam['mount'] = dict(OVERVIEW_CAM_MOUNT)
            print(f"  Auto-tagged '{name}' as overview camera")
            continue

        # Generic USB camera → likely the gripper camera
        if 'usb camera' in card.lower():
            cam['mount'] = dict(ARM101_GRIPPER_CAM_MOUNT)
            print(f"  Auto-tagged '{name}' as ARM101 gripper camera")
            continue

    return cameras


# ---------------------------------------------------------------------------
# Update robot_config.yaml with discovered device indices
# ---------------------------------------------------------------------------

def _update_robot_config(cameras: dict):
    """Update camera.device_index and gripper_camera.device_index in robot_config.yaml.

    Finds the overview camera (mount.type == 'fixed') and gripper camera
    (mount.type == 'gripper') from the discovery results and patches the
    config file so all scripts use the correct devices.
    """
    cfg_path = config_path('robot_config.yaml')
    if not os.path.exists(cfg_path):
        return

    overview_idx = None
    gripper_idx = None
    for cam in cameras.values():
        mount_type = cam.get('mount', {}).get('type', '')
        idx = cam.get('device_index')
        if idx is None:
            continue
        if mount_type == 'fixed' and overview_idx is None:
            overview_idx = idx
        elif mount_type == 'gripper' and gripper_idx is None:
            gripper_idx = idx

    if overview_idx is None and gripper_idx is None:
        return

    with open(cfg_path, 'r') as f:
        lines = f.readlines()

    changed = False
    in_camera = False
    in_gripper_camera = False

    for i, line in enumerate(lines):
        stripped = line.lstrip()

        # Track which top-level section we're in
        if stripped and not stripped.startswith('#') and not stripped.startswith('-'):
            if line[0] not in (' ', '\t'):
                in_camera = stripped.startswith('camera:')
                in_gripper_camera = stripped.startswith('gripper_camera:')

        if 'device_index:' in stripped and not stripped.startswith('#'):
            # Extract the comment portion if any
            parts = line.split('#', 1)
            indent = len(line) - len(line.lstrip())

            if in_camera and not in_gripper_camera and overview_idx is not None:
                comment = f'  # overview camera (auto-detected)'
                lines[i] = f"{' ' * indent}device_index: {overview_idx}{comment}\n"
                changed = True
            elif in_gripper_camera and gripper_idx is not None:
                comment = f'  # gripper camera (auto-detected)'
                lines[i] = f"{' ' * indent}device_index: {gripper_idx}{comment}\n"
                changed = True

    if changed:
        with open(cfg_path, 'w') as f:
            f.writelines(lines)
        print(f"\nUpdated device indices in {cfg_path}")
        if overview_idx is not None:
            print(f"  camera.device_index: {overview_idx} (overview)")
        if gripper_idx is not None:
            print(f"  gripper_camera.device_index: {gripper_idx} (gripper)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Discover connected cameras and write config/cameras.yaml')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print discovered cameras without writing file')
    parser.add_argument('--merge', action='store_true',
                        help='Merge with existing cameras.yaml (preserve mount/extrinsics)')
    parser.add_argument('--no-heuristic', action='store_true',
                        help='Skip ARM101 gripper camera auto-detection heuristic')
    args = parser.parse_args()

    print("Discovering cameras...")
    cameras = discover_all()

    if not cameras:
        print("No cameras detected.")
        return

    # Apply ARM101 heuristic unless disabled
    if not args.no_heuristic:
        cameras = _apply_arm101_heuristic(cameras)

    # Merge with existing if requested
    if args.merge:
        existing = _load_existing_cameras()
        if existing:
            print(f"  Merging with {len(existing)} existing camera entries...")
            # Match by serial or device_path
            existing_by_serial = {}
            existing_by_path = {}
            for name, cam in existing.items():
                if cam.get('serial') and cam['serial'] != 'unknown':
                    existing_by_serial[cam['serial']] = cam
                if cam.get('device_path'):
                    existing_by_path[cam['device_path']] = cam

            merged = {}
            for name, cam in cameras.items():
                old = None
                if cam.get('serial') and cam['serial'] != 'unknown':
                    old = existing_by_serial.get(cam['serial'])
                if old is None and cam.get('device_path'):
                    old = existing_by_path.get(cam['device_path'])
                merged[name] = _merge_camera(cam, old)
            cameras = merged

    # Build output
    output = {'cameras': cameras}

    if args.dry_run:
        print("\n--- Discovered cameras (dry run) ---")
        print(yaml.dump(output, default_flow_style=False, sort_keys=False))
        return

    # Write
    os.makedirs(os.path.dirname(_CAMERAS_YAML), exist_ok=True)
    with open(_CAMERAS_YAML, 'w') as f:
        f.write('# Camera registry — auto-populated by scripts/discover_cameras.py\n')
        f.write(f'# Last updated: {datetime.datetime.now().isoformat(timespec="seconds")}\n')
        f.write('# Edit mount/extrinsics by hand; re-run with --merge to preserve them.\n\n')
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)

    print(f"\nWrote {len(cameras)} camera(s) to {_CAMERAS_YAML}")
    for name, cam in cameras.items():
        mount_type = cam.get('mount', {}).get('type', 'other')
        intr_src = cam.get('intrinsics', {}).get('source', '?')
        print(f"  {name}: {cam['type']}, mount={mount_type}, intrinsics={intr_src}")

    # Update robot_config.yaml device indices to match discovered cameras
    _update_robot_config(cameras)


if __name__ == '__main__':
    main()
