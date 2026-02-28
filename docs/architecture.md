# Rod Pick-and-Stand System Architecture

## Project Goal

Build a proof-of-concept system where a fixed Intel RealSense D435i camera
detects a metal hollow black rod lying on its side, and a Dobot Nova5 robot arm
with a gripper picks it up and stands it upright.

---

## 1. System Overview

```
┌─────────────┐   USB    ┌──────────────────────────────────────────────┐
│  RealSense  │─────────>│              Host PC (Python)               │
│   D435i     │          │                                              │
└─────────────┘          │  ┌────────────┐  ┌────────────┐  ┌────────┐ │   TCP/IP    ┌───────────┐
                         │  │  Vision    │─>│ Planner /  │─>│ Robot  │─│────────────>│ Dobot     │
                         │  │  Module    │  │ Orchestr.  │  │ Driver │ │ 192.168.5.1 │ Nova5     │
                         │  └────────────┘  └────────────┘  └────────┘ │             │ + Gripper │
                         └──────────────────────────────────────────────┘             └───────────┘
```

### Data Flow

1. **Camera** streams aligned RGB + depth frames via `pyrealsense2`
2. **Vision module** detects the rod, computes its 3D pose (position + orientation) in camera frame
3. **Calibration** transforms camera-frame coordinates to robot base-frame coordinates
4. **Planner** computes a grasp strategy and a sequence of waypoints
5. **Robot driver** sends motion commands to the Dobot Nova5 over TCP/IP
6. **Gripper** closes, lifts, reorients, places, and opens

---

## 2. Hardware

| Component | Model | Interface | Notes |
|-----------|-------|-----------|-------|
| Camera | Intel RealSense D435i | USB 3.0 | Fixed mount, known extrinsics after calibration |
| Robot Arm | Dobot Nova5 (6-axis) | Ethernet TCP/IP | IP: `192.168.5.1`, 5 kg payload, 850 mm reach |
| Gripper | TBD (attached to Nova5) | Digital I/O via Nova5 | Parallel jaw or pneumatic, controlled via DO ports |
| Host PC | Linux x86_64 | USB + Ethernet | Runs all software |

### Network Configuration

- Host PC Ethernet: `192.168.5.X` (same subnet as robot)
- Robot IP: `192.168.5.1`
- Dobot TCP ports: `29999` (dashboard/control), `30003` (motion), `30004` (feedback)

---

## 3. Software Components

### 3.1 Vision Module (`vision/`)

**Purpose:** Detect the rod and estimate its 6-DOF pose in camera coordinates.

**Approach (simplest first, escalate if needed):**

1. **Color + Depth Segmentation (primary approach)**
   - The rod is black metal on a presumably lighter surface
   - HSV thresholding in low-value range to isolate dark objects
   - Depth-based filtering to isolate the table plane and objects above it
   - Contour analysis to find elongated shapes (aspect ratio >> 1)
   - Use depth map to get 3D center point and orientation axis

2. **Fallback: YOLO/ML-based detection**
   - Fine-tune a small YOLO model if segmentation proves unreliable
   - Only needed if lighting/background makes simple segmentation fail

**Key outputs:**
- Rod center position in camera frame (x, y, z) in meters
- Rod orientation vector (axis of the cylinder) in camera frame
- Confidence score

**Dependencies:** `pyrealsense2`, `opencv-python`, `numpy`

### 3.2 Hand-Eye Calibration (`calibration/`)

**Purpose:** Map camera coordinates to robot base coordinates.

**Approach:** Since the camera is fixed (eye-to-hand configuration):

1. Use a known calibration target (e.g., ArUco marker or the gripper tip itself)
2. Move robot to N known positions, record robot pose + camera observation
3. Solve `T_base_to_camera` using `cv2.calibrateHandEye()`
4. Store the 4x4 transform matrix to disk (JSON/YAML)

**Simplified alternative for PoC:**
- Manually measure camera position relative to robot base
- Define transform matrix from physical measurements
- Adequate accuracy for a PoC with a fixed camera and constrained workspace

**Key outputs:**
- `T_camera_to_base`: 4x4 homogeneous transformation matrix

**Dependencies:** `opencv-python`, `numpy`

### 3.3 Grasp Planner (`planner/`)

**Purpose:** Given the rod's 3D pose in robot-base frame, compute a grasp strategy.

**Logic:**

```
Rod is lying on its side:
  1. Approach from above, offset by safety margin (e.g., +100mm Z)
  2. Descend to grasp height (rod center Z + gripper offset)
  3. Orient gripper perpendicular to rod axis for center grasp
  4. Close gripper
  5. Lift straight up to safe height
  6. Rotate wrist to make rod vertical (90-degree reorientation)
  7. Move to placement position
  8. Lower to surface
  9. Open gripper
  10. Retract upward
```

**Waypoint types:**
- `SAFE_HOME`: Starting/resting position above workspace
- `PRE_GRASP`: Above the rod, gripper oriented for grasp
- `GRASP`: At rod height, gripper open
- `LIFT`: Rod grasped, lifted to safe height
- `REORIENT`: Wrist rotated so rod is vertical
- `PLACE`: Above placement location
- `RELEASE`: Rod standing, gripper opens

**Key outputs:**
- Ordered list of (x, y, z, rx, ry, rz) waypoints with motion type (MovJ/MovL)
- Gripper commands interleaved at appropriate steps

**Dependencies:** `numpy`

### 3.4 Robot Driver (`robot/`)

**Purpose:** Communicate with the Dobot Nova5 via TCP/IP protocol.

**Based on:** [Dobot TCP-IP-Python-V4 SDK](https://github.com/Dobot-Arm/TCP-IP-Python-V4)

**Key operations:**
- `connect()` - Establish TCP connections to ports 29999, 30003, 30004
- `enable()` - Power on and enable the robot
- `move_joint(j1..j6)` - Joint-space move (MovJ)
- `move_linear(x, y, z, rx, ry, rz)` - Cartesian linear move (MovL)
- `set_digital_output(port, value)` - Control gripper via DO
- `get_pose()` - Read current TCP position
- `get_joints()` - Read current joint angles
- `clear_error()` - Clear alarm state
- `disable()` - Safely disable robot

**Protocol details:**
- Port 29999: Dashboard commands (EnableRobot, ClearError, SetDO, etc.)
- Port 30003: Motion commands (MovJ, MovL, ServoJ, etc.)
- Port 30004: Real-time feedback (joint angles, TCP pose, status flags)

**Dependencies:** `numpy`, standard library `socket`

### 3.5 Orchestrator (`main.py`)

**Purpose:** Top-level state machine that ties all modules together.

**State machine:**

```
INIT -> DETECT -> PLAN -> APPROACH -> GRASP -> LIFT -> REORIENT -> PLACE -> RELEASE -> DONE
  |                 |                                                                    |
  |                 v                                                                    |
  |            DETECT_FAILED ──> RETRY (up to N times) ──> ABORT                        |
  |                                                                                      |
  └──────────────────────────<───────────────────────────────────────────────────────────┘
                                          (loop for next rod)
```

**States:**
| State | Action |
|-------|--------|
| `INIT` | Connect to robot, initialize camera, load calibration |
| `DETECT` | Capture frame, run vision pipeline, get rod pose |
| `PLAN` | Transform to robot frame, compute waypoints |
| `APPROACH` | Move to pre-grasp position |
| `GRASP` | Descend, close gripper |
| `LIFT` | Lift rod to safe height |
| `REORIENT` | Rotate wrist to stand rod vertical |
| `PLACE` | Move to placement location, lower |
| `RELEASE` | Open gripper, retract |
| `DONE` | Return to home, report success |

---

## 4. Coordinate Frames

```
Robot Base Frame (world frame for our system)
  Origin: Robot base center
  Z: Up
  X: Forward (toward workspace)
  Y: Left

Camera Frame
  Origin: Camera optical center
  Z: Forward (into scene)
  X: Right
  Y: Down

Transform: T_camera_to_base (from calibration)
  P_base = T_camera_to_base @ P_camera
```

---

## 5. Key Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Black rod hard to detect on dark surface | Cannot locate rod | Use depth segmentation (geometry, not color); add contrasting surface mat |
| Calibration inaccuracy | Gripper misses rod | Start with coarse calibration, add visual servoing for fine approach |
| Rod rolls during approach | Grasp fails | Approach from above (not side); use slow linear descent |
| Gripper can't hold hollow rod | Rod slips | Test grip force; use rubber pads; consider vacuum gripper |
| TCP/IP command latency | Jerky motion | Use MovL for smooth linear paths; tune acceleration parameters |
| Camera USB bandwidth | Dropped frames | Use 640x480 resolution; lower framerate (15fps sufficient) |

---

## 6. Development Phases

### Phase 1: Foundation (Day 1 morning)
- [ ] Set up Python environment and dependencies
- [ ] Verify RealSense camera streams (RGB + depth)
- [ ] Verify Dobot Nova5 TCP/IP connection and basic moves
- [ ] Test gripper open/close via digital output

### Phase 2: Perception (Day 1 afternoon)
- [ ] Implement rod detection via depth + color segmentation
- [ ] Compute 3D position and orientation from depth map
- [ ] Perform hand-eye calibration (manual measurement or automated)
- [ ] Validate camera-to-robot coordinate transform

### Phase 3: Motion Planning (Day 2 morning)
- [ ] Define safe home position and workspace bounds
- [ ] Implement grasp waypoint generation
- [ ] Implement reorientation (lying -> standing) motion
- [ ] Test full waypoint sequence with robot (no rod)

### Phase 4: Integration (Day 2 afternoon)
- [ ] Connect vision -> planner -> robot pipeline
- [ ] End-to-end test with actual rod
- [ ] Tune parameters (gripper force, approach speed, detection thresholds)
- [ ] Add error handling and recovery

### Phase 5: Demo Polish (if time permits)
- [ ] Add visualization overlay (show detection on RGB stream)
- [ ] Add simple UI/CLI for triggering pick-and-stand
- [ ] Record demo video

---

## 7. Dependencies

```
pyrealsense2>=2.50      # RealSense camera SDK
opencv-python>=4.8      # Computer vision
numpy>=1.24             # Numerical computation
```

No ROS required. No Docker required. Pure Python with TCP/IP sockets.

---

## 8. File Structure

```
202602hackathon_1/
├── docs/
│   └── architecture.md          # This document
├── src/
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── camera.py            # RealSense camera wrapper
│   │   └── rod_detector.py      # Rod detection and pose estimation
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── calibrate.py         # Hand-eye calibration routine
│   │   └── transform.py         # Coordinate frame transforms
│   ├── robot/
│   │   ├── __init__.py
│   │   ├── dobot_api.py         # TCP/IP communication with Nova5
│   │   └── gripper.py           # Gripper control abstraction
│   ├── planner/
│   │   ├── __init__.py
│   │   └── grasp_planner.py     # Waypoint generation for pick-and-stand
│   └── main.py                  # Orchestrator / state machine
├── config/
│   ├── robot_config.yaml        # Robot IP, ports, joint limits, speeds
│   └── calibration.yaml         # Camera-to-robot transform (generated)
├── scripts/
│   ├── test_camera.py           # Verify camera stream
│   ├── test_robot.py            # Verify robot connection and basic moves
│   └── calibrate.py             # Run calibration procedure
├── requirements.txt
└── README.md
```

---

## 9. Reference Links

- [Dobot TCP-IP-Python-V4 SDK](https://github.com/Dobot-Arm/TCP-IP-Python-V4) - Official 6-axis Python control
- [Dobot TCP-IP-Python-V3 SDK](https://github.com/Dobot-Arm/TCP-IP-Python-V3) - Alternative (CR/Nova V3.5.5+)
- [pyrealsense2 PyPI](https://pypi.org/project/pyrealsense2/) - Camera SDK
- [Hackathon Forgis Dobot URR](https://github.com/ForgisX/Hackathon_Forgis_Dobot_URR) - Reference architecture (Docker/REST/ROS2 approach)
- [YOLO + RealSense D435 3D detection](https://github.com/Mazhichaoruya/Object-Detection-and-location-RealsenseD435) - ML-based detection reference
