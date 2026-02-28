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
4. **Planner** computes a grasp strategy and a sequence of Cartesian waypoints
5. **Local IK** converts each waypoint to joint angles using Pinocchio + Nova5 URDF
6. **Trajectory** subdivides large joint moves into smooth quintic-interpolated steps
7. **Robot driver** sends joint-angle commands (`MovJ(joint={...})`) to the Nova5
8. **Gripper** closes, lifts, reorients, places, and opens

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

### 3.3 Local Kinematics (`kinematics/`)

**Purpose:** Compute inverse/forward kinematics locally to avoid unreliable Cartesian motion commands.

**Why:** `MovJ(pose={...})` and `MovL(pose={...})` frequently put the robot into error state. Individual joint-angle commands (`MovJ(joint={...})`) work reliably. Local IK converts Cartesian targets to joint angles before sending.

**Implementation:**
- Loads the official Nova5 URDF (`assets/nova5_robot.urdf`) via **Pinocchio** (`pin` package)
- URDF includes a configurable `tool_tip` frame for gripper offset
- Damped least-squares (Levenberg-Marquardt) iterative IK solver, ~0.8ms per solve (1200 Hz)
- Joint unwrapping (`_unwrap_to_seed`) prevents full-revolution flips on joints with ±360° limits
- Linear Cartesian interpolation: subdivides a straight-line path into small steps, solves IK at each

**Key methods:**
- `solve_ik(pos_mm, rpy_deg, seed_joints_deg)` → joint angles in degrees (or None)
- `forward_kin(joints_deg)` → (pos_mm, rpy_deg)
- `interpolate_linear(start_pos, start_rpy, end_pos, end_rpy, ...)` → list of joint configs

**Dependencies:** `pin` (Pinocchio), `numpy`

### 3.4 Trajectory Generation (`planner/trajectory.py`)

**Purpose:** Subdivide large joint moves into smooth, small steps to prevent motion errors.

**Approach:** Quintic smoothstep polynomial `s(t) = 10t³ - 15t⁴ + 6t⁵`:
- Zero velocity at start and end (smooth ramp-up/ramp-down)
- Zero acceleration at start and end (no jerk at endpoints)
- Configurable `max_step_deg` (default 5°) — larger moves get more steps

**Execution:** Each step is sent as `MovJ(joint={...})` with retry logic:
- On failure: clear error, re-enable robot, wait 500ms, retry (up to 2 retries per step)
- Logs progress at DEBUG level

**Key functions:**
- `quintic_trajectory(q_start, q_goal, max_step_deg)` → list of joint configs
- `execute_trajectory(robot, q_start, q_goal, ...)` → True/False

**Dependencies:** `numpy`

### 3.5 Grasp Planner (`planner/grasp_planner.py`)

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

### 3.6 Robot Driver (`robot/`)

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

### 3.7 Orchestrator (`main.py`)

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
PyYAML>=6.0             # Configuration files
pin>=3.0                # Pinocchio — URDF-based kinematics
scipy>=1.10             # Scientific computing
```

No ROS required. No Docker required. Pure Python with TCP/IP sockets.

---

## 8. File Structure

```
202602hackathon_4/
├── assets/
│   └── nova5_robot.urdf         # Official Nova5 URDF + gripper tool_tip frame
├── docs/
│   └── architecture.md          # This document
├── src/
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── camera.py            # RealSense camera wrapper
│   │   └── rod_detector.py      # Rod detection and pose estimation
│   ├── calibration/
│   │   ├── __init__.py
│   │   └── transform.py         # Coordinate frame transforms
│   ├── kinematics/
│   │   ├── __init__.py
│   │   └── ik_solver.py         # Local IK/FK via Pinocchio + URDF
│   ├── robot/
│   │   ├── __init__.py
│   │   ├── dobot_api.py         # TCP/IP communication with Nova5
│   │   └── gripper.py           # Gripper control abstraction
│   ├── planner/
│   │   ├── __init__.py
│   │   ├── grasp_planner.py     # Waypoint generation for pick-and-stand
│   │   └── trajectory.py        # Quintic smoothstep trajectory subdivision
│   ├── gui/                     # Shared OpenCV GUI panel
│   ├── logger.py                # Session logging (console + file)
│   ├── config_loader.py         # YAML config with local overrides
│   └── main.py                  # Orchestrator / state machine
├── config/
│   ├── robot_config.yaml        # Robot IP, speeds, gripper, kinematics config
│   ├── settings.yaml            # Local overrides (gitignored)
│   └── calibration.yaml         # Camera-to-robot transform (generated)
├── scripts/
│   ├── test_robot.py            # Robot connection and basic moves
│   ├── test_ik.py               # Validate local IK against robot FK/IK
│   ├── control_panel.py         # Interactive camera + robot GUI control
│   ├── demo_cube.py             # Random poses / cube corner demo
│   ├── detect_checkerboard.py   # Calibration via checkerboard
│   └── collect_dataset.py       # Detection debugging
├── tests/
│   ├── test_ik_solver.py        # IK solver unit tests (16 tests)
│   └── test_trajectory.py       # Trajectory unit tests (9 tests)
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
