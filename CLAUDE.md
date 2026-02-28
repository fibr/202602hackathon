# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Rod pick-and-stand system for a robotics hackathon. Uses an Intel RealSense D435i camera and Dobot Nova5 6-axis arm to detect a metal hollow black rod, pick it up, and stand it upright. Pure Python, ROS2 driver runs in Docker for motion commands.

## Setup & Run

```bash
./setup.sh                    # Create venv and install deps
./run.sh src/main.py          # Run full pick-and-stand pipeline (auto-creates venv)
```

### ROS2 Driver (for precise MovJ/MovL motion)

Without the ROS2 driver, motion uses jog pulses (slow). With it, real MovJ/MovL are available.

```bash
cp .env.example .env          # Set DOBOT_IP (default 192.168.5.1)
docker compose --profile dobot up -d   # Build & start ROS2 driver
docker compose logs dobot-driver       # Verify it's running
```

The driver auto-detects motion support on connect and logs the mode.

### Test and debug scripts
```bash
./run.sh scripts/test_robot.py           # Ping, connect, enable, jog wiggle + gripper
./run.sh scripts/control_panel.py        # Interactive keyboard control
./run.sh scripts/debug_robot.py          # Dashboard diagnostics dump
./run.sh scripts/test_camera.py [--hd]   # RealSense stream (press 'd' for detection)
./run.sh scripts/detect_checkerboard.py [--hd]  # Checkerboard calibration
./run.sh scripts/calibrate.py            # Manual hand-eye calibration
```

## Dependencies

Defined in `requirements.txt`: pyrealsense2, opencv-python, numpy, PyYAML. No other build system — just pip in a venv.

## Architecture

Pipeline: **Camera → Vision → Calibration Transform → Planner → Robot Driver**

- `src/main.py` — State machine orchestrator (INIT → DETECT → PLAN → EXECUTE → DONE)
- `src/config_loader.py` — Loads `robot_config.yaml` with `settings.yaml` overrides (deep merge)
- `src/vision/` — RealSense camera wrapper + rod detection via HSV color/depth segmentation (not ML)
- `src/calibration/` — 4×4 homogeneous transforms for camera-to-robot-base frame conversion
- `src/planner/` — Generates ordered waypoints (approach, grasp, lift, reorient, place, release)
- `src/robot/dobot_api.py` — TCP/IP driver with auto-detected motion modes (MovL/MovJ or jog fallback)
- `assets/dobot-driver/` — Docker ROS2 driver (DOBOT_6Axis_ROS2_V4) for enabling MovJ/MovL
- `docker-compose.yml` — `docker compose --profile dobot up` to start ROS2 driver
- `src/robot/gripper.py` — Electric gripper via ToolDOInstant dual-channel control
- `config/robot_config.yaml` — Shared config (robot IP, speeds, camera, detection thresholds)
- `config/settings.yaml` — Local overrides, gitignored (create to override any config value)
- `config/calibration.yaml` — Generated camera-to-base 4×4 matrix
- `docs/architecture.md` — System design, coordinate frames, risk analysis
- `docs/hackathon_api_reference.txt` — Hackathon team's ROS2/Docker/FastAPI reference implementation

## Robot Protocol (Nova5 firmware 4.6.2)

Dashboard port 29999 for state queries, gripper, enable/disable. Motion commands auto-detect the best available mode.

Response format: `code,{value},CommandName();` where code 0 = success.

### Motion modes (auto-detected on connect)
1. **`dashboard`** — `MovJ`/`MovL` work on port 29999 (ROS2 driver running)
2. **`motion_port`** — `MovJ`/`MovL` via port 30003 (ROS2 driver provides this)
3. **`jog`** — fallback to `MoveJog` pulses + `InverseKin` (no ROS2 driver)

### What always works (dashboard port 29999)
- **Joint jog**: `MoveJog(J1+)` through `MoveJog(J6-)` — **uppercase only**, lowercase silently ignored
- **Jog stop**: `MoveJog()`
- **Gripper close**: `ToolDOInstant(2,0)` then `ToolDOInstant(1,1)` — must turn off opposing channel first
- **Gripper open**: `ToolDOInstant(1,0)` then `ToolDOInstant(2,1)`
- **Enable sequence**: `DisableRobot()` → sleep 1s → `ClearError()` → `EnableRobot()` → sleep 1s
- **State queries**: `GetPose()`, `GetAngle()`, `RobotMode()`, `GetErrorID()`
- **Speed**: `SpeedFactor(1-100)`
- **Inverse kinematics**: `InverseKin(x,y,z,rx,ry,rz)` → returns joint angles in `{j1,j2,...,j6}`
- **Forward kinematics**: `PositiveKin(j1,j2,j3,j4,j5,j6)` → returns pose in `{x,y,z,rx,ry,rz}`

### What requires ROS2 driver
- `MovJ(x,y,z,rx,ry,rz)` — joint-space move to Cartesian pose (coordinated motion)
- `MovL(x,y,z,rx,ry,rz)` — linear move to Cartesian pose
- Without ROS2 driver: returns error -30001; driver falls back to IK + jog

### What never works
- Cartesian jog `MoveJog(z+)` → silently ignored; `MoveJog(Z+)` → error -6
- `DO(port, val)` → error -1 (use ToolDOInstant instead)
- `ToolDO(index, status)` → error -1 in idle mode (use ToolDOInstant)
- `ModbusRead` / `ModbusWrite` → unknown command -10000

### Robot modes
`RobotMode()` returns: 1=init, 2=brake_open, 4=disabled, **5=enabled (idle)**, 6=backdrive, 7=running, 9=error, 10=pause, 11=jog

## Code Conventions

- Dataclasses for structured data (`RodDetection`, `Waypoint`, `RobotState`)
- Enums for constants (`MotionType`, `GripperAction`)
- Type hints throughout; docstrings with Args/Returns sections
- YAML config files — no hardcoded magic numbers
- `sys.path` manipulation for local imports (no installed package)
- Context managers for resource cleanup (camera, robot connection)

## Hardware

- **Camera**: RealSense D435i at USB 3.0 (640×480 @ 15fps, fixed mount, eye-to-hand)
- **Robot**: Dobot Nova5 (firmware 4.6.2) at `192.168.5.1`, dashboard port 29999
- **Gripper**: Electric motor, dual-channel ToolDOInstant control (ch1=close, ch2=open)
