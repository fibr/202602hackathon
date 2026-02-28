# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Rod pick-and-stand system for a robotics hackathon. Uses an Intel RealSense D435i camera and Dobot Nova5 6-axis arm to detect a metal hollow black rod, pick it up, and stand it upright. Pure Python, ROS2 driver in Docker for motion commands.

## Setup & Run

```bash
./setup.sh                    # Create venv and install deps
./run.sh src/main.py          # Run full pick-and-stand pipeline (auto-creates venv)
```

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
- `src/robot/dobot_api.py` — Dual-port TCP/IP driver (dashboard 29999 + motion 30003)
- `src/robot/gripper.py` — Electric gripper via ToolDOInstant dual-channel control
- `config/robot_config.yaml` — Shared config (robot IP, speeds, camera, detection thresholds)
- `config/settings.yaml` — Local overrides, gitignored (create to override any config value)
- `config/calibration.yaml` — Generated camera-to-base 4×4 matrix
- `docs/architecture.md` — System design, coordinate frames, risk analysis
- `docs/hackathon_api_reference.txt` — Hackathon team's ROS2/Docker/FastAPI reference implementation

## Robot Protocol (Nova5 firmware 4.6.2)

Dual-port architecture. The ROS2 driver must be running for motion commands.

```bash
docker compose --profile dobot up -d   # Start ROS2 driver (required for MovJ/MovL)
```

Response format: `code,{value},CommandName();` where code 0 = success.

### Port 30003 — motion commands (requires ROS2 driver)
- `MovJ(x,y,z,rx,ry,rz)` — joint-space move to Cartesian pose
- `MovL(x,y,z,rx,ry,rz)` — linear move to Cartesian pose
- Motion is fire-and-forget; completion detected by polling joint stability via dashboard
- **Error -7**: returned if MovJ/MovL sent to dashboard port 29999 instead of motion port 30003
- V4 `pose={...}` syntax also returns -7 on dashboard — it's a port issue, not syntax

### Port 29999 — dashboard commands
- **Joint jog**: `MoveJog(J1+)` through `MoveJog(J6-)` — **uppercase only**, lowercase silently ignored
- **Jog stop**: `MoveJog()`
- **Gripper close**: `ToolDOInstant(2,0)` then `ToolDOInstant(1,1)` — must turn off opposing channel first
- **Gripper open**: `ToolDOInstant(1,0)` then `ToolDOInstant(2,1)`
- **Enable sequence**: `DisableRobot()` → sleep 1s → `ClearError()` → `EnableRobot()` → sleep 1s
- **State queries**: `GetPose()`, `GetAngle()`, `RobotMode()`, `GetErrorID()`
- **Speed**: `SpeedFactor(1-100)`
- **Inverse kinematics**: `InverseKin(x,y,z,rx,ry,rz)` → returns joint angles
- **Forward kinematics**: `PositiveKin(j1,j2,j3,j4,j5,j6)` → returns pose

### What doesn't work
- `MovJ`/`MovL` on port 29999 → error **-7** (must use port 30003)
- Cartesian jog `MoveJog(z+)` → silently ignored; `MoveJog(Z+)` → error -6
- `DO(port, val)` → error -1 (use ToolDOInstant instead)
- `ToolDO(index, status)` → error -1 in idle mode (use ToolDOInstant)
- `ModbusRead` / `ModbusWrite` → unknown command -10000

### Robot modes
`RobotMode()` returns: 1=init, 2=brake_open, 4=disabled, **5=enabled (idle)**, 6=backdrive, 7=running, 9=error, 10=pause, 11=jog

## Git

- Never add `Co-Authored-By` lines to commit messages

## Code Conventions

- Dataclasses for structured data (`RodDetection`, `Waypoint`, `RobotState`)
- Enums for constants (`MotionType`, `GripperAction`)
- Type hints throughout; docstrings with Args/Returns sections
- YAML config files — no hardcoded magic numbers
- `sys.path` manipulation for local imports (no installed package)
- Context managers for resource cleanup (camera, robot connection)

## Hardware

- **Camera**: RealSense D435i at USB 3.0 (640×480 @ 15fps, fixed mount, eye-to-hand)
- **Robot**: Dobot Nova5 (firmware 4.6.2) at `192.168.5.1`, dashboard 29999 + motion 30003
- **Gripper**: Electric motor, dual-channel ToolDOInstant control (ch1=close, ch2=open)
