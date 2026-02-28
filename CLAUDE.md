# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Rod pick-and-stand system for a robotics hackathon. Uses an Intel RealSense D435i camera and Dobot Nova5 6-axis arm to detect a metal hollow black rod, pick it up, and stand it upright. Pure Python, no ROS.

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
- `src/robot/dobot_api.py` — TCP/IP driver using V4 command syntax (MovL/MovJ with jog fallback)
- `src/robot/gripper.py` — Electric gripper via ToolDOInstant dual-channel control
- `config/robot_config.yaml` — Shared config (robot IP, speeds, camera, detection thresholds)
- `config/settings.yaml` — Local overrides, gitignored (create to override any config value)
- `config/calibration.yaml` — Generated camera-to-base 4×4 matrix
- `docs/architecture.md` — System design, coordinate frames, risk analysis
- `docs/hackathon_api_reference.txt` — Hackathon team's ROS2/Docker/FastAPI reference implementation

## Robot Protocol (Nova5 firmware 4.6.2)

All commands go through **dashboard port 29999** using V4 named-parameter syntax. No ROS2 driver or extra ports needed.

Response format: `code,{value},CommandName();` where code 0 = success.

### V4 motion commands (probed on enable)
- `MovJ(pose={x,y,z,rx,ry,rz})` — joint-space move to Cartesian pose
- `MovL(pose={x,y,z,rx,ry,rz})` — linear move to Cartesian pose
- `MovJ(joint={j1,j2,j3,j4,j5,j6})` — joint-space move to joint angles
- **Important**: V3 syntax `MovJ(x,y,z,rx,ry,rz)` returns -30001 — must use V4 named params with braces
- Driver auto-probes on `enable()`, falls back to IK + jog if V4 commands fail
- Motion is fire-and-forget; completion detected by polling joint stability

### Other dashboard commands
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
