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
./run.sh scripts/test_robot.py --diag    # Dashboard diagnostics only (no motion)
./run.sh scripts/control_panel.py        # Camera + GUI control panel (OpenCV)
./run.sh scripts/collect_dataset.py --no-robot   # Live camera feed with detection
./run.sh scripts/collect_dataset.py --snapshot   # Single-frame detection debug (6-stage images)
./run.sh scripts/detect_checkerboard.py          # Interactive calibration (GUI panel + click corners)
./run.sh scripts/detect_checkerboard.py --verify # Verify calibration: hover above board corners
./run.sh scripts/test_ik.py              # Validate local IK against robot's built-in FK/IK
./run.sh scripts/demo_cube.py            # Random reachable poses demo (default)
./run.sh scripts/demo_cube.py --mode cube # Trace cube corners
```

## Logging

All modules log to `logs/YYYYMMDD_HHMMSS.log` (auto-created per session). Console shows INFO+, file captures DEBUG with timestamps. Robot driver logs every command/response at DEBUG level.

```python
from logger import get_logger
log = get_logger(__name__)
log.info("visible on console + file")
log.debug("file only — use for protocol traces")
```

Inspect logs: `tail -f logs/*.log` or `cat logs/$(ls -t logs/ | head -1)`

## Tests

```bash
./run.sh -m pytest tests/                    # Run all tests
./run.sh -m pytest tests/test_ik_solver.py   # IK solver tests (16 tests)
./run.sh -m pytest tests/test_trajectory.py  # Trajectory tests (9 tests)
```

## Dependencies

Defined in `requirements.txt`: pyrealsense2, opencv-python, numpy, PyYAML, pin (Pinocchio), scipy. No other build system — just pip in a venv.

## Architecture

Pipeline: **Camera → Vision → Calibration Transform → Planner → Robot Driver**

- `src/main.py` — State machine orchestrator (INIT → DETECT → PLAN → EXECUTE → DONE)
- `src/config_loader.py` — Loads `robot_config.yaml` with `settings.yaml` overrides (deep merge)
- `src/gui/` — Shared OpenCV GUI panel for robot arm control (XY jog pad, Z, gripper, speed, enable/home, status)
- `src/vision/` — RealSense camera wrapper + rod detection via HSV color/depth segmentation (not ML)
- `src/calibration/` — 4×4 homogeneous transforms for camera-to-robot-base frame conversion
- `src/kinematics/ik_solver.py` — Local IK/FK using Pinocchio + Nova5 URDF (~0.8ms per solve)
- `src/planner/` — Generates ordered waypoints (approach, grasp, lift, reorient, place, release)
- `src/planner/trajectory.py` — Quintic smoothstep trajectory subdivision for smooth joint motion
- `src/robot/dobot_api.py` — TCP/IP driver, dashboard port 29999 only (V4 syntax)
- `assets/nova5_robot.urdf` — Official Nova5 URDF with configurable gripper tool_tip frame
- `src/robot/gripper.py` — Electric gripper via ToolDOInstant dual-channel control
- `config/robot_config.yaml` — Shared config (robot IP, speeds, camera, detection thresholds)
- `config/settings.yaml` — Local overrides, gitignored (create to override any config value)
- `config/calibration.yaml` — Generated camera-to-base 4×4 matrix
- `docs/architecture.md` — System design, coordinate frames, risk analysis
- `docs/hackathon_api_reference.txt` — Hackathon team's ROS2/Docker/FastAPI reference implementation

## Robot Protocol (Nova5 firmware 4.6.2)

All commands go through dashboard port 29999. V4 named-parameter syntax required.

Response format: `code,{value},CommandName();` where code 0 = success.

### V4 motion commands (dashboard port 29999)
- `MovJ(pose={x,y,z,rx,ry,rz})` — joint-space move to Cartesian pose
- `MovL(pose={x,y,z,rx,ry,rz})` — linear move to Cartesian pose
- `MovJ(joint={j1,j2,j3,j4,j5,j6})` — joint-space move to joint angles
- Motion is fire-and-forget; completion detected by polling joint stability
- **V3 syntax** `MovL(x,y,z,...)` without `pose={}` returns **-30001** — do NOT use

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
- V3 syntax `MovL(x,y,z,rx,ry,rz)` → error **-30001** (must use V4 `pose={...}` syntax)
- Cartesian jog `MoveJog(z+)` → silently ignored; `MoveJog(Z+)` → error -6
- `DO(port, val)` → error -1 (use ToolDOInstant instead)
- `ToolDO(index, status)` → error -1 in idle mode (use ToolDOInstant)
- `ModbusRead` / `ModbusWrite` → unknown command -10000

### Robot modes
`RobotMode()` returns: 1=init, 2=brake_open, 4=disabled, **5=enabled (idle)**, 6=backdrive, 7=running, 9=error, 10=pause, 11=jog

## Local Kinematics & Trajectory

Cartesian motion commands (`MovJ(pose={...})`) are unreliable and frequently error out. Instead, all motion uses **local IK + joint-angle commands**:

1. **IK Solver** (`src/kinematics/ik_solver.py`): Uses Pinocchio to load `assets/nova5_robot.urdf` and solve IK via damped least-squares (~0.8ms, 1200 Hz). Configurable tool_tip offset for gripper length (`gripper.tool_length_mm` in config).
2. **Joint unwrapping**: For joints with ±360° range (J1, J4, J5, J6), `_unwrap_to_seed()` picks the solution closest to the seed to prevent full-revolution configuration flips.
3. **Trajectory subdivision** (`src/planner/trajectory.py`): Large joint moves are subdivided into small steps (default 5°/step) using quintic smoothstep `s(t) = 10t³ - 15t⁴ + 6t⁵` — zero velocity and acceleration at endpoints.
4. **Execution**: Each step is sent as `MovJ(joint={...})` with retry logic (clear error + re-enable on failure).
5. **Linear moves**: Cartesian path is interpolated into small steps (default 5mm), each solved by IK, then executed as a sequence of small joint moves.

All public interfaces use mm and degrees. Internal math uses meters and radians.

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
- **Robot**: Dobot Nova5 (firmware 4.6.2) at `192.168.5.1`, dashboard port 29999 (V4 syntax)
- **No teach pendant/controller** — all control via dashboard TCP only
- **Gripper**: Electric motor, dual-channel ToolDOInstant control (ch1=close, ch2=open)
- Dashboard sends **no banner** on connect — code must handle `recv` timeout after connect
- Dashboard allows **only one TCP connection** at a time — "IP:Port has been occupied" if two clients try
- After power cycle, robot stays in mode 3 (init) for ~60s before reaching mode 4 (disabled)
