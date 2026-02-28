# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Rod pick-and-stand system for a robotics hackathon. Uses an Intel RealSense D435i camera and Dobot Nova5 6-axis arm to detect a metal hollow black rod, pick it up, and stand it upright. Pure Python, no ROS.

## Setup & Run

```bash
./setup.sh                    # Create venv and install deps
source .venv/bin/activate     # Activate venv
python src/main.py            # Run full pick-and-stand pipeline
```

Hardware test scripts:
```bash
python scripts/test_camera.py   # Verify RealSense stream (press 'd' for detection overlay)
python scripts/test_robot.py    # Test Nova5 connection, read pose, test gripper
python scripts/calibrate.py     # Run hand-eye calibration
```

## Dependencies

Defined in `requirements.txt`: pyrealsense2, opencv-python, numpy, PyYAML. No other build system — just pip in a venv.

## Architecture

Pipeline: **Camera → Vision → Calibration Transform → Planner → Robot Driver**

- `src/main.py` — State machine orchestrator (INIT → DETECT → PLAN → EXECUTE → DONE)
- `src/vision/` — RealSense camera wrapper + rod detection via HSV color/depth segmentation (not ML)
- `src/calibration/` — 4×4 homogeneous transforms for camera-to-robot-base frame conversion
- `src/planner/` — Generates ordered waypoints (approach, grasp, lift, reorient, place, release)
- `src/robot/` — TCP/IP driver for Nova5 (3 concurrent sockets: dashboard/motion/feedback) + gripper digital I/O
- `config/robot_config.yaml` — All runtime parameters (robot IP, speeds, camera resolution, detection thresholds)
- `config/calibration.yaml` — Generated camera-to-base 4×4 matrix
- `docs/architecture.md` — Detailed system design, coordinate frames, risk analysis

## Code Conventions

- Dataclasses for structured data (`RodDetection`, `Waypoint`, `RobotState`)
- Enums for constants (`MotionType`, `GripperAction`)
- Type hints throughout; docstrings with Args/Returns sections
- YAML config files — no hardcoded magic numbers
- `sys.path` manipulation for local imports (no installed package)
- Context managers for resource cleanup (camera pipeline)
- Background thread in `dobot_api.py` reads robot feedback continuously

## Hardware

- **Camera**: RealSense D435i at USB 3.0 (640×480 @ 15fps, fixed mount, eye-to-hand)
- **Robot**: Dobot Nova5 at `192.168.5.1` (default), ports 29999/30003/30004
- **Gripper**: Digital output port 1 on Nova5
