#!/usr/bin/env python3
"""
Digital twin of SO-ARM101 + cubes in Isaac Sim / Isaac Lab.

Loads the arm101 URDF into an Isaac Lab interactive scene with:
  - Ground plane + table surface
  - Colored cubes as manipulation targets
  - Overhead camera (fixed, looking down at workspace)
  - Wrist camera (attached to gripper link)
  - Optional: reads real arm joint angles and mirrors them in sim

Usage (via Isaac Lab launcher):
    cd ~/src/IsaacLab
    ./isaaclab.sh -p ~/src/202602hackathon/scripts/digital_twin.py
    ./isaaclab.sh -p ~/src/202602hackathon/scripts/digital_twin.py --headless  # no GUI
    ./isaaclab.sh -p ~/src/202602hackathon/scripts/digital_twin.py --enable_cameras  # render cameras
    ./isaaclab.sh -p ~/src/202602hackathon/scripts/digital_twin.py --enable_cameras --save_images  # save to disk
    ./isaaclab.sh -p ~/src/202602hackathon/scripts/digital_twin.py --mirror  # mirror real arm (requires HW)

Or via the convenience script:
    ./scripts/run_digital_twin.sh
    ./scripts/run_digital_twin.sh --headless
    ./scripts/run_digital_twin.sh --enable_cameras

Note: Camera rendering (--enable_cameras) requires:
    - NVIDIA driver >= 535.161 (for RTX renderer)
    - The scene works without cameras even on older drivers
"""

import argparse
import os
import sys

# ── Isaac Lab bootstrap (must happen before other isaaclab imports) ─────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SO-ARM101 digital twin in Isaac Sim")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--save_images", action="store_true", help="Save camera images to disk")
parser.add_argument("--mirror", action="store_true", help="Mirror real arm joint angles (requires HW)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Now safe to import Isaac Lab / torch / etc ─────────────────────────────
import math
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_PATH = os.path.join(REPO_ROOT, "assets", "so101", "so101_new_calib.urdf")
IMAGE_DIR = os.path.join(REPO_ROOT, "logs", "digital_twin_images")

# ── Robot configuration ────────────────────────────────────────────────────
# The SO-ARM101 has 5 DOF + gripper. Joint names from URDF:
#   shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
ARM101_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=URDF_PATH,
        fix_base=True,
        merge_fixed_joints=False,
        force_usd_conversion=True,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            target_type="position",
            drive_type="force",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                stiffness=40.0,
                damping=5.0,
            ),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Arm sits on the table at z=0.42m (table height)
        pos=(0.0, 0.0, 0.42),
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        },
    ),
    actuators={
        "arm_joints": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=40.0,
            damping=5.0,
        ),
        "gripper_joint": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=40.0,
            damping=5.0,
        ),
    },
)

# ── Cube configurations ───────────────────────────────────────────────────
CUBE_SIZE = 0.025  # 25mm cubes
TABLE_HEIGHT = 0.42


def make_cube_cfg(color: tuple[float, float, float], pos: tuple[float, float, float]) -> RigidObjectCfg:
    """Create a colored cube rigid object config."""
    return RigidObjectCfg(
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),  # 20g
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


# ── Scene configuration ───────────────────────────────────────────────────
@configclass
class Arm101SceneCfg(InteractiveSceneCfg):
    """SO-ARM101 digital twin scene: arm on table with cubes (no cameras)."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Dome light for ambient illumination
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.95)),
    )

    # Distant light for shadows
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=800.0,
            color=(1.0, 0.98, 0.9),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.8660, 0.0, 0.5, 0.0),  # 60° angle
        ),
    )

    # Table surface (flat box)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.6, TABLE_HEIGHT),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.35, 0.2),  # Wood-brown
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, TABLE_HEIGHT / 2),  # Half-height so top is at TABLE_HEIGHT
        ),
    )

    # Robot arm
    robot = ARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Colored cubes on the table
    cube_green = make_cube_cfg(
        color=(0.1, 0.8, 0.1),
        pos=(0.15, 0.05, TABLE_HEIGHT + CUBE_SIZE / 2 + 0.001),
    ).replace(prim_path="{ENV_REGEX_NS}/CubeGreen")

    cube_red = make_cube_cfg(
        color=(0.9, 0.1, 0.1),
        pos=(0.15, -0.05, TABLE_HEIGHT + CUBE_SIZE / 2 + 0.001),
    ).replace(prim_path="{ENV_REGEX_NS}/CubeRed")

    cube_blue = make_cube_cfg(
        color=(0.1, 0.2, 0.9),
        pos=(0.10, 0.0, TABLE_HEIGHT + CUBE_SIZE / 2 + 0.001),
    ).replace(prim_path="{ENV_REGEX_NS}/CubeBlue")


@configclass
class Arm101SceneWithCamerasCfg(Arm101SceneCfg):
    """SO-ARM101 scene with simulated cameras (requires --enable_cameras)."""

    # Overhead camera: fixed above the workspace, looking down
    overhead_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/OverheadCamera",
        update_period=0.1,  # 10 Hz
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            # Above workspace, looking down
            pos=(0.0, 0.0, 0.85),
            # Camera looks -Z in ROS convention; we want to look straight down
            # Rotation: 180° around X axis so camera Z points down
            rot=(0.0, 1.0, 0.0, 0.0),  # w,x,y,z quaternion
            convention="ros",
        ),
    )

    # Wrist camera: attached to the gripper link, looking forward/down
    wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_link/wrist_camera",
        update_period=0.1,  # 10 Hz
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.67,  # ~60° HFOV for small sensor
            focus_distance=200.0,
            horizontal_aperture=6.4,  # Gives ~60° HFOV with 3.67mm focal length
            clipping_range=(0.01, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            # Match real wrist camera mount: 10mm X, -20mm Y, -35mm Z from gripper
            pos=(0.01, -0.02, -0.035),
            # Camera faces down (-Z in gripper frame)
            rot=(0.0, 1.0, 0.0, 0.0),  # 180° around X: camera Z → -gripper Z
            convention="ros",
        ),
    )


# ── Simulation loop ───────────────────────────────────────────────────────
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Main simulation loop."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Optional: connect to real arm for mirroring
    real_arm = None
    joint_signs = None
    if args_cli.mirror:
        # ── Check runtime dependencies before attempting connection ────
        _missing_deps = []
        try:
            import serial  # noqa: F401  (pyserial)
        except ImportError:
            _missing_deps.append("pyserial")
        try:
            import scservo_sdk  # noqa: F401  (feetech-servo-sdk)
        except ImportError:
            _missing_deps.append("feetech-servo-sdk")

        if _missing_deps:
            print(
                f"[ERROR]: --mirror requires packages not found in this Python: "
                f"{', '.join(_missing_deps)}", flush=True,
            )
            print(
                "[ERROR]: Install them with:  "
                "<isaaclab_python> -m pip install feetech-servo-sdk pyserial",
                flush=True,
            )
            print(
                "[ERROR]: Or run ./setup.sh which installs them automatically.",
                flush=True,
            )
            print("[WARN]: Falling back to demo mode (no mirroring).", flush=True)
        else:
            try:
                sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
                print("[INFO]: Importing arm101 driver...", flush=True)
                from robot.lerobot_arm101 import LeRobotArm101

                # Load joint signs + URDF offsets from servo_offsets.yaml
                # (avoids importing Pinocchio which conflicts with Isaac Sim's Assimp).
                import yaml
                _SIGN_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                               "wrist_flex", "wrist_roll"]
                _DEFAULT_SIGNS = np.array([1.0, 1.0, 1.0, 1.0, -1.0])
                joint_signs = _DEFAULT_SIGNS.copy()
                urdf_offsets_deg = np.zeros(5)
                _offsets_path = os.path.expanduser(
                    "~/.config/202602hackathon/servo_offsets.yaml")
                if os.path.exists(_offsets_path):
                    with open(_offsets_path) as _f:
                        _data = yaml.safe_load(_f)
                    _sd = _data.get("joint_signs") if _data else None
                    if _sd and isinstance(_sd, dict):
                        joint_signs = np.array([
                            float(_sd.get(n, _DEFAULT_SIGNS[i]))
                            for i, n in enumerate(_SIGN_NAMES)
                        ])
                    _od = _data.get("urdf_offsets_deg") if _data else None
                    if _od and isinstance(_od, dict):
                        urdf_offsets_deg = np.array([
                            float(_od.get(n, 0.0))
                            for n in _SIGN_NAMES
                        ])
                    print(f"[INFO]: Loaded from {_offsets_path}", flush=True)
                else:
                    print(f"[WARN]: {_offsets_path} not found, using defaults", flush=True)
                print(f"[INFO]: Signs: {joint_signs}", flush=True)
                print(f"[INFO]: URDF offsets: {urdf_offsets_deg}", flush=True)

                port = LeRobotArm101.find_port()
                print(f"[INFO]: Found port {port}, connecting...", flush=True)
                real_arm = LeRobotArm101(port=port)
                real_arm.connect()
                print(f"[INFO]: Connected to real arm on {port} for mirroring", flush=True)
                print(f"[INFO]: Joint signs: {joint_signs}", flush=True)
            except Exception as e:
                import traceback
                print(f"[WARN]: Could not connect to real arm: {e}", flush=True)
                traceback.print_exc()
                print("[WARN]: Falling back to demo mode (no mirroring).", flush=True)
                real_arm = None

    # Prepare image output directory
    if args_cli.save_images:
        os.makedirs(IMAGE_DIR, exist_ok=True)
        print(f"[INFO]: Saving images to {IMAGE_DIR}")

    # Joint name ordering in URDF
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    print("[INFO]: Digital twin running. Press Ctrl+C to exit.")
    print(f"[INFO]: Scene has {scene.num_envs} environment(s)")

    try:
        while simulation_app.is_running():
            # Reset every 2000 steps (disabled in mirror mode)
            if real_arm is None and count % 2000 == 0:
                count = 0
                # Reset robot to default state
                root_state = scene["robot"].data.default_root_state.clone()
                root_state[:, :3] += scene.env_origins
                scene["robot"].write_root_pose_to_sim(root_state[:, :7])
                scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
                joint_pos, joint_vel = (
                    scene["robot"].data.default_joint_pos.clone(),
                    scene["robot"].data.default_joint_vel.clone(),
                )
                scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
                scene.reset()
                print("[INFO]: Scene reset")

            # ── Set joint targets ──────────────────────────────────────
            if real_arm is not None:
                # Mirror real arm joint angles (convert motor→URDF via signs)
                try:
                    motor_angles = real_arm.read_all_angles()  # 6 motor angles in degrees
                    # Apply signs + URDF offsets: urdf = (motor * sign + offset) * pi/180
                    motor_deg = np.array(motor_angles[:5], dtype=float)
                    urdf_deg = motor_deg * joint_signs + urdf_offsets_deg
                    urdf_rad = np.zeros(6)
                    urdf_rad[:5] = np.radians(urdf_deg)
                    # Include gripper (no sign/offset correction)
                    urdf_rad[5] = math.radians(motor_angles[5]) if len(motor_angles) > 5 else 0.0
                    targets = torch.tensor(
                        [urdf_rad.tolist()],
                        dtype=torch.float32,
                        device=scene["robot"].device,
                    )
                    scene["robot"].set_joint_position_target(targets)
                except Exception as e:
                    if count % 100 == 0:
                        print(f"[WARN]: Could not read arm angles: {e}")
            else:
                # Demo: gentle sinusoidal wave motion
                t = sim_time
                targets = scene["robot"].data.default_joint_pos.clone()
                targets[:, 0] = 0.5 * math.sin(0.5 * t)         # shoulder_pan
                targets[:, 1] = 0.3 * math.sin(0.3 * t + 0.5)   # shoulder_lift
                targets[:, 2] = 0.4 * math.sin(0.4 * t + 1.0)   # elbow_flex
                targets[:, 3] = 0.3 * math.sin(0.6 * t + 1.5)   # wrist_flex
                targets[:, 4] = 0.2 * math.sin(0.8 * t + 2.0)   # wrist_roll
                # gripper stays at 0
                scene["robot"].set_joint_position_target(targets)

            # ── Step simulation ────────────────────────────────────────
            scene.write_data_to_sim()
            sim.step()
            sim.render()
            sim_time += sim_dt
            count += 1
            scene.update(sim_dt)

            # ── Camera output ──────────────────────────────────────────
            if args_cli.enable_cameras and count % 50 == 0:
                # Print camera info periodically
                try:
                    overhead_rgb = scene["overhead_cam"].data.output["rgb"]
                    wrist_rgb = scene["wrist_cam"].data.output["rgb"]
                    print(
                        f"[t={sim_time:.1f}s] "
                        f"overhead_cam: {overhead_rgb.shape}, "
                        f"wrist_cam: {wrist_rgb.shape}, "
                        f"joints: {scene['robot'].data.joint_pos[0, :5].tolist()}"
                    )

                    if args_cli.save_images and count % 100 == 0:
                        _save_camera_images(scene, count)
                except Exception as e:
                    if count % 200 == 0:
                        print(f"[WARN]: Camera read error: {e}")

            elif not args_cli.enable_cameras and count % 500 == 0:
                # Without cameras, just print joint state
                jp = scene["robot"].data.joint_pos[0].tolist()
                print(
                    f"[t={sim_time:.1f}s] joints (rad): "
                    f"[{', '.join(f'{v:.3f}' for v in jp[:5])}] "
                    f"gripper={jp[5]:.3f}"
                )

    except KeyboardInterrupt:
        print("\n[INFO]: Shutting down...")
    finally:
        if real_arm is not None:
            try:
                real_arm.disconnect()
            except Exception:
                pass


def _save_camera_images(scene: InteractiveScene, step: int):
    """Save camera RGB images to disk as PNG files."""
    try:
        import torchvision.utils as vutils

        for cam_name in ["overhead_cam", "wrist_cam"]:
            rgb = scene[cam_name].data.output["rgb"][0]  # First env, shape (H, W, 4)
            rgb = rgb[:, :, :3]  # Drop alpha
            rgb = rgb.permute(2, 0, 1).float() / 255.0  # CHW, [0,1]
            path = os.path.join(IMAGE_DIR, f"{cam_name}_{step:06d}.png")
            vutils.save_image(rgb, path)
    except ImportError:
        # Fallback: save raw numpy
        for cam_name in ["overhead_cam", "wrist_cam"]:
            rgb = scene[cam_name].data.output["rgb"][0].cpu().numpy()[:, :, :3]
            path = os.path.join(IMAGE_DIR, f"{cam_name}_{step:06d}.npy")
            np.save(path, rgb)
    except Exception as e:
        print(f"[WARN]: Could not save images: {e}")


# ── Entry point ────────────────────────────────────────────────────────────
def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01,
        render_interval=1,
        device=args_cli.device,
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    # Create scene (with or without cameras based on --enable_cameras flag)
    if args_cli.enable_cameras:
        scene_cfg = Arm101SceneWithCamerasCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    else:
        scene_cfg = Arm101SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Start simulation
    sim.reset()

    # Set viewport camera AFTER reset (reset clears camera state)
    sim.set_camera_view(eye=[0.6, 0.6, 0.9], target=[0.0, 0.0, 0.45])

    # Warm-up: run a few render frames so the RTX renderer initializes
    for _ in range(10):
        sim.step()
        sim.render()

    print("[INFO]: Isaac Sim setup complete")
    print(f"[INFO]: URDF loaded from: {URDF_PATH}")
    print(f"[INFO]: Cameras enabled: {args_cli.enable_cameras}")
    print(f"[INFO]: Mirror mode: {args_cli.mirror}")

    run_simulator(sim, scene)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
