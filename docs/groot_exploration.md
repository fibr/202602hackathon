# GR00T Foundation Models — Exploration for ARM101 + Cubes Scenario

*Research date: 2026-03-14*

## Executive Summary

NVIDIA's **Isaac GR00T** family (N1 → N1.5 → N1.6) is a series of open **Vision-Language-Action (VLA)** foundation models that can control robot arms using natural language instructions and camera images. **GR00T N1.5/N1.6 are directly applicable to our ARM101 + cubes scenario** — the SO-100/SO-101 arms (nearly identical to our ARM101) are a first-class supported embodiment, with official fine-tuning tutorials and demo datasets for pick-and-place cube tasks.

Key takeaway: **With ~20-50 teleoperated demonstrations and a GPU with ≥25GB VRAM, we can fine-tune GR00T to do language-conditioned cube manipulation on our ARM101** — replacing our hand-coded vision pipeline + planner with a learned policy.

---

## 1. The GR00T Family

| Version | Released | Parameters | Key Advance |
|---------|----------|-----------|-------------|
| **N1** | Mar 2025 | 2B | First open humanoid foundation model; dual-system (fast reflexes + slow reasoning) |
| **N1.5** | Mid 2025 | 3B | Added single-arm + EEF control; cross-embodiment; LeRobot integration; works with 20-40 demos |
| **N1.6** | Late 2025 | 3B | 2× larger DiT head (32 layers); better long-horizon reasoning; improved real-world performance |

### Architecture (Dual-System Design)
- **System 2 (Slow)**: Vision-Language Model (VLM) that reasons about images + language instructions → produces semantic plans
- **System 1 (Fast)**: Diffusion Transformer (DiT) head that denoises continuous joint actions from the plan → outputs 8 actions per inference step
- **Input**: Camera images (1-2 views) + language instruction (e.g., "pick up the red cube and place it in the bowl")
- **Output**: Continuous joint position targets at ~10-27 Hz depending on GPU

### Training Data
Pre-trained on a diverse mix of:
- Egocentric human videos (internet-scale)
- Real robot trajectories (multiple embodiments)
- Synthetic data from NVIDIA Isaac Sim
- Simulated environments (LIBERO benchmark)

---

## 2. Relevance to Our ARM101 + Cubes Scenario

### Direct Compatibility ✅

Our ARM101 is a **LeRobot SO-ARM101** (5-DOF + gripper, Feetech STS3215 servos). GR00T has **first-class support** for SO-100/SO-101 arms:

| Feature | Our Setup | GR00T Support |
|---------|-----------|---------------|
| Robot arm | SO-ARM101 (5-DOF + gripper) | ✅ SO-100/SO-101 pre-registered embodiment |
| Servo protocol | Feetech STS3215 via scservo_sdk | ✅ LeRobot driver handles this |
| Cameras | Intel RealSense D435i (640×480) | ✅ Any USB camera at 640×480 @ 30fps |
| Task type | Pick-and-place cubes | ✅ Cube pick-and-place is a standard demo task |
| Control space | Joint positions + gripper | ✅ Single-arm + gripper via `new_embodiment` tag |

### What GR00T Replaces

Currently our pipeline is: **Camera → HSV Detection → Calibration → IK Planner → Joint Commands**

With GR00T: **Camera + Language → GR00T N1.6 → Joint Commands**

The entire vision pipeline, calibration transform, IK solver, and planner collapse into a single neural network inference call.

### What GR00T Does NOT Replace
- Low-level servo communication (still need scservo_sdk / LeRobot driver)
- Camera capture (still need pyrealsense2 or OpenCV)
- Safety limits (still need joint limits, collision checks)

---

## 3. Integration Pathway

### Option A: Via LeRobot Framework (Recommended)

LeRobot v0.4.0+ has native GR00T integration. This is the cleanest path:

```bash
# 1. Install LeRobot with GR00T support
conda create -n lerobot python=3.10
conda activate lerobot
pip install -e ".[groot]"

# 2. Collect ~30-50 teleoperation demonstrations
lerobot-record \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --dataset.repo_id=our-team/arm101-cubes \
  --dataset.num_episodes=50 \
  --dataset.single_task="Pick up the cube and place it upright" \
  --dataset.episode_time_s=30

# 3. Fine-tune GR00T N1.6 (~2000-10000 steps)
lerobot-train \
  --policy.type=groot \
  --dataset.repo_id=our-team/arm101-cubes \
  --batch_size=32 \
  --steps=10000 \
  --output_dir=./outputs/groot-arm101-cubes

# 4. Deploy and evaluate
lerobot-eval \
  --policy.path=./outputs/groot-arm101-cubes \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0
```

### Option B: Via Isaac-GR00T Repository Directly

For more control over training and a client-server deployment architecture:

```bash
# 1. Clone and install
git clone https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
pip install -e .[base]
pip install flash-attn --no-build-isolation

# 2. Fine-tune with custom modality config
python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path ./demo_data/arm101-cubes \
  --embodiment-tag new_embodiment \
  --max-steps 10000

# 3. Deploy as inference server
python gr00t/eval/run_gr00t_server.py \
  --model_path ./checkpoints/arm101-cubes

# 4. Connect robot client
python getting_started/examples/eval_lerobot.py \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --lang_instruction="Pick up the red cube and stand it upright"
```

---

## 4. Hardware Requirements

### For Fine-Tuning (Training)

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 25 GB (with `--no-tune_diffusion_model`) | 48 GB (full fine-tune) |
| GPU Models | RTX 4080/4090, A6000 | H100, L40, L20 |
| Training Time | ~2-4 hours (10K steps on 4090) | ~1 hour (H100) |
| Storage | ~20 GB (model + dataset) | 50 GB |

### For Inference (Deployment)

| Resource | Performance |
|----------|-------------|
| RTX 5090 | ~27 Hz |
| RTX 4090 | ~20 Hz |
| Jetson AGX Orin 64GB | ~10-15 Hz (TensorRT optimized) |
| RTX 4080 | ~15 Hz |

Our workstation GPU should be checked — we need at minimum an RTX 3090/4080 for inference, or a Jetson AGX Orin for edge deployment.

### For Data Collection
- LeRobot SO-100/SO-101 leader arm (for teleoperation) OR keyboard/GUI teleop
- 1-2 USB cameras at 640×480 (we already have the RealSense D435i)

---

## 5. Data Collection Strategy for Cubes

### Recommended Dataset Design

Based on the SO-101 fine-tuning tutorial and cube_to_bowl example:

1. **Camera setup**: Dual cameras — one front-facing (we have D435i), one wrist-mounted (would need to add)
2. **Episodes needed**: 30-50 demonstrations minimum (GR00T N1.5+ generalizes well in low-data regime)
3. **Task variants** to include:
   - Pick cube from random position → place at target
   - Pick cube → stand it upright
   - Different cube colors/sizes for generalization
4. **Data format**: LeRobot v2 format with `modality.json` specifying camera names, joint state dims, and action dims
5. **Episode length**: ~15-30 seconds per demonstration

### Teleoperation Options
- **Leader arm**: If we have a second SO-ARM101 configured as leader, this is the standard LeRobot teleop method
- **Keyboard teleop**: Our existing `control_panel.py --arm101` could be adapted to record LeRobot-format episodes
- **GUI teleop**: Use our existing XY jog pad interface with recording

---

## 6. Advantages Over Our Current Pipeline

| Aspect | Current (Classical) | GR00T (Learned) |
|--------|-------------------|-----------------|
| **Generalization** | Fails on new objects/lighting | Generalizes from language + vision |
| **Calibration** | Requires precise hand-eye calibration | Learned implicitly from demonstrations |
| **New tasks** | Must reprogram planner | Just record new demonstrations + fine-tune |
| **Language control** | None | "Pick up the red cube", "Move it left" |
| **Development effort** | ~2000 LOC (vision + calib + IK + planner) | ~50 demos + training script |
| **Robustness** | Brittle (HSV thresholds, lighting) | More robust (trained on visual diversity) |

### Limitations/Risks
- **Latency**: Neural inference adds latency vs. direct IK computation (~0.8ms IK vs ~50-100ms GR00T)
- **GPU dependency**: Requires GPU for inference (our current pipeline runs on CPU)
- **Reproducibility**: Learned policies can have unexpected failure modes
- **Training cost**: Need GPU time for fine-tuning (can use cloud if local GPU insufficient)
- **Precision**: VLA models may not achieve the same positioning precision as IK-based approaches
- **Debugging**: Black-box model harder to debug than explicit kinematic pipeline

---

## 7. Recommended Next Steps

### Quick Win (1-2 days)
1. **Check our GPU** — run `nvidia-smi` to see available VRAM
2. **Install Isaac-GR00T** in a separate conda env and run the SO-100 demo dataset inference to verify it works
3. **Test zero-shot** — try the pre-trained GR00T N1.6 model with our ARM101 without fine-tuning to see baseline behavior

### Medium Effort (3-5 days)
4. **Collect 30-50 demos** of cube pick-and-place using our ARM101 + existing control panel (adapted for recording)
5. **Fine-tune GR00T N1.6** on our data (~2-4 hours of GPU time)
6. **Evaluate** on novel cube positions and compare success rate vs. our classical pipeline

### Full Integration (1-2 weeks)
7. **Add language conditioning** — train with multiple task descriptions
8. **Integrate into our pipeline** as an alternative planner mode (keep classical as fallback)
9. **Optimize inference** — TensorRT quantization for faster control loop

---

## 8. Key Resources

- **GitHub**: [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) — Official repo with SO-100 examples
- **SO-101 Fine-tuning Tutorial**: [HuggingFace Blog](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning) — Step-by-step with our exact arm type
- **GR00T in LeRobot**: [HuggingFace Blog](https://huggingface.co/blog/nvidia/nvidia-isaac-gr00t-in-lerobot) — Native LeRobot integration
- **N1.6 on SO-101 + AGX Orin**: [Seeed Studio Wiki](https://wiki.seeedstudio.com/fine_tune_gr00t_n1.6_for_lerobot_so_arm_and_deploy_on_agx_orin/) — Edge deployment guide
- **Pre-trained Model**: [nvidia/GR00T-N1.5-3B](https://huggingface.co/nvidia/GR00T-N1.5-3B) on HuggingFace
- **Research Paper**: [arXiv:2503.14734](https://arxiv.org/abs/2503.14734) — GR00T N1 technical paper
- **Demo Dataset**: [cube_to_bowl_5](https://github.com/NVIDIA/Isaac-GR00T/tree/main/demo_data) — Included in the repo
- **Bimanual Cube Handover Dataset**: [pepijn223/bimanual-so100-handover-cube](https://huggingface.co/datasets/pepijn223/bimanual-so100-handover-cube)
- **GR00T N1.6 Research Page**: [research.nvidia.com](https://research.nvidia.com/labs/gear/gr00t-n1_6/)
- **6/7-DOF Arm Control Guide**: [Seeed Studio Wiki](https://wiki.seeedstudio.com/control_robotic_arm_via_gr00t/)
