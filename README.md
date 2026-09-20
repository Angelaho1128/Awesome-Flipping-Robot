# Awesome Flipping Robot

> Before long training, run the new [bounded launch search](training/baseten/LAUNCH_SEARCH.md). Full training now requires a CPU-confirmed, fault-free simulated launch baseline.

Top-down pancake detection and a shared two-joint simulation/training model. The viewer and Baseten PPO trainer now use the same XML generator, torque limits, reward and observation contract. The camera is not yet calibrated or connected to automatic motor execution.

## Fixed measured configuration

The single source is **training/baseten/settings.yaml**. `common/config.py` reads it; runtime simulation generates XML from it. `sim/assets/arm.xml` is a generated nominal snapshot for inspection, not a second runtime configuration.

| Parameter | Value |
|---|---:|
| Elbow/shoulder axis → wrist axis | 240 mm |
| Wrist axis → pan bowl centre | 100 mm |
| Pan mass | 315 g |
| Pancake mass | 25 g |
| Pancake diameter, randomized | 95–105 mm |
| Pancake centre, randomized | X and Y each ±10 mm from pan centre |
| Nominal pan rim / flat cooking diameter | 160 / 150 mm |
| Pancake full thickness, assumed fixed | 2.5 mm |
| Wrist motor mass | 700 g |
| Attachment mass, assumed fixed | 110 g |
| Link mass, assumed fixed | 180 g |
| Shoulder holding-torque reference | 3.0 N·m |
| Wrist holding-torque reference | 1.2356379 N·m |

Only pancake diameter and initial location vary across physical training cases. Mass, friction (0.45), delay (17.5 ms), inertia multiplier (1), motor limits, thickness and geometry remain fixed. Camera observation noise is disabled. PPO still explores different actions; those are choices, not randomized physical parameters.

The 3 N·m reference is the [Sienci closed-loop motor rating](https://sienci.com/product/nema-23-closed-loop-stepper-motor-3nm/), assuming your 57HYB112 is that identified Sienci variant. The [LDO-57STH56-2804AC specification](https://ldomotion.com/products/ldo57sth562804ac) gives 12.6 kgf·cm = 1.2356379 N·m and a 0.7 kg mass. Neither source establishes the actual torque-speed curve on your driver at 24 V.

Both joints currently use fixed 90°/s speed limits and 180°/s² acceleration limits. The shoulder's previously reported four-second revolution motivates 90°/s; wrist speed and acceleration are provisional. Torque decreases linearly toward zero at an assumed 450°/s cutoff. These dynamic assumptions need measurement before interpreting simulated success as reproducible hardware behavior. An H100 does not change physical speed limits.

Analytical load at the 45°/-45° home pose is approximately **2.398 N·m shoulder / 0.334 N·m wrist**. The shoulder's permitted initial load is 2.55 N·m, so this pose passes narrowly. Horizontally extending the same load would require roughly 3.253 N·m at the shoulder. Static support does not guarantee enough acceleration for a flip.

## What changed

- Replaced the viewer's separate 40 cm + 40 cm model and amplified actuators with the canonical 24 cm + 10 cm model. Actuator gear is 1, with explicit 3.0 / 1.2356379 N·m zero-speed caps.
- Replaced the 28 cm pan and 19 cm × 24 mm pancake with the dimensions above. Pan mass is explicitly assigned; rim geometry cannot silently add extra mass.
- Replaced the direct two-torque environment with Gymnasium-compatible **one-observation, one-complete-toss** action selection: 23 observation values and seven trajectory parameters.
- The viewer now replays a canonical baseline or a saved policy using recorded simulation timestamps. Wall-clock rendering speed does not change the physics. Old discontinuous wrist snaps were removed.
- Preparation now raises the shoulder from 45° toward 45–60°, instead of lowering it into higher gravity load. Launch excursion spans 10–40°; all trajectories are checked against joint, pitch, speed and acceleration limits.
- Added self-contained Baseten/MuJoCo Warp training under `training/baseten/`: 128 CUDA physics worlds, PPO on CUDA, periodic checkpoints, refreshed case banks and CPU/GPU parity gates.
- Batching preserves per-world compiled parameters and retains the audited `dof_length` fix with simulator sleeping explicitly disabled.
- Added `common.observation.from_measurements(diameter_mm, x_mm, y_mm)` for calibrated measurements; it does not convert pixels automatically.
- Fixed the detector's duplicate camera-switch function and broken P-key handler. Colour segmentation and shape detection are otherwise unchanged.

Reward still distinguishes launch, rotation and settled opposite-side landing. A true success requires airborne motion, approximately a half-turn, containment and settling. It does not use the old negative distance-to-world-origin reward. Failure to establish a valid starting state does not enter PPO as a training transition.

## Baseten launch: commands on your laptop

These commands upload source and start execution remotely. Do not run `run.sh` locally.

```sh
cd /Users/andrewdai/Programming/Awesome-Flipping-Robot/training/baseten
baseten train push --config config_smoke.py
baseten train job logs --job-id SMOKE_JOB_ID --tail
```

Replace `SMOKE_JOB_ID` with the returned ID. Smoke runs cloud regression tests, CPU/GPU comparison, and 16 one-toss attempts on four GPU worlds. Require completed smoke, `gpu_parity.json` with `passed: true`, and `COMPLETE.json`. A few failed flips do not imply installation failure; failed parity or initial-state setup must be resolved.

Then start a **new policy**:

```sh
baseten train push --config config.py
baseten train job logs --job-id TRAIN_JOB_ID --tail
```

Leave both resume fields as `None`. Previous policies have incompatible geometry/action contracts. Training runs indefinitely on one H100 with 128 worlds × 8 steps = 1,024 samples per rollout, minibatches of 512 and five PPO epochs. Validation occurs every 9,216 attempts on separate seeded size/location cases. The best mean-reward checkpoint and its success count are saved. First-time kernel compilation adds startup time; no throughput promise is made.

```sh
baseten train job stop --job-id TRAIN_JOB_ID
baseten train checkpoint list --job-id TRAIN_JOB_ID
baseten train checkpoint files --job-id TRAIN_JOB_ID --output jsonl > checkpoint-urls.jsonl
python3 download_artifacts.py checkpoint-urls.jsonl --output downloaded
```

`push` submits AND starts the job. Nothing was submitted automatically by this repository update. Expiring checkpoint URL files, secrets and downloaded models are excluded from source uploads.

## Camera and viewer

Install root `requirements.txt` only if you intend to run the camera/viewer yourself. Cloud installs its separate pinned requirements through `run.sh`.

```sh
python3 vision/pancake_detection.py --source oak
python3 oak_test.py
python3 demo.py --source oak
```

OpenCV cameras use `--source 0`, `1`, or `auto`. Click the pancake to calibrate colour; Space pauses, M toggles the mask, N switches camera, C shows candidates, S saves a screenshot, P prints detection information and Q exits. Measurements are in resized-image pixels and shape scores are not calibrated probabilities.

On macOS, explicitly view a trained checkpoint with:

```sh
mjpython sim/view_sim.py --policy /absolute/path/to/policy.zip
```

Keep its metadata.json next to policy.zip. Without a policy the viewer displays a midpoint bounded baseline and prints its actual outcome. Neither mode establishes real-world success.

## Plan after the smoke test

1. Inspect baseline replay and motor saturation/faults using this corrected model. A useful physical motion must exist before a long RL run can find it.
2. Verify actual speed, acceleration, tracking and load response on the robot; replace the fixed provisional limits with measurements. Keep the sourced holding torque separate from moving torque.
3. Inspect airborne rate, success count and validation trajectories, not just mean reward. If it again settles near zero reward without launching, diagnose the trajectory/mechanical envelope before spending more GPU time.
4. Calibrate the top-down camera into pan-relative millimetres, then feed diameter/XY through the shared observation adapter. The policy selects a full trajectory; local robot control executes it.
5. Add synchronized trial video/telemetry and human-verified landing labels before deploying policies to hardware.

`robot/controller.py`, `robot/gcode.py`, `robot/safety.py`, pan calibration, automatic inference integration and trial recording remain unimplemented. The camera and viewer still run as separate processes. No hot-pan or motor execution is enabled by this update.

## Verification

Source/config/XML consistency and Baseten SDK configuration loading are checked statically. New cloud regression tests and GPU parity must run on Baseten. No local simulation or training was executed for this update. The simulator is a rigid-pancake research model; it does not simulate sticking, folding, tearing or a measured motor controller.
