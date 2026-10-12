# XLeRobot-Pro

<img width="894" height="455" alt="The robot removing a screw from an electric-vehicle battery pack under VR teleoperation" src="https://github.com/user-attachments/assets/d8b911e5-4e95-4397-965e-9f6c243c8a38" />

XLeRobot-Pro is [XLeRobot](https://github.com/Vector-Wangel/XLeRobot) with the computer on board. It keeps XLeRobot's two SO-101 arms, omni-wheel base, and power station, and adds a Jetson Orin Nano Super, a RealSense D435, separate power outlets for the arms, the head and wheels, and the computer, and stronger printed arm links. The parts cost $1,247, and the robot can record demonstrations, train a policy, and run it on battery without a second machine.

This repository holds the software, the servo settings, and the print files. The measurements and protocols behind the paper live in the companion data repository, [xlerobot-pro-data](https://github.com/Minko82/xlerobot-pro-data). The [project website](https://minko82.github.io/xlerobot-pro-website/) has the full build guide with photos.

## What changed from XLeRobot

In its $660 configuration, XLeRobot runs its learned policies on a laptop that you supply; an optional Raspberry Pi can run a small ACT policy, but training and larger models need another computer. Putting the computer on the robot exposed two problems, and fixing them is most of what this build adds.

- **Power.** In XLeRobot's wiring, each arm shares a servo bus with either the head or the wheels, and each bus draws from its own USB-C outlet. Moving an arm and the head together collapsed the shared bus supply in 5 of 5 stress trials, and each collapse needed a power cycle. Now both arms draw from the power station's 12 V car outlet, the head and wheels from one USB-C outlet, and the Jetson from another. Under the same protocol, with the firmware settings below, the supply held in all 5 trials. Those trials don't separate the outlet change from the firmware limits.
- **Printed links.** On the default print profile, the arm broke at a printed link before the servo gave out. Printing the load-bearing links with 4 perimeters instead of 2 raised the arm's failure load by 67% (18.5 N against 11.1 N) for 80 g more per arm, and the failure moved to the servo.
- **Compute.** A Jetson Orin Nano Super sits in a ducted enclosure at the center of the cart, with a RealSense D435 on the neck. Together they account for $594 of the total.

## What it can do

These are the paper's measurements. Each holds for the conditions next to it.

| | Result | Conditions |
|---|---|---|
| Learned grasping | 49 of 50 trials | ACT trained on the Jetson in under an hour; the left arm lifts a 100 mL glass bottle from a shelf; one marked bottle pose; base stationary; on battery |
| Policy timing | ACT 104.5 ms, Diffusion Policy 540 ms, SmolVLA 997 ms | One pass on the Jetson in MAXN SUPER; 10 denoising steps for Diffusion Policy and SmolVLA |
| Shoulder hold | 0.7 kg for 30 min | One shoulder servo, load 18.6 cm from the joint, 24.4 °C room. 0.8 kg reached 65 °C after 27.8 min, 0.9 kg after 22.1 min, and 1 kg slipped. The 0.7 kg hold probably ran with the power station on pass-through charging, which kept a matched 0.8 kg hold 2 °C cooler |
| Teleoperated grasping | 74 of 75 attempts | Five objects (a plastic apple, a cube, a drill, a water bottle, and a screwdriver), 15 attempts each, with the Fin-Ray fingers; the one miss was the 17 g apple |
| Battery | About 13–16 h of SmolVLA inference, about 17 h holding loads | Projected from whole-percent charge readings on the 288 Wh power station, not run to depletion |

The holding results come from one servo, and the grasping policy was tested at one bottle pose with a fixed camera image. Navigation, autonomous two-arm coordination, and mobile manipulation were not evaluated.

## Hardware

### Parts

| Subsystem | Component | Qty | USD |
|---|---|---|---|
| Compute | Jetson Orin Nano Super + microSD | 1 | 260.23 |
| Perception | Intel RealSense D435 | 1 | 333.75 |
| Actuation | Feetech STS3215, 12 V (ST-3215-C018) + 2 bus driver boards | 17 | 292.93 |
| Power | Anker SOLIX C300 DC power station | 1 | 159.99 |
| Structure | IKEA RÅSKOG steel cart, 3 omni wheels, PLA (incl. Jetson enclosure), fasteners | 1 | 129.95 |
| Cabling | USB-C hub, power cables, connectors | 7 | 70.42 |
| **Total** | | | **1,247.27** |

These are the prices we paid. The total leaves out the 3D printer, TPU filament for the fingers, and the VR headset used for teleoperation. The [components page](https://minko82.github.io/xlerobot-pro-website/components.html) lists every line item with a link.

### Printing

Print everything in PLA at 0.2 mm layers. Every printed part of both arms gets **4 perimeters and 15% infill** (High-Shell), everything else stays on the default 2 perimeters, and the Fin-Ray fingers are TPU 95A. The 3MF project in [`hardware/`](hardware/README.md) already applies these settings. The test results behind them, the Jetson mounting frame, and an editable STEP file of the arm base are in [`hardware/`](hardware/README.md) too.

### Power and buses

| Supply | Loads | Outlet | Cable | Torque_Limit | Acceleration |
|---|---|---|---|---|---|
| Bus 1 (`/dev/xle_arms`) | 12 arm servos | 12 V car outlet | Car socket to DC5521 | 450 | 40 |
| Bus 2 (`/dev/xle_head`) | 2 head and 3 wheel servos | USB-C | USB-C to DC5521 (PD) | 650 | 20 |
| Jetson | | A second USB-C | USB-C to DC5525, 140 W (PD) | | |

- Plug both bus adapters straight into the Jetson. Behind USB hubs shared with the camera they failed to enumerate, which looks exactly like a dead bus.
- The outlets share one battery, but the arms no longer share a bus with the head or the wheels. The build adds no fuses; each outlet on the power station shuts off on overload.
- The arm torque setting comes from a simple current estimate (Eq. 1 in the paper). With all 12 arm servos at no-load current and two of them loaded at 450, the estimate is about 4.4 A (4.6 A with the datasheet's listed torque constant), under a 5 A budget that is half the car outlet's 10 A rating. One arm peaked at 1.41 A in testing. Bus 2's current margin was not measured.
- All limits live in [`src/xlerobot_pro/firmware_limits.py`](src/xlerobot_pro/firmware_limits.py) and are applied automatically whenever a robot connects.

### Motor IDs

| Bus | IDs |
|---|---|
| Bus 1 | Left arm 1–6, right arm 7–12. In each arm: shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, gripper |
| Bus 2 | Head: 1 tilt, 2 pan. Wheels: 3 left, 4 back, 5 right |

## Getting started

These steps take you from a fresh Jetson (JetPack 6, Python 3.10) to a calibrated robot. The [software guide](https://minko82.github.io/xlerobot-pro-website/software.html) covers the same ground in more detail, including flashing the Jetson and working over SSH.

**1. Clone and make an environment.**

```bash
git clone https://github.com/Minko82/xlerobot-pro.git
cd xlerobot-pro
python3 -m venv ~/.venvs/xlerobot-pro
source ~/.venvs/xlerobot-pro/bin/activate
pip install --upgrade pip
```

Activate the environment in every new terminal before running anything.

**2. Install PyTorch (Jetson only).** The PyPI `torch` wheel is CPU-only on the Jetson, so install NVIDIA's JetPack 6 build first:

```bash
pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
python -c "import torch; print(torch.cuda.is_available())"   # should print True
```

**3. Install the robot stack and the vision and IK libraries.**

```bash
pip install -e ".[all]"
pip install -r requirements/requirements-vision.txt
```

The vision requirements pin the versions that ran on the robot, including Pink 4.3.0. On the Jetson, leave out `open3d`: run `grep -v '^open3d' requirements/requirements-vision.txt > /tmp/req-vision.txt` and install from that file. The lockfiles in `requirements/` reproduce a known-good desktop install.

**4. Check the install without hardware.**

```bash
make smoke                             # compiles everything and runs the tests
python tests/vision_offline_suite.py   # perception, transform, IK, FK; expect 49/49
```

**5. Give the buses stable names.** Both adapters use the same USB chip, so Linux names them `/dev/ttyACM0` and `/dev/ttyACM1` in whatever order they enumerate. Everything here opens `/dev/xle_arms` and `/dev/xle_head` instead. With both adapters plugged in and the servos powered:

```bash
sudo usermod -aG dialout $USER                 # then log out and back in
python diagnostics/detect_buses.py --write     # finds which adapter is which and installs udev rules
sudo udevadm control --reload-rules && sudo udevadm trigger
ls -l /dev/xle_arms /dev/xle_head
```

**6. Check that every servo answers.**

```bash
python diagnostics/check_bus.py --motors arm
python diagnostics/check_bus.py --motors head
```

If a servo is missing or a new one needs an ID, connect only that servo and run `python set_motor_id.py --port /dev/ttyACM0 --id <id> --model sts3215`.

**7. Calibrate.** One calibration covers both arms, the head, and the wheels:

```bash
python scripts/calibrate_xlerobot.py
```

It asks you to center the arms, sweep every arm joint to both end stops, and center the head. Reference calibrations from a working robot are in [`calibration/`](calibration/). The software guide shows how to check a calibration, which can finish without errors and still be wrong.

**8. Move it.** Start small:

```bash
python diagnostics/base_drive_check.py --vx 0.05 --seconds 2   # wheels
python examples/pygame_test.py                                # is the gamepad detected?
python examples/5_xlerobot_teleop_xbox_new_wiring.py          # whole robot from an Xbox controller
python diagnostics/verify_motor_limits.py                     # confirm the limits the servos stored
```

**9. Set the Jetson's power mode.** The paper's measurements used MAXN SUPER with the clocks left dynamic:

```bash
sudo nvpmodel -m 2      # MAXN SUPER. On the Orin Nano Super, mode 0 is 15 W and mode 1 (the default) is 25 W
sudo jetson_clocks      # optional: pins the clocks; SmolVLA ran about 6.5% faster
```

Long training runs keep the Jetson near 22 W for hours, which is more than the passive enclosure was designed for. One 60,000-step run ended in a thermal shutdown about two hours in. For long jobs, leave the clocks dynamic, save checkpoints often, and put a fan on the board.

## Learning on the robot

The paper's grasping result came from this loop, run entirely on the Jetson.

**Record.** The demonstrations were kinesthetic: the operator moved the left arm by hand while the robot recorded the neck camera and the joint positions at 30 Hz. The recording and dataset tools are in the data repository's `scripts/` folder (`record_kinesthetic.py`, `drop_episodes.py`, `mask_dataset_videos.py`, `freeze_dataset_videos.py`). Because the operator's hand shows up in kinesthetic demonstrations, the images were masked (the mask is [`calibration/hand_mask_grey88.png`](calibration/hand_mask_grey88.png)) and each episode's first frame was held fixed.

**Train.** These are the settings from the paper's run, on 49 demonstrations:

```bash
lerobot-train --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
    --dataset.repo_id=<your dataset> --output_dir=outputs/act_run \
    --batch_size=4 --steps=30000 --save_freq=10000 --num_workers=2
```

ACT's defaults supply the rest (100-action chunks, a ResNet-18 backbone). The deployed policy is the checkpoint at step 10,000, about an hour into training.

**Run.**

```bash
python examples/policies/act_policy_control.py run \
    --checkpoint outputs/act_run/checkpoints/010000/pretrained_model \
    --port /dev/xle_arms --arm left --duration 25 --n-action-steps 100 \
    --freeze-frame --overlay calibration/hand_mask_grey88.png \
    --task "pick up the glass bottle from the shelf" \
    --camera-serial <your D435 serial>
```

Each pass returns 100 joint targets, which the arm follows open-loop for 3.3 s. The runner applies the servo limits before anything moves and prints the inference time and the realized control rate at the end. The policy was only tested at one marked bottle pose, so collect your own demonstrations for anything else.

## Timing other policies

[`benchmarks/`](benchmarks/README.md) holds the two scripts behind the paper's SmolVLA and Diffusion Policy timings. Both time the full loop from camera capture to action on the Jetson.

## Teleoperation

[`examples/teleop_hand_and_arm.py`](examples/teleop_hand_and_arm.py) drives the arms from a VR headset. It builds on Open-TeleVision through Unitree's [xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate) packages (TeleVuer and teleimager), which you install separately, and the headset is not in the parts list. Run it with `--arm SO101`. It was written for the earlier wiring with one bus per arm (`--left-port`, `--right-port`); on the two-bus build, the right arm's IDs 7–12 need the `SO101FollowerArm` class in [`scripts/xle_arms.py`](scripts/xle_arms.py).

Without a headset, the Xbox example in step 8 drives the whole robot.

## Mapping and navigation

The robot also runs RTAB-Map and Nav2 in a separate ROS 2 workspace. That workspace is not part of this repository, and the paper does not evaluate navigation.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Permission denied: /dev/ttyACM0` | Add yourself to `dialout` and log in again (step 5) |
| `could not open port /dev/xle_arms` | The udev names do not exist yet (step 5) |
| A servo is missing from a bus scan | Check its ID (step 6) and the cable to its neighbor |
| Servos answer but never move | The bus has logic power but no servo power. Read `Present_Voltage`: about 120–135 means 12 V and is healthy; about 50 means the bus is on a 5 V feed |
| A bus disappears after the base drives | A cable was tugged loose. Look for `error -110` or `unable to enumerate` in `dmesg`, and add strain relief before driving again |
| An arm drops when a script ends | Torque was cut instead of bled off; see [`diagnostics/README.md`](diagnostics/README.md) |
| `ModuleNotFoundError: lerobot` | Activate the environment and run `pip install -e ".[all]"` again |
| `import pinocchio` fails over a missing `liburdfdom` library | Reinstall from `requirements/requirements-vision.txt`, which pins compatible versions |
| RealSense frames time out | Replug the USB 3 cable and check `rs-enumerate-devices` |

More bus, motor, and frame-transform tools are in [`diagnostics/`](diagnostics/README.md).

## Repository layout

| Path | Contents |
|---|---|
| [`src/lerobot/`](src/lerobot/) | LeRobot fork with this robot's classes and teleoperators |
| [`src/xlerobot_pro/`](src/xlerobot_pro/) | Vision and IK library, robot model, and the servo limits |
| [`examples/`](examples/README.md) | Bring-up, teleoperation, policy, and vision examples |
| [`benchmarks/`](benchmarks/README.md) | The SmolVLA and Diffusion Policy timing scripts used in the paper |
| [`scripts/`](scripts/) | Whole-robot calibration and the two-arm bus helper |
| [`diagnostics/`](diagnostics/README.md) | Bus, motor, and frame-transform debugging tools |
| [`calibration/`](calibration/) | Reference calibrations and the hand mask used for the grasping policy |
| [`hardware/`](hardware/README.md) | Print files, print settings, and the URDF |
| [`requirements/`](requirements/) | Vision requirements and lockfiles |
| [`tests/`](tests/) | Hardware-free tests (`make smoke`) |

## Data

[xlerobot-pro-data](https://github.com/Minko82/xlerobot-pro-data) holds the raw logs and the scripts that produced them:

- `A1/`: startup trials and the peak arm-bus current
- `A2/`: the shoulder holding tests, Jetson temperature and power, and the SmolVLA timing sweep
- `B1/`: base acceleration with a bottle in the gripper
- `trials/`: the 50 grasping trials and the training logs

## Citation

The paper describing this platform, "Cutting the Cord: System Design and Characterization of a Low-Cost, Untethered Bimanual Mobile Manipulator," is under review. A citation will be added here.

## Acknowledgements

This project builds on [XLeRobot](https://github.com/Vector-Wangel/XLeRobot), [LeRobot](https://github.com/huggingface/lerobot), the [SO-101 arm](https://github.com/TheRobotStudio/SO-ARM100), [LeKiwi](https://github.com/SIGRobotics-UIUC/LeKiwi), [Open-TeleVision](https://github.com/OpenTeleVision/TeleVision), [Pink](https://github.com/pink-kinematics/pink), and [Pinocchio](https://github.com/stack-of-tasks/pinocchio). Thank you to everyone behind them.

## License

Apache 2.0, inherited from LeRobot. See [`LICENSE`](LICENSE).
