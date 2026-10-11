# Examples

Scripts for bringing up the hardware, checking communication, controlling the SO-101 arms, and running the policies from the paper.

All examples assume:

- You have installed the software stack (see [Getting Started](../README.md#getting-started)).
- Your `lerobot` environment is activated.
- The robot is powered and wired as described in the [hardware section of the main README](../README.md#hardware).

<br>

---

## How to run

From the repo root:

```bash
cd examples
python3 0_so100_keyboard_joint_control.py
```

<br>

---

## Tested examples

- **`0_so100_keyboard_joint_control.py`** → basic single-arm joint control.
- **`2_dual_so100_keyboard_ee_control.py`** → dual-arm, joint + end-effector control.
- **`9_dual_wrist_camera.py`** → checks two wrist cameras (optional hardware, not part of the paper's build).

## Teleoperation

- **`5_xlerobot_teleop_xbox_new_wiring.py`** → full-robot Xbox-controller teleoperation for this build's two buses (arms on `/dev/xle_arms`, head and wheels on `/dev/xle_head`). The plain `5_xlerobot_teleop_xbox.py` targets the older per-arm wiring.
- **`teleop_hand_and_arm.py`** → VR teleoperation built on Open-TeleVision, using Unitree's [TeleVuer](https://github.com/unitreerobotics/televuer) and [teleimager](https://github.com/unitreerobotics/teleimager) packages from [xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate). Run it with `--arm SO101`. It maps each wrist pose to the arm with a two-link planar IK and opens each arm on its own port (`--left-port`, `--right-port`).
- **`pygame_test.py`** → verify a game controller is detected before running Xbox teleop.

## Subdirectories

- **[`vision/`](vision/README.md)** → perception + IK demos (cube grasping and scripted motions). Part of the stack, but not evaluated in the paper.
- **`policies/`**
  - `act_policy_control.py` runs an ACT policy on the robot. This is the runner behind the paper's 49-of-50 grasp test; the main README lists the exact settings.
  - `smolvla_policy_control.py` replays a recorded episode from a LeRobot dataset on one arm. It does not run SmolVLA and is not the paper's SmolVLA benchmark.
  - `diffusion_policy_control.py` is a self-contained toy diffusion model for rough timing on random inputs. It is not the source of the paper's Diffusion Policy number.
  - `serve_policy.py` serves `lerobot/smolvla_base` over a websocket from a separate GPU machine. The paper ran everything on the Jetson instead.
- **`xlerobot_examples/`** → additional, unverified upstream references; may require updates to work with XLeRobot-Pro.

## Benchmarks

- **`so100_precision_test.py`** → repeatability test (results in `precision_test_results.csv`; not used in the paper).
