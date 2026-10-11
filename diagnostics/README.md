# Diagnostics

Standalone tools for debugging the robot hardware and the vision/IK pipeline.
All of them import from the `xlerobot_pro` and `lerobot` libraries, so run
them from anywhere after the install in the [main README](../README.md#getting-started).

Scripts open the buses by their stable names — `/dev/xle_arms` (12 arm motors)
and `/dev/xle_head` (2 neck + 3 wheel motors). These are udev symlinks keyed on
each adapter's USB serial; **create them first with `detect_buses.py` below**, or
nothing here will connect. Override with `XLEROBOT_ARMS_PORT` /
`XLEROBOT_HEAD_PORT` (or `--port` where offered) if you must use raw device names.

## First-time setup

| Script | Purpose |
| --- | --- |
| `detect_buses.py` | Identify which USB adapter is which by pinging motors, and emit the udev rules that give them stable names. **Run this before anything else.** `--write` installs the rules (needs sudo). |

## Bus and motors

| Script | Purpose |
| --- | --- |
| `check_bus.py` | Verify which motors respond on a bus (`--port`, `--motors arm\|head`) |
| `read_motors.py` | Print current calibrated motor positions |
| `read_head.py` | Print calibrated head motor positions (for URDF zero offsets) |
| `read_head_robot.py` | Same, via the `XLerobot` robot class |
| `lock_motors.py` / `unlock_motors.py` | Enable / release torque |
| `verify_motor_limits.py` | Check configured joint limits against the hardware |
| `debug_motors.py` | Measure neutral offsets and joint directions |
| `base_drive_check.py` | Confirm the omni base drives: wheel direction, odometry agreement, and that the wheels reliably stop. Wheels only — the arms are untouched. Capped at 0.15 m/s and 10 s. Run after building, before any mobile work |

**Traps worth knowing before writing anything that touches motors:**

- **`bus.disconnect()` disables torque by default.** A read-only check that
  connects and disconnects will cut holding torque on every motor in its dict and
  drop a loaded arm. Pass `disconnect(disable_torque=False)` unless you mean it.
- **Never cut torque on a loaded arm.** `disable_torque` removes it in one write
  and the arm falls. Step `Torque_Limit` down to zero over several seconds
  instead, then restore it.
- **The speed register is `Present_Velocity`, not `Present_Speed`**, and
  `OperatingMode` imports from `lerobot.motors.feetech`, not `lerobot.motors`.
- **Servos with logic power but no motor power still answer the bus.** They report
  load pinned at `1024 + Torque_Limit` with current 0 and never move. Check
  `Present_Voltage`: ~120–135 means 12 V, ~50 means the bus is on a 5 V rail.
- **A single corrupted read can inflate a `max()`** — by 32 °C in one observed
  case. Use a high percentile when reporting peaks.

## Frame transforms and vision

| Script | Purpose |
| --- | --- |
| `check_vision_live.py` | Live end-to-end check: camera detection vs. transformed base coordinates |
| `diagnose_fk_vs_vision.py` | Place the gripper on the target, compare FK vs. vision position |
| `debug_transform_chain.py` | Dump every intermediate matrix in the camera-to-base transform |
| `calibrate_head_offsets.py` | Calibrate head-camera offsets (writes `calibration/t_base_camera.json`) |
| `measure_ee.py` | Measure the end-effector position with the arm posed by hand |
| `debug_pincer_transform.py` | Debug the reduced-arm-model transform helpers |

## Visualizers

| Script | Purpose |
| --- | --- |
| `visualize_mujoco.py` | Simulate the full transform + IK pipeline in MuJoCo |
| `visualize_color_detect.py` | Overlay color-detection results on the last capture |
| `visualize_point_cloud.py` | Display the last captured point cloud (needs `open3d`) |

## Experiments

The measurement tooling (A1 power integrity, A2 thermal and holding tests, B1
base acceleration with a held bottle, C2 inference timing) lives **outside this
repo**, in the companion data repository
([xlerobot-pro-data](https://github.com/Minko82/xlerobot-pro-data)) next to the
raw telemetry it produced. This repository is for operating the robot; that one
holds the measurements behind the paper.

Those scripts still `import xlerobot_pro`, so an editable install of this package
is a prerequisite for running them.

Capture artifacts are read from and written to `./outputs/` in the current
working directory (gitignored).
