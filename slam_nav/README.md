# XLeRobot SLAM & Navigation

## File structure

```
slam_nav/
├── fast_lio_nav2_integration/      # ROS 2 (ament_python) package
│   ├── launch/
│   │   ├── 0_lekiwi_mapping.launch.py
│   │   ├── 1_lekiwi_mapping_filtered.launch.py
│   │   ├── 2_lekiwi_mapping_scan.launch.py
│   │   ├── lekiwi_nav2.launch.py
│   │   ├── test_lekiwi_nav2.launch.py
│   │   └── test_lekiwi_nav2_wheels_only.launch.py
│   ├── config/
│   │   ├── lekiwi_nav2_config.yaml
│   │   └── nav2_rviz.rviz
│   ├── scripts/
│   │   ├── xlerobot_cmd_vel_bridge.py
│   │   └── xlerobot_wheels_only_cmd_vel_bridge.py
│   ├── fast_lio_nav2_integration/
│   │   └── odom_quat_normalizer.py
│   ├── package.xml
│   └── setup.py
├── tests/
│   ├── test_wheels_identify.py
│   └── test_wheels_kinematics.py
└── square_open_loop.sh
```

### `launch/`: mapping (SLAM)
| File | What it does |
|---|---|
| `0_lekiwi_mapping.launch.py` | RealSense + RTAB-Map with visual odometry. The 2D grid comes from the depth image; RealSense filters are off. |
| `1_lekiwi_mapping_filtered.launch.py` | Same as 0, plus a `base_link -> camera_link` static TF and the RealSense decimation/spatial/temporal/hole-filling filters on. |
| `2_lekiwi_mapping_scan.launch.py` | Converts the point cloud to a pseudo-LiDAR `/scan` (0.1–1.5 m height slice) and has RTAB-Map draw the 2D grid from it. |

All three start a fresh map on each launch (`--delete_db_on_start`) and save it to `~/.ros/rtabmap.db`.

### `launch/`: navigation
| File | What it does |
|---|---|
| `lekiwi_nav2.launch.py` | RTAB-Map localization on the saved map + `map_server` + `nav2_bringup`. Doesn't drive the robot. |
| `test_lekiwi_nav2.launch.py` | Localization + Nav2 servers + the **full-robot** `/cmd_vel` bridge. |
| `test_lekiwi_nav2_wheels_only.launch.py` | Localization + Nav2 + the **wheels-only** bridge + RViz. The newest and most-tested launch file. |

### `config/`
- `lekiwi_nav2_config.yaml`: Nav2 parameters (MPPI controller, Smac Hybrid planner, behaviors, local/global costmaps).
- `nav2_rviz.rviz`: RViz layout for map, costmaps, paths and goal setting.

### `scripts/`: `/cmd_vel` to base motors
- `xlerobot_cmd_vel_bridge.py`: drives the base through lerobot's `XLerobot` class (requires the full robot).
- `xlerobot_wheels_only_cmd_vel_bridge.py`: talks to only the 3 wheel motors on the Feetech bus and never touches the arms or head. Use it when only the base is assembled.

### Other
- `odom_quat_normalizer.py`: relays `/odom_raw` to `/odom`, normalizing quaternions and dropping all-zero ones.
- `tests/test_wheels_identify.py`: spins each wheel motor in turn so you can map motor IDs to physical wheels.
- `tests/test_wheels_kinematics.py`: drives the base through a set of moves without ROS to check directions.
- `square_open_loop.sh`: publishes a timed open-loop square on `/cmd_vel` to check odometry and drift.

## Important details

- **Default paths:** map `~/.ros/rtabmap.yaml` / `.pgm`, database `~/.ros/rtabmap.db`, base serial port `/dev/xle_base`.
- **Localization:** RTAB-Map runs on the saved map without adding to it (`Mem/IncrementalMemory false`) and is constrained to 2D (`Reg/Force3DoF true`). It publishes `map -> odom`. `/initialpose` is relayed to `/rtabmap/initialpose`, so RViz "2D Pose Estimate" works.
- **Camera mount:** set the camera's position on the base with the `camera_tx/ty/tz/roll/pitch/yaw` launch arguments. They all default to 0.
- **Wheel motor IDs:** 3 = right, 4 = left, 5 = back (STS3215, velocity mode).
- **Kinematics:** 3 omniwheels at 240°/0°/120°, wheel radius 0.05 m, base radius 0.125 m. ROS `linear.x` (forward) maps to the kinematics y-axis.
- **Bridge safety:** the wheels stop if no `/cmd_vel` arrives for 0.25 s. Speeds are clamped to 0.6 m/s and 1.8 rad/s. Use `linear_gain`, `angular_gain` and `invert_linear` to tune.
- **Python environment:** the bridges import `lerobot`. The wheels-only launch runs its bridge with `~/miniconda3/envs/lerobot/bin/python`. `test_lekiwi_nav2.launch.py` uses plain `python3`, so that interpreter needs `lerobot` installed.
- **RealSense workaround:** the point cloud is re-enabled 12 s after launch (`pointcloud__neon_.enable`) because the driver's parameter name doesn't match the launch argument.
- **Robot outline:** `lekiwi_nav2_config.yaml` still uses a 5 cm placeholder square for the robot's outline. Set it to the real base size.
- **Package name:** the package is still called `fast_lio_nav2_integration` because it was ported from `fast_lio_nav2_humanoid`. FAST_LIO itself is not used.
