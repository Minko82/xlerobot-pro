# XLeRobot SLAM & Navigation

RTAB-Map visual SLAM and Nav2 autonomous navigation for the XLeRobot omniwheel base,
using an Intel RealSense depth camera (ROS 2 Humble). Originally developed in
`fast_lio_nav2_humanoid`; the ROS package keeps the name `fast_lio_nav2_integration`.

## Layout

```
slam_nav/
├── fast_lio_nav2_integration/          # ROS 2 (ament_python) package
│   ├── launch/
│   │   ├── 0_lekiwi_mapping.launch.py          # RealSense + RTAB-Map mapping, depth-image grid, no filters
│   │   ├── 1_lekiwi_mapping_filtered.launch.py # same, with base_link->camera TF and RealSense filters
│   │   ├── 2_lekiwi_mapping_scan.launch.py     # mapping using a pseudo-LiDAR /scan for the 2D grid
│   │   ├── lekiwi_nav2.launch.py               # RTAB-Map localization + map_server + nav2_bringup
│   │   ├── test_lekiwi_nav2.launch.py          # localization + Nav2 + full XLeRobot cmd_vel bridge
│   │   └── test_lekiwi_nav2_wheels_only.launch.py  # same, base-only bridge + RViz (latest, most tested)
│   ├── config/
│   │   ├── lekiwi_nav2_config.yaml     # Nav2 params: MPPI controller, Smac Hybrid planner, costmaps
│   │   └── nav2_rviz.rviz
│   ├── scripts/
│   │   ├── xlerobot_cmd_vel_bridge.py              # /cmd_vel -> XLerobot (full robot via lerobot)
│   │   └── xlerobot_wheels_only_cmd_vel_bridge.py  # /cmd_vel -> 3 base wheel motors only (IDs 3-5)
│   └── fast_lio_nav2_integration/odom_quat_normalizer.py  # /odom_raw -> /odom, normalizes quaternions
├── tests/
│   ├── test_wheels_identify.py     # spin each wheel motor to map IDs to physical wheels
│   └── test_wheels_kinematics.py   # drive the base through moves without ROS
└── square_open_loop.sh             # open-loop square on /cmd_vel (odometry/drift check)
```

## Pipeline

1. **Sensing**: `realsense2_camera` publishes aligned RGB-D (640x480 @ 30 Hz) and the point cloud.
   A static TF places `camera_link` on `base_link` (`camera_tx/ty/tz/roll/pitch/yaw` args).
2. **Pseudo-LiDAR**: `pointcloud_to_laserscan` slices the cloud between 0.1 m and 1.5 m into `/scan`.
3. **SLAM (mapping)**: RTAB-Map with visual odometry builds `~/.ros/rtabmap.db` and a 2D occupancy grid.
4. **Localization**: RTAB-Map runs on the saved database in localization mode
   (`Mem/IncrementalMemory false`, `Reg/Force3DoF true`), publishing `map -> odom`.
   `/initialpose` is relayed to `/rtabmap/initialpose` so RViz "2D Pose Estimate" works.
5. **Navigation**: Nav2 (map_server, Smac Hybrid planner, MPPI controller, behaviors, BT navigator).
6. **Actuation**: the bridge converts `/cmd_vel` (m/s, rad/s) into Feetech STS3215 wheel velocities using
   3-omniwheel kinematics (wheels at 240°/0°/120°, wheel radius 0.05 m, base radius 0.125 m).
   It stops the wheels if no `/cmd_vel` arrives within 0.25 s.

## Usage

```bash
# build
cd ~/ros2_ws/src && ln -s /path/to/xlerobot-pro/slam_nav/fast_lio_nav2_integration .
cd ~/ros2_ws && colcon build --packages-select fast_lio_nav2_integration && source install/setup.bash

# 1) map the space (drive with teleop); the db is saved to ~/.ros/rtabmap.db
ros2 launch fast_lio_nav2_integration 1_lekiwi_mapping_filtered.launch.py
# export the 2D map for map_server
ros2 run nav2_map_server map_saver_cli -f ~/.ros/rtabmap

# 2) localize and navigate (base-only bridge, needs the lerobot conda env at ~/miniconda3/envs/lerobot)
ros2 launch fast_lio_nav2_integration test_lekiwi_nav2_wheels_only.launch.py \
    camera_tz:=0.25 camera_pitch:=0.0 \
    bridge_linear_gain:=1.0 bridge_angular_gain:=1.0
```

Defaults: map `~/.ros/rtabmap.yaml`, database `~/.ros/rtabmap.db`, base serial port `/dev/xle_base`.

Before trusting Nav2 on new hardware, run `tests/test_wheels_identify.py` to confirm motor IDs
and `tests/test_wheels_kinematics.py` to confirm the motion directions.

## Notes

- The RealSense point cloud is re-enabled 12 s after launch (`ros2 param set /camera/camera
  "pointcloud__neon_.enable" true`) as a workaround for a parameter name mismatch in the driver.
- The bridges import `lerobot`, so they must run with a Python interpreter that has both `lerobot` and `rclpy`.
- The footprint in `lekiwi_nav2_config.yaml` is still a 5 cm placeholder square; update it to the real base size.
