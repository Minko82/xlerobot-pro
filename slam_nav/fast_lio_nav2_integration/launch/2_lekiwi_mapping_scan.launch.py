import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    
    # --- 1. RealSense Camera (Data Source) ---
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')
        ),
        launch_arguments={
            'initial_reset': 'true',
            
            # --- Your Essential Arguments ---
            'align_depth.enable': 'true',        # Align RGB rows with Depth rows
            'publish_tf': 'true',                # Publish internal camera TF
            'pointcloud.enable': 'true',         # Needed for obstacle detection
            
            # --- Performance Profiles (Keep CPU usage low) ---
            'depth_module.depth_profile': '640x480x30',
            'rgb_camera.color_profile': '640x480x30',
            
            # --- Synchronization & IMU ---
            'enable_sync': 'true',               # Hardware sync RGB and Depth
            'unite_imu_method': '2',             # Merge Gyro/Accel (Optional, good for VSLAM)
            'enable_gyro': 'true',
            'enable_accel': 'true',
            
            # --- Filters (Clean up the noise) ---
            'decimation_filter.enable': 'true',  # Downsample to save CPU
            'spatial_filter.enable': 'true',     # Smooths noisy depth pixels
            'temporal_filter.enable': 'true',    # Stabilizes depth over time
            'hole_filling_filter.enable': 'true'
        }.items()
    )

    # --- 2. Static TF (Robot -> Camera) ---
    # NOTE: Update '0.0's with your actual mounting position in meters
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', 'base_link', 'camera_link']
    )

    # --- 3. Pseudo-LiDAR (For 2D Obstacle Map) ---
    pointcloud_to_laserscan = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        parameters=[{
            'target_frame': 'base_link',
            'min_height': 0.1,  # Ignore floor (points < 10cm up)
            'max_height': 1.5,  # Ignore ceiling (points > 1.5m up)
            'range_max': 4.0,   # Max reliable range
            'use_inf': True
        }],
        remappings=[
            ('cloud_in', '/camera/camera/depth/color/points'),
            ('scan', '/scan')
        ]
    )

    # --- 4. RTAB-Map (MAPPING MODE) ---
    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('rtabmap_launch'), 'launch', 'rtabmap.launch.py')
        ),
        launch_arguments={
            # Delete old DB to start fresh
            'rtabmap_args': '--delete_db_on_start --Grid/Sensor 0 --Grid/RangeMax 4.0',
            
            'rgb_topic': '/camera/camera/color/image_raw',
            'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/camera/color/camera_info',
            'frame_id': 'base_link',
            
            # Visual Odometry (Calculates movement)
            'visual_odometry': 'true',
            'odom_topic': '/odom',
            'odom_frame_id': 'odom',
            
            # Sync & QoS
            'approx_sync': 'true',
            'approx_sync_max_interval': '0.05',
            'qos': '1', # Reliable
            
            # We use the /scan topic to draw the 2D walls
            'subscribe_scan': 'true',
            'scan_topic': '/scan',
            
            'wait_imu_to_init': 'false'
        }.items()
    )

    return LaunchDescription([
        realsense_launch,
        static_tf,
        pointcloud_to_laserscan,
        rtabmap_launch
    ])