import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    
    # --- 1. RealSense Camera ---
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')
        ),
        launch_arguments={
            'initial_reset': 'true',
            'align_depth.enable': 'true',        
            'publish_tf': 'true',                
            'pointcloud.enable': 'true',         
            
            # CPU Savers
            'depth_module.depth_profile': '640x480x30',
            'rgb_camera.color_profile': '640x480x30',
            
            # Sync & IMU
            'enable_sync': 'true',               
            'unite_imu_method': '2',             
            'enable_gyro': 'true',
            'enable_accel': 'true',
            
            # Filters (Crucial for Depth-to-Grid mapping)
            'decimation_filter.enable': 'true', 
            'spatial_filter.enable': 'true',     
            'temporal_filter.enable': 'true',    
            'hole_filling_filter.enable': 'true'
        }.items()
    )

    # --- 2. Static TF ---
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', 'base_link', 'camera_link']
    )

    # --- 3. RTAB-Map (VISUAL MAPPING MODE) ---
    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('rtabmap_launch'), 'launch', 'rtabmap.launch.py')
        ),
        launch_arguments={
            # Ultra-Clean Config
            'rtabmap_args': '--delete_db_on_start '
                            '--Grid/Sensor 1 '                  # Use Depth Image (not Laser)
                            '--Grid/RayTracing true '           # Clear empty space
                            '--Grid/NormalsSegmentation true '  # Detect flat ground
                            '--Grid/MaxGroundAngle 45 '         # Ignore ramps
                            '--Grid/MinClusterSize 20 '         # Filter speckles
                            '--Grid/NoiseFilteringRadius 0.05 ' # Filter isolated pixels
                            '--Grid/RangeMax 3.5',              # Cap range
            
            'rgb_topic': '/camera/camera/color/image_raw',
            'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/camera/color/camera_info',
            'frame_id': 'base_link',
            
            # Visual Odometry
            'visual_odometry': 'true',
            'odom_topic': '/odom',
            'odom_frame_id': 'odom',
            
            'approx_sync': 'true',
            'approx_sync_max_interval': '0.05',
            'qos': '1',
            
            # DISABLED for CPU Savings (We use Depth-to-Grid)
            'subscribe_scan': 'false', 
            
            'wait_imu_to_init': 'false'
        }.items()
    )

    return LaunchDescription([
        realsense_launch,
        static_tf,
        # pointcloud_to_laserscan is REMOVED
        rtabmap_launch
    ])