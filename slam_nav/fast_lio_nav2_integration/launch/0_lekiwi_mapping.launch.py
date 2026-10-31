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
            
            # --- FILTERS DISABLED (Matches your working command line) ---
            'decimation_filter.enable': 'false', 
            'spatial_filter.enable': 'false',     
            'temporal_filter.enable': 'false',    
            'hole_filling_filter.enable': 'false'
        }.items()
    )

    # --- 2. RTAB-Map (VISUAL MAPPING MODE) ---
    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('rtabmap_launch'), 'launch', 'rtabmap.launch.py')
        ),
        launch_arguments={
            # Ultra-Clean Config (Matched to your Command Line)
            'rtabmap_args': '--delete_db_on_start '
                            '--Grid/Sensor 1 '                  # Use Depth Image
                            '--Grid/RayTracing true '           
                            '--Grid/NormalsSegmentation true '  
                            '--Grid/MaxGroundAngle 45 '         
                            '--Grid/MinClusterSize 20 '         
                            '--Grid/NoiseFilteringRadius 0.05 ' 
                            '--Grid/NoiseFilteringMinNeighbors 5 ' 
                            '--Grid/RangeMax 3.5',              
            
            'rgb_topic': '/camera/camera/color/image_raw',
            'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/camera/color/camera_info',
            
            # CHANGED: Use camera_link to match your working command line
            'frame_id': 'camera_link',
            
            # Visual Odometry
            'visual_odometry': 'true',
            'odom_topic': '/odom',
            'odom_frame_id': 'odom',
            
            'approx_sync': 'true',
            'approx_sync_max_interval': '0.05',
            
            # CHANGED: Matched to your command line (Best Effort)
            'qos': '2',
            
            'subscribe_scan': 'false', 
            'wait_imu_to_init': 'false'
        }.items()
    )

    return LaunchDescription([
        realsense_launch,
        rtabmap_launch
    ])