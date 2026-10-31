import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node

def generate_launch_description():
    # --- 1. Define Launch Configurations ---
    use_sim_time = LaunchConfiguration('use_sim_time')
    map_yaml_file = LaunchConfiguration('map_yaml_file')
    nav2_params_file = LaunchConfiguration('nav2_params_file')
    database_path = LaunchConfiguration('database_path')
    enable_controller = LaunchConfiguration('enable_controller') # <--- NEW
    enable_xlerobot_bridge = LaunchConfiguration('enable_xlerobot_bridge')
    xlerobot_bridge_script = LaunchConfiguration('xlerobot_bridge_script')
    bridge_linear_gain = LaunchConfiguration('bridge_linear_gain')
    bridge_angular_gain = LaunchConfiguration('bridge_angular_gain')
    bridge_invert_linear = LaunchConfiguration('bridge_invert_linear')
    
    # Frames
    base_frame = LaunchConfiguration('base_frame')
    camera_frame = LaunchConfiguration('camera_frame')
    odom_frame = LaunchConfiguration('odom_frame')
    map_frame = LaunchConfiguration('map_frame')
    
    # Topics
    scan_topic = LaunchConfiguration('scan_topic')
    rgb_topic = LaunchConfiguration('rgb_topic')
    depth_topic = LaunchConfiguration('depth_topic')
    camera_info_topic = LaunchConfiguration('camera_info_topic')
    
    # Camera Transform
    camera_tx = LaunchConfiguration('camera_tx')
    camera_ty = LaunchConfiguration('camera_ty')
    camera_tz = LaunchConfiguration('camera_tz')
    camera_roll = LaunchConfiguration('camera_roll')
    camera_pitch = LaunchConfiguration('camera_pitch')
    camera_yaw = LaunchConfiguration('camera_yaw')

    # --- 2. Declare Launch Arguments (Kept at top as requested) ---
    declare_enable_controller = DeclareLaunchArgument(
        'enable_controller', 
        default_value='true',
        description='Enable Nav2 Controller')

    declare_enable_xlerobot_bridge = DeclareLaunchArgument(
        'enable_xlerobot_bridge',
        default_value='true',
        description='Run xlerobot_cmd_vel_bridge (subscribes to /cmd_vel and commands the base)')

    declare_xlerobot_bridge_script = DeclareLaunchArgument(
        'xlerobot_bridge_script',
        default_value=os.path.join(get_package_share_directory('fast_lio_nav2_integration'), 'scripts', 'xlerobot_cmd_vel_bridge.py'),
        description='Path to xlerobot_cmd_vel_bridge.py')

    declare_bridge_linear_gain = DeclareLaunchArgument('bridge_linear_gain', default_value='1.0')
    declare_bridge_angular_gain = DeclareLaunchArgument('bridge_angular_gain', default_value='1.0')
    declare_bridge_invert_linear = DeclareLaunchArgument('bridge_invert_linear', default_value='false')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='false', description='Use simulation time')

    declare_map_yaml = DeclareLaunchArgument(
        'map_yaml_file', 
        default_value=os.path.expanduser('~/.ros/rtabmap.yaml'),
        description='Path to the 2D occupancy grid map (.yaml)')

    declare_nav2_params = DeclareLaunchArgument(
        'nav2_params_file', 
        default_value=os.path.join(get_package_share_directory('fast_lio_nav2_integration'), 'config', 'lekiwi_nav2_config.yaml'),
        description='Path to Nav2 parameters file')

    declare_database_path = DeclareLaunchArgument(
        'database_path', 
        default_value=os.path.expanduser('~/.ros/rtabmap.db'),
        description='Path to RTAB-Map database for localization')

    declare_base_frame = DeclareLaunchArgument(
        'base_frame', default_value='base_link', description='Robot base frame')
    declare_camera_frame = DeclareLaunchArgument(
        'camera_frame', default_value='camera_link', description='Camera frame')
    declare_odom_frame = DeclareLaunchArgument(
        'odom_frame', default_value='odom', description='Odometry frame')
    declare_map_frame = DeclareLaunchArgument(
        'map_frame', default_value='map', description='Map frame')

    declare_scan_topic = DeclareLaunchArgument(
        'scan_topic', default_value='/scan', description='LaserScan topic')
    declare_rgb_topic = DeclareLaunchArgument(
        'rgb_topic', default_value='/camera/camera/color/image_raw', description='RGB topic')
    declare_depth_topic = DeclareLaunchArgument(
        'depth_topic', default_value='/camera/camera/aligned_depth_to_color/image_raw', description='Aligned Depth topic')
    declare_camera_info_topic = DeclareLaunchArgument(
        'camera_info_topic', default_value='/camera/camera/color/camera_info', description='Camera Info topic')

    declare_camera_tx = DeclareLaunchArgument('camera_tx', default_value='0.0')
    declare_camera_ty = DeclareLaunchArgument('camera_ty', default_value='0.0')
    declare_camera_tz = DeclareLaunchArgument('camera_tz', default_value='0.0')
    declare_camera_roll = DeclareLaunchArgument('camera_roll', default_value='0.0')
    declare_camera_pitch = DeclareLaunchArgument('camera_pitch', default_value='0.0')
    declare_camera_yaw = DeclareLaunchArgument('camera_yaw', default_value='0.0')

    # --- 3. Define Nodes and Launches ---

    # A. RealSense Camera
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')
        ),
        launch_arguments={
            'initial_reset': 'true',
            'align_depth.enable': 'true',
            'publish_tf': 'true',
            'pointcloud.enable': 'true',
            'depth_module.depth_profile': '640x480x30',
            'rgb_camera.color_profile': '640x480x30',
            'enable_sync': 'true',
            'unite_imu_method': '2',
            'enable_gyro': 'true',
            'enable_accel': 'true',
            'decimation_filter.enable': 'true',
            'spatial_filter.enable': 'true',
            'temporal_filter.enable': 'true',
            'hole_filling_filter.enable': 'true'
        }.items()
    )

    # B. Static Transform
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_camera_broadcaster',
        arguments=[camera_tx, camera_ty, camera_tz, camera_roll, camera_pitch, camera_yaw, base_frame, camera_frame],
        output='screen'
    )

    # C. Pseudo-LiDAR
    pointcloud_to_laserscan = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        parameters=[{
            'target_frame': base_frame,
            'min_height': 0.1,
            'max_height': 1.5,
            'range_max': 4.0,
            'use_inf': True
        }],
        remappings=[
            ('cloud_in', '/camera/camera/depth/color/points'),
            ('scan', scan_topic)
        ]
    )

    # D. RTAB-Map (Localization)
    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('rtabmap_launch'), 'launch', 'rtabmap.launch.py')
        ),
        launch_arguments={
            'rtabmap_args': '--Mem/IncrementalMemory false --Loc/GlobalAliasingMaxNode 20',
            'database_path': database_path,
            'rgb_topic': rgb_topic,
            'depth_topic': depth_topic,
            'camera_info_topic': camera_info_topic,
            'frame_id': base_frame,
            'visual_odometry': 'true',
            'odom_topic': '/odom',
            'odom_frame_id': odom_frame,
            'map_frame_id': map_frame,
            'approx_sync': 'true',
            'approx_sync_max_interval': '0.05',
            'qos': '1', 
            'subscribe_scan': 'false', 
            'wait_imu_to_init': 'false'
        }.items()
    )

    # E. Nav2 Nodes
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time,'yaml_filename': map_yaml_file}]
    )

    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[nav2_params_file]
    )

    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[nav2_params_file],
        condition=IfCondition(enable_controller)
    )

    # F. XLeRobot cmd_vel -> base bridge (runs as a separate process)
    xlerobot_cmd_vel_bridge = ExecuteProcess(
        condition=IfCondition(enable_xlerobot_bridge),
        cmd=[
            'python3', xlerobot_bridge_script,
            '--ros-args',
            '-p', 'cmd_vel_topic:=/cmd_vel',
            '-p', PythonExpression(["'linear_gain:=' + '", bridge_linear_gain, "'"]),
            '-p', PythonExpression(["'angular_gain:=' + '", bridge_angular_gain, "'"]),
            '-p', PythonExpression(["'invert_linear:=' + '", bridge_invert_linear, "'"]),
        ],
        output='screen'
    )

    behaviors_server = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        output='screen',
        parameters=[nav2_params_file]
    )

    bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[nav2_params_file]
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time},
                    {'autostart': True},
                    {'node_names': ['map_server', 'planner_server', 'behavior_server', 'bt_navigator']}]
    )

    lifecycle_manager_controller = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_controller',
        output='screen',
        condition=IfCondition(enable_controller),
        parameters=[{'use_sim_time': use_sim_time},
                    {'autostart': True},
                    {'node_names': ['controller_server']}]
    )


    # --- 4. Build Launch Description ---
    ld = LaunchDescription()

    # Add Declarations
    ld.add_action(declare_enable_controller)
    ld.add_action(declare_enable_xlerobot_bridge)
    ld.add_action(declare_xlerobot_bridge_script)
    ld.add_action(declare_bridge_linear_gain)
    ld.add_action(declare_bridge_angular_gain)
    ld.add_action(declare_bridge_invert_linear)
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_map_yaml)
    ld.add_action(declare_nav2_params)
    ld.add_action(declare_database_path)
    ld.add_action(declare_base_frame)
    ld.add_action(declare_camera_frame)
    ld.add_action(declare_odom_frame)
    ld.add_action(declare_map_frame)
    ld.add_action(declare_scan_topic)
    ld.add_action(declare_rgb_topic)
    ld.add_action(declare_depth_topic)
    ld.add_action(declare_camera_info_topic)
    ld.add_action(declare_camera_tx)
    ld.add_action(declare_camera_ty)
    ld.add_action(declare_camera_tz)
    ld.add_action(declare_camera_roll)
    ld.add_action(declare_camera_pitch)
    ld.add_action(declare_camera_yaw)

    # Add Nodes
    ld.add_action(realsense_launch)
    ld.add_action(static_tf)
    ld.add_action(pointcloud_to_laserscan)
    ld.add_action(rtabmap_launch)
    ld.add_action(map_server)
    ld.add_action(planner_server)
    ld.add_action(controller_server)
    ld.add_action(xlerobot_cmd_vel_bridge)
    ld.add_action(behaviors_server)
    ld.add_action(bt_navigator)
    ld.add_action(lifecycle_manager)
    ld.add_action(lifecycle_manager_controller)

    return ld