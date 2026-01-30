import time
import argparse
from multiprocessing import Value, Array, Lock
import threading
import logging_mp
logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from televuer import TeleVuerWrapper
from teleimager.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.ipc import IPC_Server
from sshkeyboard import listen_keyboard, stop_listening
from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig
import math

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: when READY is True, set RECORD_TOGGLE=True to transition.
#  --> auto  : Auto-transition after saving data.

def on_press(key):
    global STOP, START, RECORD_TOGGLE
    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and START == True:
        RECORD_TOGGLE = True
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
    }

# ADD this function (from SO101 code, lines 49-111):
def so101_inverse_kinematics(x, y, l1=0.1159, l2=0.1350):
    """
    Calculate inverse kinematics for SO101 2-link arm
    Returns shoulder_lift and elbow_flex angles in degrees
    """
    theta1_offset = math.atan2(0.028, 0.11257)
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset
    
    r = math.sqrt(x**2 + y**2)
    r_max = l1 + l2
    r_min = abs(l1 - l2)
    
    if r > r_max:
        scale_factor = r_max / r
        x *= scale_factor
        y *= scale_factor
        r = r_max
    
    if r < r_min and r > 0:
        scale_factor = r_min / r
        x *= scale_factor
        y *= scale_factor
        r = r_min
    
    cos_theta2 = -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_theta2 = max(-1, min(1, cos_theta2))  # Clamp for numerical stability
    theta2 = math.pi - math.acos(cos_theta2)
    
    beta = math.atan2(y, x)
    gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = beta + gamma
    
    joint2 = theta1 + theta1_offset
    joint3 = theta2 + theta2_offset
    
    joint2 = max(-0.1, min(3.45, joint2))
    joint3 = max(-0.2, min(math.pi, joint3))
    
    joint2_deg = 90 - math.degrees(joint2)
    joint3_deg = math.degrees(joint3) - 90
    
    return joint2_deg, joint3_deg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1', 'SO101'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco'], help='Select end effector controller')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--affinity', action = 'store_true', help = 'Enable high priority and set CPU affinity mode')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--task-dir', type = str, default = './utils/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')
    # SO101
    parser.add_argument('--left-port', type=str, default='/dev/ttyACM0', help='Serial port for left SO101 arm')
    parser.add_argument('--right-port', type=str, default='/dev/ttyACM1', help='Serial port for right SO101 arm')
    parser.add_argument('--left-arm-id', type=str, default='left_arm', help='ID for left arm calibration file')
    parser.add_argument('--right-arm-id', type=str, default='right_arm', help='ID for right arm calibration file')

    args = parser.parse_args()
    logger_mp.info(f"args: {args}")

    try:
        left_arm_config = SO100FollowerConfig(port=args.left_port, id=args.left_arm_id)
        right_arm_config = SO100FollowerConfig(port=args.right_port, id=args.right_arm_id)

        left_arm_robot = SO100Follower(left_arm_config)
        right_arm_robot = SO100Follower(right_arm_config)

        logger_mp.info(f"Connecting left arm on {args.left_port}...")
        left_arm_robot.connect()
        logger_mp.info(f"Connecting right arm on {args.right_port}...")
        right_arm_robot.connect()

        robots = {'left': left_arm_robot, 'right': right_arm_robot}

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press,get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                                      kwargs={"on_press": on_press, "until": None, "sequential": False,}, 
                                                      daemon=True)
            listen_keyboard_thread.start()

        # image client
        img_client = ImageClient(host=args.img_server_ip)
        camera_config = img_client.get_cam_config()
        logger_mp.debug(f"Camera config: {camera_config}")
        xr_need_local_img = not (args.display_mode == 'pass-through' or camera_config['head_camera']['enable_webrtc'])

        # televuer_wrapper: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", 
                                     binocular=camera_config['head_camera']['binocular'],
                                     img_shape=camera_config['head_camera']['image_shape'],
                                     # maybe should decrease fps for better performance?
                                     # https://github.com/unitreerobotics/xr_teleoperate/issues/172
                                     # display_fps=camera_config['head_camera']['fps'] ? args.frequency? 30.0?
                                     display_mode=args.display_mode,
                                     zmq=camera_config['head_camera']['enable_zmq'],
                                     webrtc=camera_config['head_camera']['enable_webrtc'],
                                     webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
                                     )
        
        # # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        # if args.motion:
        #     if args.input_mode == "controller":
        #         loco_wrapper = LocoClientWrapper()
        # else:
        #     motion_switcher = MotionSwitcher()
        #     status, result = motion_switcher.Enter_Debug_Mode()
        #     logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")


        # affinity mode (if you dont know what it is, then you probably don't need it)
        if args.affinity:
            import psutil
            p = psutil.Process(os.getpid())
            p.cpu_affinity([0,1,2,3]) # Set CPU affinity to cores 0-3
            try:
                p.nice(-20)           # Set highest priority
                logger_mp.info("Set high priority successfully.")
            except psutil.AccessDenied:
                logger_mp.warning("Failed to set high priority. Please run as root.")
                
            for child in p.children(recursive=True):
                try:
                    logger_mp.info(f"Child process {child.pid} name: {child.name()}")
                    child.cpu_affinity([5,6])
                    child.nice(-20)
                except psutil.AccessDenied:
                    pass


        # record + headless / non-headless mode
        if args.record:
            recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),
                                     task_goal = args.task_goal,
                                     task_desc = args.task_desc,
                                     task_steps = args.task_steps,
                                     frequency = args.frequency, 
                                     rerun_log = not args.headless)

        logger_mp.info("----------------------------------------------------------------")
        logger_mp.info("🟢  Press [r] to start syncing the robot with your movements.")
        if args.record:
            logger_mp.info("🟡  Press [s] to START or SAVE recording (toggle cycle).")
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        READY = True                  # now ready to (1) enter START state
        while not START and not STOP: # wait for start or stop signal.
            time.sleep(0.033)
            if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                head_img, _ = img_client.get_head_frame()
                tv_wrapper.render_to_xr(head_img)

        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
        arm_ctrl.speed_gradual_max()
        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()
            # get image
            if camera_config['head_camera']['enable_zmq']:
                if args.record or xr_need_local_img:
                    head_img, head_img_fps = img_client.get_head_frame()
                if xr_need_local_img:
                    tv_wrapper.render_to_xr(head_img)
            if camera_config['left_wrist_camera']['enable_zmq']:
                if args.record:
                    left_wrist_img, _ = img_client.get_left_wrist_frame()
            if camera_config['right_wrist_camera']['enable_zmq']:
                if args.record:
                    right_wrist_img, _ = img_client.get_right_wrist_frame()

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if recorder.create_episode():
                        RECORD_RUNNING = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    recorder.save_episode()
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)

            # get xr's tele data
            tele_data = tv_wrapper.get_tele_data()
            if (args.ee == "dex3" or args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee == "dex1" and args.input_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_ctrl_triggerValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif args.ee == "dex1" and args.input_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass
            
            # # high level control
            # if args.input_mode == "controller" and args.motion:
            #     # quit teleoperate
            #     if tele_data.right_ctrl_aButton:
            #         START = False
            #         STOP = True
            #     # command robot to enter damping mode. soft emergency stop function
            #     if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
            #         loco_wrapper.Damp()
            #     # https://github.com/unitreerobotics/xr_teleoperate/issues/135, control, limit velocity to within 0.3
            #     loco_wrapper.Move(-tele_data.left_ctrl_thumbstickValue[1] * 0.3,
            #                       -tele_data.left_ctrl_thumbstickValue[0] * 0.3,
            #                       -tele_data.right_ctrl_thumbstickValue[0]* 0.3)

            # Get current joint positions from SO101 arms
            def get_so101_observation(robot):
                """Extract joint positions from SO101 observation"""
                obs = robot.get_observation()
                positions = {}
                for key, value in obs.items():
                    if key.endswith('.pos'):
                        motor_name = key.removesuffix('.pos')
                        positions[motor_name] = value
                return positions

            left_obs = get_so101_observation(robots['left'])
            right_obs = get_so101_observation(robots['right'])

            # Convert to arrays matching the joint order
            joint_order = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
            current_left_q = np.array([left_obs.get(j, 0.0) for j in joint_order])
            current_right_q = np.array([right_obs.get(j, 0.0) for j in joint_order])
            current_lr_arm_q = np.concatenate([current_left_q, current_right_q])  # [12] total

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            # Extract wrist positions from VR tracking data
            # tele_data.left_wrist_pose and right_wrist_pose are typically 4x4 transformation matrices
            # Extract the position (x, y for planar IK)

            def extract_xy_from_pose(wrist_pose):
                """Extract x, y coordinates from wrist pose for SO101 planar IK"""
                # Assuming wrist_pose is a 4x4 matrix, position is in the last column
                # You may need to transform coordinates based on your setup
                x = wrist_pose[0, 3]  # or appropriate index
                y = wrist_pose[2, 3]  # depth becomes y in the arm's plane
                return x, y

            # Solve IK for left arm
            left_x, left_y = extract_xy_from_pose(tele_data.left_wrist_pose)
            left_shoulder_lift, left_elbow_flex = so101_inverse_kinematics(left_x, left_y)

            # Solve IK for right arm  
            right_x, right_y = extract_xy_from_pose(tele_data.right_wrist_pose)
            right_shoulder_lift, right_elbow_flex = so101_inverse_kinematics(right_x, right_y)

            # Extract other DOFs from wrist orientation (you'll need to implement this based on your needs)
            # For now, using placeholder values
            left_shoulder_pan = 0.0   # Could map from wrist yaw
            left_wrist_flex = -left_shoulder_lift - left_elbow_flex  # Keep end-effector level
            left_wrist_roll = 0.0     # Could map from wrist roll
            left_gripper = tele_data.left_hand_pinchValue * 100 if hasattr(tele_data, 'left_hand_pinchValue') else 0.0

            right_shoulder_pan = 0.0
            right_wrist_flex = -right_shoulder_lift - right_elbow_flex
            right_wrist_roll = 0.0
            right_gripper = tele_data.right_hand_pinchValue * 100 if hasattr(tele_data, 'right_hand_pinchValue') else 0.0

            # Build action dictionaries for SO101
            left_action = {
                'shoulder_pan.pos': left_shoulder_pan,
                'shoulder_lift.pos': left_shoulder_lift,
                'elbow_flex.pos': left_elbow_flex,
                'wrist_flex.pos': left_wrist_flex,
                'wrist_roll.pos': left_wrist_roll,
                'gripper.pos': left_gripper,
            }

            right_action = {
                'shoulder_pan.pos': right_shoulder_pan,
                'shoulder_lift.pos': right_shoulder_lift,
                'elbow_flex.pos': right_elbow_flex,
                'wrist_flex.pos': right_wrist_flex,
                'wrist_roll.pos': right_wrist_roll,
                'gripper.pos': right_gripper,
            }

            # Send commands to SO101 arms
            robots['left'].send_action(left_action)
            robots['right'].send_action(right_action)

            # For recording compatibility
            sol_q = np.array([
                left_shoulder_pan, left_shoulder_lift, left_elbow_flex, 
                left_wrist_flex, left_wrist_roll, left_gripper,
                right_shoulder_pan, right_shoulder_lift, right_elbow_flex,
                right_wrist_flex, right_wrist_roll, right_gripper
            ])

            # record data
            if args.record:
                READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state

                left_ee_state = []
                right_ee_state = []
                left_hand_action = []
                right_hand_action = []
                current_body_state = []
                current_body_action = []

                # arm state and action
                left_arm_state  = current_lr_arm_q[:6]
                right_arm_state = current_lr_arm_q[-6:]
                left_arm_action = sol_q[:6]
                right_arm_action = sol_q[-6:]
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    if camera_config['head_camera']['binocular']:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img[:, :camera_config['head_camera']['image_shape'][1]//2]
                            colors[f"color_{1}"] = head_img[:, camera_config['head_camera']['image_shape'][1]//2:]
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{2}"] = left_wrist_img
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{3}"] = right_wrist_img
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    else:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{1}"] = left_wrist_img
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{2}"] = right_wrist_img
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   [],                          
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   [],                          
                            "torque": [],                         
                        },                        
                        "left_ee": {                                                                    
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                        "body": {
                            "qpos": current_body_state,
                        }, 
                    }
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],       
                        },                         
                        "left_ee": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body": {
                            "qpos": current_body_action,
                        }, 
                    }
                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        try:
            # Move SO101 arms to home/zero position
            def move_so101_to_home(robots, duration=3.0):
                """Move both SO101 arms to zero position"""
                zero_positions = {
                    'shoulder_pan.pos': 0.0,
                    'shoulder_lift.pos': 0.0,
                    'elbow_flex.pos': 0.0,
                    'wrist_flex.pos': 0.0,
                    'wrist_roll.pos': 0.0,
                    'gripper.pos': 0.0
                }
                
                control_freq = 50
                total_steps = int(duration * control_freq)
                kp = 0.5
                
                for step in range(total_steps):
                    for arm_name, robot in robots.items():
                        obs = robot.get_observation()
                        action = {}
                        for key, target in zero_positions.items():
                            joint = key.removesuffix('.pos')
                            current = obs.get(f"{joint}.pos", 0.0)
                            error = target - current
                            new_pos = current + kp * error
                            action[key] = new_pos
                        robot.send_action(action)
                    time.sleep(1.0 / control_freq)

            move_so101_to_home(robots)
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")
        
        try:
            for arm_name, robot in robots.items():
                logger_mp.info(f"Disconnecting {arm_name} arm...")
                robot.disconnect()
        except Exception as e:
            logger_mp.error(f"Failed to disconnect SO101 arms: {e}")

        try:
            if args.ipc:
                ipc_server.stop()
            else:
                stop_listening()
                listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        try:
            if args.sim:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")
        
        try:
            if args.record:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)