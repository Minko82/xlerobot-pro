from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig
import numpy as np
from ik_solver import IK_SO101
import time

# Connect to robot
config = SO100FollowerConfig(port="/dev/ttyACM1", use_degrees=True)
robot = SO100Follower(config)
robot.connect()

ik_solve = IK_SO101()

dt = 0.01
test_dt = 0.1

RAD2DEG = 180.0 / np.pi

ARM_JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

def traj_to_action(q_deg: np.ndarray) -> dict:
    # Convert list of values to dict for lerobot usage
    assert q_deg.shape[0] == len(ARM_JOINT_KEYS)

    return {joint: float(q_deg[i]) for i, joint in enumerate(ARM_JOINT_KEYS)}



trajectory_rad = ik_solve.generate_ik([0.30, 0.0, 0.05], [0.0, 0.0, 0.0])
traj_rad_stack = np.stack(trajectory_rad)
trajectory = traj_rad_stack * RAD2DEG

for step in trajectory:
    print(step)

actions = [traj_to_action(q_deg) for q_deg in trajectory]

for action in actions:
    robot.send_action(action)
    time.sleep(dt)
robot.disconnect()
