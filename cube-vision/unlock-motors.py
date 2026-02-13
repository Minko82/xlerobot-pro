from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
import numpy as np
from ik_solver import IK_SO101
import time

# Connect to robot
config = SO101FollowerConfig(port="/dev/ttyACM0", use_degrees=True)
robot = SO101Follower(config)
robot.connect()
robot.disconnect()
