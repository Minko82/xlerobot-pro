from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
import numpy as np
from ik_solver import IK_SO101
import time

# Connect to robot
config = SO101FollowerConfig(port="/dev/ttyACM0", use_degrees=True)
robot = SO101Follower(config)
robot.connect()

# Limits and restrictions
BUS_AB_MAX_ACCELERATION = 40
BUS_AB_MAX_TORQUE = 800
BUS_AB_MAX_VELOCITY = 100

# Disabling torque before writing to EPROM
for motor_name in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]:
    robot.bus.disable_torque(motor_name)

# Maximum_Acceleration 0-254 - EPROM (permanently written to motor, persistent)
for motor_name in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]:
    robot.bus.write("Maximum_Acceleration", motor_name, BUS_AB_MAX_ACCELERATION)
# Acceleration is non-persistent SRAM version

# Max_Torque_Limit 0-1000 - EPROM (permanently written to motor, persistent)
for motor_name in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]:
    robot.bus.write("Max_Torque_Limit", motor_name, BUS_AB_MAX_TORQUE)
# Torque_Limit is non-persistent SRAM version

# Maximum_Velocity_Limit 0-254 - EPROM (permanently written to motor, persistent)
for motor_name in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]:
    robot.bus.write("Maximum_Velocity_Limit", motor_name, BUS_AB_MAX_VELOCITY)
# Goal_Velocity is non-persistent SRAM version

# Verify EPROM values were set correctly
print("\n" + "=" * 60)
print("VERIFYING EPROM VALUES")
print("=" * 60)
for motor_name in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]:
    max_accel = robot.bus.read("Maximum_Acceleration", motor_name, normalize=False)
    max_vel = robot.bus.read("Maximum_Velocity_Limit", motor_name, normalize=False)
    max_torque = robot.bus.read("Max_Torque_Limit", motor_name, normalize=False)

    print(f"\n{motor_name}:")
    print(f"  Maximum_Acceleration:   {max_accel:>3} (expected: {BUS_AB_MAX_ACCELERATION})")
    print(f"  Maximum_Velocity_Limit: {max_vel:>3} (expected: {BUS_AB_MAX_VELOCITY})")

    expected_torque = BUS_AB_MAX_TORQUE if motor_name != "gripper" else 500
    print(f"  Max_Torque_Limit:       {max_torque:>4} (expected: {expected_torque})")

    # Check for mismatches
    if max_accel != BUS_AB_MAX_ACCELERATION:
        print(f"WARNING: Maximum_Acceleration mismatch!")
    if max_vel != BUS_AB_MAX_VELOCITY:
        print(f"WARNING: Maximum_Velocity_Limit mismatch!")
    if max_torque != expected_torque:
        print(f"WARNING: Max_Torque_Limit mismatch!")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE - Re-enabling torque")
print("=" * 60 + "\n")

# Re-enable torque after EPROM writes
for motor_name in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]:
    robot.bus.enable_torque(motor_name)

ik_solve = IK_SO101()

dt = 0.01
test_dt = 0.1

trajectory_rad = ik_solve.generate_ik([0.35, 0.0, 0.0], [-0.05, -0.01, -0.0808])
# default position tolerance of 1e-3. timesteps at 500
# Move individual joints (degrees)
RAD2DEG = 180.0 / np.pi
traj_rad_stack = np.stack(trajectory_rad)
trajectory = traj_rad_stack * RAD2DEG

ARM_JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

for step in trajectory:
    print(step)


def traj_to_action(q_deg: np.ndarray) -> dict:
    # Convert list of values to dict for lerobot usage
    assert q_deg.shape[0] == len(ARM_JOINT_KEYS)

    return {joint: float(q_deg[i]) for i, joint in enumerate(ARM_JOINT_KEYS)}


actions = [traj_to_action(q_deg) for q_deg in trajectory]

for action in actions:
    action["gripper.pos"] = 100.0
    robot.send_action(action)
    time.sleep(dt)


hold_action = {k: v for k, v in actions[-1].items() if k != "gripper.pos"}
for grip in range(100, 5, -5):
    action = dict(hold_action)
    action["gripper.pos"] = float(grip)
    robot.send_action(action)
    time.sleep(0.05)

trajectory_rad = ik_solve.generate_ik([0.30, 0.0, 0.10], [-0.05, -0.01, -0.0808])
traj_rad_stack = np.stack(trajectory_rad)
trajectory = traj_rad_stack * RAD2DEG


actions = [traj_to_action(q_deg) for q_deg in trajectory]

for action in actions:
    robot.send_action(action)
    time.sleep(dt)


# Movement 2
trajectory_rad = ik_solve.generate_ik([0.10, 0.0, 0.0], [-0.05, -0.01, -0.0808])
traj_rad_stack = np.stack(trajectory_rad)
trajectory = traj_rad_stack * RAD2DEG


actions = [traj_to_action(q_deg) for q_deg in trajectory]

for action in actions:
    robot.send_action(action)
    time.sleep(dt)

hold_action = {k: v for k, v in actions[-1].items() if k != "gripper.pos"}

for grip in range(5, 100, 5):
    action = dict(hold_action)
    action["gripper.pos"] = float(grip)
    robot.send_action(action)
    time.sleep(0.05)

robot.disconnect()
