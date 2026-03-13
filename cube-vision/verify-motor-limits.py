from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig

# Connect to robot
config = SO100FollowerConfig(port="/dev/ttyACM1", use_degrees=True)
robot = SO100Follower(config)
robot.connect()

BUS_AB_MAX_ACCELERATION = 40
BUS_AB_MAX_TORQUE = 800
BUS_AB_MAX_VELOCITY = 100

# Set the values
for motor_name in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]:
    robot.bus.write("Acceleration", motor_name, BUS_AB_MAX_ACCELERATION)

for motor_name in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]:
    robot.bus.write("Max_Torque_Limit", motor_name, BUS_AB_MAX_TORQUE)

for motor_name in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]:
    robot.bus.write("Maximum_Velocity_Limit", motor_name, BUS_AB_MAX_VELOCITY)

# Now read back and verify
print("\n" + "=" * 60)
print("VERIFYING MOTOR LIMITS")
print("=" * 60)

for motor_name in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]:
    print(f"\n{motor_name.upper()}:")

    # Read SRAM values
    accel = robot.bus.read("Acceleration", motor_name, normalize=False)
    print(f"  Acceleration (SRAM):           {accel:>3} (expected: {BUS_AB_MAX_ACCELERATION})")

    # Read EPROM values
    max_accel = robot.bus.read("Maximum_Acceleration", motor_name, normalize=False)
    print(f"  Maximum_Acceleration (EPROM):  {max_accel:>3} (should be >= {BUS_AB_MAX_ACCELERATION})")

    max_torque = robot.bus.read("Max_Torque_Limit", motor_name, normalize=False)
    expected_torque = BUS_AB_MAX_TORQUE if motor_name != "gripper" else 500
    print(f"  Max_Torque_Limit (EPROM):      {max_torque:>4} (expected: {expected_torque})")

    max_vel = robot.bus.read("Maximum_Velocity_Limit", motor_name, normalize=False)
    print(f"  Maximum_Velocity_Limit (EPROM): {max_vel:>3} (expected: {BUS_AB_MAX_VELOCITY})")

    # Check what the motor will actually use
    effective_accel = min(accel, max_accel)
    print(f"  → Effective Acceleration:       {effective_accel:>3}")

    if accel != BUS_AB_MAX_ACCELERATION:
        print(f"  ⚠️  WARNING: Acceleration mismatch!")
    if max_torque != expected_torque:
        print(f"  ⚠️  WARNING: Max_Torque_Limit mismatch!")
    if max_vel != BUS_AB_MAX_VELOCITY:
        print(f"  ⚠️  WARNING: Maximum_Velocity_Limit mismatch!")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)

robot.disconnect()
