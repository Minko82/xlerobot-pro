from lerobot.robots.xlerobot import XLerobot, XLerobotConfig

config = XLerobotConfig(port1="/dev/tty.usbmodem5A680135181", use_degrees=True)
robot = XLerobot(config)
robot.connect()

# Read current state
state = robot.get_observation()

head_motor_1 = state["head_motor_1.pos"]
head_motor_2 = state["head_motor_2.pos"]

print({"head_motor_1": head_motor_1, "head_motor_2": head_motor_2})
robot.disconnect()
