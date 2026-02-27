from lerobot.robots.xlerobot import XLerobot, XLerobotConfig

xlerobot_config = XLerobotConfig(port1="/dev/ttyACM0", use_degrees=True)
xlerobot = XLerobot(xlerobot_config)
xlerobot.bus1.connect()

xlerobot.bus1.enable_torque()
