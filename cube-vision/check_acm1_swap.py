from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorNormMode

# BOARD SWAP TEST: ACM1 now has arm motors 7-12
bus = FeetechMotorsBus(
    port="/dev/ttyACM1",
    motors={
        "shoulder_pan":  Motor(7,  "sts3215", MotorNormMode.DEGREES),
        "shoulder_lift": Motor(8,  "sts3215", MotorNormMode.DEGREES),
        "elbow_flex":    Motor(9,  "sts3215", MotorNormMode.DEGREES),
        "wrist_flex":    Motor(10, "sts3215", MotorNormMode.DEGREES),
        "wrist_roll":    Motor(11, "sts3215", MotorNormMode.DEGREES),
        "gripper":       Motor(12, "sts3215", MotorNormMode.DEGREES),
    },
)

try:
    bus.connect()
    print("Connected to /dev/ttyACM1 (swap test - arm motors 7-12)")
    pos = bus.sync_read("Present_Position", list(bus.motors.keys()))
    for name, val in pos.items():
        print(f"  {name} (ID {bus.motors[name].id}) raw pos: {val}")
except Exception as e:
    print(f"Failed to connect to /dev/ttyACM1: {e}")
finally:
    if bus.is_connected:
        bus.disconnect()
