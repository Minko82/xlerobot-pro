from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorNormMode

bus = FeetechMotorsBus(
    port="/dev/ttyACM1",
    motors={
        "head_motor_1": Motor(1, "sts3215", MotorNormMode.DEGREES),
        "head_motor_2": Motor(2, "sts3215", MotorNormMode.DEGREES),
    },
)

try:
    bus.connect()
    print("Connected to /dev/ttyACM1")
    pos = bus.sync_read("Present_Position", ["head_motor_1", "head_motor_2"])
    print(f"  head_motor_1 raw pos: {pos['head_motor_1']}")
    print(f"  head_motor_2 raw pos: {pos['head_motor_2']}")
except Exception as e:
    print(f"Failed to connect to /dev/ttyACM1: {e}")
finally:
    if bus.is_connected:
        try:
            bus.disconnect()
        except Exception:
            pass
