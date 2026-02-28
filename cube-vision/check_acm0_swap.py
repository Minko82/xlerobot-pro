from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorNormMode

# BOARD SWAP TEST: ACM0 now has head motors 1-2
bus = FeetechMotorsBus(
    port="/dev/ttyACM0",
    motors={
        "head_motor_1": Motor(1, "sts3215", MotorNormMode.DEGREES),
        "head_motor_2": Motor(2, "sts3215", MotorNormMode.DEGREES),
    },
)

try:
    bus.connect()
    print("Connected to /dev/ttyACM0 (swap test - head motors 1-2)")
    pos = bus.sync_read("Present_Position", ["head_motor_1", "head_motor_2"])
    print(f"  head_motor_1 raw pos: {pos['head_motor_1']}")
    print(f"  head_motor_2 raw pos: {pos['head_motor_2']}")
except Exception as e:
    print(f"Failed to connect to /dev/ttyACM0: {e}")
finally:
    if bus.is_connected:
        try:
            bus.disconnect()
        except Exception:
            pass
