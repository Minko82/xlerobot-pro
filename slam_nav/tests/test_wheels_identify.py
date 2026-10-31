#!/usr/bin/env python3
"""
Motor ID identification script for the XLeRobot 3-omniwheel base.

Spins each motor one at a time so you can see which physical wheel
corresponds to which ID. Run this before trusting any kinematics.

Usage:
    python test_wheels_identify.py
    python test_wheels_identify.py --port /dev/ttyACM1
    python test_wheels_identify.py --ids 1 2 3   # if motors are IDs 1,2,3
"""
import argparse
import time
import numpy as np

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

SPEED = 500          # raw ticks/s — slow enough to be safe
DURATION = 1.5       # seconds each motor spins


def spin_one(bus, motor_name: str, speed: int, duration: float):
    print(f"  Spinning motor '{motor_name}' at raw speed {speed} ...")
    bus.sync_write("Goal_Velocity", {motor_name: speed})
    time.sleep(duration)
    bus.sync_write("Goal_Velocity", {motor_name: 0})
    time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--ids", nargs=3, type=int, default=[3, 4, 5],
                        metavar=("ID1", "ID2", "ID3"),
                        help="Motor IDs to test (default: 7 8 9). "
                             "Try '1 2 3' if the defaults produce no movement.")
    args = parser.parse_args()

    id1, id2, id3 = args.ids
    motor_names = [f"wheel_{id1}", f"wheel_{id2}", f"wheel_{id3}"]

    print(f"\nConnecting on {args.port} with motor IDs {args.ids} ...")
    bus = FeetechMotorsBus(
        port=args.port,
        motors={
            motor_names[0]: Motor(id1, "sts3215", MotorNormMode.RANGE_M100_100),
            motor_names[1]: Motor(id2, "sts3215", MotorNormMode.RANGE_M100_100),
            motor_names[2]: Motor(id3, "sts3215", MotorNormMode.RANGE_M100_100),
        },
    )
    bus.connect()
    bus.configure_motors()
    for name in motor_names:
        bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value)
    bus.enable_torque()
    print("Connected.\n")

    try:
        for name in motor_names:
            input(f"Press ENTER to spin motor ID {name.split('_')[1]} ...")
            spin_one(bus, name, SPEED, DURATION)
            answer = input(f"  Which wheel moved? (left / right / back / none): ").strip().lower()
            print(f"  → {name} = {answer}\n")

        print("Identification complete. Note the mapping above and update")
        print("the bridge script motor IDs accordingly.")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        bus.sync_write("Goal_Velocity", dict.fromkeys(motor_names, 0), num_retry=5)
        bus.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
