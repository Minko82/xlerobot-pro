"""
Set the ID of a Feetech motor.

Connects to a motor on the given port, scans all baudrates to find it,
and reprograms it with the specified ID at the default baudrate (1 MHz).

Usage:
    python examples/set_motor_id.py --port /dev/tty.usbmodem... --id 3

    # For a non-sts3215 motor:
    python examples/set_motor_id.py --port /dev/tty.usbmodem... --id 3 --model scs0009

Make sure only ONE motor is connected when running this.
"""

import argparse

from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode


def main():
    parser = argparse.ArgumentParser(description="Set the ID of a Feetech motor.")
    parser.add_argument("--port", required=True, help="Serial port (e.g. /dev/tty.usbmodem575E0031751)")
    parser.add_argument("--id", required=True, type=int, help="Target motor ID to assign (1-253)")
    parser.add_argument(
        "--model",
        default="sts3215",
        choices=["sts3215", "sts3250", "scs0009", "sm8512bl", "sts_series", "scs_series", "sms_series"],
        help="Motor model (default: sts3215)",
    )
    args = parser.parse_args()

    if not (1 <= args.id <= 253):
        parser.error("--id must be between 1 and 253")

    bus = FeetechMotorsBus(
        port=args.port,
        motors={"motor": Motor(args.id, args.model, MotorNormMode.RANGE_M100_100)},
    )

    print(f"Scanning {args.port} for motor (model: {args.model})...")
    bus.setup_motor("motor")
    print(f"Done. Motor ID set to {args.id} at {bus.default_baudrate} bps.")


if __name__ == "__main__":
    main()
