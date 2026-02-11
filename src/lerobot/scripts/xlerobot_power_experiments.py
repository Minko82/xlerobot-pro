import argparse
import signal
import sys
import time
import traceback

from lerobot.motors import MotorCalibration
from lerobot.motors.feetech import OperatingMode
from lerobot.robots.xlerobot import XLerobot, XLerobotConfig


def cleanup(robot: XLerobot):
    print("\nUnlocking arm and disconnecting...")
    try:
        if robot.bus1.is_connected:
            robot.bus1.disconnect(robot.config.disable_torque_on_disconnect)
    except Exception:
        pass

    # In case the script is run against dual-arm hardware in the future.
    try:
        if robot.bus2.is_connected:
            robot.bus2.disconnect(robot.config.disable_torque_on_disconnect)
    except Exception:
        pass

    print("Done. Arm is free.")


def apply_limits(robot: XLerobot, torque: int, acceleration: int, p: int, i: int, d: int):
    bus = robot.bus1
    motors = list(bus.motors.keys())

    print("Applying motor limits:")
    print(f"    Torque_Limit = {torque} / 1000")
    print(f"    Acceleration = {acceleration} / 254")
    print(f"    P={p}, I={i}, D={d}\n")

    bus.disable_torque()
    for name in motors:
        bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
        bus.write("Torque_Limit", name, torque)
        bus.write("Acceleration", name, acceleration)
        bus.write("P_Coefficient", name, p)
        bus.write("I_Coefficient", name, i)
        bus.write("D_Coefficient", name, d)
    bus.enable_torque()


def calibrate_bus1(robot: XLerobot):
    bus = robot.bus1
    motors = list(bus.motors.keys())

    print("\nRunning bus1 calibration...")
    bus.disable_torque()
    for name in motors:
        bus.write("Operating_Mode", name, OperatingMode.POSITION.value)

    input("Move bus1 joints to the middle of their range of motion and press ENTER....")
    homing_offsets = bus.set_half_turn_homings(motors)

    print(
        "Move all bus1 joints sequentially through their entire ranges of motion.\n"
        "Recording positions. Press ENTER to stop..."
    )
    range_mins, range_maxes = bus.record_ranges_of_motion(motors)

    calibration_bus1 = {}
    for name, motor in bus.motors.items():
        calibration_bus1[name] = MotorCalibration(
            id=motor.id,
            drive_mode=0,
            homing_offset=homing_offsets[name],
            range_min=range_mins[name],
            range_max=range_maxes[name],
        )

    bus.write_calibration(calibration_bus1)
    robot.calibration = {**{k: v for k, v in robot.calibration.items() if k not in bus.motors}, **calibration_bus1}
    robot._save_calibration()
    print("Calibration saved to", robot.calibration_fpath)


def maybe_restore_or_calibrate_bus1(robot: XLerobot):
    bus = robot.bus1
    calibration_bus1 = {k: v for k, v in robot.calibration.items() if k in bus.motors}

    if calibration_bus1:
        user_input = input(
            f"Calibration found for robot id '{robot.id}'. Press ENTER to restore it, "
            "or type 'c' and press ENTER to recalibrate bus1: "
        )
        if user_input.strip().lower() != "c":
            bus.calibration = calibration_bus1
            bus.write_calibration(calibration_bus1)
            print("Bus1 calibration restored from file.\n")
            return

        calibrate_bus1(robot)
        print()
        return

    user_input = input(
        f"No bus1 calibration found at {robot.calibration_fpath}. "
        "Type 'c' and press ENTER to run calibration now, or any other key to abort: "
    )
    if user_input.strip().lower() != "c":
        raise RuntimeError("Calibration required for safe testing. Aborting.")
    calibrate_bus1(robot)
    print()


def step_response_test(robot: XLerobot, step_steps: int, hold_time: float):
    bus = robot.bus1
    motors = list(bus.motors.keys())

    positions = bus.sync_read("Present_Position", motors)
    print("Current positions (raw):")
    for name, pos in positions.items():
        print(f"    {name:16s} = {pos}")
    print()

    print(f"Step response test: moving each joint ±{step_steps} steps(~{step_steps * 360 / 4096:.1f}deg)")
    print("Observe smooth joint motions")

    for name in motors:
        home = positions[name]

        target_fwd = home + step_steps
        print(f"    {name}: {home} -> {target_fwd} (forward)")
        bus.write("Goal_Position", name, target_fwd)
        time.sleep(hold_time)

        target_bwd = home - step_steps
        print(f"    {name}: {target_fwd} -> {target_bwd} (backward)")
        bus.write("Goal_Position", name, target_bwd)
        time.sleep(hold_time)

        print(f"    {name}: {target_bwd} -> {home} (home)")
        bus.write("Goal_Position", name, home)
        time.sleep(hold_time)
        print()

    print("Present_Load after step resposne test:")
    loads = bus.sync_read("Present_Load", motors)
    for name, load in loads.items():
        print(f"    {name:16s} load = {load}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Motor limit test with configurable torque limit, acceleration, and PID"
    )
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port (e.g., /dev/ttyACM0)")
    parser.add_argument("--torque", type=int, default=200, help="Torque limit (0-1000)")
    parser.add_argument("--acceleration", type=int, default=10, help="Acceleration (0-254)")
    parser.add_argument("--p", type=int, default=8, help="P coefficient")
    parser.add_argument("--i", type=int, default=0, help="I coefficient")
    parser.add_argument("--d", type=int, default=32, help="D coefficient")
    parser.add_argument("--step-steps", type=int, default=40, help="Step size (raw steps)")
    parser.add_argument("--hold-time", type=float, default=1.5, help="Hold time (seconds)")
    return parser.parse_args()


def main():
    args = parse_args()

    config = XLerobotConfig(port1=args.port)
    robot = XLerobot(config)

    # <C-c> handler
    def signal_handler(sig, frame):
        cleanup(robot)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # step 1: connect bus1 only and configure motors
        print(f"Connecting bus1 on {args.port}...")
        robot.bus1.connect()
        maybe_restore_or_calibrate_bus1(robot)
        robot.bus1.disable_torque()
        robot.bus1.configure_motors()
        robot.bus1.enable_torque()
        print("Bus1 connected and configured!\n")

        # step 2: override with test limits
        apply_limits(robot, args.torque, args.acceleration, args.p, args.i, args.d)
        print("Torque enabled with test limits.\n")

        # step 3: step response test
        step_response_test(robot, args.step_steps, args.hold_time)

        cleanup(robot)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        cleanup(robot)


if __name__ == "__main__":
    main()