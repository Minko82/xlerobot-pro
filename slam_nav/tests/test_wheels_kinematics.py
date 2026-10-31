#!/usr/bin/env python3
"""
Standalone kinematics test for the XLeRobot 3-omniwheel base.

Runs a sequence of movements WITHOUT ROS so you can verify the
kinematics is correct before using it with Nav2.

Usage (from the lerobot conda env):
    python test_wheels_kinematics.py
    python test_wheels_kinematics.py --port /dev/ttyACM0
"""
import argparse
import time
import numpy as np

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode


# ---------------------------------------------------------------------------
# Kinematics  (same as the bridge)
# ---------------------------------------------------------------------------

def _degps_to_raw(degps: float) -> int:
    steps_per_deg = 4096.0 / 360.0
    speed_int = int(round(degps * steps_per_deg))
    return max(-0x8000, min(0x7FFF, speed_int))


def _body_to_wheel_raw(x, y, theta,
                       wheel_radius=0.05, base_radius=0.125, max_raw=3000):
    """
    x     : forward velocity  (m/s)   — pass ROS linear.x here as y (see bridge)
    y     : lateral velocity  (m/s)
    theta : rotation          (deg/s)
    """
    theta_rad = theta * (np.pi / 180.0)
    velocity_vector = np.array([x, y, theta_rad])

    angles = np.radians(np.array([240, 0, 120]))
    m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

    wheel_linear_speeds = m.dot(velocity_vector)
    wheel_angular_speeds = wheel_linear_speeds / wheel_radius
    wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

    raw_floats = [abs(d) * 4096.0 / 360.0 for d in wheel_degps]
    if max(raw_floats) > max_raw:
        wheel_degps = wheel_degps * (max_raw / max(raw_floats))

    wheel_raw = [_degps_to_raw(d) for d in wheel_degps]
    return {
        "base_left_wheel":  wheel_raw[0],
        "base_back_wheel":  wheel_raw[1],
        "base_right_wheel": wheel_raw[2],
    }


WHEEL_MOTORS = ["base_left_wheel", "base_back_wheel", "base_right_wheel"]
STOP = dict.fromkeys(WHEEL_MOTORS, 0)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def move(bus, x=0.0, y=0.0, theta=0.0, duration=1.0, label=""):
    """Send a constant velocity command for `duration` seconds."""
    cmd = _body_to_wheel_raw(x, y, theta)
    print(f"  {label:30s}  x={x:+.2f} y={y:+.2f} θ={theta:+.1f}°/s  "
          f"→ raw {cmd}")
    bus.sync_write("Goal_Velocity", cmd)
    time.sleep(duration)
    bus.sync_write("Goal_Velocity", STOP)
    time.sleep(0.4)          # brief pause between moves


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0",
                        help="Serial port for wheel motors (default: /dev/ttyACM0)")
    parser.add_argument("--speed", type=float, default=0.15,
                        help="Linear speed in m/s (default: 0.15)")
    parser.add_argument("--turn", type=float, default=40.0,
                        help="Rotation speed in deg/s (default: 40)")
    parser.add_argument("--duration", type=float, default=1.5,
                        help="Duration of each move in seconds (default: 1.5)")
    args = parser.parse_args()

    print(f"\nConnecting to wheel motors on {args.port} ...")
    bus = FeetechMotorsBus(
        port=args.port,
        motors={
            "base_left_wheel":  Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),  # ID 8 = left
            "base_back_wheel":  Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),  # ID 9 = back
            "base_right_wheel": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),  # ID 7 = right
        },
    )
    bus.connect()
    bus.configure_motors()
    for name in WHEEL_MOTORS:
        bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value)
    bus.enable_torque()
    print("Connected. Starting test sequence.\n")

    v = args.speed
    w = args.turn
    d = args.duration

    try:
        # ---- NOTE on axis convention ----------------------------------------
        # In the bridge, ROS linear.x (forward) is passed as the *y* argument
        # to _body_to_wheel_raw because the kinematics x-axis points physically
        # to the robot's right.  The same convention is used here:
        #   forward/backward  →  y argument  (positive = forward)
        #   strafe left/right →  x argument  (positive = robot's physical right)
        #   rotation          →  theta arg   (positive = counter-clockwise)
        # ---------------------------------------------------------------------

        print("=== FORWARD / BACKWARD ===")
        move(bus, x=0,  y=+v, theta=0,  duration=d, label="FORWARD")
        move(bus, x=0,  y=-v, theta=0,  duration=d, label="BACKWARD")

        print("\n=== ROTATION ===")
        move(bus, x=0,  y=0,  theta=+w, duration=d, label="ROTATE CCW (left)")
        move(bus, x=0,  y=0,  theta=-w, duration=d, label="ROTATE CW  (right)")

        print("\n=== STRAFE ===")
        move(bus, x=+v, y=0,  theta=0,  duration=d, label="STRAFE RIGHT")
        move(bus, x=-v, y=0,  theta=0,  duration=d, label="STRAFE LEFT")

        print("\nTest sequence complete.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        bus.sync_write("Goal_Velocity", STOP, num_retry=5)
        bus.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
