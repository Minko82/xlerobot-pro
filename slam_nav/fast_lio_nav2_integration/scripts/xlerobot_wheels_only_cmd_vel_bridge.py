#!/usr/bin/env python3
"""
cmd_vel bridge for the XLeRobot 3-omniwheel base only.

Connects directly to Bus 2 (/dev/ttyACM1) with only the 3 wheel motors
(IDs 3-5), completely bypassing the arms and head. Use this when the robot
is in an 'incomplete' state: only the base (3 wheels + motors) is assembled.

Drop-in replacement for xlerobot_cmd_vel_bridge.py with the same ROS
parameters so the launch file needs only one argument change.
"""
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode


# ---------------------------------------------------------------------------
# Omniwheel kinematics (identical to XLerobot._body_to_wheel_raw / _degps_to_raw)
# ---------------------------------------------------------------------------

def _degps_to_raw(degps: float) -> int:
    steps_per_deg = 4096.0 / 360.0
    speed_int = int(round(degps * steps_per_deg))
    return max(-0x8000, min(0x7FFF, speed_int))


def _body_to_wheel_raw(
    x: float,
    y: float,
    theta: float,
    wheel_radius: float = 0.05,
    base_radius: float = 0.125,
    max_raw: int = 3000,
) -> dict:
    """Convert body-frame velocities (m/s, m/s, deg/s) to raw wheel commands."""
    theta_rad = theta * (np.pi / 180.0)
    velocity_vector = np.array([x, y, theta_rad])

    angles = np.radians(np.array([240, 0, 120]))
    m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

    wheel_linear_speeds = m.dot(velocity_vector)
    wheel_angular_speeds = wheel_linear_speeds / wheel_radius
    wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

    steps_per_deg = 4096.0 / 360.0
    raw_floats = [abs(d) * steps_per_deg for d in wheel_degps]
    max_raw_computed = max(raw_floats)
    if max_raw_computed > max_raw:
        wheel_degps = wheel_degps * (max_raw / max_raw_computed)

    wheel_raw = [_degps_to_raw(d) for d in wheel_degps]
    return {
        "base_left_wheel": wheel_raw[0],
        "base_back_wheel": wheel_raw[1],
        "base_right_wheel": wheel_raw[2],
    }


# ---------------------------------------------------------------------------
# ROS 2 node
# ---------------------------------------------------------------------------

class WheelsOnlyCmdVelBridge(Node):
    WHEEL_MOTORS = ["base_left_wheel", "base_back_wheel", "base_right_wheel"]

    def __init__(self):
        super().__init__("xlerobot_wheels_only_bridge")

        # Declare parameters (same names as the original bridge where applicable)
        self.declare_parameter("port", "/dev/ttyACM0")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("send_rate_hz", 30.0)
        self.declare_parameter("timeout_s", 0.25)
        self.declare_parameter("max_linear_mps", 0.6)
        self.declare_parameter("max_angular_rps", 1.8)
        self.declare_parameter("linear_gain", 1.0)
        self.declare_parameter("angular_gain", 1.0)
        self.declare_parameter("invert_linear", False)

        port = self.get_parameter("port").value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value

        self._last_twist = Twist()
        self._last_twist_time = 0.0

        # Connect to Bus 2 with ONLY the 3 wheel motors (IDs 3, 4, 5).
        # Other motors on the same physical bus are simply not listed here,
        # so they are never pinged or commanded.
        self.bus = FeetechMotorsBus(
            port=port,
            motors={
                "base_left_wheel":  Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),  # ID 4 = left
                "base_back_wheel":  Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),  # ID 5 = back
                "base_right_wheel": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),  # ID 3 = right
            },
        )

        self.bus.connect()
        self.get_logger().info(f"Connected to wheel motors on {port}")

        # Configure wheels for velocity mode (mirrors XLerobot.configure())
        self.bus.configure_motors()
        for name in self.WHEEL_MOTORS:
            self.bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value)
        self.bus.enable_torque()

        self.get_logger().info("Wheel motors set to VELOCITY mode, torque enabled")

        self.sub = self.create_subscription(Twist, cmd_vel_topic, self._on_twist, 10)

        period = 1.0 / float(self.get_parameter("send_rate_hz").value)
        self.timer = self.create_timer(period, self._tick)

    # ------------------------------------------------------------------

    def _on_twist(self, msg: Twist):
        self._last_twist = msg
        self._last_twist_time = time.time()

    def _tick(self):
        now = time.time()
        timeout_s = float(self.get_parameter("timeout_s").value)

        if now - self._last_twist_time > timeout_s:
            lin, ang = 0.0, 0.0
        else:
            lin = float(self._last_twist.linear.x)
            ang = float(self._last_twist.angular.z)

        max_lin = float(self.get_parameter("max_linear_mps").value)
        max_ang = float(self.get_parameter("max_angular_rps").value)
        lin = max(-max_lin, min(max_lin, lin))
        ang = max(-max_ang, min(max_ang, ang))

        linear_gain  = float(self.get_parameter("linear_gain").value)
        angular_gain = float(self.get_parameter("angular_gain").value)
        invert_linear = bool(self.get_parameter("invert_linear").value)

        x_cmd = lin * linear_gain
        if invert_linear:
            x_cmd = -x_cmd

        # Twist angular.z is rad/s; kinematics expect deg/s
        theta_cmd = (ang * 180.0 / math.pi) * angular_gain

        # ROS linear.x (forward) maps to the kinematics y-axis.
        # The kinematics x-axis points physically to the robot's right.
        wheel_raw = _body_to_wheel_raw(0.0, x_cmd, theta_cmd)
        self.bus.sync_write("Goal_Velocity", wheel_raw)

    # ------------------------------------------------------------------

    def destroy_node(self):
        try:
            self.bus.sync_write(
                "Goal_Velocity",
                dict.fromkeys(self.WHEEL_MOTORS, 0),
                num_retry=5,
            )
            self.bus.disconnect()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = WheelsOnlyCmdVelBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
