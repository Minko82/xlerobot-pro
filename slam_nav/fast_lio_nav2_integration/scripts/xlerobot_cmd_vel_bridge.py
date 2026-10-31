#!/usr/bin/env python3
import time
import numpy as np
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

from lerobot.robots.xlerobot import XLerobotConfig, XLerobot


def _infer_key_from_examples(a_pos: dict, a_neg: dict):
    """
    Find a single key that exists in both dicts and has opposite sign.
    Returns (key, scale) where scale converts 'Twist units' to action units.
    """
    common = set(a_pos.keys()) & set(a_neg.keys())
    for k in common:
        vp = float(a_pos[k])
        vn = float(a_neg[k])
        if vp == 0.0 or vn == 0.0:
            continue
        if np.sign(vp) == -np.sign(vn):
            # If pressing forward produces +0.1, then 1.0 m/s should map to 0.1/??.
            # We treat vp as "unit command magnitude" and scale relative to Twist input.
            # So action_value = twist_value * (vp / 1.0)
            return k, vp
    return None, None


class XLeRobotCmdVelBridge(Node):
    def __init__(self):
        super().__init__("xlerobot_cmd_vel_bridge")

        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("send_rate_hz", 30.0)
        self.declare_parameter("timeout_s", 0.25)
        self.declare_parameter("max_linear_mps", 0.6)
        self.declare_parameter("max_angular_rps", 1.8)
        self.declare_parameter("linear_gain", 1.0)
        self.declare_parameter("angular_gain", 1.0)
        self.declare_parameter("invert_linear", False)

        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value

        self._last_twist = Twist()
        self._last_twist_time = 0.0

        # --- START FIX: Bypass Calibration Prompt ---
        import builtins
        original_input = builtins.input
        # Determine what to return. The prompt says "Press ENTER... or type 'c'".
        # We return "" to simulate pressing ENTER.
        builtins.input = lambda prompt="": "" 
        self.get_logger().info("Bypassing calibration prompt via input mocking...")
        # --- END FIX ---

        # Connect robot
        cfg = XLerobotConfig()
        self.robot = XLerobot(cfg)
        self.robot.connect()
        self.get_logger().info("Connected to XLeRobot")

        builtins.input = original_input # Restore original input function

        # Infer action keys for linear / angular using the same key scheme as your teleop :contentReference[oaicite:8]{index=8}
        fwd = self.robot._from_keyboard_to_base_action(np.array(["k"])) or {}
        back = self.robot._from_keyboard_to_base_action(np.array(["i"])) or {}
        left = self.robot._from_keyboard_to_base_action(np.array(["u"])) or {}
        right = self.robot._from_keyboard_to_base_action(np.array(["o"])) or {}

        self.linear_key, self.linear_unit = _infer_key_from_examples(fwd, back)
        self.angular_key, self.angular_unit = _infer_key_from_examples(left, right)

        if not self.linear_key or not self.angular_key:
            raise RuntimeError(
                f"Failed to infer base keys. fwd={fwd}, back={back}, left={left}, right={right}"
            )

        self.get_logger().info(
            f"Inferred linear_key={self.linear_key} (unit={self.linear_unit}), "
            f"angular_key={self.angular_key} (unit={self.angular_unit})"
        )

        self.sub = self.create_subscription(Twist, cmd_vel_topic, self._on_twist, 10)

        period = 1.0 / float(self.get_parameter("send_rate_hz").value)
        self.timer = self.create_timer(period, self._tick)

    def _on_twist(self, msg: Twist):
        self._last_twist = msg
        self._last_twist_time = time.time()

    def _tick(self):
        now = time.time()
        timeout_s = float(self.get_parameter("timeout_s").value)

        if now - self._last_twist_time > timeout_s:
            lin = 0.0
            ang = 0.0
        else:
            lin = float(self._last_twist.linear.x)
            ang = float(self._last_twist.angular.z)

        # Clamp
        max_lin = float(self.get_parameter("max_linear_mps").value)
        max_ang = float(self.get_parameter("max_angular_rps").value)
        lin = max(-max_lin, min(max_lin, lin))
        ang = max(-max_ang, min(max_ang, ang))

        # Convert Twist (m/s, rad/s) -> XLeRobot base action
        linear_gain = float(self.get_parameter('linear_gain').value)
        angular_gain = float(self.get_parameter('angular_gain').value)
        invert_linear = bool(self.get_parameter('invert_linear').value)

        x_cmd = lin * linear_gain
        if invert_linear:
            x_cmd = -x_cmd

        theta_cmd = (ang * 180.0 / math.pi) * angular_gain

        action = {
            self.linear_key: x_cmd,
            self.angular_key: theta_cmd,
        }

        self.robot.send_action(action)

    def destroy_node(self):
        try:
            self.robot.disconnect()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = XLeRobotCmdVelBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()