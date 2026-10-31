"""
Relay node: subscribes to /odom_raw, normalizes the quaternion, republishes to /odom.
Drops messages where the quaternion is zero (uninitialized).
"""
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry

_BEST_EFFORT_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


class OdomQuatNormalizer(Node):
    def __init__(self):
        super().__init__('odom_quat_normalizer')
        self.sub = self.create_subscription(Odometry, 'odom_raw', self._cb, _BEST_EFFORT_QOS)
        self.pub = self.create_publisher(Odometry, 'odom', _BEST_EFFORT_QOS)

    def _cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        norm2 = q.x**2 + q.y**2 + q.z**2 + q.w**2
        if norm2 < 1e-6:
            return  # drop uninitialized zero quaternion
        if abs(norm2 - 1.0) > 1e-6:
            scale = 1.0 / math.sqrt(norm2)
            msg.pose.pose.orientation.x *= scale
            msg.pose.pose.orientation.y *= scale
            msg.pose.pose.orientation.z *= scale
            msg.pose.pose.orientation.w *= scale
        self.pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(OdomQuatNormalizer())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
