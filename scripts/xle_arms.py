#!/usr/bin/env python3
"""SO-101 follower bound to either arm of the bimanual robot.

Both arms share one bus: the left is motor IDs 1-6, the right 7-12. Stock
``SO101Follower`` hardcodes 1-6, so it can only ever drive the left one.

This subclasses it and rebuilds the bus with the right arm's IDs, keeping the
joint NAMES identical (``shoulder_pan`` ... ``gripper``). That matters more than
it looks: the dataset features, the policy's input and output layout, and the
deployment script all key off those names, so a policy trained on one arm has the
same interface as one trained on the other. Only the wire addresses differ.

Calibration files are keyed on the robot ``id``, not the class, so give the two
arms different ids (``left_follower`` / ``right_follower``) or they will overwrite
each other.
"""

from __future__ import annotations

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower

#: Wire addresses per arm. Names are deliberately identical between the two.
ARM_IDS = {
    "left": (1, 2, 3, 4, 5, 6),
    "right": (7, 8, 9, 10, 11, 12),
}

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


class SO101FollowerArm(SO101Follower):
    """SO101Follower whose motor IDs can be either arm's."""

    def __init__(self, config: SO101FollowerConfig, arm: str = "left"):
        super().__init__(config)
        if arm not in ARM_IDS:
            raise ValueError(f"arm must be one of {list(ARM_IDS)}, got {arm!r}")
        self.arm = arm
        if arm == "left":
            return  # the inherited bus is already correct

        norm = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        ids = ARM_IDS[arm]
        self.bus = FeetechMotorsBus(
            port=config.port,
            motors={
                name: Motor(ids[i], "sts3215",
                            MotorNormMode.RANGE_0_100 if name == "gripper" else norm)
                for i, name in enumerate(JOINTS)
            },
            calibration=self.calibration,
        )


def make_arm(port: str, robot_id: str, arm: str, cameras: dict | None = None) -> SO101FollowerArm:
    """Build a follower for the named arm."""
    cfg = SO101FollowerConfig(port=port, id=robot_id, cameras=cameras or {})
    return SO101FollowerArm(cfg, arm=arm)
