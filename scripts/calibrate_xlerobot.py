#!/usr/bin/env python3
"""Calibrate the whole robot (both arms, head, and wheels) in one run.

``lerobot-calibrate`` cannot load this robot: its ``--robot.type`` choices are
built before the XLeRobot classes register themselves, so ``xlerobot_new_wiring``
never appears. This script drives the robot class directly instead.

    python scripts/calibrate_xlerobot.py

The run prompts four times, in this order:

1. Move both arms to the middle of their range and press Enter.
2. Sweep every arm joint through its full travel, to each end stop, on both arms.
3. Center the head and press Enter.
4. The wheels are calibrated automatically as continuous-rotation motors.

The result is written to
``~/.cache/huggingface/lerobot/calibration/robots/xlerobot_new_wiring/xlerobot.json``.
Reference calibrations from a known-good robot are in ``calibration/`` at the
repo root, as a sanity check or fallback.
"""

from lerobot.robots.xlerobot.config_xlerobot import XLerobotNewWiringConfig
from lerobot.robots.xlerobot.xlerobot_new_wiring import XLerobotNewWiring

config = XLerobotNewWiringConfig(
    id="xlerobot",
    port1="/dev/xle_arms",  # Bus 1: left arm (IDs 1-6) and right arm (IDs 7-12)
    port2="/dev/xle_head",  # Bus 2: head (IDs 1-2) and wheels (IDs 3-5)
)
robot = XLerobotNewWiring(config)
robot.connect(calibrate=False)
try:
    robot.calibrate()
finally:
    robot.disconnect()
