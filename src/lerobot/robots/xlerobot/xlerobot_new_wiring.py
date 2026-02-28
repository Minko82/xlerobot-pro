#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============================================================================
# NEW WIRING DESIGN
# Bus 1 (port1 = /dev/xle_arms): Motors 1-6 = left arm, Motors 7-12 = right arm
# Bus 2 (port2 = /dev/xle_head): Motors 1-2 = head, Motors 3-5 = base wheels
# =============================================================================

import logging
import time
from functools import cached_property
from itertools import chain
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_xlerobot import XLerobotNewWiringConfig

logger = logging.getLogger(__name__)


class XLerobotNewWiring(Robot):
    """
    NEW WIRING DESIGN variant of XLerobot.

    Bus 1 (port1): left arm (IDs 1-6) + right arm (IDs 7-12)
    Bus 2 (port2): head (IDs 1-2) + base wheels (IDs 3-5)
    """

    config_class = XLerobotNewWiringConfig
    name = "xlerobot_new_wiring"

    def __init__(self, config: XLerobotNewWiringConfig):
        super().__init__(config)
        self.config = config
        self.teleop_keys = config.teleop_keys
        # Define three speed levels and a current index
        self.speed_levels = [
            {"xy": 0.1, "theta": 30},  # slow
            {"xy": 0.2, "theta": 60},  # medium
            {"xy": 0.3, "theta": 90},  # fast
        ]
        self.speed_index = 0  # Start at slow
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        # --- Bus 1 calibration: left arm + right arm ---
        if self.calibration.get("left_arm_shoulder_pan") is not None:
            calibration1 = {
                "left_arm_shoulder_pan": self.calibration.get("left_arm_shoulder_pan"),
                "left_arm_shoulder_lift": self.calibration.get("left_arm_shoulder_lift"),
                "left_arm_elbow_flex": self.calibration.get("left_arm_elbow_flex"),
                "left_arm_wrist_flex": self.calibration.get("left_arm_wrist_flex"),
                "left_arm_wrist_roll": self.calibration.get("left_arm_wrist_roll"),
                "left_arm_gripper": self.calibration.get("left_arm_gripper"),
                "right_arm_shoulder_pan": self.calibration.get("right_arm_shoulder_pan"),
                "right_arm_shoulder_lift": self.calibration.get("right_arm_shoulder_lift"),
                "right_arm_elbow_flex": self.calibration.get("right_arm_elbow_flex"),
                "right_arm_wrist_flex": self.calibration.get("right_arm_wrist_flex"),
                "right_arm_wrist_roll": self.calibration.get("right_arm_wrist_roll"),
                "right_arm_gripper": self.calibration.get("right_arm_gripper"),
            }
        else:
            calibration1 = self.calibration

        # Bus 1: left arm (IDs 1-6) + right arm (IDs 7-12)
        self.bus1 = FeetechMotorsBus(
            port=self.config.port1,
            motors={
                # left arm
                "left_arm_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "left_arm_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "left_arm_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "left_arm_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "left_arm_wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "left_arm_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
                # right arm
                "right_arm_shoulder_pan": Motor(7, "sts3215", norm_mode_body),
                "right_arm_shoulder_lift": Motor(8, "sts3215", norm_mode_body),
                "right_arm_elbow_flex": Motor(9, "sts3215", norm_mode_body),
                "right_arm_wrist_flex": Motor(10, "sts3215", norm_mode_body),
                "right_arm_wrist_roll": Motor(11, "sts3215", norm_mode_body),
                "right_arm_gripper": Motor(12, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=calibration1,
        )

        # --- Bus 2 calibration: head + base wheels ---
        if self.calibration.get("head_motor_1") is not None:
            calibration2 = {
                "head_motor_1": self.calibration.get("head_motor_1"),
                "head_motor_2": self.calibration.get("head_motor_2"),
                "base_left_wheel": self.calibration.get("base_left_wheel"),
                "base_back_wheel": self.calibration.get("base_back_wheel"),
                "base_right_wheel": self.calibration.get("base_right_wheel"),
            }
        else:
            calibration2 = self.calibration

        # Bus 2: head (IDs 1-2) + base wheels (IDs 3-5)
        self.bus2 = FeetechMotorsBus(
            port=self.config.port2,
            motors={
                # head
                "head_motor_1": Motor(1, "sts3215", norm_mode_body),
                "head_motor_2": Motor(2, "sts3215", norm_mode_body),
                # base wheels
                "base_left_wheel": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_back_wheel": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_right_wheel": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
            },
            calibration=calibration2,
        )

        self.left_arm_motors = [motor for motor in self.bus1.motors if motor.startswith("left_arm")]
        self.right_arm_motors = [motor for motor in self.bus1.motors if motor.startswith("right_arm")]
        self.head_motors = [motor for motor in self.bus2.motors if motor.startswith("head")]
        self.base_motors = [motor for motor in self.bus2.motors if motor.startswith("base")]
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(
            (
                "left_arm_shoulder_pan.pos",
                "left_arm_shoulder_lift.pos",
                "left_arm_elbow_flex.pos",
                "left_arm_wrist_flex.pos",
                "left_arm_wrist_roll.pos",
                "left_arm_gripper.pos",
                "right_arm_shoulder_pan.pos",
                "right_arm_shoulder_lift.pos",
                "right_arm_elbow_flex.pos",
                "right_arm_wrist_flex.pos",
                "right_arm_wrist_roll.pos",
                "right_arm_gripper.pos",
                "head_motor_1.pos",
                "head_motor_2.pos",
                "x.vel",
                "y.vel",
                "theta.vel",
            ),
            float,
        )

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        return self.bus1.is_connected and self.bus2.is_connected and all(
            cam.is_connected for cam in self.cameras.values()
        )

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus1.connect()
        self.bus2.connect()

        # Check if calibration file exists and ask user if they want to restore it
        if self.calibration_fpath.is_file():
            logger.info(f"Calibration file found at {self.calibration_fpath}")
            user_input = input(
                f"Press ENTER to restore calibration from file, or type 'c' and press ENTER to run manual calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info("Attempting to restore calibration from file...")
                try:
                    self.bus1.calibration = {k: v for k, v in self.calibration.items() if k in self.bus1.motors}
                    self.bus2.calibration = {k: v for k, v in self.calibration.items() if k in self.bus2.motors}
                    logger.info("Calibration data loaded into bus memory successfully!")

                    self.bus1.write_calibration({k: v for k, v in self.calibration.items() if k in self.bus1.motors})
                    self.bus2.write_calibration({k: v for k, v in self.calibration.items() if k in self.bus2.motors})
                    logger.info("Calibration restored successfully from file!")

                except Exception as e:
                    logger.warning(f"Failed to restore calibration from file: {e}")
                    if calibrate:
                        logger.info("Proceeding with manual calibration...")
                        self.calibrate()
            else:
                logger.info("User chose manual calibration...")
                if calibrate:
                    self.calibrate()
        elif calibrate:
            logger.info("No calibration file found, proceeding with manual calibration...")
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus1.is_calibrated and self.bus2.is_calibrated

    def calibrate(self) -> None:
        logger.info(f"\nRunning calibration of {self}")

        # --- Bus 1: left arm + right arm ---
        arm_motors = self.left_arm_motors + self.right_arm_motors
        self.bus1.disable_torque()
        for name in arm_motors:
            self.bus1.write("Operating_Mode", name, OperatingMode.POSITION.value)
        input(
            "Move left arm and right arm motors to the middle of their range of motion and press ENTER...."
        )
        homing_offsets = self.bus1.set_half_turn_homings(arm_motors)
        homing_offsets.update(dict.fromkeys(self.head_motors + self.base_motors, 0))

        print(
            f"Move all left arm and right arm joints sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus1.record_ranges_of_motion(arm_motors)

        calibration_bus1 = {}
        for name, motor in self.bus1.motors.items():
            calibration_bus1[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets[name],
                range_min=range_mins[name],
                range_max=range_maxes[name],
            )

        self.bus1.write_calibration(calibration_bus1)

        # --- Bus 2: head + base wheels ---
        self.bus2.disable_torque(self.head_motors)
        for name in self.head_motors:
            self.bus2.write("Operating_Mode", name, OperatingMode.POSITION.value)

        input(
            "Move head motors to the middle of their range of motion and press ENTER...."
        )

        homing_offsets_bus2 = self.bus2.set_half_turn_homings(self.head_motors)
        homing_offsets_bus2.update(dict.fromkeys(self.base_motors, 0))

        full_turn_motors = [motor for motor in self.base_motors if "wheel" in motor]
        unknown_range_motors = [motor for motor in self.head_motors]

        print(
            f"Move all head joints sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins_bus2, range_maxes_bus2 = self.bus2.record_ranges_of_motion(unknown_range_motors)
        for name in full_turn_motors:
            range_mins_bus2[name] = 0
            range_maxes_bus2[name] = 4095

        calibration_bus2 = {}
        for name, motor in self.bus2.motors.items():
            calibration_bus2[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets_bus2[name],
                range_min=range_mins_bus2[name],
                range_max=range_maxes_bus2[name],
            )

        self.bus2.write_calibration(calibration_bus2)
        self.calibration = {**calibration_bus1, **calibration_bus2}
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self):
        self.bus1.disable_torque()
        self.bus2.disable_torque()
        self.bus1.configure_motors()
        self.bus2.configure_motors()

        # Bus 1: left arm (position mode)
        for name in self.left_arm_motors:
            self.bus1.write("Operating_Mode", name, OperatingMode.POSITION.value)
            self.bus1.write("P_Coefficient", name, 16)
            self.bus1.write("I_Coefficient", name, 0)
            self.bus1.write("D_Coefficient", name, 43)

        # Bus 1: right arm (position mode)
        for name in self.right_arm_motors:
            self.bus1.write("Operating_Mode", name, OperatingMode.POSITION.value)
            self.bus1.write("P_Coefficient", name, 16)
            self.bus1.write("I_Coefficient", name, 0)
            self.bus1.write("D_Coefficient", name, 43)

        # Bus 2: head (position mode)
        for name in self.head_motors:
            self.bus2.write("Operating_Mode", name, OperatingMode.POSITION.value)
            self.bus2.write("P_Coefficient", name, 16)
            self.bus2.write("I_Coefficient", name, 0)
            self.bus2.write("D_Coefficient", name, 43)

        # Bus 2: base wheels (velocity mode)
        for name in self.base_motors:
            self.bus2.write("Operating_Mode", name, OperatingMode.VELOCITY.value)

        self.bus1.enable_torque()
        self.bus2.enable_torque()

    def setup_motors(self) -> None:
        # Bus 1: left arm then right arm
        for motor in chain(reversed(self.left_arm_motors), reversed(self.right_arm_motors)):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus1.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus1.motors[motor].id}")

        # Bus 2: head then base wheels
        for motor in chain(reversed(self.head_motors), reversed(self.base_motors)):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus2.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus2.motors[motor].id}")

    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        speed_in_steps = degps * steps_per_deg
        speed_int = int(round(speed_in_steps))
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF
        elif speed_int < -0x8000:
            speed_int = -0x8000
        return speed_int

    @staticmethod
    def _raw_to_degps(raw_speed: int) -> float:
        steps_per_deg = 4096.0 / 360.0
        degps = raw_speed / steps_per_deg
        return degps

    def _body_to_wheel_raw(
        self,
        x: float,
        y: float,
        theta: float,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
        max_raw: int = 3000,
    ) -> dict:
        theta_rad = theta * (np.pi / 180.0)
        velocity_vector = np.array([x, y, theta_rad])
        angles = np.radians(np.array([240, 0, 120]) - 90)
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])
        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)
        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]
        max_raw_computed = max(raw_floats)
        if max_raw_computed > max_raw:
            scale = max_raw / max_raw_computed
            wheel_degps = wheel_degps * scale
        wheel_raw = [self._degps_to_raw(deg) for deg in wheel_degps]
        return {
            "base_left_wheel": wheel_raw[0],
            "base_back_wheel": wheel_raw[1],
            "base_right_wheel": wheel_raw[2],
        }

    def _wheel_raw_to_body(
        self,
        left_wheel_speed,
        back_wheel_speed,
        right_wheel_speed,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
    ) -> dict[str, Any]:
        wheel_degps = np.array(
            [
                self._raw_to_degps(left_wheel_speed),
                self._raw_to_degps(back_wheel_speed),
                self._raw_to_degps(right_wheel_speed),
            ]
        )
        wheel_radps = wheel_degps * (np.pi / 180.0)
        wheel_linear_speeds = wheel_radps * wheel_radius
        angles = np.radians(np.array([240, 0, 120]) - 90)
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])
        m_inv = np.linalg.inv(m)
        velocity_vector = m_inv.dot(wheel_linear_speeds)
        x, y, theta_rad = velocity_vector
        theta = theta_rad * (180.0 / np.pi)
        return {
            "x.vel": x,
            "y.vel": y,
            "theta.vel": theta,
        }

    def _from_keyboard_to_base_action(self, pressed_keys: np.ndarray):
        if self.teleop_keys["speed_up"] in pressed_keys:
            self.speed_index = min(self.speed_index + 1, 2)
        if self.teleop_keys["speed_down"] in pressed_keys:
            self.speed_index = max(self.speed_index - 1, 0)
        speed_setting = self.speed_levels[self.speed_index]
        xy_speed = speed_setting["xy"]
        theta_speed = speed_setting["theta"]

        x_cmd = 0.0
        y_cmd = 0.0
        theta_cmd = 0.0

        if self.teleop_keys["forward"] in pressed_keys:
            x_cmd += xy_speed
        if self.teleop_keys["backward"] in pressed_keys:
            x_cmd -= xy_speed
        if self.teleop_keys["left"] in pressed_keys:
            y_cmd += xy_speed
        if self.teleop_keys["right"] in pressed_keys:
            y_cmd -= xy_speed
        if self.teleop_keys["rotate_left"] in pressed_keys:
            theta_cmd += theta_speed
        if self.teleop_keys["rotate_right"] in pressed_keys:
            theta_cmd -= theta_speed

        return {
            "x.vel": x_cmd,
            "y.vel": y_cmd,
            "theta.vel": theta_cmd,
        }

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        # Both arms on bus1
        left_arm_pos = self.bus1.sync_read("Present_Position", self.left_arm_motors)
        right_arm_pos = self.bus1.sync_read("Present_Position", self.right_arm_motors)
        # Head and base wheels on bus2
        head_pos = self.bus2.sync_read("Present_Position", self.head_motors)
        base_wheel_vel = self.bus2.sync_read("Present_Velocity", self.base_motors)

        base_vel = self._wheel_raw_to_body(
            base_wheel_vel["base_left_wheel"],
            base_wheel_vel["base_back_wheel"],
            base_wheel_vel["base_right_wheel"],
        )

        left_arm_state = {f"{k}.pos": v for k, v in left_arm_pos.items()}
        right_arm_state = {f"{k}.pos": v for k, v in right_arm_pos.items()}
        head_state = {f"{k}.pos": v for k, v in head_pos.items()}
        obs_dict = {**left_arm_state, **right_arm_state, **head_state, **base_vel}

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        left_arm_pos = {k: v for k, v in action.items() if k.startswith("left_arm_") and k.endswith(".pos")}
        right_arm_pos = {k: v for k, v in action.items() if k.startswith("right_arm_") and k.endswith(".pos")}
        head_pos = {k: v for k, v in action.items() if k.startswith("head_") and k.endswith(".pos")}
        base_goal_vel = {k: v for k, v in action.items() if k.endswith(".vel")}
        base_wheel_goal_vel = self._body_to_wheel_raw(
            base_goal_vel.get("x.vel", 0.0),
            base_goal_vel.get("y.vel", 0.0),
            base_goal_vel.get("theta.vel", 0.0),
        )

        if self.config.max_relative_target is not None:
            # Both arms on bus1; head on bus2
            present_pos_left = self.bus1.sync_read("Present_Position", self.left_arm_motors)
            present_pos_right = self.bus1.sync_read("Present_Position", self.right_arm_motors)
            present_pos_head = self.bus2.sync_read("Present_Position", self.head_motors)

            present_pos = {**present_pos_left, **present_pos_right, **present_pos_head}

            goal_present_pos = {
                key: (g_pos, present_pos[key])
                for key, g_pos in chain(left_arm_pos.items(), right_arm_pos.items(), head_pos.items())
            }
            safe_goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

            left_arm_pos = {k: v for k, v in safe_goal_pos.items() if k in left_arm_pos}
            right_arm_pos = {k: v for k, v in safe_goal_pos.items() if k in right_arm_pos}
            head_pos = {k: v for k, v in safe_goal_pos.items() if k in head_pos}

        left_arm_pos_raw = {k.replace(".pos", ""): v for k, v in left_arm_pos.items()}
        right_arm_pos_raw = {k.replace(".pos", ""): v for k, v in right_arm_pos.items()}
        head_pos_raw = {k.replace(".pos", ""): v for k, v in head_pos.items()}

        # Both arms write to bus1
        if left_arm_pos_raw:
            self.bus1.sync_write("Goal_Position", left_arm_pos_raw)
        if right_arm_pos_raw:
            self.bus1.sync_write("Goal_Position", right_arm_pos_raw)
        # Head and base write to bus2
        if head_pos_raw:
            self.bus2.sync_write("Goal_Position", head_pos_raw)
        if base_wheel_goal_vel:
            self.bus2.sync_write("Goal_Velocity", base_wheel_goal_vel)

        return {
            **left_arm_pos,
            **right_arm_pos,
            **head_pos,
            **base_goal_vel,
        }

    def stop_base(self):
        self.bus2.sync_write("Goal_Velocity", dict.fromkeys(self.base_motors, 0), num_retry=5)
        logger.info("Base motors stopped")

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.stop_base()
        self.bus1.disconnect(self.config.disable_torque_on_disconnect)
        self.bus2.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
