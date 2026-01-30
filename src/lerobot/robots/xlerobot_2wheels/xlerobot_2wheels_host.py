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

import json
import logging
import time
from typing import Any

import numpy as np
import zmq

from lerobot.utils.robot_utils import busy_wait

from .xlerobot_2wheels import XLerobot2Wheels
from .config_xlerobot_2wheels import XLerobot2WheelsConfig, XLerobot2WheelsHostConfig

logger = logging.getLogger(__name__)


class XLerobot2WheelsHost:
    """
    Host for XLerobot2Wheels that runs on the robot hardware.
    Receives commands via ZMQ and sends them to the robot.
    Sends observations back via ZMQ.
    """

    def __init__(self, robot_config: XLerobot2WheelsConfig, host_config: XLerobot2WheelsHostConfig):
        self.robot_config = robot_config
        self.host_config = host_config
        
        self.robot = XLerobot2Wheels(robot_config)
        
        # ZMQ setup
        self.zmq_context = None
        self.zmq_cmd_socket = None
        self.zmq_observation_socket = None
        
        self._is_running = False
        self.last_cmd_time = time.time()

    def connect(self):
        """Connect to robot hardware and setup ZMQ sockets"""
        logger.info("Connecting to robot hardware...")
        self.robot.connect()
        
        logger.info("Setting up ZMQ sockets...")
        self.zmq_context = zmq.Context()
        
        # Command socket (PULL - receives commands)
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.bind(f"tcp://*:{self.host_config.port_zmq_cmd}")
        
        # Observation socket (PUSH - sends observations)
        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.bind(f"tcp://*:{self.host_config.port_zmq_observations}")
        
        logger.info(f"ZMQ sockets bound to ports {self.host_config.port_zmq_cmd} and {self.host_config.port_zmq_observations}")
        self._is_running = True

    def run(self):
        """Main control loop"""
        if not self._is_running:
            raise RuntimeError("Host not connected. Call connect() first.")
        
        logger.info("Starting XLerobot2Wheels host control loop...")
        start_time = time.time()
        
        try:
            while self._is_running:
                loop_start = time.time()
                
                # Check for commands with timeout
                if self.zmq_cmd_socket.poll(timeout=1):  # 1ms timeout
                    try:
                        cmd_string = self.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                        cmd = json.loads(cmd_string)
                        self._process_command(cmd)
                        self.last_cmd_time = time.time()
                    except zmq.Again:
                        pass  # No command available
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode command: {e}")
                
                # Check watchdog timeout
                if time.time() - self.last_cmd_time > self.host_config.watchdog_timeout_ms / 1000.0:
                    logger.warning("Watchdog timeout - stopping base motors")
                    self.robot.stop_base()
                    self.last_cmd_time = time.time()  # Reset to avoid spam
                
                # Get observation and send it
                try:
                    obs = self.robot.get_observation()
                    self._send_observation(obs)
                except Exception as e:
                    logger.error(f"Failed to get observation: {e}")
                
                # Check if we should stop
                if time.time() - start_time > self.host_config.connection_time_s:
                    logger.info("Connection time limit reached, stopping host")
                    break
                
                # Control loop frequency
                loop_duration = time.time() - loop_start
                target_dt = 1.0 / self.host_config.max_loop_freq_hz
                if loop_duration < target_dt:
                    busy_wait(target_dt - loop_duration)
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, stopping host")
        finally:
            self.stop()

    def _process_command(self, cmd: dict[str, Any]):
        """Process a received command"""
        try:
            # Send command to robot
            self.robot.send_action(cmd)
        except Exception as e:
            logger.error(f"Failed to process command: {e}")

    def _send_observation(self, obs: dict[str, Any]):
        """Send observation via ZMQ"""
        try:
            # Convert images to base64 for transmission
            obs_for_transmission = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray) and len(value.shape) == 3:  # Image
                    # Encode image as base64
                    import cv2
                    import base64
                    _, buffer = cv2.imencode('.jpg', value)
                    obs_for_transmission[key] = base64.b64encode(buffer).decode('utf-8')
                else:
                    obs_for_transmission[key] = value
            
            # Send observation
            obs_string = json.dumps(obs_for_transmission)
            self.zmq_observation_socket.send_string(obs_string)
            
        except Exception as e:
            logger.error(f"Failed to send observation: {e}")

    def stop(self):
        """Stop the host and disconnect"""
        logger.info("Stopping XLerobot2Wheels host...")
        self._is_running = False
        
        if self.robot.is_connected:
            self.robot.disconnect()
        
        if self.zmq_cmd_socket:
            self.zmq_cmd_socket.close()
        if self.zmq_observation_socket:
            self.zmq_observation_socket.close()
        if self.zmq_context:
            self.zmq_context.term()
        
        logger.info("XLerobot2Wheels host stopped")


def main():
    """Main function for running the host"""
    import argparse
    
    parser = argparse.ArgumentParser(description="XLerobot2Wheels Host")
    parser.add_argument("--robot.id", type=str, default="xlerobot_2wheels", help="Robot ID")
    parser.add_argument("--robot.port1", type=str, default="/dev/ttyACM0", help="Port 1")
    parser.add_argument("--robot.port2", type=str, default="/dev/ttyACM1", help="Port 2")
    parser.add_argument("--host.port_zmq_cmd", type=int, default=5555, help="ZMQ command port")
    parser.add_argument("--host.port_zmq_observations", type=int, default=5556, help="ZMQ observation port")
    parser.add_argument("--host.connection_time_s", type=int, default=3600, help="Connection time limit")
    parser.add_argument("--host.watchdog_timeout_ms", type=int, default=500, help="Watchdog timeout")
    parser.add_argument("--host.max_loop_freq_hz", type=int, default=30, help="Max loop frequency")
    
    args = parser.parse_args()
    
    # Create configs
    robot_config = XLerobot2WheelsConfig(
        id=args.robot.id,
        port1=args.robot.port1,
        port2=args.robot.port2,
    )
    
    host_config = XLerobot2WheelsHostConfig(
        port_zmq_cmd=args.host.port_zmq_cmd,
        port_zmq_observations=args.host.port_zmq_observations,
        connection_time_s=args.host.connection_time_s,
        watchdog_timeout_ms=args.host.watchdog_timeout_ms,
        max_loop_freq_hz=args.host.max_loop_freq_hz,
    )
    
    # Create and run host
    host = XLerobot2WheelsHost(robot_config, host_config)
    
    try:
        host.connect()
        host.run()
    except Exception as e:
        logger.error(f"Host error: {e}")
        host.stop()


if __name__ == "__main__":
    main()
