#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""
Standalone camera frame server.

Opens hardware cameras once and publishes frames over ZMQ PUB so multiple
consumers (robot client, agent notebook, etc.) can share the same camera.

Usage:
    python -m lerobot.cameras.zmq.camera_server \
        --port 5555 --fps 30 \
        --cameras '{
            "base":  {"type": "realsense", "serial_number_or_name": "838212073725", "width": 640, "height": 480},
            "wrist": {"type": "opencv",    "index_or_path": "/dev/video0",           "width": 640, "height": 480}
        }'

Consumers connect as ZMQ SUB clients:
    ZMQCameraConfig(server_address="localhost", port=5555, camera_name="base")
"""

import argparse
import json
import logging

from lerobot.cameras.zmq.image_server import ImageServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Camera frame server — opens cameras once and publishes over ZMQ PUB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--port", type=int, default=5555, help="ZMQ PUB port (default: 5555)")
    parser.add_argument("--fps", type=int, default=30, help="Target publish rate (default: 30)")
    parser.add_argument(
        "--cameras",
        type=str,
        required=True,
        help=(
            "JSON dict of cameras to open. "
            'RealSense: {"type": "realsense", "serial_number_or_name": "...", "width": 640, "height": 480}. '
            'OpenCV:    {"type": "opencv",    "index_or_path": "/dev/video0", "width": 640, "height": 480}.'
        ),
    )
    args = parser.parse_args()

    cameras = json.loads(args.cameras)
    config = {"fps": args.fps, "cameras": cameras}
    ImageServer(config, port=args.port).run()


if __name__ == "__main__":
    main()
