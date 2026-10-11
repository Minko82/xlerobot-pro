"""Shared configuration for the XLeRobot-Pro vision/IK stack.

All paths and hardware defaults used across the package live here so that
scripts, diagnostics, and examples stay consistent.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Hardware defaults (override per-script with CLI flags where provided)
# ---------------------------------------------------------------------------

#: Bus 1: left arm (IDs 1-6) + right arm (IDs 7-12).
ARMS_PORT = os.getenv("XLEROBOT_ARMS_PORT", "/dev/xle_arms")

#: Bus 2: head (ID 1 = tilt, ID 2 = pan) + base wheels (IDs 3-5).
HEAD_PORT = os.getenv("XLEROBOT_HEAD_PORT", "/dev/xle_head")

#: Alias kept for scripts that only drive the arm bus.
BUS_PORT = os.getenv("XLEROBOT_BUS_PORT", ARMS_PORT)

# ---------------------------------------------------------------------------
# Model assets (shipped as package data)
# ---------------------------------------------------------------------------

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

#: Canonical MJCF model of the robot (used by FK, IK, and MuJoCo tooling).
MJCF_PATH = ASSETS_DIR / "xlerobot" / "xlerobot.xml"

#: URDF variants of the same robot.
URDF_PATH = ASSETS_DIR / "xlerobot" / "xlerobot.urdf"
FRONT_URDF_PATH = ASSETS_DIR / "xlerobot" / "xlerobot_front.urdf"

# ---------------------------------------------------------------------------
# Capture artifacts
# ---------------------------------------------------------------------------

# Where realsense.capture() writes RGBD frames and where color_detect reads them.
# These MUST agree: the writer used a path relative to the working directory while
# the reader used one relative to the package, so a capture run from anywhere other
# than the package directory silently left the reader looking at stale or absent
# files. Defining it once here removes the mismatch.
CAPTURE_DIR = Path(os.getenv("XLEROBOT_CAPTURE_DIR", "outputs")) / "realsense_capture"

# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

# This repository is used via an editable install, so the versioned
# calibration/ directory at the repo root is the default. Override with
# XLEROBOT_CALIBRATION_DIR for non-repo deployments.
_REPO_CALIBRATION = Path(__file__).resolve().parents[2] / "calibration"
CALIBRATION_DIR = Path(os.getenv("XLEROBOT_CALIBRATION_DIR", _REPO_CALIBRATION))

DEFAULT_CALIBRATION_FILE = CALIBRATION_DIR / "single_bus.json"
HEAD_CALIBRATION_FILE = CALIBRATION_DIR / "head.json"
