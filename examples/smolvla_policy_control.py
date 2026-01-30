#!/usr/bin/env python3
"""
Replay one episode from: lerobot/svla_so101_pickplace

- Connects to SO100Follower (same pattern as your controller).
- Optionally connects to top+wrist OpenCV cameras for live monitoring.
- Loads a HuggingFace dataset episode and replays stored actions step-by-step.

Run:
  python replay_svla_episode.py --episode 0 --robot_port /dev/ttyACM0
"""

import argparse
import logging
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# HF datasets (LeRobot datasets are hosted on the Hub)
from datasets import load_dataset

# Optional live monitoring (same camera stack you used)
import cv2
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

# Robot follower (same as your controller)
from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("replay")


# --- Copied structure from your controller ---
CAMERA_ID_TOP = 6
CAMERA_ID_WRIST = 8

ORDERED_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Same calibration table as your controller (used for optional inverse mapping on actions)
JOINT_CALIBRATION = [
    ["shoulder_pan", 6.0, 1.0],
    ["shoulder_lift", 2.0, 0.97],
    ["elbow_flex", 0.0, 1.05],
    ["wrist_flex", 0.0, 0.94],
    ["wrist_roll", 0.0, 0.5],
    ["gripper", 0.0, 1.0],
]


def _calib_params(joint_name: str) -> Tuple[float, float]:
    for name, offset, scale in JOINT_CALIBRATION:
        if name == joint_name:
            return float(offset), float(scale)
    return 0.0, 1.0


def inverse_joint_calibration(joint_name: str, calibrated_position: float) -> float:
    """Inverse of (raw - offset) * scale -> raw = calibrated/scale + offset"""
    offset, scale = _calib_params(joint_name)
    if scale == 0:
        return float(calibrated_position)  # should never happen
    return float(calibrated_position) / float(scale) + float(offset)


def move_to_zero_position(robot: SO100Follower, sleep_s: float = 2.0) -> None:
    logger.info("Moving to Zero Position...")
    zero_action = {f"{name}.pos": 0.0 for name in ORDERED_JOINTS}
    robot.send_action(zero_action)
    time.sleep(sleep_s)


def find_episode_key(column_names: List[str]) -> str:
    # Common conventions across HF / LeRobot datasets
    candidates = [
        "episode_index",
        "episode_id",
        "episode",
        "traj_id",
        "trajectory_id",
        "demo_id",
    ]
    for c in candidates:
        if c in column_names:
            return c

    # Fallback: look for something that smells like an episode column
    for c in column_names:
        lc = c.lower()
        if "episode" in lc and ("id" in lc or "index" in lc):
            return c

    raise KeyError(f"Couldn't find an episode column in: {column_names}")


def find_time_key(column_names: List[str]) -> Optional[str]:
    candidates = ["timestamp", "time", "t", "frame_time", "wall_time"]
    for c in candidates:
        if c in column_names:
            return c
    return None


def _maybe_get_nested(d: Any, path: str) -> Any:
    """Get nested dict fields like 'action/joints' or 'action.joints'."""
    if not isinstance(d, dict):
        return None
    if path in d:
        return d[path]
    # try split by '/' then '.'
    cur = d
    for sep in ["/", "."]:
        cur = d
        ok = True
        for part in path.split(sep):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            return cur
    return None


def extract_action_vector(row: Dict[str, Any], ordered_joints: List[str]) -> List[float]:
    """
    Tries hard to extract a 6D action vector in ORDERED_JOINTS order.

    Supported patterns:
      - row["action"] is list/np array
      - row["actions"] is list/np array
      - row["action"] is dict with joint keys like "shoulder_pan.pos" or "shoulder_pan"
      - per-joint columns like "action/shoulder_pan.pos" or "action.shoulder_pan.pos"
    """
    # 1) direct vector fields
    for k in ["action", "actions", "robot_action", "target_action"]:
        if k in row:
            a = row[k]
            if isinstance(a, (list, tuple, np.ndarray)):
                a = list(a)
                if len(a) >= len(ordered_joints):
                    return [float(a[i]) for i in range(len(ordered_joints))]
            if isinstance(a, dict):
                # 1a) dict keyed by joint name or joint.pos
                vec = []
                for j in ordered_joints:
                    if f"{j}.pos" in a:
                        vec.append(float(a[f"{j}.pos"]))
                    elif j in a:
                        vec.append(float(a[j]))
                    else:
                        # maybe nested dict under 'pos'
                        v = _maybe_get_nested(a, f"{j}.pos")
                        if v is None:
                            v = _maybe_get_nested(a, j)
                        if v is None:
                            raise KeyError(f"Action dict missing joint '{j}' keys. Keys={list(a.keys())[:30]}")
                        vec.append(float(v))
                return vec

    # 2) per-joint columns
    colnames = set(row.keys())
    vec = []
    for j in ordered_joints:
        per_joint_candidates = [
            f"action/{j}.pos",
            f"action.{j}.pos",
            f"actions/{j}.pos",
            f"actions.{j}.pos",
            f"{j}.pos_action",
            f"{j}_pos_action",
        ]
        found = None
        for c in per_joint_candidates:
            if c in colnames:
                found = c
                break
        if found is None:
            # last-resort: scan columns that end with this joint.pos and contain 'action'
            for c in colnames:
                if c.endswith(f"{j}.pos") and ("action" in c.lower() or "actions" in c.lower()):
                    found = c
                    break
        if found is None:
            raise KeyError(f"Couldn't locate per-joint action for '{j}'. Available keys: {sorted(list(colnames))[:40]} ...")
        vec.append(float(row[found]))
    return vec


def select_episode_rows(ds, episode_key: str, episode_ordinal: int):
    """
    Returns (episode_value, episode_ds).
    episode_ordinal picks the Nth unique episode value (sorted).
    """
    # Unique episode ids (works for int or string)
    ep_values = ds.unique(episode_key)
    try:
        ep_values = sorted(ep_values)
    except Exception:
        # if values aren't sortable, keep original order
        pass

    if episode_ordinal < 0 or episode_ordinal >= len(ep_values):
        raise IndexError(f"--episode {episode_ordinal} out of range. Dataset has {len(ep_values)} episodes.")

    ep_val = ep_values[episode_ordinal]

    # Filter rows for this episode (batched for speed)
    episode_ds = ds.filter(lambda x: x[episode_key] == ep_val, batched=False)
    return ep_val, episode_ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="lerobot/svla_so101_pickplace")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--episode", type=int, default=0, help="Nth episode (0-based) among unique episode ids.")
    ap.add_argument("--dt", type=float, default=0.1, help="Fallback control period if dataset has no timestamps.")
    ap.add_argument("--robot_port", type=str, default="/dev/ttyACM0")
    ap.add_argument("--no_robot", action="store_true", help="Don't connect/send actions; just print + optionally show cameras.")
    ap.add_argument("--no_cameras", action="store_false", help="Don't open live cameras.")
    ap.add_argument("--top_cam", type=int, default=CAMERA_ID_TOP)
    ap.add_argument("--wrist_cam", type=int, default=CAMERA_ID_WRIST)
    ap.add_argument("--inverse_calib_actions", action="store_false",
                    help="Apply inverse calibration before sending actions (calibrated->raw).")
    ap.add_argument("--dry_run", action="store_true", help="Print actions but don't send.")
    args = ap.parse_args()

    robot = None
    cam_top = None
    cam_wrist = None

    try:
        # 1) Load dataset
        logger.info(f"Loading dataset: {args.dataset} [{args.split}] ...")
        ds = load_dataset(args.dataset, split=args.split)

        episode_key = find_episode_key(ds.column_names)
        time_key = find_time_key(ds.column_names)

        ep_val, ep_ds = select_episode_rows(ds, episode_key, args.episode)
        n_steps = len(ep_ds)
        logger.info(f"Selected episode #{args.episode}: {episode_key}={ep_val} with {n_steps} steps.")

        # 2) Connect robot (optional)
        if not args.no_robot:
            logger.info("Connecting to robot...")
            robot_config = SO100FollowerConfig(port=args.robot_port, id="left_arm")
            robot = SO100Follower(robot_config)
            robot.connect()
            logger.info("✅ Robot connected.")

            if input("Recalibrate robot? (y/n): ").strip().lower() in ["y", "yes"]:
                robot.calibrate()
                logger.info("Calibration completed.")

            if input("Move to Zero Position? (y/n): ").strip().lower() in ["y", "yes"]:
                move_to_zero_position(robot)

        # 3) Open cameras for live monitoring (optional)
        if not args.no_cameras:
            logger.info("Connecting cameras...")
            cam_top = OpenCVCamera(OpenCVCameraConfig(index_or_path=args.top_cam, fps=30, width=640, height=480))
            cam_wrist = OpenCVCamera(OpenCVCameraConfig(index_or_path=args.wrist_cam, fps=30, width=640, height=480))
            cam_top.connect()
            cam_wrist.connect()
            logger.info(f"✅ Cameras connected (Top: {args.top_cam}, Wrist: {args.wrist_cam}).")

        input("Ready to replay this episode. Press Enter to start...")

        # 4) Replay loop
        prev_t = None
        for i in range(n_steps):
            row = ep_ds[i]

            # show live views for safety / monitoring
            if cam_top is not None and cam_wrist is not None:
                ft = cam_top.read()
                fw = cam_wrist.read()
                if ft is not None:
                    cv2.imshow("Top View (LIVE)", ft)
                if fw is not None:
                    cv2.imshow("Wrist View (LIVE)", fw)
                # press 'q' to abort
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.warning("Aborted by user (q).")
                    break

            action_vec = extract_action_vector(row, ORDERED_JOINTS)

            # optional inverse calibration
            if args.inverse_calib_actions:
                action_vec = [
                    inverse_joint_calibration(j, a) for j, a in zip(ORDERED_JOINTS, action_vec)
                ]

            logger.info(f"[{i+1:04d}/{n_steps:04d}] action={np.round(action_vec, 4).tolist()}")

            if (not args.no_robot) and (not args.dry_run) and robot is not None:
                robot_action = {f"{j}.pos": float(a) for j, a in zip(ORDERED_JOINTS, action_vec)}
                robot.send_action(robot_action)

            # timing
            dt = args.dt
            if time_key is not None:
                try:
                    t = float(row[time_key])
                    if prev_t is not None:
                        dt = max(0.0, min(0.5, t - prev_t))
                    prev_t = t
                except Exception:
                    pass
            time.sleep(dt)

        logger.info("Replay done.")

    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if robot is not None:
            try:
                if input("Return to zero at end? (y/n): ").strip().lower() in ["y", "yes"]:
                    move_to_zero_position(robot)
            except Exception:
                pass
            try:
                robot.disconnect()
            except Exception:
                pass
        if cam_top is not None:
            try:
                cam_top.disconnect()
            except Exception:
                pass
        if cam_wrist is not None:
            try:
                cam_wrist.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()
