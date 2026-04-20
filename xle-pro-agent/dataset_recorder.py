#!/usr/bin/env python
"""
Headless-compatible data recorder for SO101 follower + SO101 leader.

Episode flow:
  1. Teleop runs freely — position the arm
  2. Press Enter to start recording
  3. Episode records until time limit or Enter/s to save early
  4. Reset period — teleop runs, move arm back to start
  5. Episode saved in background while you position for the next one

Controls (type + Enter):
    Enter / s  → proceed / end episode early (save)
    r          → discard and re-record episode
    q          → stop all recording
"""

import concurrent.futures
import select
import sys
import threading
import time

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

# ── Config ─────────────────────────────────────────────────────────────────────
NUM_EPISODES   = 50
FPS            = 30
EPISODE_TIME_S = 60
RESET_TIME_S   = 15
TASK           = "Pick screw from bin and place in container"
HF_REPO_ID     = "robotnana/screw_picking"

FOLLOWER_PORT  = "/dev/ttyACM1"
LEADER_PORT    = "/dev/ttyACM0"

CAMERAS = {
    "base":  RealSenseCameraConfig(serial_number_or_name="838212073725", width=640, height=480, fps=30),
    "wrist": OpenCVCameraConfig(index_or_path="/dev/video9", width=640, height=480, fps=30),
}


# ── Headless keyboard listener ─────────────────────────────────────────────────
def _drain_stdin():
    """Discard any buffered stdin lines (e.g. extra Enters from previous episode)."""
    while select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()


def _init_keyboard(events, proceed_event):
    print(
        "\n  Controls (type + Enter):\n"
        "    Enter / s  → proceed / end episode early\n"
        "    r          → discard and re-record\n"
        "    q          → stop all recording\n"
    )

    _drain_stdin()

    def _run():
        try:
            for line in sys.stdin:
                cmd = line.strip().lower()
                if cmd in ("q", "quit"):
                    events["stop_recording"] = True
                    events["exit_early"] = True
                    proceed_event.set()
                    print("  → Stopping …")
                elif events.get("waiting_for_proceed"):
                    # Any input starts the episode
                    proceed_event.set()
                elif events.get("in_episode"):
                    # Only accept control keys while an episode is active
                    if cmd in ("", "s"):
                        events["exit_early"] = True
                        print("  → Saving episode early …")
                    elif cmd == "r":
                        events["rerecord_episode"] = True
                        events["exit_early"] = True
                        print("  → Re-recording …")
                # Ignore all other input between episodes
        except EOFError:
            pass

    threading.Thread(target=_run, daemon=True).start()


# ── Pre-episode positioning loop ───────────────────────────────────────────────
def _positioning_loop(follower, leader, fps, proceed_event, events):
    """Run teleop without saving until proceed_event fires or stop requested."""
    consecutive_errors = 0
    while not proceed_event.is_set() and not events["stop_recording"]:
        loop_t = time.perf_counter()
        try:
            follower.get_observation()   # keeps cameras + bus warm
            act = leader.get_action()
            follower.send_action(act)
            consecutive_errors = 0
        except ConnectionError as e:
            consecutive_errors += 1
            if consecutive_errors > 5:
                raise
            print(f"  [warn] motor read error ({consecutive_errors}/5), retrying …")
            time.sleep(0.05)
            continue
        precise_sleep(max(0.0, 1 / fps - (time.perf_counter() - loop_t)))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    follower = SO101Follower(SO101FollowerConfig(
        port=FOLLOWER_PORT,
        id="left_arm",
        cameras=CAMERAS,
    ))
    leader = SO101Leader(SO101LeaderConfig(
        port=LEADER_PORT,
        id="bmw_leader_arm",
    ))

    # Build features from config (no connection needed) and create dataset
    # before connecting hardware — avoids the motor bus going idle during
    # the slow dataset initialisation.
    teleop_proc, robot_proc, obs_proc = make_default_processors()

    features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_proc,
            initial_features=create_initial_features(action=follower.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=obs_proc,
            initial_features=create_initial_features(observation=follower.observation_features),
            use_videos=True,
        ),
    )

    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        robot_type=follower.name,
        features=features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4 * len(CAMERAS),
        batch_encoding_size=1,
        vcodec="h264",
    )

    # Connect hardware now — immediately before the control loop so the motor
    # bus doesn't go idle waiting for dataset initialisation.
    follower.connect()
    leader.connect()

    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "waiting_for_proceed": False,
        "in_episode": False,  # True only during recording/reset — gates exit_early
    }
    proceed_event = threading.Event()
    _init_keyboard(events, proceed_event)

    save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pending_save = None

    def _wait_save():
        nonlocal pending_save
        if pending_save is not None:
            pending_save.result()
            pending_save = None

    try:
        with VideoEncodingManager(dataset):
            recorded = 0
            while recorded < NUM_EPISODES and not events["stop_recording"]:
                # ── Position arm (teleop, no recording) ───────────────────────
                # Start positioning immediately — previous episode's encoding
                # runs in background during this time.
                print(
                    f"\n  Position arm, then press Enter to start "
                    f"episode {recorded + 1}/{NUM_EPISODES}  (q = quit) …"
                )
                proceed_event.clear()
                events["waiting_for_proceed"] = True
                _positioning_loop(follower, leader, FPS, proceed_event, events)
                events["waiting_for_proceed"] = False

                if events["stop_recording"]:
                    break

                # Wait for previous save/encode to finish before writing new frames
                if pending_save is not None:
                    print("  Waiting for previous episode encoding to finish …")
                    _wait_save()

                # Flush stale camera frames and any buffered stdin from positioning
                for _ in range(3):
                    follower.get_observation()
                _drain_stdin()
                events["exit_early"] = False

                # ── Record episode ────────────────────────────────────────────
                log_say(f"Recording episode {dataset.num_episodes}")
                print(
                    f"\n  Recording episode {recorded + 1}/{NUM_EPISODES} …"
                    "  (Enter/s = save early, r = re-record, q = quit)"
                )

                events["in_episode"] = True
                record_loop(
                    robot=follower,
                    events=events,
                    fps=FPS,
                    teleop_action_processor=teleop_proc,
                    robot_action_processor=robot_proc,
                    robot_observation_processor=obs_proc,
                    teleop=leader,
                    dataset=dataset,
                    control_time_s=EPISODE_TIME_S,
                    single_task=TASK,
                )
                events["in_episode"] = False

                if events["rerecord_episode"]:
                    print("  Re-recording …")
                    dataset.clear_episode_buffer()
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    continue

                if not dataset.episode_buffer or dataset.episode_buffer.get("size", 0) == 0:
                    print("  No frames recorded — skipping.")
                    dataset.clear_episode_buffer()
                    continue

                # ── Reset period (teleop, no recording) ───────────────────────
                if not events["stop_recording"] and recorded < NUM_EPISODES - 1:
                    log_say("Reset environment")
                    print("  Resetting …  (r = re-record this episode, Enter = skip reset)")
                    events["in_episode"] = True
                    record_loop(
                        robot=follower,
                        events=events,
                        fps=FPS,
                        teleop_action_processor=teleop_proc,
                        robot_action_processor=robot_proc,
                        robot_observation_processor=obs_proc,
                        teleop=leader,
                        dataset=None,
                        control_time_s=RESET_TIME_S,
                        single_task=TASK,
                    )
                    events["in_episode"] = False

                if events["rerecord_episode"]:
                    print("  Re-recording …")
                    dataset.clear_episode_buffer()
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    continue

                # ── Save episode in background ────────────────────────────────
                log_say("Saving episode")
                pending_save = save_executor.submit(dataset.save_episode)
                recorded += 1
                print(f"  Episode {recorded}/{NUM_EPISODES} saving in background …")

            _wait_save()

    finally:
        _wait_save()
        save_executor.shutdown(wait=False)
        log_say("Stop recording", blocking=True)
        if follower.is_connected:
            follower.disconnect()
        if leader.is_connected:
            leader.disconnect()

    print(f"\n  Done — {recorded} episodes saved to: {dataset.root}")


if __name__ == "__main__":
    main()
