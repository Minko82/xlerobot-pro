"""
Leader-follower teleoperation recording with per-episode YOLO screw detection.

Usage (same flags as lerobot-record, plus optional --yolo_target and --yolo_confidence):

    python record_with_yolo.py \\
        --robot.type=so100_follower \\
        --robot.port=/dev/ttyUSB0 \\
        --robot.id=follower \\
        --robot.cameras='{
            realsense: {type: intelrealsense, serial_number_or_name: "", fps: 30, width: 640, height: 480},
            wrist:     {type: opencv, index_or_path: 1, fps: 30, width: 640, height: 480}
        }' \\
        --teleop.type=so100_leader \\
        --teleop.port=/dev/ttyUSB1 \\
        --teleop.id=leader \\
        --dataset.repo_id=<hf_user>/<dataset_name> \\
        --dataset.single_task="Pick up screw and place in hole" \\
        --dataset.num_episodes=50 \\
        --dataset.fps=30 \\
        --dataset.push_to_hub=false

What this adds vs. lerobot-record:
  1. Before each episode: runs YOLO on the RealSense frame to detect screws.
  2. Stores the result as 'observation.yolo.screw_xyz' — a float32 array of shape
     (MAX_SCREWS * 3,) = (15,) — containing [x0,y0,z0, x1,y1,z1, ...] for up to
     MAX_SCREWS detections, zero-padded.  x/y are normalised image coords (-0.5…+0.5),
     z is metres from the camera.
  3. That array is stamped onto every frame of the episode so the policy always has
     the initial screw layout as context alongside the live camera streams.
"""

import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np

# Make lerobot importable
sys.path.insert(0, "/home/xle/lerobot/src")
# Make yolo_camera importable
sys.path.insert(0, str(Path(__file__).parent))

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import Robot, RobotConfig, make_robot_from_config, so_follower  # noqa: F401
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
    so_leader,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from yolo_camera import MAX_SCREWS, observe_screws

logger = logging.getLogger(__name__)

# Feature key injected into every frame
YOLO_FEATURE_KEY = "observation.yolo.screw_xyz"
YOLO_FEATURE = {
    "dtype": "float32",
    "shape": (MAX_SCREWS * 3,),
    "names": ["coord"],
}


# ---------------------------------------------------------------------------
# Config dataclasses (mirrors lerobot_record.py)
# ---------------------------------------------------------------------------

@dataclass
class DatasetRecordConfig:
    repo_id: str
    single_task: str
    root: str | Path | None = None
    fps: int = 30
    episode_time_s: int | float = 60
    reset_time_s: int | float = 60
    num_episodes: int = 50
    video: bool = True
    push_to_hub: bool = True
    private: bool = False
    tags: list[str] | None = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    vcodec: str = "libsvtav1"
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("Provide --dataset.single_task.")


@dataclass
class RecordWithYoloConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    teleop: TeleoperatorConfig | None = None
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    play_sounds: bool = True
    resume: bool = False
    # YOLO settings
    yolo_target: str = "screw"
    yolo_confidence: float = 0.25

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            from lerobot.configs.policies import PreTrainedConfig
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            self.policy = None

        if self.teleop is None and not hasattr(self, "policy"):
            raise ValueError("Provide --teleop or --policy.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


# ---------------------------------------------------------------------------
# record_loop (identical to upstream, no changes needed)
# ---------------------------------------------------------------------------

@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor=None,
    postprocessor=None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"dataset.fps ({dataset.fps}) != requested fps ({fps}).")

    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)

        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        if policy is not None and preprocessor is not None and postprocessor is not None:
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )
            act_processed_policy: RobotAction = make_robot_action(action_values, dataset.features)
        elif policy is None and teleop is not None:
            act = teleop.get_action()
            act_processed_teleop = teleop_action_processor((act, obs))
        else:
            logger.info("No policy or teleop provided — skipping action.")
            continue

        if policy is not None and act_processed_policy is not None:
            action_values = act_processed_policy
            robot_action_to_send = robot_action_processor((act_processed_policy, obs))
        else:
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        robot.send_action(robot_action_to_send)

        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)  # monkey-patched to inject YOLO context

        if display_data:
            log_rerun_data(
                observation=obs_processed,
                action=action_values,
                compress_images=display_compressed_images,
            )

        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(max(1 / fps - dt_s, 0.0))
        timestamp = time.perf_counter() - start_episode_t


# ---------------------------------------------------------------------------
# Main record function
# ---------------------------------------------------------------------------

@parser.wrap()
def record(cfg: RecordWithYoloConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )
    if cfg.display_data:
        init_rerun(session_name="recording_yolo", ip=cfg.display_ip, port=cfg.display_port)

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Build the standard features, then inject the YOLO feature
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )
    dataset_features[YOLO_FEATURE_KEY] = YOLO_FEATURE
    logger.info(f"Dataset features: {list(dataset_features.keys())}")

    dataset = None
    listener = None

    # Mutable holder for the current episode's YOLO context
    yolo_ctx: dict[str, np.ndarray] = {"xyz": np.zeros(MAX_SCREWS * 3, dtype=np.float32)}

    try:
        if cfg.resume:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
            )
            if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                dataset.start_image_writer(
                    num_processes=cfg.dataset.num_image_writer_processes,
                    num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=(
                    cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras)
                ),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
            )

        # ── Monkey-patch add_frame to stamp every frame with this episode's YOLO context ──
        _original_add_frame = dataset.add_frame

        def _patched_add_frame(frame: dict) -> None:
            frame[YOLO_FEATURE_KEY] = yolo_ctx["xyz"].copy()
            return _original_add_frame(frame)

        dataset.add_frame = _patched_add_frame

        # Load policy if provided
        policy = None
        preprocessor = postprocessor = None
        if cfg.policy is not None:
            policy = make_policy(cfg.policy, ds_meta=dataset.meta)
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=cfg.policy,
                pretrained_path=cfg.policy.pretrained_path,
                dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
                preprocessor_overrides={
                    "device_processor": {"device": cfg.policy.device},
                    "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
                },
            )

        robot.connect()
        if teleop is not None:
            teleop.connect()

        listener, events = init_keyboard_listener()

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:

                # ── Pre-episode: YOLO scan ────────────────────────────────────
                log_say(f"Scanning for screws before episode {dataset.num_episodes}…", cfg.play_sounds)
                result = observe_screws(
                    target_object=cfg.yolo_target,
                    confidence_threshold=cfg.yolo_confidence,
                )
                yolo_ctx["xyz"] = result["screw_xyz"]
                logger.info(
                    f"Episode {dataset.num_episodes}: detected {result['count']} screw(s). "
                    f"depth_status={result['depth_status']}  "
                    f"xyz={yolo_ctx['xyz']}"
                )
                if result["error"]:
                    logger.warning(f"YOLO scan warning: {result['error']}")
                # ─────────────────────────────────────────────────────────────

                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_compressed_images=display_compressed_images,
                )

                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds)
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1

    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)
        if dataset:
            dataset.finalize()
        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()
        if not is_headless() and listener:
            listener.stop()
        if cfg.dataset.push_to_hub:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
        log_say("Exiting", cfg.play_sounds)

    return dataset


def main():
    record()


if __name__ == "__main__":
    main()
