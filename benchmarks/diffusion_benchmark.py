#!/usr/bin/env python3
"""
DiffusionPolicy PushT end-to-end inference speed benchmark.

Measures the full pipeline: camera capture → batch construction → model forward.
Runs as fast as possible with no rate limiting.
"""

import argparse
import signal
import time
from datetime import datetime

import cv2
import numpy as np
import torch

from diffusers import DDIMScheduler
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

DEFAULT_CAMERA   = "/dev/video0"
DEFAULT_MODEL    = "lerobot/diffusion_pusht"
DEFAULT_DURATION = 30    # 30s default; use --duration 3600 for 1 hour
ACTION_CHUNK     = 5     # n_action_steps override
WARMUP_STEPS     = 5     # discard first N steps from stats
LOG_INTERVAL     = 20


def parse_args():
    p = argparse.ArgumentParser(description="DiffusionPolicy PushT inference speed benchmark")
    p.add_argument("--model",          default=DEFAULT_MODEL)
    p.add_argument("--camera",         default=DEFAULT_CAMERA)
    p.add_argument("--duration",       type=int, default=DEFAULT_DURATION, help="Benchmark duration in seconds")
    p.add_argument("--chunk",          type=int, default=ACTION_CHUNK, help="chunk_size: how many actions the model outputs per forward pass")
    p.add_argument("--n-action-steps", type=int, default=None, help="how many actions to execute before re-planning (defaults to --chunk)")
    p.add_argument("--num-steps",      type=int, default=None, help="denoising steps per forward pass (default: from model config, typically 100)")
    p.add_argument("--no-ddim",        action="store_true", help="use original DDPM scheduler instead of DDIM")
    p.add_argument("--device",         default=None, help="'cpu', 'cuda', or None for auto")
    p.add_argument("--width",          type=int, default=320)
    p.add_argument("--height",         type=int, default=240)
    return p.parse_args()


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def get_device(requested: str | None) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── model ─────────────────────────────────────────────────────────────────────

def load_policy(model_id: str, device: str) -> DiffusionPolicy:
    print(f"[{ts()}] Loading '{model_id}' …")
    # Load on CPU first to prevent CUDA allocator spikes, then convert to fp16
    policy = DiffusionPolicy.from_pretrained(model_id)
    policy = policy.to(dtype=torch.float16, device=device)
    policy.eval()
    print(f"[{ts()}] Policy ready on {device} (fp16)")
    return policy


# ── camera ────────────────────────────────────────────────────────────────────

def open_camera(path: str, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {path}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[{ts()}] Camera {path} opened at {actual_w}×{actual_h}")
    return cap


def capture_frame(cap: cv2.VideoCapture, device: str, resize: tuple[int, int] | None = None) -> torch.Tensor:
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Camera read failed")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if resize is not None:
        rgb = cv2.resize(rgb, resize, interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(rgb).half().div(255.0)
    t   = t.permute(2, 0, 1).unsqueeze(0)   # [1, 3, H, W]
    return t.to(device)


# ── batch ─────────────────────────────────────────────────────────────────────

def build_batch(
    policy:     DiffusionPolicy,
    img_tensor: torch.Tensor,
    device:     str,
) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {}
    for key in policy.config.image_features:
        batch[key] = img_tensor
    state_dim = policy.config.input_features[OBS_STATE].shape[0]
    batch[OBS_STATE] = torch.zeros(1, state_dim, dtype=torch.float16, device=device)
    # Stack images and populate obs queues (mirrors select_action logic)
    batch[OBS_IMAGES] = torch.stack([batch[key] for key in policy.config.image_features], dim=-4)
    policy._queues = populate_queues(policy._queues, batch)
    return batch


# ── stats ─────────────────────────────────────────────────────────────────────

class Timer:
    """Accumulates timing samples and computes percentile stats."""

    def __init__(self, name: str):
        self.name    = name
        self.samples: list[float] = []

    def record(self, elapsed_s: float):
        self.samples.append(elapsed_s * 1000.0)   # store in ms

    def report(self, indent: str = "  ") -> str:
        if not self.samples:
            return f"{indent}{self.name}: no data"
        a = np.array(self.samples)
        return (
            f"{indent}{self.name}:\n"
            f"{indent}  n      = {len(a)}\n"
            f"{indent}  mean   = {a.mean():.2f} ms\n"
            f"{indent}  median = {np.median(a):.2f} ms\n"
            f"{indent}  std    = {a.std():.2f} ms\n"
            f"{indent}  min    = {a.min():.2f} ms\n"
            f"{indent}  p90    = {np.percentile(a, 90):.2f} ms\n"
            f"{indent}  p95    = {np.percentile(a, 95):.2f} ms\n"
            f"{indent}  p99    = {np.percentile(a, 99):.2f} ms\n"
            f"{indent}  max    = {a.max():.2f} ms"
        )

    def hz(self) -> float:
        if not self.samples:
            return 0.0
        return 1000.0 / np.mean(self.samples)


# ── main ──────────────────────────────────────────────────────────────────────

def run(args):
    device = get_device(args.device)
    print(f"[{ts()}] Device: {device}")
    torch.set_default_dtype(torch.float16)
    policy = load_policy(args.model, device)

    # Override action chunk size
    from collections import deque
    n_action_steps = args.n_action_steps if args.n_action_steps is not None else args.chunk
    policy.config.n_action_steps = n_action_steps
    policy._queues["action"] = deque(maxlen=n_action_steps)
    if not args.no_ddim:
        cfg = policy.config
        policy.diffusion.noise_scheduler = DDIMScheduler(
            num_train_timesteps=cfg.num_train_timesteps,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            beta_schedule=cfg.beta_schedule,
            clip_sample=cfg.clip_sample,
            clip_sample_range=cfg.clip_sample_range,
            prediction_type=cfg.prediction_type,
        )
    if args.num_steps is not None:
        policy.diffusion.num_inference_steps = args.num_steps

    scheduler_name = policy.config.noise_scheduler_type if args.no_ddim else "DDIM"
    img_features    = policy.config.image_features
    img_keys        = list(img_features.keys())
    _, img_h, img_w = next(iter(img_features.values())).shape
    action_dim      = policy.config.action_feature.shape[0]

    print(f"[{ts()}] chunk_size={args.chunk}, n_action_steps={n_action_steps}, num_steps={policy.diffusion.num_inference_steps}, scheduler={scheduler_name}")
    print(f"[{ts()}] Image keys : {img_keys}  ({img_h}×{img_w})")
    print(f"[{ts()}] Action dim : {action_dim}")

    cap = open_camera(args.camera, args.width, args.height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Always resize to model's expected image size
    resize = (img_w, img_h) if (img_w != actual_w or img_h != actual_h) else None
    if resize:
        print(f"[{ts()}] SW resize   : {actual_w}×{actual_h} → {img_w}×{img_h}")

    t_capture  = Timer("capture + preprocess")
    t_batch    = Timer("batch construction")
    t_infer    = Timer("model forward (diffusion denoising)")
    t_end2end  = Timer("end-to-end (capture → action)")

    step       = 0
    start_time = time.monotonic()
    deadline   = start_time + args.duration

    print(f"\n[{ts()}] Warming up for {WARMUP_STEPS} steps, then benchmarking for {args.duration}s …\n")

    stop = False
    def _sigint(sig, frame):
        nonlocal stop
        print(f"\n[{ts()}] Interrupted.")
        stop = True
    signal.signal(signal.SIGINT, _sigint)

    try:
        while not stop and time.monotonic() < deadline:
            step += 1
            measuring = step > WARMUP_STEPS

            e2e_start = time.monotonic()

            # ── 1. capture ────────────────────────────────────────────────────
            t0  = time.monotonic()
            img = capture_frame(cap, device, resize)
            dt_capture = time.monotonic() - t0

            # ── 2. batch ──────────────────────────────────────────────────────
            t0    = time.monotonic()
            batch = build_batch(policy, img, device)
            dt_batch = time.monotonic() - t0

            # ── 3. model forward ──────────────────────────────────────────────
            t0 = time.monotonic()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                chunk = policy.predict_action_chunk(batch)  # (1, horizon, action_dim)
            dt_infer = time.monotonic() - t0

            dt_e2e = time.monotonic() - e2e_start

            if measuring:
                t_capture.record(dt_capture)
                t_batch.record(dt_batch)
                t_infer.record(dt_infer)
                t_end2end.record(dt_e2e)

            # ── print ─────────────────────────────────────────────────────────
            tag = "bench" if measuring else "warm "
            chunk_np = chunk.squeeze(0).cpu().numpy()  # (horizon, action_dim)
            print(
                f"[{ts()}] #{step:>5d} [{tag}] "
                f"e2e={dt_e2e*1000:7.1f}ms  "
                f"infer={dt_infer*1000:7.1f}ms  "
                f"cap={dt_capture*1000:5.1f}ms"
            )
            for i, act in enumerate(chunk_np):
                vals = "  ".join(f"{v:+.4f}" for v in act)
                print(f"    t+{i:<3d} | {vals}")

            if measuring and len(t_end2end.samples) % LOG_INTERVAL == 0:
                n = len(t_end2end.samples)
                mean_e2e_ms = np.mean(t_end2end.samples)
                max_replan_hz = 1000.0 / mean_e2e_ms
                max_ctrl_hz   = max_replan_hz * n_action_steps
                print(
                    f"\n  [live] after {n} bench steps — "
                    f"e2e mean={mean_e2e_ms:.1f}ms  |  "
                    f"max replan freq={max_replan_hz:.2f} Hz  |  "
                    f"max control freq={max_ctrl_hz:.2f} Hz "
                    f"(n_action_steps={n_action_steps})\n"
                )

    finally:
        cap.release()

    # ── final report ──────────────────────────────────────────────────────────
    wall_s = time.monotonic() - start_time
    bench_n = len(t_end2end.samples)

    print(f"\n{'='*60}")
    print(f"  DiffusionPolicy PushT benchmark — final report")
    print(f"  model  : {args.model}")
    print(f"  device : {device}")
    print(f"  scheduler: {scheduler_name}")
    print(f"  num_steps: {policy.diffusion.num_inference_steps}")
    print(f"{'='*60}")
    print(f"  wall time      : {wall_s:.1f}s")
    print(f"  warmup steps   : {WARMUP_STEPS}")
    print(f"  benchmark steps: {bench_n}")
    print()
    print(t_capture.report())
    print()
    print(t_batch.report())
    print()
    print(t_infer.report())
    print()
    print(t_end2end.report())
    print()
    mean_e2e_ms   = np.mean(t_end2end.samples)
    max_replan_hz = 1000.0 / mean_e2e_ms
    max_ctrl_hz   = max_replan_hz * n_action_steps
    print(f"  chunk_size     : {args.chunk}")
    print(f"  n_action_steps : {n_action_steps}")
    print()
    print(f"  >> max replan frequency : {max_replan_hz:.3f} Hz  (1 / e2e latency)")
    print(f"  >> max control frequency: {max_ctrl_hz:.3f} Hz  (replan_hz × n_action_steps)")
    print(f"{'='*60}")


if __name__ == "__main__":
    run(parse_args())
