#!/usr/bin/env python3
"""
SmolVLA-base end-to-end inference speed benchmark.

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

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import (
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

DEFAULT_CAMERA   = "/dev/video0"
DEFAULT_TASK     = "pick up the object and place it in the box"
DEFAULT_MODEL    = "lerobot/smolvla_base"
DEFAULT_DURATION = 30    # 5 minutes default; use --duration 3600 for 1 hour
ACTION_CHUNK     = 5      # n_action_steps override
ACTION_DIM       = 6       # how many action dims to print (model outputs 32)
WARMUP_STEPS     = 5      # discard first N steps from stats (JIT / cache warmup)
LOG_INTERVAL     = 20


def parse_args():
    p = argparse.ArgumentParser(description="SmolVLA-base inference speed benchmark")
    p.add_argument("--model",    default=DEFAULT_MODEL)
    p.add_argument("--camera",   default=DEFAULT_CAMERA)
    p.add_argument("--task",     default=DEFAULT_TASK)
    p.add_argument("--duration", type=int,   default=DEFAULT_DURATION,  help="Benchmark duration in seconds")
    p.add_argument("--chunk",      type=int, default=ACTION_CHUNK, help="chunk_size: how many actions the model outputs per forward pass")
    p.add_argument("--n-action-steps", type=int, default=None, help="how many actions to execute before re-planning (defaults to --chunk)")
    p.add_argument("--num-steps",  type=int, default=None, help="denoising steps per forward pass (default: from model config, typically 10)")
    p.add_argument("--device",         default=None, help="'cpu', 'cuda', or None for auto")
    p.add_argument("--width",          type=int, default=320)
    p.add_argument("--height",         type=int, default=240)
    p.add_argument("--resize-imgs",    type=int, default=None, help="override resize_imgs_with_padding to NxN (default: from model config, typically 512)")
    p.add_argument("--compile", action="store_true",
                   help="wrap the model in torch.compile(mode='reduce-overhead') to capture CUDA graphs")
    p.add_argument("--warmup", type=int, default=None,
                   help="override warmup steps; raise it with --compile, since compilation happens on the first calls")
    p.add_argument("--quiet", action="store_true",
                   help="suppress the per-step action dump (chunk 50 prints 50 lines per step)")
    p.add_argument("--json-out", default=None, help="write final metrics to this path as JSON")
    return p.parse_args()


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def get_device(requested: str | None) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── model ─────────────────────────────────────────────────────────────────────

def load_policy(model_id: str, device: str) -> SmolVLAPolicy:
    print(f"[{ts()}] Loading '{model_id}' …")
    
    # Load on CPU first to prevent CUDA allocator spikes, then convert to fp16.
    # The checkpoint config pins device="cuda", which makes from_pretrained pull
    # the safetensors straight onto the GPU at full precision -- that peak OOMs
    # an 8 GB Jetson before the fp16 cast below can shrink it. Overriding the
    # config to CPU keeps the load in system RAM and halves what reaches the GPU.
    from lerobot.configs.policies import PreTrainedConfig

    config = PreTrainedConfig.from_pretrained(model_id)
    config.device = "cpu"
    policy = SmolVLAPolicy.from_pretrained(model_id, config=config)
    policy = policy.to(dtype=torch.float16, device=device)
    policy.eval()
    print(f"[{ts()}] Policy ready on {device} (fp16)")
    return policy


def tokenize_task(policy: SmolVLAPolicy, task: str, device: str):
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    enc = tokenizer(
        task,
        padding="max_length",
        max_length=policy.config.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device).bool()


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
    policy:      SmolVLAPolicy,
    img_tensor:  torch.Tensor,
    lang_tokens: torch.Tensor,
    lang_mask:   torch.Tensor,
    device:      str,
) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {}

    img_keys = list(policy.config.image_features.keys()) or [f"{OBS_IMAGES}.top"]
    for key in img_keys:
        batch[key] = img_tensor

    state_features = {k: v for k, v in policy.config.input_features.items() if k == OBS_STATE}
    state_dim = state_features[OBS_STATE].shape[0] if state_features else policy.config.max_state_dim
    batch[OBS_STATE] = torch.zeros(1, state_dim, dtype=torch.float16, device=device)

    batch[OBS_LANGUAGE_TOKENS]         = lang_tokens
    batch[OBS_LANGUAGE_ATTENTION_MASK] = lang_mask
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
    policy.config.chunk_size = args.chunk
    policy.config.n_action_steps = n_action_steps
    policy._queues["action"] = deque(maxlen=n_action_steps)
    if args.num_steps is not None:
        policy.config.num_steps = args.num_steps

    if args.resize_imgs is not None:
        policy.config.resize_imgs_with_padding = (args.resize_imgs, args.resize_imgs)

    compiled = False
    if args.compile:
        # reduce-overhead captures CUDA graphs. This is the only untested lever on
        # the ~195 ms dispatch floor, so fail loudly rather than silently falling back
        # to eager and reporting it as a compiled result.
        try:
            policy.model = torch.compile(policy.model, mode="reduce-overhead")
            compiled = True
            print(f"[{ts()}] torch.compile(mode='reduce-overhead') applied to policy.model")
        except Exception as exc:
            print(f"[{ts()}] [warn] torch.compile FAILED: {type(exc).__name__}: {exc}")

    warmup_steps = args.warmup if args.warmup is not None else (WARMUP_STEPS + 10 if args.compile else WARMUP_STEPS)

    print(f"[{ts()}] chunk_size={args.chunk}, n_action_steps={n_action_steps}, num_steps={policy.config.num_steps}")
    print(f"[{ts()}] resize_imgs={policy.config.resize_imgs_with_padding}")

    img_keys   = list(policy.config.image_features.keys())
    action_dim = policy.config.action_feature.shape[0]
    print(f"[{ts()}] Image keys : {img_keys}")
    print(f"[{ts()}] Action dim : {action_dim}")

    lang_tokens, lang_mask = tokenize_task(policy, args.task, device)
    print(f"[{ts()}] Task       : '{args.task}'")

    cap = open_camera(args.camera, args.width, args.height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resize = (args.width, args.height) if (args.width != actual_w or args.height != actual_h) else None
    if resize:
        print(f"[{ts()}] SW resize   : {actual_w}×{actual_h} → {args.width}×{args.height}")

    t_capture  = Timer("capture + preprocess")
    t_batch    = Timer("batch construction")
    t_infer    = Timer("model forward (select_action)")
    t_end2end  = Timer("end-to-end (capture → action)")

    step       = 0
    start_time = time.monotonic()
    deadline   = None                       # set when the first measured step begins
    hard_stop  = start_time + args.duration + 1200   # backstop if compilation hangs

    print(f"\n[{ts()}] Warming up for {warmup_steps} steps, then benchmarking for {args.duration}s …\n")

    stop = False
    def _sigint(sig, frame):
        nonlocal stop
        print(f"\n[{ts()}] Interrupted.")
        stop = True
    signal.signal(signal.SIGINT, _sigint)

    try:
        while not stop and time.monotonic() < hard_stop and (deadline is None or time.monotonic() < deadline):
            step += 1
            measuring = step > warmup_steps
            if measuring and deadline is None:
                deadline = time.monotonic() + args.duration

            e2e_start = time.monotonic()

            # ── 1. capture ────────────────────────────────────────────────────
            t0  = time.monotonic()
            img = capture_frame(cap, device, resize)
            dt_capture = time.monotonic() - t0

            # ── 2. batch ──────────────────────────────────────────────────────
            t0    = time.monotonic()
            batch = build_batch(policy, img, lang_tokens, lang_mask, device)
            dt_batch = time.monotonic() - t0

            # ── 3. model forward ──────────────────────────────────────────────
            t0 = time.monotonic()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                chunk = policy.predict_action_chunk(batch)  # (1, chunk_size, action_dim)
            dt_infer = time.monotonic() - t0

            dt_e2e = time.monotonic() - e2e_start

            if measuring:
                t_capture.record(dt_capture)
                t_batch.record(dt_batch)
                t_infer.record(dt_infer)
                t_end2end.record(dt_e2e)

            # ── print ─────────────────────────────────────────────────────────
            tag = "bench" if measuring else "warm "
            print(
                f"[{ts()}] #{step:>5d} [{tag}] "
                f"e2e={dt_e2e*1000:7.1f}ms  "
                f"infer={dt_infer*1000:7.1f}ms  "
                f"cap={dt_capture*1000:5.1f}ms"
            )
            if not args.quiet:
                chunk_np = chunk.squeeze(0).cpu().numpy()  # (chunk_size, action_dim)
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
    if bench_n == 0:
        print("\n  NO BENCHMARK SAMPLES — the window expired during warmup/compilation.")
        print("  Raise --duration or lower --warmup. Nothing written.")
        return

    print(f"\n{'='*60}")
    print(f"  SmolVLA-base inference benchmark — final report")
    print(f"  model  : {args.model}")
    print(f"  device : {device}")
    print(f"  num_steps: {policy.config.num_steps}")
    print(f"  task   : {args.task}")
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

    if args.json_out:
        import json as _json, os as _os
        payload = {
            "model": args.model,
            "device": device,
            "num_steps": policy.config.num_steps,
            "chunk": args.chunk,
            "n_action_steps": n_action_steps,
            "compile_requested": bool(args.compile),
            "compiled": compiled,
            "warmup_steps": warmup_steps,
            "bench_steps": bench_n,
            "wall_s": round(wall_s, 2),
            "e2e_mean_ms": float(np.mean(t_end2end.samples)),
            "e2e_median_ms": float(np.median(t_end2end.samples)),
            "e2e_p90_ms": float(np.percentile(t_end2end.samples, 90)),
            "infer_median_ms": float(np.median(t_infer.samples)),
            "capture_median_ms": float(np.median(t_capture.samples)),
            "replan_hz": float(max_replan_hz),
            "control_hz": float(max_ctrl_hz),
        }
        _os.makedirs(_os.path.dirname(_os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as fh:
            _json.dump(payload, fh, indent=2)
        print(f"  wrote {args.json_out}")


if __name__ == "__main__":
    run(parse_args())
