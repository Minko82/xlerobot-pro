# Inference benchmarks

These are the two scripts behind the paper's onboard timings for SmolVLA and
Diffusion Policy (Tables VI and VII). They are copied unchanged from the robot.
Both time the full loop on the Jetson: camera capture, batch construction, and
the model's forward pass. Each prints a per-stage breakdown (mean, median, SD,
and percentiles) at the end.

Run them in MAXN SUPER mode (`sudo nvpmodel -m 2`) with dynamic clocks, as in
the paper. Pass `--camera` with your RealSense color device; the default is
`/dev/video0`.

## SmolVLA

```bash
python benchmarks/smolvla_benchmark.py --num-steps 10 --chunk 5 \
    --width 640 --height 480 --resize-imgs 512 --duration 1800 --camera /dev/video2
```

This is the configuration of the paper's 30-minute run (1,791 passes,
996.7 ms model-forward median). The step-count and chunk-length rows of
Table VII come from 60-second runs at `--num-steps` 2, 3, 5, and 10 and
`--chunk` 5, 25, and 50. The paper used a fine-tuned 450M SmolVLA checkpoint;
pass it with `--model`. The default is `lerobot/smolvla_base`, which has the same
architecture.

## Diffusion Policy

```bash
python benchmarks/diffusion_benchmark.py --num-steps 10 --camera /dev/video2
```

This loads LeRobot's `DiffusionPolicy` with the public `lerobot/diffusion_pusht`
checkpoint. The paper's Diffusion Policy row (539.6 ms) is an end-to-end time,
from camera capture to action output, at 10 sampling steps.

## A note on the printed rates

Both scripts also print a "max control frequency", computed as replanning rate
× actions per chunk. That is an upper bound that ignores the time spent
executing each chunk. The paper's command throughput uses Eq. (2) instead,
f_cmd = K / (K / f_exec + T_inf), which can never exceed the 30 Hz execution rate.
