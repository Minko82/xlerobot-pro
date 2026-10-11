# Perception + IK demos

These demos are part of the stack described in the paper, but the paper does not evaluate them. Demos for the `xlerobot_pro` pipeline: an Intel RealSense camera detects a
colored object, the detection is transformed into the arm's base frame, and
the differential-IK solver drives the SO-101 arm to it.

```
RealSense capture ─▶ color detection ─▶ frame transform ─▶ IK ─▶ motor bus
```

Install the vision extras first (`pip install -r requirements/requirements-vision.txt`) and
validate without hardware: `python tests/vision_offline_suite.py` (expected:
49/49 passed).

## Demos

- **`grab_cube.py`** — the main demo: detect a cube, plan with IK, grasp it.
  Requires a RealSense camera and one SO-101 arm.
- **`scripted/`** — pre-programmed single-arm motions (no camera needed):
  `so101_simple.py`, `so101_reach_forward.py`, `so101_pen_pickup.py`,
  `so101_pen_drop.py`, `so101_grab_cube.py`, `so101_grab_apple.py`,
  `so101_hang.py`.

**Wiring caveat.** `grab_cube.py` opens the arms bus (`/dev/xle_arms`, from
`xlerobot_pro.config`) but builds it from `MOTOR_DEFS` in `calibration.py`, which
still follows the original single-bus wiring (head on IDs 1-2, right arm on 7-12).
On this build's two-bus wiring, IDs 1-2 on the arms bus are the left arm's
shoulder pan and lift, so do not run `grab_cube.py` on this wiring until
`MOTOR_DEFS` is updated.

The scripted demos open `/dev/ttyACM0` and drive motor IDs 1-6; change `PORT` (or
the `port=` argument) at the top of each script to match your wiring. The two that
load a calibration file use `calibration/single_bus.json` at the repo root.
