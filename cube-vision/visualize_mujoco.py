#!/usr/bin/env python3
"""MuJoCo visualizer for the xlerobot IK pipeline.

Simulates the full control pipeline (frame_transform + IK solver) on an
imaginary cube on a table, without requiring hardware.

Usage:
    python visualize_mujoco.py                              # default cube 25cm from arm on table
    python visualize_mujoco.py --cube-x 0.05 --cube-y -0.25 --cube-z 0.02
    python visualize_mujoco.py --use-transform              # run frame_transform pipeline
    python visualize_mujoco.py --speed 0.5                  # slower playback
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# On macOS, MuJoCo's launch_passive requires the script to run under mjpython.
# Auto-relaunch if we detect we're on macOS and not already under mjpython.
# ---------------------------------------------------------------------------
if platform.system() == "Darwin" and "MJPYTHON" not in os.environ:
    mjpython = shutil.which("mjpython")
    if mjpython:
        env = os.environ.copy()
        env["MJPYTHON"] = "1"  # prevent infinite re-launch
        result = subprocess.run([mjpython] + sys.argv, env=env)
        sys.exit(result.returncode)
    else:
        print("ERROR: mjpython not found. On macOS, MuJoCo viewer requires mjpython.")
        print("       It should be installed with mujoco: pip install mujoco")
        sys.exit(1)

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("ERROR: mujoco package not found. Install with: pip install mujoco")
    sys.exit(1)

# Ensure project root is on path so we can import ik_solver / frame_transform
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ik_solver import IK_SO101

_MJCF_PATH = _PROJECT_ROOT / "frame_transform" / "xlerobot" / "xlerobot.xml"

# Offset converts IK world positions (URDF) to MJCF world positions.
# URDF Base X=-0.135, MJCF Base X=-0.09; URDF Base Y=-0.088, MJCF Base Y=-0.11
_MJCF_OFFSET = np.array([0.045, -0.022, 0.015])

# The 5 IK joints in the order returned by generate_ik()
_IK_JOINT_NAMES = [
    "Rotation_L",
    "Pitch_L",
    "Elbow_L",
    "Wrist_Pitch_L",
    "Wrist_Roll_L",
]


def _resolve_joint_qpos_indices(model: "mujoco.MjModel") -> list[int]:
    """Map each IK joint name to its qpos index in the MuJoCo model."""
    indices = []
    for name in _IK_JOINT_NAMES:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jnt_id == -1:
            raise RuntimeError(f"Joint '{name}' not found in MJCF model")
        indices.append(model.jnt_qposadr[jnt_id])
    return indices


def _update_target_cube(model, data, world_xyz: np.ndarray):
    """Move the mocap target cube to the given world-frame position."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")
    if body_id == -1:
        return
    # mocap bodies have their own index
    mocap_id = model.body_mocapid[body_id]
    if mocap_id >= 0:
        data.mocap_pos[mocap_id] = world_xyz


def run_visualization(
    cube_base: list[float],
    gripper_offset: list[float],
    use_transform: bool = False,
    speed: float = 1.0,
):
    """Run the full IK pipeline and animate in MuJoCo viewer."""

    # ------------------------------------------------------------------
    # Optionally run the frame_transform pipeline with a synthetic point
    # ------------------------------------------------------------------
    if use_transform:
        from frame_transform.frame_transform import camera_xyz_to_base_xyz

        # Synthesize a camera-frame point that roughly maps to our target.
        # Use head at neutral (pan=0, tilt=0 in motor convention = pan≈1°, tilt≈14° motor).
        # We just pass 0,0 in radians as a simple demo.
        joint_values = {"head_pan_joint": 0.0, "head_tilt_joint": 0.0}

        # A point 30cm forward, slightly down in camera optical frame
        cam_x, cam_y, cam_z = 0.0, 0.05, 0.30
        bx, by, bz = camera_xyz_to_base_xyz(cam_x, cam_y, cam_z, joint_values)
        cube_base = [bx, by, bz]
        print(f"[frame_transform] Camera ({cam_x}, {cam_y}, {cam_z}) "
              f"-> Base ({bx:.4f}, {by:.4f}, {bz:.4f})")

    print(f"Target in Base frame: {cube_base}")

    # ------------------------------------------------------------------
    # Run IK solver
    # ------------------------------------------------------------------
    print("Running IK solver...")
    ik = IK_SO101()
    trajectory = ik.generate_ik(
        target_xyz=cube_base,
        gripper_offset_xyz=gripper_offset,
    )

    if not trajectory:
        print("ERROR: IK solver returned empty trajectory (target may be unreachable).")
        sys.exit(1)

    print(f"IK converged in {len(trajectory)} steps.")
    final_q = trajectory[-1]
    print(f"Final joint angles (deg): {np.rad2deg(final_q).round(2).tolist()}")

    # Convert target to world frame for cube placement
    target_world_ik = ik.base_to_world(np.asarray(cube_base) + np.asarray(gripper_offset))
    target_world = target_world_ik + _MJCF_OFFSET
    print(f"Target in world frame: [{target_world[0]:.4f}, {target_world[1]:.4f}, {target_world[2]:.4f}]")

    # ------------------------------------------------------------------
    # Load MuJoCo model
    # ------------------------------------------------------------------
    print(f"Loading MJCF model from {_MJCF_PATH} ...")
    # Load from the XML's own directory so meshdir="./assets/" resolves correctly
    prev_cwd = os.getcwd()
    os.chdir(_MJCF_PATH.parent)
    model = mujoco.MjModel.from_xml_path(str(_MJCF_PATH))
    os.chdir(prev_cwd)
    data = mujoco.MjData(model)

    qpos_indices = _resolve_joint_qpos_indices(model)

    # Place the target cube in world frame
    _update_target_cube(model, data, target_world)

    # Set initial pose and forward
    mujoco.mj_forward(model, data)

    # ------------------------------------------------------------------
    # Interpolate trajectory for smooth animation
    # ------------------------------------------------------------------
    # IK may converge in very few steps (e.g. 5). Interpolate so the
    # animation always takes a reasonable amount of time.
    MIN_ANIM_STEPS = 200
    if len(trajectory) < MIN_ANIM_STEPS and len(trajectory) >= 2:
        orig = np.array(trajectory)
        t_orig = np.linspace(0, 1, len(orig))
        t_new = np.linspace(0, 1, MIN_ANIM_STEPS)
        interp = np.zeros((MIN_ANIM_STEPS, orig.shape[1]))
        for j in range(orig.shape[1]):
            interp[:, j] = np.interp(t_new, t_orig, orig[:, j])
        trajectory = [interp[i] for i in range(MIN_ANIM_STEPS)]
        print(f"Interpolated to {len(trajectory)} display steps for smooth animation.")

    # ------------------------------------------------------------------
    # Animate in passive viewer
    # ------------------------------------------------------------------
    print("Launching MuJoCo viewer... (close the window to exit)")
    print("  Label toggle keys in viewer:  i = body/link  j = joint  u = off")

    # Target ~3 seconds for full animation at speed=1
    dt_display = max(3.0 / len(trajectory), 0.01) / speed

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # -- Enable labels & frames for joints and links -----------------
        # mjtLabel: 0=none, 1=body, 2=joint, 3=geom, 4=site …
        # mjtFrame: 0=none, 1=body, 2=geom, 3=site, 4=world, 6=joint
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_BODY   # show body/link names
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY    # show body coord frames

        # Give the viewer a moment to initialize
        time.sleep(0.5)

        # Ensure target cube is placed
        _update_target_cube(model, data, target_world)

        # Loop: animate trajectory, hold 8s, reset, repeat
        loop_count = 0
        while viewer.is_running():
            loop_count += 1
            print(f"Animation loop {loop_count}...")

            # Animate through trajectory
            for step_i, q_step in enumerate(trajectory):
                if not viewer.is_running():
                    break
                for idx, q_val in zip(qpos_indices, q_step):
                    data.qpos[idx] = q_val
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(dt_display)

            # Hold final pose for 8 seconds
            hold_start = time.time()
            while viewer.is_running() and time.time() - hold_start < 8.0:
                _update_target_cube(model, data, target_world)
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.05)

            # Reset to neutral before next loop
            if viewer.is_running():
                for idx in qpos_indices:
                    data.qpos[idx] = 0.0
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.5)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo visualizer for xlerobot IK pipeline",
    )
    parser.add_argument(
        "--cube-x", type=float, default=0.0,
        help="Target cube X in Base frame (left/right). Default: 0.0",
    )
    parser.add_argument(
        "--cube-y", type=float, default=-0.25,
        help="Target cube Y in Base frame (-Y is forward). Default: -0.20",
    )
    parser.add_argument(
        "--cube-z", type=float, default=0.0,
        help="Target cube Z in Base frame (up/down). Default: 0.0",
    )
    parser.add_argument(
        "--gripper-offset-x", type=float, default=0.0,
        help="Gripper offset X in Base frame. Default: 0.0",
    )
    parser.add_argument(
        "--gripper-offset-y", type=float, default=0.0,
        help="Gripper offset Y in Base frame. Default: 0.0",
    )
    parser.add_argument(
        "--gripper-offset-z", type=float, default=0.0,
        help="Gripper offset Z in Base frame. Default: 0.0",
    )
    parser.add_argument(
        "--use-transform", action="store_true",
        help="Run frame_transform pipeline with synthetic camera point",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed multiplier. Default: 1.0",
    )

    args = parser.parse_args()

    cube_base = [args.cube_x, args.cube_y, args.cube_z]
    gripper_offset = [args.gripper_offset_x, args.gripper_offset_y, args.gripper_offset_z]

    run_visualization(
        cube_base=cube_base,
        gripper_offset=gripper_offset,
        use_transform=args.use_transform,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()
