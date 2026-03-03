import time
import numpy as np
from pathlib import Path

import pinocchio as pin

import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask

try:
    from pinocchio.visualize import MeshcatVisualizer
    import meshcat.geometry as g
    import meshcat.transformations as tf
except ModuleNotFoundError:
    MeshcatVisualizer = None

# Path to the MJCF model (shared with frame_transform)
_MJCF_PATH = Path(__file__).resolve().parent.parent / "frame_transform" / "xlerobot" / "xlerobot.xml"

# Joints to keep in the reduced model (first arm only)
_ARM_JOINTS = {"Rotation_L", "Pitch_L", "Elbow_L", "Wrist_Pitch_L", "Wrist_Roll_L"}


class IK_SO101:
    def __init__(self) -> None:
        # Build reduced model from MJCF with only the first arm's 5 joints
        full_model = pin.buildModelFromMJCF(str(_MJCF_PATH))
        q_neutral = pin.neutral(full_model)

        lock_ids = [
            i for i in range(1, full_model.njoints)
            if full_model.names[i] not in _ARM_JOINTS
        ]
        self.model = pin.buildReducedModel(full_model, lock_ids, q_neutral)
        self.data = self.model.createData()

        # EE frame
        self.EE_FRAME = "Fixed_Jaw"

        # Precompute the fixed Base -> world transform (for converting targets)
        q = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        base_oMf = self.data.oMf[self.model.getFrameId("Base")]
        self._base_R = base_oMf.rotation.copy()
        self._base_t = base_oMf.translation.copy()

        # IK timestep
        self.dt = 0.01  # 100 Hz

        # Initial configuration
        self.q = pin.neutral(self.model)
        self.configuration = pink.Configuration(self.model, self.data, self.q)

        # Pink tasks
        self.ee_task = FrameTask(self.EE_FRAME, position_cost=10.0, orientation_cost=0.0)
        self.posture_task = PostureTask(cost=1e-2)
        self.tasks = [self.ee_task, self.posture_task]

        # Preferred "elbow-up" posture (MJCF deg): shoulder and elbow high
        # so Pink biases toward concave poses where the gripper approaches from above.
        self._elbow_up_q = np.deg2rad([0.0, 90.0, 90.0, 0.0, 0.0])

    def base_to_world(self, p_base: np.ndarray) -> np.ndarray:
        """Convert a point from Base frame to the pinocchio world frame."""
        return self._base_R @ np.asarray(p_base) + self._base_t

    # Seed configurations for multi-start IK (degrees, converted to rad at use).
    # Neutral + "arm raised" seeds to escape local minima for high targets.
    _SEED_CONFIGS_DEG = [
        [0.0, 0.0, 0.0, 0.0, 0.0],         # neutral (arm extended forward)
        [0.0, 90.0, 90.0, 0.0, 0.0],        # shoulder + elbow at 90°
        [0.0, 135.0, 135.0, 45.0, 0.0],     # arm folded back / raised
    ]

    def _run_ik_from_seed(
        self,
        q_seed: np.ndarray,
        target_transform: "pin.SE3",
        position_tolerance: float,
        max_timesteps: int,
    ) -> tuple[list[np.ndarray], float]:
        """Run IK from a given seed configuration. Returns (trajectory, final_error)."""
        self.configuration = pink.Configuration(self.model, self.data, q_seed.copy())
        self.ee_task.set_target(target_transform)
        self.posture_task.set_target(self._elbow_up_q)

        trajectory: list[np.ndarray] = []
        ee_frame_id = self.model.getFrameId(self.EE_FRAME)

        for step in range(max_timesteps):
            pin.forwardKinematics(self.model, self.data, self.configuration.q)
            pin.updateFramePlacements(self.model, self.data)

            pos_error = target_transform.translation - self.data.oMf[ee_frame_id].translation
            error_norm = np.linalg.norm(pos_error)

            if error_norm < position_tolerance:
                break

            try:
                dq = solve_ik(self.configuration, self.tasks, self.dt, solver="quadprog")
            except Exception as e:
                print(f"IK Solver Failed at Step{step}. Error: {e}")
                break

            self.configuration.integrate_inplace(dq, self.dt)
            trajectory.append(self.configuration.q.copy())

        # Compute final error
        pin.forwardKinematics(self.model, self.data, self.configuration.q)
        pin.updateFramePlacements(self.model, self.data)
        final_error = np.linalg.norm(
            target_transform.translation - self.data.oMf[ee_frame_id].translation
        )
        return trajectory, final_error

    def generate_ik(
        self,
        target_xyz: list[float],  # [x, y, z] in Base frame
        gripper_offset_xyz: list[float],  # [x, y, z] in Base frame
        position_tolerance: float = 1e-3,
        max_timesteps: int = 1000,
        seed_q_rad: np.ndarray | None = None,
    ):
        base_xyz = np.asarray(target_xyz) + np.asarray(gripper_offset_xyz)
        xyz = self.base_to_world(base_xyz)
        target_transform = pin.SE3(np.eye(3), xyz)

        best_traj: list[np.ndarray] = []
        best_error = float("inf")

        # Build seed list: caller-provided seed first, then hardcoded defaults
        seeds_rad = []
        if seed_q_rad is not None:
            seeds_rad.append(np.asarray(seed_q_rad, dtype=float))
        for seed_deg in self._SEED_CONFIGS_DEG:
            seeds_rad.append(np.deg2rad(seed_deg))

        for q_seed in seeds_rad:
            q_seed = np.clip(
                q_seed,
                self.model.lowerPositionLimit,
                self.model.upperPositionLimit,
            )
            traj, error = self._run_ik_from_seed(
                q_seed, target_transform, position_tolerance, max_timesteps,
            )
            if error < best_error:
                best_error = error
                best_traj = traj
            if best_error < position_tolerance:
                break

        # Update instance state to match the best result
        if best_traj:
            self.q = best_traj[-1].copy()
            self.configuration = pink.Configuration(self.model, self.data, self.q)

        return best_traj

    def visualize_ik(self, trajectory: list, object_xyz):
        if MeshcatVisualizer is None:
            print("Meshcat failed to import.")
            return

        ee_frame_id = self.model.getFrameId(self.EE_FRAME)

        collision_model = pin.GeometryModel()
        visual_model = pin.GeometryModel()
        viz = MeshcatVisualizer(self.model, collision_model, visual_model)
        viz.initViewer(open=True)
        viz.loadViewerModel()
        viz.display(self.q)

        viewer = viz.viewer
        cube = g.Box([0.017, 0.017, 0.017])
        material = g.MeshLambertMaterial(color=0x00FFFF, opacity=0.8)
        cube_pos = np.array(object_xyz)
        viewer["target_cube"].set_object(cube, material)
        viewer["target_cube"].set_transform(tf.translation_matrix(cube_pos))

        viz.viewer["ee_point"].set_object(g.Sphere(0.005), g.MeshLambertMaterial(color=0xFF0000))
        pos = self.data.oMf[ee_frame_id].translation
        viz.viewer["ee_point"].set_transform(tf.translation_matrix(pos))

        for q_step in trajectory:
            viz.display(q_step)
            pin.forwardKinematics(self.model, self.data, q_step)
            pin.updateFramePlacements(self.model, self.data)
            time.sleep(self.dt)
            ee_pos = self.data.oMf[ee_frame_id].translation
            viz.viewer["ee_point"].set_transform(tf.translation_matrix(ee_pos))


if __name__ == "__main__":
    arm = IK_SO101()

    target_base = [0.0, 0.30, 0.01]

    print(f"Target in Base frame: {target_base}")
    print(f"Generating IK trajectory...")

    traj = arm.generate_ik(target_xyz=target_base, gripper_offset_xyz=[0, 0, 0])

    if len(traj) > 0:
        print(f"Success! Trajectory has {len(traj)} steps.")
        target_world = arm.base_to_world(target_base).tolist()
        arm.visualize_ik(traj, object_xyz=target_world)
    else:
        print("IK Failed or target out of reach.")
