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

# Path to the URDF model (shared with frame_transform)
_URDF_PATH = Path(__file__).resolve().parent.parent / "frame_transform" / "xlerobot" / "xlerobot.urdf"

# Joints to keep in the reduced model (second arm only)
_ARM_JOINTS = {"Rotation_2", "Pitch_2", "Elbow_2", "Wrist_Pitch_2", "Wrist_Roll_2"}


class IK_SO101:
    def __init__(self) -> None:
        # Build reduced model from URDF with only the second arm's 5 joints
        full_model = pin.buildModelFromUrdf(str(_URDF_PATH))
        q_neutral = pin.neutral(full_model)

        lock_ids = [
            i for i in range(1, full_model.njoints)
            if full_model.names[i] not in _ARM_JOINTS
        ]
        self.model = pin.buildReducedModel(full_model, lock_ids, q_neutral)
        self.data = self.model.createData()

        # EE frame
        self.EE_FRAME = "Fixed_Jaw_2"

        # Precompute the fixed Base_2 -> world transform (for converting targets)
        q = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        base2_oMf = self.data.oMf[self.model.getFrameId("Base_2")]
        self._base2_R = base2_oMf.rotation.copy()
        self._base2_t = base2_oMf.translation.copy()

        # IK timestep
        self.dt = 0.01  # 100 Hz

        # Initial configuration
        self.q = pin.neutral(self.model)
        self.configuration = pink.Configuration(self.model, self.data, self.q)

        # Pink tasks
        self.ee_task = FrameTask(self.EE_FRAME, position_cost=10.0, orientation_cost=0.0)
        self.posture_task = PostureTask(cost=1e-4)
        self.tasks = [self.ee_task, self.posture_task]

    def base2_to_world(self, p_base2: np.ndarray) -> np.ndarray:
        """Convert a point from Base_2 frame to the pinocchio world frame.

        Base_2 frame convention (from URDF):
            -Y is forward (arm reach direction)
            +X is left
            +Z is up
        Base_2 is rotated 180° around Z from the world frame.
        """
        return self._base2_R @ np.asarray(p_base2) + self._base2_t

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
        self.posture_task.set_target(q_seed)

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
        target_xyz: list[float],  # [x, y, z] in Base_2 frame
        gripper_offset_xyz: list[float],  # [x, y, z] in Base_2 frame
        position_tolerance: float = 1e-3,
        max_timesteps: int = 1000,
    ):
        base2_xyz = np.asarray(target_xyz) + np.asarray(gripper_offset_xyz)
        xyz = self.base2_to_world(base2_xyz)
        target_transform = pin.SE3(np.eye(3), xyz)

        best_traj: list[np.ndarray] = []
        best_error = float("inf")

        for seed_deg in self._SEED_CONFIGS_DEG:
            q_seed = np.deg2rad(seed_deg)
            # Clamp seed to joint limits
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
            # Early exit if we already converged
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

    # Target in Base_2 frame: -Y is forward, +X is left, +Z is up
    target_base2 = [0.0, -0.30, 0.01]

    print(f"Target in Base_2 frame: {target_base2}")
    print(f"Generating IK trajectory...")

    traj = arm.generate_ik(target_xyz=target_base2, gripper_offset_xyz=[0, 0, 0])

    if len(traj) > 0:
        print(f"Success! Trajectory has {len(traj)} steps.")
        target_world = arm.base2_to_world(target_base2).tolist()
        arm.visualize_ik(traj, object_xyz=target_world)
    else:
        print("IK Failed or target out of reach.")
