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
_MJCF_PATH = Path(__file__).resolve().parent.parent / "assets" / "xlerobot.xml"

# Joints to keep in the reduced model (second arm only)
_ARM_JOINTS = {"Rotation_R", "Pitch_R", "Elbow_R", "Wrist_Pitch_R", "Wrist_Roll_R"}


class IK_SO101:
    def __init__(self) -> None:
        # Build reduced model from MJCF with only the second arm's 5 joints
        full_model = pin.buildModelFromMJCF(str(_MJCF_PATH))
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
        self.posture_task = PostureTask(cost=1e-2)
        self.tasks = [self.ee_task, self.posture_task]

    def base2_to_world(self, p_base2: np.ndarray) -> np.ndarray:
        """Convert a point from Base_2 frame to the reduced model's world frame."""
        return self._base2_R @ np.asarray(p_base2) + self._base2_t

    def generate_ik(
        self,
        target_xyz: list[float],  # [x, y, z] in reduced model world frame
        gripper_offset_xyz: list[float],  # [x, y, z]
        position_tolerance: float = 1e-3,
        max_timesteps: int = 500,
    ):
        xyz = np.asarray(target_xyz) + np.asarray(gripper_offset_xyz)
        target_transform = pin.SE3(np.eye(3), xyz)
        self.ee_task.set_target(target_transform)
        self.posture_task.set_target(self.configuration.q)

        trajectory: list[np.ndarray] = []

        ee_frame_id = self.model.getFrameId(self.EE_FRAME)

        for step in range(max_timesteps):
            pin.forwardKinematics(self.model, self.data, self.configuration.q)
            pin.updateFramePlacements(self.model, self.data)

            transform_current = self.data.oMf[ee_frame_id]
            pos_error = target_transform.translation - transform_current.translation

            if np.linalg.norm(pos_error) < position_tolerance:
                break

            self.posture_task.set_target(self.configuration.q)

            try:
                dq = solve_ik(self.configuration, self.tasks, self.dt, solver="quadprog")
            except Exception as e:
                print(f"IK Solver Failed at Step{step}. Error: {e}")
                break

            dq_max = 1.0
            dq = np.clip(dq, -dq_max, dq_max)
            dq *= 0.2

            self.configuration.integrate_inplace(dq, self.dt)
            trajectory.append(self.configuration.q.copy())

        return trajectory

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

    # Target in Base_2 frame, then convert to world
    target_base2 = [0.0, -0.30, 0.01]
    target_world = arm.base2_to_world(target_base2).tolist()

    print(f"Target in Base_2 frame: {target_base2}")
    print(f"Target in world frame:  {target_world}")
    print(f"Generating IK trajectory...")

    traj = arm.generate_ik(target_xyz=target_world, gripper_offset_xyz=[0, 0, 0])

    if len(traj) > 0:
        print(f"Success! Trajectory has {len(traj)} steps.")
        arm.visualize_ik(traj, object_xyz=target_world)
    else:
        print("IK Failed or target out of reach.")
