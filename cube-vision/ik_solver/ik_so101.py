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

# Joints to keep in the reduced model (both arms, 5 per arm)
_LEFT_ARM_JOINTS = ["Rotation_L", "Pitch_L", "Elbow_L", "Wrist_Pitch_L", "Wrist_Roll_L"]
_RIGHT_ARM_JOINTS = ["Rotation_R", "Pitch_R", "Elbow_R", "Wrist_Pitch_R", "Wrist_Roll_R"]
_ARM_JOINTS = set(_LEFT_ARM_JOINTS + _RIGHT_ARM_JOINTS)


class IK_SO101:
    def __init__(self) -> None:
        # Build reduced model from MJCF keeping only the 10 arm joints
        full_model = pin.buildModelFromMJCF(str(_MJCF_PATH))
        q_neutral = pin.neutral(full_model)

        # Lock everything except arm joints (lock head, wheels, jaw joints)
        lock_ids = [
            i for i in range(1, full_model.njoints)
            if full_model.names[i] not in _ARM_JOINTS
        ]
        self.model = pin.buildReducedModel(full_model, lock_ids, q_neutral)
        self.data = self.model.createData()

        # Map joint names to indices in reduced model q vector
        self._joint_q_idx = {}
        for i in range(1, self.model.njoints):
            name = self.model.names[i]
            self._joint_q_idx[name] = self.model.joints[i].idx_q

        # Compute q-vector index slices for each arm
        self._left_q_indices = np.array([self._joint_q_idx[j] for j in _LEFT_ARM_JOINTS])
        self._right_q_indices = np.array([self._joint_q_idx[j] for j in _RIGHT_ARM_JOINTS])

        # EE frame names
        self.EE_LEFT = "Fixed_Jaw_tip"
        self.EE_RIGHT = "Fixed_Jaw_tip_2"

        # Precompute fixed Base and Base_2 transforms (at neutral config)
        q = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        base_oMf = self.data.oMf[self.model.getFrameId("Base")]
        self._base_R = base_oMf.rotation.copy()
        self._base_t = base_oMf.translation.copy()

        base2_oMf = self.data.oMf[self.model.getFrameId("Base_2")]
        self._base2_R = base2_oMf.rotation.copy()
        self._base2_t = base2_oMf.translation.copy()

        # IK timestep
        self.dt = 0.01  # 100 Hz

        # Initial configuration
        self.q = pin.neutral(self.model)
        self.configuration = pink.Configuration(self.model, self.data, self.q)

        # Lock Wrist_Roll on both arms at 0 so grippers can't turn sideways
        for wrist_roll_joint in ["Wrist_Roll_L", "Wrist_Roll_R"]:
            idx = self._joint_q_idx[wrist_roll_joint]
            self.model.lowerPositionLimit[idx] = 0.0
            self.model.upperPositionLimit[idx] = 0.0

        # Pink tasks: one FrameTask per EE + posture
        self.ee_left_task = FrameTask(self.EE_LEFT, position_cost=10.0, orientation_cost=0.0)
        self.ee_right_task = FrameTask(self.EE_RIGHT, position_cost=10.0, orientation_cost=0.0)
        self.posture_task = PostureTask(cost=1e-4)
        self.tasks = [self.ee_left_task, self.ee_right_task, self.posture_task]

        # Preferred posture: elbow up, wrist_roll=0 to prevent sideways twist
        self._elbow_up_q_5 = np.deg2rad([0.0, 90.0, 90.0, 0.0, 0.0])

    def base_to_world(self, p_base: np.ndarray) -> np.ndarray:
        """Convert a point from Base frame to the pinocchio world frame."""
        return self._base_R @ np.asarray(p_base) + self._base_t

    def base2_to_world(self, p_base2: np.ndarray) -> np.ndarray:
        """Convert a point from Base_2 frame to the pinocchio world frame."""
        return self._base2_R @ np.asarray(p_base2) + self._base2_t

    def choose_arm(self, target_base_xyz: np.ndarray, target_base2_xyz: np.ndarray) -> str:
        """Pick the closer arm by comparing distance from target to each arm's base.

        Both targets should be in their respective arm's Base frame.
        Returns "left" or "right".
        """
        dist_left = np.linalg.norm(target_base_xyz)
        dist_right = np.linalg.norm(target_base2_xyz)
        chosen = "left" if dist_left <= dist_right else "right"
        print(f"Arm selection: left dist={dist_left:.4f}, right dist={dist_right:.4f} → {chosen}")
        return chosen

    def _get_current_ee_world_pos(self, ee_frame_name: str) -> np.ndarray:
        """Get current EE position in world frame from the current configuration."""
        pin.forwardKinematics(self.model, self.data, self.configuration.q)
        pin.updateFramePlacements(self.model, self.data)
        fid = self.model.getFrameId(ee_frame_name)
        return self.data.oMf[fid].translation.copy()

    # Seed configurations for multi-start IK (degrees, converted to rad at use).
    _SEED_CONFIGS_DEG = [
        [0.0, 0.0, 0.0, 0.0, 0.0],         # neutral
        [0.0, 90.0, 90.0, 0.0, 0.0],        # shoulder + elbow at 90°
        [0.0, 135.0, 135.0, 45.0, 0.0],     # arm folded back / raised
    ]

    def _build_seed_q(self, seed_deg_5: list[float], arm: str) -> np.ndarray:
        """Build a full 10-joint q_seed with the given 5-DOF seed for one arm,
        and neutral (zeros) for the other arm."""
        q_seed = pin.neutral(self.model)
        seed_rad = np.deg2rad(seed_deg_5)
        if arm == "left":
            q_seed[self._left_q_indices] = seed_rad
        else:
            q_seed[self._right_q_indices] = seed_rad
        # Clamp to joint limits
        q_seed = np.clip(q_seed, self.model.lowerPositionLimit, self.model.upperPositionLimit)
        return q_seed

    def _run_ik_from_seed(
        self,
        q_seed: np.ndarray,
        active_ee_frame: str,
        target_transform: "pin.SE3",
        idle_ee_frame: str,
        idle_target: "pin.SE3",
        position_tolerance: float,
        max_timesteps: int,
    ) -> tuple[list[np.ndarray], float]:
        """Run IK from a given seed. Returns (trajectory of full q, final_error)."""
        self.configuration = pink.Configuration(self.model, self.data, q_seed.copy())

        # Set active arm EE to desired target
        active_task = self.ee_left_task if active_ee_frame == self.EE_LEFT else self.ee_right_task
        idle_task = self.ee_right_task if active_ee_frame == self.EE_LEFT else self.ee_left_task

        active_task.set_target(target_transform)
        idle_task.set_target(idle_target)

        # High cost on idle arm to keep it still
        active_task.position_cost = 10.0
        idle_task.position_cost = 100.0

        # Use preferred elbow-up posture for both arms
        posture_q = pin.neutral(self.model)
        posture_q[self._left_q_indices] = self._elbow_up_q_5
        posture_q[self._right_q_indices] = self._elbow_up_q_5
        self.posture_task.set_target(posture_q)

        trajectory: list[np.ndarray] = []
        ee_frame_id = self.model.getFrameId(active_ee_frame)

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

        # Final error
        pin.forwardKinematics(self.model, self.data, self.configuration.q)
        pin.updateFramePlacements(self.model, self.data)
        final_error = np.linalg.norm(
            target_transform.translation - self.data.oMf[ee_frame_id].translation
        )
        return trajectory, final_error

    def generate_ik_bimanual(
        self,
        target_xyz: list[float],
        arm: str = "left",
        gripper_offset_xyz: list[float] | None = None,
        position_tolerance: float = 1e-3,
        max_timesteps: int = 1000,
        seed_q_rad: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Solve IK for one arm while holding the other.

        Parameters
        ----------
        target_xyz : [x, y, z] in the active arm's Base frame
        arm : "left" or "right"
        gripper_offset_xyz : optional offset in Base frame
        position_tolerance : convergence threshold (meters)
        max_timesteps : max IK iterations
        seed_q_rad : optional 5-DOF seed config (radians) for the active arm

        Returns
        -------
        Trajectory of 5-joint configs (radians) for the active arm only.
        """
        offset = np.asarray(gripper_offset_xyz) if gripper_offset_xyz else np.zeros(3)
        base_xyz = np.asarray(target_xyz) + offset

        if arm == "left":
            world_xyz = self.base_to_world(base_xyz)
            active_ee = self.EE_LEFT
            idle_ee = self.EE_RIGHT
            active_indices = self._left_q_indices
        else:
            world_xyz = self.base2_to_world(base_xyz)
            active_ee = self.EE_RIGHT
            idle_ee = self.EE_LEFT
            active_indices = self._right_q_indices

        target_transform = pin.SE3(np.eye(3), world_xyz)

        # Get idle arm's current EE position as hold target
        idle_world_pos = self._get_current_ee_world_pos(idle_ee)
        idle_target = pin.SE3(np.eye(3), idle_world_pos)

        best_traj: list[np.ndarray] = []
        best_error = float("inf")

        # Build seed list: caller-provided seed first, then hardcoded defaults
        seeds = []
        if seed_q_rad is not None:
            # Preserve current full-body state and only reseed active arm joints.
            # This keeps the idle arm continuous across sequential IK calls.
            q_custom = self.configuration.q.copy()
            q_custom[active_indices] = np.asarray(seed_q_rad, dtype=float)
            q_custom = np.clip(q_custom, self.model.lowerPositionLimit, self.model.upperPositionLimit)
            seeds.append(q_custom)
        for seed_deg in self._SEED_CONFIGS_DEG:
            seeds.append(self._build_seed_q(seed_deg, arm))

        for q_seed in seeds:
            traj, error = self._run_ik_from_seed(
                q_seed, active_ee, target_transform,
                idle_ee, idle_target,
                position_tolerance, max_timesteps,
            )
            if error < best_error:
                best_error = error
                best_traj = traj
            if best_error < position_tolerance:
                break

        if len(best_traj) >= max_timesteps:
            print(f"IK did not converge: error={best_error*1000:.1f}mm, steps={len(best_traj)}/{max_timesteps}")
            return []
        if best_error >= position_tolerance:
            print(f"IK failed to reach tolerance: error={best_error*1000:.1f}mm, tol={position_tolerance*1000:.1f}mm")
            return []

        # Update state
        if best_traj:
            self.q = best_traj[-1].copy()
            self.configuration = pink.Configuration(self.model, self.data, self.q)

        # Extract active arm's joints only
        active_traj = [q[active_indices] for q in best_traj]
        return active_traj

    # Keep the single-arm generate_ik for backwards compatibility
    def generate_ik(
        self,
        target_xyz: list[float],
        gripper_offset_xyz: list[float],
        position_tolerance: float = 1e-3,
        max_timesteps: int = 1000,
        seed_q_rad: np.ndarray | None = None,
    ):
        return self.generate_ik_bimanual(
            target_xyz, arm="left",
            gripper_offset_xyz=gripper_offset_xyz,
            position_tolerance=position_tolerance,
            max_timesteps=max_timesteps,
            seed_q_rad=seed_q_rad,
        )

    def visualize_ik(self, trajectory: list, object_xyz):
        if MeshcatVisualizer is None:
            print("Meshcat failed to import.")
            return

        ee_frame_id = self.model.getFrameId(self.EE_LEFT)

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

    print(f"Reduced model has {arm.model.njoints - 1} joints:")
    for i in range(1, arm.model.njoints):
        name = arm.model.names[i]
        idx = arm.model.joints[i].idx_q
        print(f"  [{idx}] {name}")

    print(f"\nLeft arm q indices: {arm._left_q_indices}")
    print(f"Right arm q indices: {arm._right_q_indices}")

    target_base = [0.0, 0.30, 0.01]
    print(f"\nTarget in Base frame: {target_base}")
    print(f"Generating bimanual IK for left arm...")

    traj = arm.generate_ik_bimanual(target_xyz=target_base, arm="left")

    if len(traj) > 0:
        print(f"Success! Trajectory has {len(traj)} steps.")
        print(f"Final joint config (deg): {np.rad2deg(traj[-1])}")
    else:
        print("IK Failed or target out of reach.")
