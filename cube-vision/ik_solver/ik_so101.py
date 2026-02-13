import time
import numpy as np
from pathlib import Path

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask

import meshcat.geometry as g
import meshcat.transformations as tf

try:
    from pinocchio.visualize import MeshcatVisualizer
except ModuleNotFoundError:
    MeshcatVisualizer = None


class IK_SO101:
    def __init__(self) -> None:
        # File paths for model urdf and frame data
        self.BASE_DIR = Path(__file__).resolve().parent
        self.URDF_PATH = str(self.BASE_DIR / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf")
        self.MESH_DIR = str(self.BASE_DIR / "SO-ARM100" / "Simulation" / "SO101")
        # ID of end effector
        self.EE_FRAME = "gripper_frame_link"

        # Sets change in time per control loop
        self.dt = 0.01  # 100 hertz

        # configuring pink off of URDF
        full_model = pin.buildModelFromUrdf(str(self.URDF_PATH))

        # Build a reduced model with the gripper joint locked at neutral position
        # so the IK solver only operates on the arm joints
        q_neutral = pin.neutral(full_model)
        gripper_joint_id = full_model.getJointId("gripper")
        self.model = pin.buildReducedModel(full_model, [gripper_joint_id], q_neutral)
        self.data = self.model.createData()
        self.q = pin.neutral(self.model)

        # # Set wrist roll to 90 degrees (π/2 radians) for sideways orientation
        # jid = self.model.getJointId("wrist_roll")
        # idx = self.model.joints[jid].idx_q
        # self.q[idx] = np.pi / 2
        self.configuration = pink.Configuration(self.model, self.data, self.q)

        # pink tasks that can be used to create sets of frames to reach the final goal position
        #
        # ee_task gets gets end effector to desired position
        self.ee_task = FrameTask(self.EE_FRAME, position_cost=10.0, orientation_cost=0.0)
        # posture tasks describes what position it should bias towards while moving
        self.posture_task = PostureTask(cost=1e-2)
        # make a list of the needed tasks
        self.tasks = [self.ee_task, self.posture_task]

        # self.gripper_offset = np.array([-0.015, 0.0, 0.03])

    def generate_ik(  # TODO: Add support for target position for better grasping
        self,
        target_xyz: list[float],  # [x, y, z]
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
            # calculates forward kinematics
            pin.forwardKinematics(self.model, self.data, self.configuration.q)
            pin.updateFramePlacements(self.model, self.data)

            # pin fills data.omf with forward kinematics and frame placements
            transform_current = self.data.oMf[ee_frame_id]

            # error found by comparing target transform to current location
            pos_error = target_transform.translation - transform_current.translation

            # if we enter within the position tolerance, break and return trajectory
            if np.linalg.norm(pos_error) < position_tolerance:
                break

            self.posture_task.set_target(self.configuration.q)

            # wrapped in try except so impossible calcs exit for safety
            try:
                dq = solve_ik(self.configuration, self.tasks, self.dt, solver="quadprog")
            except Exception as e:
                print(f"IK Solver Failed at Step{step}. Error: {e}")
                break

            # clipping
            dq_max = 1.0
            dq = np.clip(dq, -dq_max, dq_max)

            # damping
            dq *= 0.2

            # update  q in place
            self.configuration.integrate_inplace(dq, self.dt)
            # add latest joint positions to list of steps
            trajectory.append(self.configuration.q.copy())

        return trajectory

    def visualize_ik(
        self,
        trajectory: list,
        object_xyz,
    ):
        if MeshcatVisualizer is not None:
            # generates physical model
            visual_model = pin.buildGeomFromUrdf(
                self.model,
                self.URDF_PATH,
                pin.GeometryType.VISUAL,
                package_dirs=[self.MESH_DIR],
            )
            # generates collisions
            collision_model = pin.buildGeomFromUrdf(
                self.model,
                str(self.URDF_PATH),
                pin.GeometryType.COLLISION,
                package_dirs=[str(self.MESH_DIR)],
            )

            ee_frame_id = self.model.getFrameId(self.EE_FRAME)

            # initiates visualizer, displays model
            viz = MeshcatVisualizer(self.model, collision_model, visual_model)
            viz.initViewer(open=True)
            viz.loadViewerModel()
            viz.display(self.q)

            # creates cube at object target point
            viewer = viz.viewer
            cube = g.Box([0.017, 0.017, 0.017])  # .17 cm cube
            material = g.MeshLambertMaterial(color=0x00FFFF, opacity=0.8)
            cube_pos = np.array(object_xyz)
            viewer["target_cube"].set_object(cube, material)
            viewer["target_cube"].set_transform(tf.translation_matrix(cube_pos))

            viz.viewer["ee_point"].set_object(g.Sphere(0.005), g.MeshLambertMaterial(color=0xFF0000))

            pos = self.data.oMf[ee_frame_id].translation
            viz.viewer["ee_point"].set_transform(tf.translation_matrix(pos))

            # runs through the generated trajectory and visualizes it
            for q_step in trajectory:
                viz.display(q_step)
                pin.forwardKinematics(self.model, self.data, q_step)
                pin.updateFramePlacements(self.model, self.data)
                time.sleep(self.dt)
                ee_pos = self.data.oMf[ee_frame_id].translation
                viz.viewer["ee_point"].set_transform(tf.translation_matrix(ee_pos))

        else:
            print("Meshcat failed to import.")


if __name__ == "__main__":
    # 1. Create the solver
    arm = IK_SO101()

    # 2. target position (x, y, z) in meters
    target = [0.30, 0.0, 0.0125]

    gripper_offset = [0.0, 0.0, 0.0]

    print(f"Generating IK trajectory to: {target}")

    # 3. Solve
    traj = arm.generate_ik(target_xyz=target, gripper_offset_xyz=gripper_offset)

    if len(traj) > 0:
        print(f"Success! Trajectory has {len(traj)} steps.")
        print("Opening visualizer...")

        # 4. Visualize
        arm.visualize_ik(traj, object_xyz=target)
    else:
        print("IK Failed or target out of reach.")
