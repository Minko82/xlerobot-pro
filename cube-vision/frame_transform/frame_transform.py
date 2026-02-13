from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import pinocchio as pin


@dataclass
class FrameTransform:
    model: pin.Model
    data: pin.Data
    base_frame_id: int
    cam_frame_id: int

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str,
        base_frame: str = "Base",
        cam_frame: str = "head_camera_rgb_optical_frame",
    ) -> "FrameTransform":
        model = pin.buildModelFromUrdf(urdf_path)
        data = model.createData()

        # Verify frames exist
        base_frame_id = model.getFrameId(base_frame)
        cam_frame_id = model.getFrameId(cam_frame)

        if base_frame_id == len(model.frames):
            raise ValueError(f"Base frame '{base_frame}' not found in URDF frames.")
        if cam_frame_id == len(model.frames):
            raise ValueError(f"Camera frame '{cam_frame}' not found in URDF frames.")

        return cls(model=model, data=data, base_frame_id=base_frame_id, cam_frame_id=cam_frame_id)

    def neutral_q(self) -> np.ndarray:
        return pin.neutral(self.model)

    def set_joint_1d(self, q: np.ndarray, joint_name: str, value: float) -> None:
        """
        Set a 1-DoF joint (revolute/prismatic/continuous) by name into the configuration vector q.
        Pinocchio stores joints in model.joints; each joint has an idx_q and nq.
        """
        jid = self.model.getJointId(joint_name)
        if jid == 0:
            # Pinocchio uses 0 for "universe"
            raise ValueError(f"Joint '{joint_name}' not found in model joints.")
        j = self.model.joints[jid]
        if j.nq != 1:
            raise ValueError(f"Joint '{joint_name}' has nq={j.nq}; expected 1.")
        q[j.idx_q] = float(value)

    def build_q(self, joint_values: Dict[str, float], q0: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Build a configuration vector q from a dict of joint_name -> value.
        Any joints not provided remain at the default (neutral or q0).
        """
        q = self.neutral_q() if q0 is None else np.array(q0, dtype=float, copy=True)

        for name, val in joint_values.items():
            self.set_joint_1d(q, name, val)

        return q

    def base_T_cam(self, q: np.ndarray) -> pin.SE3:
        """
        Compute transform ^Base T_cam (i.e., Base <- Camera).
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        oT_base = self.data.oMf[self.base_frame_id]  # world <- Base
        oT_cam = self.data.oMf[self.cam_frame_id]    # world <- Cam

        # Base <- Cam  =  (world<-Base)^-1 * (world<-Cam)
        return oT_base.inverse() * oT_cam

    def transform_point_cam_to_base(
        self,
        p_cam: np.ndarray,
        q: np.ndarray,
    ) -> np.ndarray:
        """
        Convert a 3D point in camera frame into Base frame using ^Base T_cam.
        """
        p_cam = np.asarray(p_cam, dtype=float).reshape(3)
        T = self.base_T_cam(q)
        return np.asarray(T.rotation, dtype=float) @ p_cam + np.asarray(T.translation, dtype=float).ravel()


# -------------------------
# Convenience wrapper
# -------------------------

def camera_xyz_to_base_xyz(
    x: float,
    y: float,
    z: float,
    joint_values: Dict[str, float],
    base_frame: str = "Base",
    cam_frame: str = "head_camera_rgb_optical_frame",
    urdf_path: str = str(Path(__file__).parent / "xlerobot/xlerobot.urdf"),
) -> Tuple[float, float, float]:
    """
    Transform (x,y,z) from cam_frame into base_frame.

    joint_values MUST include any joints on the kinematic path between base_frame and cam_frame
    that are not fixed. For your URDF, that is at least:
      - head_pan_joint (radians)
      - head_tilt_joint (radians)

    If you are using root_x_axis_joint / root_y_axis_joint / root_z_rotation_joint in your runtime
    and they are not zero, include them too (meters, meters, radians respectively).
    """
    tfm = FrameTransform.from_urdf(urdf_path, base_frame=base_frame, cam_frame=cam_frame)
    q = tfm.build_q(joint_values)

    p_base = tfm.transform_point_cam_to_base(np.array([x, y, z]), q)
    return float(p_base[0]), float(p_base[1]), float(p_base[2])


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    URDF = str(Path(__file__).parent / "xlerobot/xlerobot.urdf")

    # Example camera point in head_camera_rgb_optical_frame (meters)
    x, y, z = 0.02, -0.01, 0.55

    # Supply current joint values for the chain.
    # If your head is fixed and never moves, set both to 0.0 and keep them there.
    joint_vals = {
        "head_pan_joint": 0.0,     # rad
        "head_tilt_joint": 0.0,    # rad

        # Only include these if you actually use them (nonzero) in your runtime:
        # "root_x_axis_joint": 0.0,        # meters
        # "root_y_axis_joint": 0.0,        # meters
        # "root_z_rotation_joint": 0.0,    # rad
    }

    xb, yb, zb = camera_xyz_to_base_xyz(URDF, x, y, z, joint_vals)
    print("p_base =", (xb, yb, zb))
