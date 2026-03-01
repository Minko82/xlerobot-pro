from pathlib import Path

URDF_PATH = Path(__file__).resolve().parent.parent / "assets" / "xlerobot" / "xlerobot_front.urdf"

BASE_FRAME = "Base"
CAMERA_FRAME = "head_camera_rgb_optical_frame"
EE_FRAME = "Fixed_Jaw_tip"

ARM_MOTORS = [
    "left_arm_shoulder_pan",
    "left_arm_shoulder_lift",
    "left_arm_elbow_flex",
    "left_arm_wrist_flex",
    "left_arm_wrist_roll",
]
ARM_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]

HEAD_MOTORS = ["head_motor_1", "head_motor_2"]
HEAD_JOINTS = ["head_pan_joint", "head_tilt_joint"]
