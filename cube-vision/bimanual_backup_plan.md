 Bimanual IK: Choose Closest Arm                        

 Context

 The robot has two arms — Base (IDs 7-12) and Base_2 (IDs 1-6) — mirrored in Y (Base at Y=-0.11,
 Base_2 at Y=+0.11). Currently only the Base arm has IK. We need to detect objects, determine which
 arm is closer, run IK on that arm, and hold the other arm still.

 Changes

 1. Parameterize IK solver (ik_solver/ik_so101.py)

 - Add an arm parameter to IK_SO101.__init__() accepting "left" or "right" (default "left" for
 backwards compat)
 - Right arm uses joints: Rotation_R, Pitch_R, Elbow_R, Wrist_Pitch_R, Wrist_Roll_R
 - Right arm uses EE frame: Fixed_Jaw_2, base frame: Base_2
 - Extract _ARM_JOINTS and EE_FRAME/base frame name into the constructor based on the arm parameter

 2. Add camera_xyz_to_base2_xyz() to frame_transform (frame_transform/frame_transform.py)

 - Add _BASE_2_FRAME_ID = _model.getFrameId("Base_2")
 - Add a new function camera_xyz_to_base2_xyz() that transforms to the Base_2 frame (same logic,
 different frame ID)
 - Or better: refactor camera_xyz_to_base_xyz to accept an optional base_frame parameter, defaulting
  to "Base"

 3. Add mjcf_to_motor_2() in control script (control_single_bus.py)

 - The right arm (Base_2) joint axes may differ from the left arm. Looking at the XML:
   - Rotation_R axis: 0 -1 0 (same as Rotation_L 0 -1 0) — same sign convention
   - Pitch_R axis: -1 0 0 (same as Pitch_L -1 0 0)
   - Elbow_R axis: 1 0 0 (same as Elbow_L 1 0 0)
   - So the joint conventions are identical — mjcf_to_motor should work for both arms
 - However, Base_2 euler is the same as Base (0 0 -1.5708), arms are symmetric, so the same
 mjcf_to_motor() conversion applies
 - Add ARM_JOINT_KEYS_2 for the _2 motor names and a traj_to_goals_2() that maps to those keys

 4. Update control logic (control_single_bus.py)

 - Create two IK solver instances: ik_left = IK_SO101(arm="left"), ik_right = IK_SO101(arm="right")
 - Transform detected object to both Base and Base_2 frames
 - Compare distances to pick the closer arm
 - Run IK on the chosen arm
 - Send trajectory to the chosen arm's motors
 - Hold the other arm at its current position (read current pos, keep sending it)

 Files modified

 - ik_solver/ik_so101.py — parameterize arm selection
 - frame_transform/frame_transform.py — add Base_2 frame support
 - control_single_bus.py — bimanual arm selection logic

 Verification

 - Run python calibrate.py --bus arm to calibrate
 - Run the control script, place object to the left and right of center to verify each arm is
 selected appropriately

