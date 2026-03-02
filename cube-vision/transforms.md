  1. Camera optical frame → camera_xyz_to_base_xyz() → Base frame
    - Camera sees object in optical convention (Z forward, X right, Y down)
    - Pinocchio FK computes the camera-to-Base transform using head pan/tilt joint angles
    - Motor angle offsets are applied (pan negated, tilt offset by 14°)
  2. Base frame → base_to_world() → World frame
    - -Y is forward, +Z is up (quirk of the SO101 mesh)
    - FK-derived rotation/translation maps Base coords to pinocchio world coords
  3. World frame → generate_ik() → Joint angles
    - Pink IK solver finds joint angles that put the end effector (Fixed_Jaw) at the world target
    - Multi-seed approach to avoid local minima
    - Only uses the 5 arm joints (reduced model, head/wheels locked)
  4. Joint angles (MJCF rad) → mjcf_to_motor() → Motor degrees
    - Pitch: 90° - value
    - Elbow: value - 90°
    - Rotation, wrist pitch, wrist roll pass through
