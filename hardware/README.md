# Hardware files

| File | What it is |
|---|---|
| `XLeRobot-Pro.3mf` | PrusaSlicer project with every printed part laid out, print settings included (set up for a Prusa CORE One) |
| `XLeRobot-Pro.stl` | The same parts as a single STL |
| `nvidia-jetson-mounting-frame.stl` | Mounting frame for the Jetson Orin Nano Super |
| `XLeRobot040_armbase.step` | Arm base, as editable CAD (STEP) |
| `xlerobot/` | URDF and meshes for visualization |

The 3MF also holds the passively ducted Jetson enclosure, the arm-base mounts, the neck, and the wheel hubs and motor mounts.

## Print settings

All parts in the paper were printed in PLA at 0.2 mm layers. The 3MF already applies the settings below. The 18 parts of the two SO-101 arms are named "(arm, High-Shell)" in the object list and carry 4 perimeters; everything else uses the default profile.

| Parts | Perimeters | Infill | Notes |
|---|---|---|---|
| Every printed part of both arms (High-Shell) | 4 | 15% | The profile tested in the paper |
| Everything else | 2 | 15% | The default profile |
| Gripper fingers | — | — | Print the Fin-Ray fingers in TPU 95A |

Wall count matters much more than infill for links in bending. In five cantilever tests per profile, loading each arm 18.6 cm from the shoulder:

| Profile | Failure load | Arm mass | Where it broke |
|---|---|---|---|
| Baseline: 2 perimeters, 15% | 11.06 ± 0.63 N | 758 g | Printed link |
| High-Infill: 2 perimeters, 50% | 12.00 ± 1.28 N | 996 g | — |
| High-Shell: 4 perimeters, 15% | 18.49 ± 2.15 N | 838 g | Servo or gearbox, every time |

Because no High-Shell link broke, 18.49 N is a lower bound on the links themselves: the servo now sets the limit of the arm.

## About the URDF

`xlerobot/xlerobot.urdf` includes wrist cameras and cart wheels, and it leaves out the Jetson, the power station, the omni-wheel base, and the neck RealSense. It is good for visualization but is not the exact configuration in the paper. The model the vision and IK code uses is in `src/xlerobot_pro/assets/xlerobot/`.
