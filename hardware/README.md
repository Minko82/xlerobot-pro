# Hardware files

| File | What it is |
|---|---|
| `XLeRobot-Pro.3mf` | PrusaSlicer project with every printed part laid out (set up for a Prusa CORE One) |
| `XLeRobot-Pro.stl` | The same parts as a single STL |
| `xlerobot/` | URDF and meshes for visualization |

## Print settings

All parts in the paper were printed in PLA at 0.2 mm layers. The 3MF project uses
the default profile throughout (2 perimeters, 15% infill), so **change the arm
links before you slice**:

| Parts | Perimeters | Infill | Notes |
|---|---|---|---|
| Load-bearing links of both SO-101 arms (High-Shell) | 4 | 15% | The profile tested in the paper |
| Parts that are meant to flex | 2 | 15% | Keep the default, or they lose their compliance |
| Gripper fingers | — | — | Print the Fin-Ray fingers in TPU 95A |

Wall count matters much more than infill for links in bending. In five
cantilever tests per profile, loading each arm 18.6 cm from the shoulder:

| Profile | Failure load | Arm mass | Where it broke |
|---|---|---|---|
| Baseline: 2 perimeters, 15% | 11.06 ± 0.63 N | 758 g | Printed link |
| High-Infill: 2 perimeters, 50% | 12.00 ± 1.28 N | 996 g | — |
| High-Shell: 4 perimeters, 15% | 18.49 ± 2.15 N | 838 g | Servo or gearbox, every time |

Because no High-Shell link broke, 18.49 N is a lower bound on the links
themselves: the servo now sets the limit of the arm.

## About the URDF

`xlerobot/xlerobot.urdf` includes wrist cameras and cart wheels, and it leaves out
the Jetson, the power station, the omni-wheel base, and the neck RealSense. It is
good for visualization but is not the exact configuration in the paper. The
model the vision and IK code uses is in `src/xlerobot_pro/assets/xlerobot/`.
