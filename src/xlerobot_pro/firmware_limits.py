"""Firmware limits for the robot's servos.

This file is the one place where the torque, acceleration, and speed limits
for every servo are set. The torque and acceleration values are the ones in
Table III of the paper:

    Supply                      Torque_Limit   Acceleration   Outlet
    Bus 1: 12 arm servos        450            40             12 V car outlet (rated 10 A)
    Bus 2: head + 3 wheels      650            20             USB-C outlet
    Jetson                      --             --             a second USB-C outlet

The arm setting comes from the paper's current estimate, Eq. (1):

    I_est = N * I0 + N_a * T / K_t,    K_t = T_stall / (I_stall - I0) = 1.167 N*m/A

With the datasheet values (T_stall = 2.94 N*m, I0 = 0.18 A, I_stall = 2.7 A),
two arm joints loaded at Torque_Limit 450 (1.32 N*m nominal) and the other ten
at I0 give about 4.43 A, under a 5 A design budget for the arm bus (half the car
outlet's 10 A rating). At 650 the same case gives about 5.44 A. This is a
nominal estimate, not a stall guarantee: twelve arm joints all loaded at 450
would draw far more.

On the arm bus, an inline meter on one arm's supply lead (FNIRSI 2C53T, peak
hold) read 1.41 A over five six-joint moves at these settings, or about 2.8 A
for two arms by extrapolation (xlerobot-pro-data, A1/bus_b_peak_left_run4).
Bus 2's current margin has not been measured.

The speed caps (ARM_MAX_VELOCITY, WHEEL_MAX_RAW_SPEED) and the 65 C ceiling are
extra software limits. They are not part of Table III.

Edit the constants below to change the limits. They are applied automatically
wherever motors are configured: by the robot classes at connect()/configure()
time and by the vision demos before any motion. To check what the motors
actually stored, run ``python diagnostics/verify_motor_limits.py``.

Worth knowing before you change anything:

- The base accelerates at about 0.15 m/s^2 at most. WHEEL_NECK_ACCELERATION = 20
  units x 8.7 deg/s^2 x 0.05 m wheel radius = 0.152 m/s^2, and commanding 0.25
  and 0.5 m/s^2 produced 0.151 and 0.136 m/s^2. Asking for more does not help.
- With 366 g in the gripper, right_shoulder_lift follows only about 1 % of a
  commanded slew at Torque_Limit 450, while right_elbow_flex reaches about 29 %.
  These slew runs are not in the paper. Choose a different joint rather than
  raising the limit.

Register semantics (Feetech STS3215):

- ``Torque_Limit``: 0-1000 in 0.1 % steps. It limits PWM duty, so the torque it
  allows is a nominal estimate, about 2.94 N*m x value / 1000: 1.32 N*m at 450,
  1.91 N*m at 650.
- ``Acceleration``: 0-254, ramp steepness (one unit is about 8.7 deg/s^2).
  Lower values give softer starts and smaller current spikes; this is the main
  speed governor for position moves.
- ``Present_Temperature``: whole degrees C from the servo's internal sensor.
  The firmware cuts torque above 70 C. The 65 C ceiling below ends a run first.
"""

# ─────────────────────────────────────────────────────────────────────
# Bus 2: head and wheels (USB-C outlet on the power station)
# ─────────────────────────────────────────────────────────────────────

#: Maximum torque for the head and wheel motors.
#: Firmware units 0-1000; 650 ~= 1.91 N*m nominal on the STS3215.
#: The USB-C outlet's current margin was not measured, so raise this with care.
WHEEL_NECK_TORQUE_LIMIT = 650

#: Maximum acceleration for the head and wheel motors ("Medium" ramp).
#: Firmware units 0-254; lower = gentler starts = smaller current spikes.
WHEEL_NECK_ACCELERATION = 20

#: Maximum raw wheel velocity command (firmware ticks) for the mobile
#: base. Goal_Velocity commands above this are scaled down uniformly,
#: capping the robot's top driving speed.
WHEEL_MAX_RAW_SPEED = 3000

# ─────────────────────────────────────────────────────────────────────
# Bus 1: arms (12 V car outlet, 5 A design budget)
# ─────────────────────────────────────────────────────────────────────

#: Maximum torque for the arm motors.
#: Firmware units 0-1000; 450 ~= 1.32 N*m nominal on the STS3215. Chosen with
#: Eq. (1) so the arm-bus estimate stays under the 5 A budget. In the paper,
#: one shoulder at this setting held 0.7 kg, 18.6 cm from the joint, for
#: 30 min. That is a measured hold, not a payload rating.
ARM_TORQUE_LIMIT = 450

#: RETIRED, do not use. Kept only so that old imports fail loudly instead of
#: silently falling back to a default.
#:
#: It let shoulder_lift follow a slew trajectory under load by raising two arm
#: joints to 650. By Eq. (1) that moves the arm-bus estimate from 4.43 A to
#: 5.44 A, past the 5 A budget. B1 arm-slew runs whose run_info.json says
#: boosted=true were collected outside these limits; report them as such, or
#: repeat them at 450.
SLEW_TORQUE_LIMIT = None

#: Maximum acceleration for the arm motors ("Soft" ramp), chosen to
#: soften current spikes at the start of each move.
ARM_ACCELERATION = 40

#: Maximum speed for the arm motors (``Maximum_Velocity_Limit``
#: register). Caps how fast a position move may run regardless of the
#: commanded trajectory.
ARM_MAX_VELOCITY = 100

# ─────────────────────────────────────────────────────────────────────
# Thermal
# ─────────────────────────────────────────────────────────────────────

#: Temperature ceiling for sustained-load work, degrees C.
#: The STS3215 firmware cuts torque above 70 C. A run should stop before
#: that: a firmware trip is abrupt and gives no controlled descent, whereas
#: stopping here leaves margin to lower the arms gently. This is also the
#: stop condition used in the paper's holding tests.
SERVO_TEMP_CEILING_C = 65

#: Seconds over which to bleed torque away when releasing a loaded arm.
#: Cutting Torque_Limit to zero in one write drops the arm; stepping it
#: down lets gravity lower it against progressively weaker resistance.
TORQUE_RELEASE_SECONDS = 8.0

# ─────────────────────────────────────────────────────────────────────
# Power-on defaults (EPROM)
# ─────────────────────────────────────────────────────────────────────

#: ``Max_Torque_Limit`` EPROM ceiling — the value the firmware copies
#: into ``Torque_Limit`` at power-on, before the software limits above
#: are applied at configure() time. Must be >= both torque limits above.
MAX_TORQUE_EPROM = 800

# ─────────────────────────────────────────────────────────────────────
# Application helpers (used by robot classes and demos — no need to
# call these yourself unless you are writing a new control script)
# ─────────────────────────────────────────────────────────────────────


def apply_arm_limits(bus, motor_names) -> None:
    """Write the arm (Bus 1) limits to the given motors.

    Call with bus torque disabled (e.g. during ``configure()``); the
    caller re-enables torque afterwards.
    """
    for name in motor_names:
        bus.write("Torque_Limit", name, ARM_TORQUE_LIMIT)
        bus.write("Acceleration", name, ARM_ACCELERATION)
        bus.write("Maximum_Velocity_Limit", name, ARM_MAX_VELOCITY)


def apply_wheel_neck_limits(bus, motor_names) -> None:
    """Write the head and wheel (Bus 2) limits to the given motors.

    Call with bus torque disabled (e.g. during ``configure()``); the
    caller re-enables torque afterwards.
    """
    for name in motor_names:
        bus.write("Torque_Limit", name, WHEEL_NECK_TORQUE_LIMIT)
        bus.write("Acceleration", name, WHEEL_NECK_ACCELERATION)
