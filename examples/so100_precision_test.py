#!/usr/bin/env python3
"""
SO100/SO101 Robot Arm Precision Test Script

This script tests the repeatability and precision of the robot arm by:
1. Moving to random positions within the workspace
2. Then moving to a fixed target position
3. Recording the final position error
4. Repeating N times and computing statistics
"""

import time
import logging
import traceback
import math
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TestConfig:
    """Configuration for precision test"""
    # Number of test trials
    num_trials: int = 10
    
    # Target end effector position - FIXED TO ZERO POSITION
    # x0, y0 = 0.1629, 0.1131 from original script (zero position)
    target_x: float = 0.1629
    target_y: float = 0.1131
    target_z: float = 0.0
    
    # Target shoulder_pan angle (degrees) - zero
    target_shoulder_pan: float = 0.0
    
    # Target pitch adjustment
    target_pitch: float = 0.0
    
    # Workspace limits for random position generation (meters)
    x_min: float = 0.10
    x_max: float = 0.24
    y_min: float = 0.05
    y_max: float = 0.20
    
    # Joint limits for random positions (degrees)
    shoulder_pan_min: float = -45.0
    shoulder_pan_max: float = 45.0
    
    # Control parameters (2x faster than previous: kp 0.1 -> 0.2)
    kp: float = 0.2
    control_freq: int = 50
    
    # Movement timing (2x faster: 15s -> 7.5s)
    move_duration: float = 7.5   # Time to move to each position (seconds)
    settle_time: float = 0.5     # Time to wait for arm to settle (seconds)
    
    # Convergence threshold (degrees)
    convergence_threshold: float = 1.0
    
    # Link lengths (meters)
    l1: float = 0.1159
    l2: float = 0.1350


# ============================================================================
# Joint Calibration (copied from original script)
# ============================================================================

JOINT_CALIBRATION = [
    ['shoulder_pan', 6.0, 1.0],
    ['shoulder_lift', 2.0, 0.97],
    ['elbow_flex', 0.0, 1.05],
    ['wrist_flex', 0.0, 0.94],
    ['wrist_roll', 0.0, 0.5],
    ['gripper', 0.0, 1.0],
]


def apply_joint_calibration(joint_name: str, raw_position: float) -> float:
    """Apply joint calibration coefficients"""
    for joint_cal in JOINT_CALIBRATION:
        if joint_cal[0] == joint_name:
            offset = joint_cal[1]
            scale = joint_cal[2]
            calibrated_position = (raw_position - offset) * scale
            return calibrated_position
    return raw_position


# ============================================================================
# Kinematics
# ============================================================================

def inverse_kinematics(x: float, y: float, l1: float = 0.1159, l2: float = 0.1350) -> Tuple[float, float]:
    """
    Calculate inverse kinematics for a 2-link robotic arm
    
    Returns:
        joint2_deg, joint3_deg: Joint angles in degrees
    """
    theta1_offset = math.atan2(0.028, 0.11257)
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset
    
    r = math.sqrt(x**2 + y**2)
    r_max = l1 + l2
    r_min = abs(l1 - l2)
    
    # Clamp to workspace
    if r > r_max:
        scale_factor = r_max / r
        x *= scale_factor
        y *= scale_factor
        r = r_max
    if r < r_min and r > 0:
        scale_factor = r_min / r
        x *= scale_factor
        y *= scale_factor
        r = r_min
    
    cos_theta2 = -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))  # Clamp for numerical stability
    
    theta2 = math.pi - math.acos(cos_theta2)
    
    beta = math.atan2(y, x)
    gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = beta + gamma
    
    joint2 = theta1 + theta1_offset
    joint3 = theta2 + theta2_offset
    
    joint2 = max(-0.1, min(3.45, joint2))
    joint3 = max(-0.2, min(math.pi, joint3))
    
    joint2_deg = math.degrees(joint2)
    joint3_deg = math.degrees(joint3)
    
    joint2_deg = 90 - joint2_deg
    joint3_deg = joint3_deg - 90
    
    return joint2_deg, joint3_deg


def forward_kinematics(joint2_deg: float, joint3_deg: float, l1: float = 0.1159, l2: float = 0.1350) -> Tuple[float, float]:
    """
    Calculate forward kinematics to get end effector position
    
    Returns:
        x, y: End effector position in meters
    """
    theta1_offset = math.atan2(0.028, 0.11257)
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset
    
    # Convert from joint angles back to theta
    joint2_rad = math.radians(90 - joint2_deg)
    joint3_rad = math.radians(joint3_deg + 90)
    
    theta1 = joint2_rad - theta1_offset
    theta2 = joint3_rad - theta2_offset
    
    # Forward kinematics
    x = l1 * math.cos(theta1) + l2 * math.cos(theta1 + theta2)
    y = l1 * math.sin(theta1) + l2 * math.sin(theta1 + theta2)
    
    return x, y


# ============================================================================
# Robot Control Functions
# ============================================================================

def get_current_positions(robot) -> Dict[str, float]:
    """Get current joint positions with calibration applied"""
    current_obs = robot.get_observation()
    current_positions = {}
    for key, value in current_obs.items():
        if key.endswith('.pos'):
            motor_name = key.removesuffix('.pos')
            calibrated_value = apply_joint_calibration(motor_name, value)
            current_positions[motor_name] = calibrated_value
    return current_positions


def get_raw_positions(robot) -> Dict[str, float]:
    """Get current joint positions without calibration"""
    current_obs = robot.get_observation()
    raw_positions = {}
    for key, value in current_obs.items():
        if key.endswith('.pos'):
            motor_name = key.removesuffix('.pos')
            raw_positions[motor_name] = value
    return raw_positions


def move_to_position(robot, target_positions: Dict[str, float], config: TestConfig, 
                     timeout: float = None) -> Dict[str, float]:
    """
    Move robot to target position using P control
    
    Returns:
        Final joint positions after movement
    """
    if timeout is None:
        timeout = config.move_duration
    
    control_period = 1.0 / config.control_freq
    total_steps = int(timeout * config.control_freq)
    
    for step in range(total_steps):
        current_positions = get_current_positions(robot)
        
        robot_action = {}
        total_error = 0
        
        for joint_name, target_pos in target_positions.items():
            if joint_name in current_positions:
                current_pos = current_positions[joint_name]
                error = target_pos - current_pos
                total_error += abs(error)
                
                control_output = config.kp * error
                new_position = current_pos + control_output
                robot_action[f"{joint_name}.pos"] = new_position
        
        if robot_action:
            robot.send_action(robot_action)
        
        # Check for early convergence
        if total_error < config.convergence_threshold * len(target_positions):
            break
        
        time.sleep(control_period)
    
    # Return final positions
    return get_current_positions(robot)


def move_to_zero_position(robot, config: TestConfig):
    """Move robot to zero position"""
    print("Moving to zero position...")
    zero_positions = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0,
        'gripper': 0.0
    }
    move_to_position(robot, zero_positions, config, timeout=3.0)
    print("Reached zero position")


# ============================================================================
# Random Position Generation
# ============================================================================

def generate_random_position(config: TestConfig) -> Tuple[float, float, float]:
    """
    Generate a random position within the workspace
    
    Returns:
        x, y, shoulder_pan: Random position parameters
    """
    # Generate random x, y within workspace
    x = random.uniform(config.x_min, config.x_max)
    y = random.uniform(config.y_min, config.y_max)
    
    # Check if position is reachable
    r = math.sqrt(x**2 + y**2)
    r_max = config.l1 + config.l2
    r_min = abs(config.l1 - config.l2)
    
    # Regenerate if outside workspace
    attempts = 0
    while (r > r_max * 0.95 or r < r_min * 1.1) and attempts < 100:
        x = random.uniform(config.x_min, config.x_max)
        y = random.uniform(config.y_min, config.y_max)
        r = math.sqrt(x**2 + y**2)
        attempts += 1
    
    # Random shoulder pan
    shoulder_pan = random.uniform(config.shoulder_pan_min, config.shoulder_pan_max)
    
    return x, y, shoulder_pan


def compute_target_joint_positions(x: float, y: float, shoulder_pan: float, 
                                    pitch: float = 0.0) -> Dict[str, float]:
    """
    Compute full joint positions for a given end effector position
    
    Returns:
        Dictionary of joint positions
    """
    joint2_deg, joint3_deg = inverse_kinematics(x, y)
    
    # Compute wrist_flex to maintain pitch
    wrist_flex = -joint2_deg - joint3_deg + pitch
    
    return {
        'shoulder_pan': shoulder_pan,
        'shoulder_lift': joint2_deg,
        'elbow_flex': joint3_deg,
        'wrist_flex': wrist_flex,
        'wrist_roll': 0.0,
        'gripper': 0.0
    }


# ============================================================================
# Precision Test
# ============================================================================

@dataclass
class TrialResult:
    """Result from a single trial"""
    trial_num: int
    
    # Starting position (random)
    start_x: float
    start_y: float
    start_z: float
    start_shoulder_pan: float
    
    # Target position
    target_x: float
    target_y: float
    target_z: float
    target_shoulder_pan: float
    
    # Final achieved position
    final_joint2: float
    final_joint3: float
    final_shoulder_pan: float
    final_x: float
    final_y: float
    final_z: float
    
    # Errors
    x_error: float
    y_error: float
    z_error: float
    euclidean_error: float
    shoulder_pan_error: float


def run_single_trial(robot, trial_num: int, config: TestConfig) -> TrialResult:
    """
    Run a single precision test trial
    
    1. Move to random starting position
    2. Move to target position
    3. Record final position error
    """
    print(f"\n{'='*60}")
    print(f"TRIAL {trial_num + 1} / {config.num_trials}")
    print(f"{'='*60}")
    
    # Generate random starting position
    start_x, start_y, start_shoulder_pan = generate_random_position(config)
    start_z = config.target_z  # Use same z for random start
    print(f"Random start position: x={start_x:.4f}m, y={start_y:.4f}m, z={start_z:.4f}m, pan={start_shoulder_pan:.1f}°")
    
    # Compute joint positions for starting position
    start_joints = compute_target_joint_positions(start_x, start_y, start_shoulder_pan, config.target_pitch)
    print(f"Start joint targets: shoulder_lift={start_joints['shoulder_lift']:.2f}°, "
          f"elbow_flex={start_joints['elbow_flex']:.2f}°")
    
    # Move to random starting position
    print("Moving to random start position (slow)...")
    move_to_position(robot, start_joints, config)
    time.sleep(config.settle_time)
    
    # Now move to target position
    print(f"\nTarget position: x={config.target_x:.4f}m, y={config.target_y:.4f}m, z={config.target_z:.4f}m, "
          f"pan={config.target_shoulder_pan:.1f}°")
    
    target_joints = compute_target_joint_positions(
        config.target_x, config.target_y, 
        config.target_shoulder_pan, config.target_pitch
    )
    print(f"Target joint angles: shoulder_lift={target_joints['shoulder_lift']:.2f}°, "
          f"elbow_flex={target_joints['elbow_flex']:.2f}°")
    
    # Move to target position
    print("Moving to target position (slow)...")
    move_to_position(robot, target_joints, config)
    time.sleep(config.settle_time)
    
    # Record final position
    final_positions = get_current_positions(robot)
    
    final_joint2 = final_positions.get('shoulder_lift', 0.0)
    final_joint3 = final_positions.get('elbow_flex', 0.0)
    final_shoulder_pan = final_positions.get('shoulder_pan', 0.0)
    
    # Compute final end effector position
    final_x, final_y = forward_kinematics(final_joint2, final_joint3)
    final_z = config.target_z  # Z is fixed for this arm
    
    # Compute errors
    x_error = final_x - config.target_x
    y_error = final_y - config.target_y
    z_error = final_z - config.target_z
    euclidean_error = math.sqrt(x_error**2 + y_error**2 + z_error**2)
    shoulder_pan_error = final_shoulder_pan - config.target_shoulder_pan
    
    print(f"\nFinal position: x={final_x:.4f}m, y={final_y:.4f}m, z={final_z:.4f}m, pan={final_shoulder_pan:.1f}°")
    print(f"Position error: Δx={x_error*1000:.2f}mm, Δy={y_error*1000:.2f}mm, Δz={z_error*1000:.2f}mm, "
          f"Euclidean={euclidean_error*1000:.2f}mm")
    print(f"Shoulder pan error: {shoulder_pan_error:.2f}°")
    
    return TrialResult(
        trial_num=trial_num,
        start_x=start_x,
        start_y=start_y,
        start_z=start_z,
        start_shoulder_pan=start_shoulder_pan,
        target_x=config.target_x,
        target_y=config.target_y,
        target_z=config.target_z,
        target_shoulder_pan=config.target_shoulder_pan,
        final_joint2=final_joint2,
        final_joint3=final_joint3,
        final_shoulder_pan=final_shoulder_pan,
        final_x=final_x,
        final_y=final_y,
        final_z=final_z,
        x_error=x_error,
        y_error=y_error,
        z_error=z_error,
        euclidean_error=euclidean_error,
        shoulder_pan_error=shoulder_pan_error
    )


def compute_statistics(results: List[TrialResult]) -> Dict:
    """Compute statistics from trial results"""
    x_errors = [r.x_error * 1000 for r in results]  # Convert to mm
    y_errors = [r.y_error * 1000 for r in results]
    z_errors = [r.z_error * 1000 for r in results]
    euclidean_errors = [r.euclidean_error * 1000 for r in results]
    pan_errors = [r.shoulder_pan_error for r in results]
    
    return {
        'x_error': {
            'mean': np.mean(x_errors),
            'std': np.std(x_errors),
            'min': np.min(x_errors),
            'max': np.max(x_errors),
            'abs_mean': np.mean(np.abs(x_errors))
        },
        'y_error': {
            'mean': np.mean(y_errors),
            'std': np.std(y_errors),
            'min': np.min(y_errors),
            'max': np.max(y_errors),
            'abs_mean': np.mean(np.abs(y_errors))
        },
        'z_error': {
            'mean': np.mean(z_errors),
            'std': np.std(z_errors),
            'min': np.min(z_errors),
            'max': np.max(z_errors),
            'abs_mean': np.mean(np.abs(z_errors))
        },
        'euclidean_error': {
            'mean': np.mean(euclidean_errors),
            'std': np.std(euclidean_errors),
            'min': np.min(euclidean_errors),
            'max': np.max(euclidean_errors)
        },
        'shoulder_pan_error': {
            'mean': np.mean(pan_errors),
            'std': np.std(pan_errors),
            'min': np.min(pan_errors),
            'max': np.max(pan_errors),
            'abs_mean': np.mean(np.abs(pan_errors))
        }
    }


def print_results_summary(results: List[TrialResult], stats: Dict):
    """Print a summary of all results"""
    print("\n")
    print("=" * 70)
    print("PRECISION TEST RESULTS SUMMARY")
    print("=" * 70)
    
    print("\n--- Individual Trial Results ---")
    print(f"{'Trial':<6} {'Start (x,y,z,pan)':<30} {'Final (x,y,z)':<25} {'Error (mm)':<15}")
    print("-" * 80)
    
    for r in results:
        start_str = f"({r.start_x:.3f}, {r.start_y:.3f}, {r.start_z:.3f}, {r.start_shoulder_pan:.1f}°)"
        final_str = f"({r.final_x:.4f}, {r.final_y:.4f}, {r.final_z:.4f})"
        error_str = f"{r.euclidean_error*1000:.2f}"
        print(f"{r.trial_num+1:<6} {start_str:<30} {final_str:<25} {error_str:<15}")
    
    print("\n--- Statistical Summary ---")
    print(f"\nX-axis error (mm):")
    print(f"  Mean:     {stats['x_error']['mean']:+.3f} mm")
    print(f"  Std Dev:  {stats['x_error']['std']:.3f} mm")
    print(f"  Range:    [{stats['x_error']['min']:.3f}, {stats['x_error']['max']:.3f}] mm")
    print(f"  Abs Mean: {stats['x_error']['abs_mean']:.3f} mm")
    
    print(f"\nY-axis error (mm):")
    print(f"  Mean:     {stats['y_error']['mean']:+.3f} mm")
    print(f"  Std Dev:  {stats['y_error']['std']:.3f} mm")
    print(f"  Range:    [{stats['y_error']['min']:.3f}, {stats['y_error']['max']:.3f}] mm")
    print(f"  Abs Mean: {stats['y_error']['abs_mean']:.3f} mm")
    
    print(f"\nZ-axis error (mm):")
    print(f"  Mean:     {stats['z_error']['mean']:+.3f} mm")
    print(f"  Std Dev:  {stats['z_error']['std']:.3f} mm")
    print(f"  Range:    [{stats['z_error']['min']:.3f}, {stats['z_error']['max']:.3f}] mm")
    print(f"  Abs Mean: {stats['z_error']['abs_mean']:.3f} mm")
    
    print(f"\nEuclidean position error (mm):")
    print(f"  Mean:     {stats['euclidean_error']['mean']:.3f} mm")
    print(f"  Std Dev:  {stats['euclidean_error']['std']:.3f} mm")
    print(f"  Range:    [{stats['euclidean_error']['min']:.3f}, {stats['euclidean_error']['max']:.3f}] mm")
    
    print(f"\nShoulder pan error (degrees):")
    print(f"  Mean:     {stats['shoulder_pan_error']['mean']:+.3f}°")
    print(f"  Std Dev:  {stats['shoulder_pan_error']['std']:.3f}°")
    print(f"  Range:    [{stats['shoulder_pan_error']['min']:.3f}°, {stats['shoulder_pan_error']['max']:.3f}°]")
    print(f"  Abs Mean: {stats['shoulder_pan_error']['abs_mean']:.3f}°")
    
    print("\n" + "=" * 70)
    print(f"REPEATABILITY (2σ): ±{2*stats['euclidean_error']['std']:.3f} mm")
    print("=" * 70)


def save_results_to_csv(results: List[TrialResult], stats: Dict, filename: str = "precision_test_results.csv"):
    """Save results to CSV file"""
    with open(filename, 'w') as f:
        # Header
        f.write("trial,start_x,start_y,start_z,start_pan,target_x,target_y,target_z,target_pan,")
        f.write("final_x,final_y,final_z,final_pan,x_error_mm,y_error_mm,z_error_mm,euclidean_error_mm,pan_error_deg\n")
        
        # Data rows
        for r in results:
            f.write(f"{r.trial_num+1},{r.start_x:.6f},{r.start_y:.6f},{r.start_z:.6f},{r.start_shoulder_pan:.3f},")
            f.write(f"{r.target_x:.6f},{r.target_y:.6f},{r.target_z:.6f},{r.target_shoulder_pan:.3f},")
            f.write(f"{r.final_x:.6f},{r.final_y:.6f},{r.final_z:.6f},{r.final_shoulder_pan:.3f},")
            f.write(f"{r.x_error*1000:.4f},{r.y_error*1000:.4f},{r.z_error*1000:.4f},{r.euclidean_error*1000:.4f},{r.shoulder_pan_error:.4f}\n")
        
        # Statistics
        f.write("\n# Statistics\n")
        f.write(f"# Euclidean Error - Mean: {stats['euclidean_error']['mean']:.4f} mm, Std: {stats['euclidean_error']['std']:.4f} mm\n")
        f.write(f"# Repeatability (2σ): ±{2*stats['euclidean_error']['std']:.4f} mm\n")
    
    print(f"\nResults saved to: {filename}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main function to run precision tests"""
    print("=" * 70)
    print("SO100/SO101 ROBOT ARM PRECISION TEST")
    print("=" * 70)
    
    # Create configuration
    config = TestConfig()
    
    print(f"\nTest Configuration:")
    print(f"  Number of trials: {config.num_trials}")
    print(f"  Target: ZERO POSITION (x={config.target_x:.4f}m, y={config.target_y:.4f}m, z={config.target_z:.4f}m)")
    print(f"  Target shoulder pan: {config.target_shoulder_pan}°")
    print(f"  Workspace: x=[{config.x_min}, {config.x_max}]m, y=[{config.y_min}, {config.y_max}]m")
    print(f"  Control: Kp={config.kp}, freq={config.control_freq}Hz")
    
    try:
        # Import robot modules
        from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig
        
        # Get port
        port = input("\nPlease enter the USB port for SO100 robot (e.g., /dev/ttyACM0): ").strip()
        if not port:
            port = "/dev/ttyACM0"
            print(f"Using default port: {port}")
        
        # Configure and connect robot
        robot_config = SO100FollowerConfig(port=port)
        robot = SO100Follower(robot_config)
        robot.connect()
        print("Robot connected successfully!")
        
        # Ask about calibration
        while True:
            calibrate_choice = input("Do you want to recalibrate the robot? (y/n): ").strip().lower()
            if calibrate_choice in ['y', 'yes']:
                print("Starting recalibration...")
                robot.calibrate()
                print("Calibration completed!")
                break
            elif calibrate_choice in ['n', 'no']:
                print("Using previous calibration file")
                break
            else:
                print("Please enter y or n")
        
        # Move to zero position first
        move_to_zero_position(robot, config)
        
        # Ask to confirm start
        input("\nPress Enter to start precision test...")
        
        # Run trials
        results = []
        for trial_num in range(config.num_trials):
            try:
                result = run_single_trial(robot, trial_num, config)
                results.append(result)
            except KeyboardInterrupt:
                print("\n\nTest interrupted by user!")
                break
            except Exception as e:
                print(f"\nError in trial {trial_num + 1}: {e}")
                traceback.print_exc()
                continue
        
        # Return to zero position
        print("\nReturning to zero position...")
        move_to_zero_position(robot, config)
        
        # Compute and display results
        if results:
            stats = compute_statistics(results)
            print_results_summary(results, stats)
            
            # Save results
            save_results_to_csv(results, stats)
        else:
            print("\nNo results to display (all trials failed or interrupted)")
        
        # Disconnect
        robot.disconnect()
        print("\nRobot disconnected. Test complete!")
        
    except Exception as e:
        print(f"\nProgram execution failed: {e}")
        traceback.print_exc()
        print("\nPlease check:")
        print("1. Whether the robot is properly connected")
        print("2. Whether the USB port is correct")
        print("3. Whether you have sufficient permissions to access USB devices")


if __name__ == "__main__":
    main()