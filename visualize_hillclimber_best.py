import os
import json
import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import csv
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from glob import glob

from robot_config import (
    JOINT_MAPPING,
    HIP_MAX_FORCE,
    KNEE_MAX_FORCE,
    POSITION_GAIN,
    VELOCITY_GAIN,
    FLOOR_FRICTION_COEFFICIENT,
    A_MIN,
    A_MAX,
    B_MIN,
    B_MAX,
    C_MIN,
    C_MAX,
    OMEGA_MIN,
    OMEGA_MAX
)

# ===========================
# Configuration Constants
# ===========================

# Robot Parameters
ROBOT_URDF = "urdf-assembly.urdf"
START_POS = [0, 0, 0.5]
START_ORIENTATION = [math.radians(90), 0, 0]

# Joint Configuration
HIP_JOINTS = [0, 2, 4, 6]
KNEE_JOINTS = [1, 3, 5, 7]
ALL_JOINTS = HIP_JOINTS + KNEE_JOINTS

# Simulation Parameters
SIMULATION_FPS = 240
SIMULATION_DURATION = 10  # seconds
TOTAL_STEPS = SIMULATION_FPS * SIMULATION_DURATION
SETTLING_STEPS = int(0.2 * SIMULATION_FPS)

# Camera Configuration
CAMERA_DISTANCE = 1.5
CAMERA_YAW = 45
CAMERA_PITCH = -30
CAMERA_TARGET = [0, 0, 0]

# Directory Paths
DEFAULT_RESULTS_DIR = "results/hillclimber"
DEFAULT_TRAINING_DIR = "training_runs/hillclimber"

def load_training_data(run_name, results_dir=DEFAULT_RESULTS_DIR, training_dir=DEFAULT_TRAINING_DIR):
    """
    Load and analyze all training data from logs.
    
    Returns:
        tuple: (metrics_df, params_df, best_params)
    """
    log_dir = os.path.join(training_dir, run_name, "logs")
    metrics_files = glob(os.path.join(log_dir, "metrics_*.csv"))
    params_files = glob(os.path.join(log_dir, "params_*.csv"))
    
    if not metrics_files or not params_files:
        print(f"No log files found in {log_dir}")
        return None, None, None
        
    metrics_file = max(metrics_files, key=os.path.getctime)
    params_file = max(params_files, key=os.path.getctime)
    
    print(f"Loading metrics from: {metrics_file}")
    print(f"Loading parameters from: {params_file}")
    
    metrics_df = pd.read_csv(metrics_file)
    params_df = pd.read_csv(params_file)
    
    best_idx = metrics_df['best_distance'].idxmax()
    best_distance = metrics_df.loc[best_idx, 'best_distance']
    best_params = params_df.iloc[best_idx].to_dict()
    
    print(f"Best distance found: {best_distance:.3f} m at iteration {best_idx}")
    
    return metrics_df, params_df, best_params

def setup_pybullet(gui=True):
    """Initialize PyBullet simulation."""
    if gui:
        physicsClient = p.connect(p.GUI)
    else:
        physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    return physicsClient

def reset_simulation(robot_urdf, start_pos, start_orientation):
    """Load and reset the simulation environment."""
    plane_id = p.loadURDF("plane.urdf")
    p.changeDynamics(plane_id, -1, lateralFriction=FLOOR_FRICTION_COEFFICIENT)
    
    quat = p.getQuaternionFromEuler(start_orientation)
    robot_id = p.loadURDF(robot_urdf, start_pos, quat)
    
    for link in range(p.getNumJoints(robot_id)):
        p.changeDynamics(robot_id, link, lateralFriction=FLOOR_FRICTION_COEFFICIENT)
    
    for _ in range(SETTLING_STEPS):
        p.stepSimulation()
        if p.getConnectionInfo()['connectionMethod'] == p.GUI:
            time.sleep(1/SIMULATION_FPS)
    
    return robot_id

def adjust_camera():
    """Set camera position for visualization."""
    p.resetDebugVisualizerCamera(
        cameraDistance=CAMERA_DISTANCE,
        cameraYaw=CAMERA_YAW,
        cameraPitch=CAMERA_PITCH,
        cameraTargetPosition=CAMERA_TARGET
    )

def apply_gait(robot_id, gait_params, omega, total_steps):
    """Apply gait parameters and record robot motion."""
    data_records = []
    previous_x = None
    speed = 0.0

    for step in range(total_steps):
        t = step / SIMULATION_FPS

        for joint in ALL_JOINTS:
            joint_dict = gait_params.get(str(joint), {})
            a = joint_dict.get('a', 0.0)
            b = joint_dict.get('b', 0.0)
            c = joint_dict.get('c', 0.0)
            
            theta = a + b * math.sin(omega * t + c)
            theta_rad = math.radians(theta)
            
            max_force = HIP_MAX_FORCE if joint in HIP_JOINTS else KNEE_MAX_FORCE
            
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=theta_rad,
                force=max_force,
                positionGain=POSITION_GAIN,
                velocityGain=VELOCITY_GAIN
            )
        
        p.stepSimulation()
        if p.getConnectionInfo()['connectionMethod'] == p.GUI:
            time.sleep(0.001)
        
        if step % 10 == 0:
            base_pos, base_orient = p.getBasePositionAndOrientation(robot_id)
            base_euler = p.getEulerFromQuaternion(base_orient)
            
            if previous_x is not None:
                dx = base_pos[0] - previous_x
                speed = dx / (10 / SIMULATION_FPS)
            previous_x = base_pos[0]
            
            record = {
                'time': t,
                'x': base_pos[0],
                'y': base_pos[1],
                'z': base_pos[2],
                'pitch': math.degrees(base_euler[0]),
                'yaw': math.degrees(base_euler[1]),
                'roll': math.degrees(base_euler[2]),
                'speed': speed
            }
            data_records.append(record)
    
    return data_records

def plot_training_progress(metrics_df, save_dir):
    """Plot optimization progress over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df['iteration'], metrics_df['current_distance'], 
             alpha=0.5, label='Current Distance', color='blue')
    plt.plot(metrics_df['iteration'], metrics_df['best_distance'], 
             label='Best Distance', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Distance (m)')
    plt.title('Hill Climber Training Progress')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'training_progress.png'))
    plt.close()

def plot_parameter_evolution(params_df, save_dir):
    """Plot parameter evolution during optimization."""
    fig, axes = plt.subplots(8, 3, figsize=(15, 20))
    fig.suptitle('Joint Parameter Evolution')
    
    for joint in range(8):
        axes[joint, 0].plot(params_df[f'a_{joint}'])
        axes[joint, 0].set_title(f'Joint {joint} - a')
        axes[joint, 0].grid(True)
        axes[joint, 0].set_ylim(A_MIN, A_MAX)
        
        axes[joint, 1].plot(params_df[f'b_{joint}'])
        axes[joint, 1].set_title(f'Joint {joint} - b')
        axes[joint, 1].grid(True)
        axes[joint, 1].set_ylim(B_MIN, B_MAX)
        
        axes[joint, 2].plot(params_df[f'c_{joint}'])
        axes[joint, 2].set_title(f'Joint {joint} - c')
        axes[joint, 2].grid(True)
        axes[joint, 2].set_ylim(C_MIN, C_MAX)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'parameter_evolution.png'))
    plt.close()

def plot_robot_motion(data_records, save_dir):
    """Plot robot position and orientation over time."""
    times = [record['time'] for record in data_records]
    positions = {
        'x': [record['x'] for record in data_records],
        'y': [record['y'] for record in data_records],
        'z': [record['z'] for record in data_records]
    }
    orientations = {
        'pitch': [record['pitch'] for record in data_records],
        'yaw': [record['yaw'] for record in data_records],
        'roll': [record['roll'] for record in data_records]
    }
    speeds = [record['speed'] for record in data_records]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Position plot
    for coord, values in positions.items():
        ax1.plot(times, values, label=f'{coord.upper()} Position')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.set_title('Robot Position Over Time')
    ax1.legend()
    ax1.grid(True)
    
    # Orientation plot
    for angle, values in orientations.items():
        ax2.plot(times, values, label=angle.capitalize())
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angle (degrees)')
    ax2.set_title('Robot Orientation Over Time')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'robot_motion.png'))
    plt.close()

def plot_joint_functions(best_params, save_dir):
    """Plot the sinusoidal functions for each joint."""
    t = np.linspace(0, SIMULATION_DURATION, SIMULATION_FPS * SIMULATION_DURATION)
    
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    fig.suptitle('Joint Angle Functions')
    
    for joint in range(8):
        row = joint // 2
        col = joint % 2
        
        a = best_params[f'a_{joint}']
        b = best_params[f'b_{joint}']
        c = best_params[f'c_{joint}']
        omega = best_params['omega']
        
        theta = a + b * np.sin(omega * t + c)
        
        axes[row, col].plot(t, theta)
        axes[row, col].set_title(f'Joint {joint}')
        axes[row, col].set_xlabel('Time (s)')
        axes[row, col].set_ylabel('Angle (degrees)')
        axes[row, col].grid(True)
        axes[row, col].set_ylim(-180, 180)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'joint_functions.png'))
    plt.close()

def save_analysis_data(data_records, filename, save_dir):
    """Save analysis results to CSV."""
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, filename), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data_records[0].keys())
        writer.writeheader()
        writer.writerows(data_records)

def params_dict_to_gait_format(best_params):
    """Convert parameter dictionary to gait format."""
    return {
        str(joint): {
            'a': best_params[f'a_{joint}'],
            'b': best_params[f'b_{joint}'],
            'c': best_params[f'c_{joint}']
        }
        for joint in range(8)
    }

def visualize_and_analyze(run_name):
    """Main analysis function."""
    print(f"\nAnalyzing run: {run_name}")
    
    # Load training data
    metrics_df, params_df, best_params = load_training_data(run_name)
    if metrics_df is None:
        return
    
    # Create analysis directory
    analysis_dir = os.path.join(DEFAULT_RESULTS_DIR, run_name, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Generate analysis plots
    print("\nGenerating analysis plots...")
    plot_training_progress(metrics_df, analysis_dir)
    plot_parameter_evolution(params_df, analysis_dir)
    plot_joint_functions(best_params, analysis_dir)
    
    # Run simulation with best parameters
    print("\nRunning simulation with best parameters...")
    physicsClient = setup_pybullet(gui=True)
    robot_id = reset_simulation(ROBOT_URDF, START_POS, START_ORIENTATION)
    adjust_camera()
    
    gait_params = params_dict_to_gait_format(best_params)
    data_records = apply_gait(robot_id, gait_params, best_params['omega'], TOTAL_STEPS)
    
    # Save and plot results
    save_analysis_data(data_records, 'simulation_data.csv', analysis_dir)
    plot_robot_motion(data_records, analysis_dir)
    
    p.disconnect()
    print(f"\nAnalysis complete! Results saved in: {analysis_dir}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Hill Climber Training Run")
    parser.add_argument('--run', type=str, required=True, help="Name of the training run")
    args = parser.parse_args()
    
    visualize_and_analyze(args.run)

if __name__ == "__main__":
    main()