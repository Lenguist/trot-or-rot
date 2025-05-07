# plot_results.py

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import csv

def load_run_logs(logs_dir):
    """
    Load all run logs from the specified directory.
    
    Args:
        logs_dir (str): Path to the logs_hill_climber directory.
    
    Returns:
        list: List of dictionaries containing run data.
    """
    run_logs = []
    for entry in os.listdir(logs_dir):
        run_path = os.path.join(logs_dir, entry, 'log.json')
        if os.path.isfile(run_path):
            with open(run_path, 'r') as f:
                data = json.load(f)
                run_logs.append(data)
    return run_logs

def plot_best_distance(run_logs):
    """
    Plot the best distance achieved over iterations.
    
    Args:
        run_logs (list): List of run data dictionaries.
    """
    # Sort runs by run_number
    run_logs_sorted = sorted(run_logs, key=lambda x: x['run_number'])
    
    best_distance = []
    current_best = -np.inf
    for run in run_logs_sorted:
        distance = run['distance_traveled_m']
        if distance > current_best:
            current_best = distance
        best_distance.append(current_best)
    
    plt.figure(figsize=(10, 6))
    plt.plot(best_distance, label='Best Distance')
    plt.xlabel('Run Number')
    plt.ylabel('Distance Traveled (meters)')
    plt.title('Best Distance Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/best_distance_over_iterations.png')
    plt.show()

def plot_best_servo_functions(best_run, run_duration=20, fps=240):
    """
    Plot the target joint angle functions for each servo based on the best run's parameters.
    
    Args:
        best_run (dict): Dictionary containing the best run's data.
        run_duration (int): Duration of the simulation in seconds.
        fps (int): Frames per second of the simulation.
    """
    gait_params = best_run['gait_parameters_deg']
    omega = best_run['omega']
    total_steps = run_duration * fps
    time_steps = np.linspace(0, run_duration, total_steps)
    
    HIP_JOINTS = [0, 2, 4, 6]
    KNEE_JOINTS = [1, 3, 5, 7]
    ALL_JOINTS = HIP_JOINTS + KNEE_JOINTS
    
    plt.figure(figsize=(14, 10))
    
    for joint in ALL_JOINTS:
        a = gait_params[str(joint)]['a']
        b = gait_params[str(joint)]['b']
        c = gait_params[str(joint)]['c']
        theta = a + b * np.sin(omega * time_steps + c)
        
        plt.plot(time_steps, theta, label=f'Joint {joint}')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Target Joint Angle (degrees)')
    plt.title('Target Joint Angle Functions for Each Servo (Best Run)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/best_servo_functions.png')
    plt.show()

def plot_parameter_evolution(run_logs):
    """
    Plot how gait parameters (a, b, c, omega) changed over iterations for each joint.
    
    Args:
        run_logs (list): List of run data dictionaries.
    """
    # Sort runs by run_number
    run_logs_sorted = sorted(run_logs, key=lambda x: x['run_number'])
    
    run_numbers = [run['run_number'] for run in run_logs_sorted]
    
    HIP_JOINTS = [0, 2, 4, 6]
    KNEE_JOINTS = [1, 3, 5, 7]
    ALL_JOINTS = HIP_JOINTS + KNEE_JOINTS
    
    # Initialize dictionaries to store parameter values
    a_values = {joint: [] for joint in ALL_JOINTS}
    b_values = {joint: [] for joint in ALL_JOINTS}
    c_values = {joint: [] for joint in ALL_JOINTS}
    omega_values = []
    
    for run in run_logs_sorted:
        gait_params = run['gait_parameters_deg']
        omega = run['omega']
        omega_values.append(omega)
        for joint in ALL_JOINTS:
            a = gait_params[str(joint)]['a']
            b = gait_params[str(joint)]['b']
            c = gait_params[str(joint)]['c']
            a_values[joint].append(a)
            b_values[joint].append(b)
            c_values[joint].append(c)
    
    # Plot 'a' parameters
    plt.figure(figsize=(14, 10))
    for joint in ALL_JOINTS:
        plt.plot(run_numbers, a_values[joint], label=f'Joint {joint} a')
    plt.xlabel('Run Number')
    plt.ylabel('a Parameter (degrees)')
    plt.title('Evolution of a Parameters Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/parameter_evolution_a.png')
    plt.show()
    
    # Plot 'b' parameters
    plt.figure(figsize=(14, 10))
    for joint in ALL_JOINTS:
        plt.plot(run_numbers, b_values[joint], label=f'Joint {joint} b')
    plt.xlabel('Run Number')
    plt.ylabel('b Parameter (degrees)')
    plt.title('Evolution of b Parameters Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/parameter_evolution_b.png')
    plt.show()
    
    # Plot 'c' parameters
    plt.figure(figsize=(14, 10))
    for joint in ALL_JOINTS:
        plt.plot(run_numbers, c_values[joint], label=f'Joint {joint} c')
    plt.xlabel('Run Number')
    plt.ylabel('c Parameter (radians)')
    plt.title('Evolution of c Parameters Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/parameter_evolution_c.png')
    plt.show()
    
    # Plot 'omega' parameter
    plt.figure(figsize=(14, 6))
    plt.plot(run_numbers, omega_values, label='Omega', color='black')
    plt.xlabel('Run Number')
    plt.ylabel('Omega (rad/s)')
    plt.title('Evolution of Omega Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/parameter_evolution_omega.png')
    plt.show()

def plot_robot_motion(best_run_data_path):
    """
    Plot the robot's x, y, z positions and pitch, yaw, roll orientations over time for the best run.
    
    Args:
        best_run_data_path (str): Path to the CSV file containing best run data.
    """
    # Load data from CSV
    time_vals = []
    x_vals = []
    y_vals = []
    z_vals = []
    pitch_vals = []
    yaw_vals = []
    roll_vals = []
    
    with open(best_run_data_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            time_vals.append(float(row['time']))
            x_vals.append(float(row['x']))
            y_vals.append(float(row['y']))
            z_vals.append(float(row['z']))
            pitch_vals.append(float(row['pitch']))
            yaw_vals.append(float(row['yaw']))
            roll_vals.append(float(row['roll']))
    
    # Plot positions
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    plt.plot(time_vals, x_vals, label='X Position')
    plt.plot(time_vals, y_vals, label='Y Position')
    plt.plot(time_vals, z_vals, label='Z Position')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Position (meters)')
    plt.title('Robot Position Over Time (Best Run)')
    plt.legend()
    plt.grid(True)
    
    # Plot orientations
    plt.subplot(2, 1, 2)
    plt.plot(time_vals, pitch_vals, label='Pitch')
    plt.plot(time_vals, yaw_vals, label='Yaw')
    plt.plot(time_vals, roll_vals, label='Roll')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Orientation (degrees)')
    plt.title('Robot Orientation Over Time (Best Run)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/robot_motion_best_run.png')
    plt.show()

def plot_servo_parameters_over_time(best_run, best_run_data_path, run_duration=20, fps=240):
    """
    Plot how the servo parameters change over time based on the best run's parameters.
    
    Args:
        best_run (dict): Dictionary containing the best run's data.
        best_run_data_path (str): Path to the CSV file containing best run data.
        run_duration (int): Duration of the simulation in seconds.
        fps (int): Frames per second of the simulation.
    """
    gait_params = best_run['gait_parameters_deg']
    omega = best_run['omega']
    total_steps = run_duration * fps
    time_steps = np.linspace(0, run_duration, total_steps)
    
    HIP_JOINTS = [0, 2, 4, 6]
    KNEE_JOINTS = [1, 3, 5, 7]
    ALL_JOINTS = HIP_JOINTS + KNEE_JOINTS
    
    plt.figure(figsize=(14, 10))
    
    for joint in ALL_JOINTS:
        a = gait_params[str(joint)]['a']
        b = gait_params[str(joint)]['b']
        c = gait_params[str(joint)]['c']
        theta = a + b * np.sin(omega * time_steps + c)
        
        plt.plot(time_steps, theta, label=f'Joint {joint}')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Target Joint Angle (degrees)')
    plt.title('Target Joint Angle Functions Over Time (Best Run)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/servo_parameters_over_time.png')
    plt.show()

def plot_results():
    """
    Function to plot all required results:
    - Best distance over iterations
    - Best servo functions
    - Parameter evolution over iterations
    - Robot's position and orientation over time for the best run
    """
    logs_dir = 'logs_hill_climber'
    
    # Ensure plots directory exists
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load all run logs
    run_logs = load_run_logs(logs_dir)
    
    if not run_logs:
        print("No run logs found. Please ensure that the logs are correctly saved in 'logs_hill_climber/run_X/log.json'.")
        return
    
    # Plot best distance over iterations
    plot_best_distance(run_logs)
    
    # Load best run log
    best_run_path = os.path.join(logs_dir, 'best_run', 'best_log.json')
    if not os.path.exists(best_run_path):
        print(f"Best run log not found at {best_run_path}.")
        return
    
    with open(best_run_path, 'r') as f:
        best_run = json.load(f)
    
    # Plot best servo functions
    plot_best_servo_functions(best_run)
    
    # Plot parameter evolution over iterations
    plot_parameter_evolution(run_logs)
    
    # Plot robot's position and orientation over time for the best run
    best_run_data_path = os.path.join(logs_dir, 'best_run', 'best_run_data.csv')
    if not os.path.exists(best_run_data_path):
        print(f"Best run data not found at {best_run_data_path}. Please run 'visualize_best_run.py' to generate this data.")
        return
    
    plot_robot_motion(best_run_data_path)

if __name__ == "__main__":
    plot_results()
