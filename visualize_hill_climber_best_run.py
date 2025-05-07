# visualize_hillclimber_best_run.py

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

def load_best_run(run_name, results_dir="results/hillclimber"):
    """
    Load the best run's data from the results directory.

    Args:
        run_name (str): The unique name of the training run.
        results_dir (str): Base directory where results are stored.

    Returns:
        dict: Dictionary containing the best run's data.
    """
    values_json_path = os.path.join(results_dir, run_name, "values.json")
    if not os.path.exists(values_json_path):
        print(f"Values JSON file '{values_json_path}' does not exist.")
        return None

    with open(values_json_path, 'r') as f:
        best_run = json.load(f)
    return best_run

def setup_pybullet(gui=True):
    """
    Initialize PyBullet simulation.

    Args:
        gui (bool): Whether to use GUI mode.

    Returns:
        int: PyBullet client ID.
    """
    if gui:
        physicsClient = p.connect(p.GUI)
    else:
        physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    return physicsClient

def reset_simulation(robot_urdf, start_pos, start_orientation):
    """
    Load the plane and robot into the simulation and reset their positions.

    Args:
        robot_urdf (str): Path to the robot URDF file.
        start_pos (list): Initial position of the robot [x, y, z].
        start_orientation (list): Initial orientation as Euler angles [roll, pitch, yaw].

    Returns:
        int: Robot's unique ID.
    """
    # Load the plane
    plane_id = p.loadURDF("plane.urdf")
    
    # Convert Euler to quaternion
    quat = p.getQuaternionFromEuler(start_orientation)
    
    # Load the robot
    robot_id = p.loadURDF(robot_urdf, start_pos, quat)
    
    # Allow some time for the robot to settle
    settling_steps = int(0.2 * 240)  # 0.2 seconds at 240 FPS
    for _ in range(settling_steps):
        p.stepSimulation()
        time.sleep(1./240.)
    
    return robot_id

def apply_gait(robot_id, gait_params, omega, total_steps, time_step=1./240.):
    """
    Apply the gait parameters to the robot over the simulation steps and record data.

    Args:
        robot_id (int): PyBullet unique ID for the robot.
        gait_params (dict): Dictionary with joint parameters 'a', 'b', 'c'.
        omega (float): Angular frequency.
        total_steps (int): Total simulation steps.
        time_step (float): Simulation time step.

    Returns:
        list: List of dictionaries containing time, position, orientation, and speed data.
    """
    # Define joint indices based on previous scripts
    HIP_JOINTS = [0, 2, 4, 6]
    KNEE_JOINTS = [1, 3, 5, 7]
    ALL_JOINTS = HIP_JOINTS + KNEE_JOINTS
    
    # Initialize data recording
    data_records = []
    previous_x = None
    speed = 0.0
    
    # Simulation loop
    for step in range(total_steps):
        t = step * time_step  # Current time in seconds
        
        for joint in ALL_JOINTS:
            a = gait_params[str(joint)]['a']
            b = gait_params[str(joint)]['b']
            c = gait_params[str(joint)]['c']
            
            # Calculate target angle in radians
            theta = a + b * math.sin(omega * t + c)
            theta_rad = math.radians(theta)
            
            # Apply joint control
            force = 250 if joint in HIP_JOINTS else 250  # Adjust force as needed
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=int(joint),
                controlMode=p.POSITION_CONTROL,
                targetPosition=theta_rad,
                force=force,
                positionGain=0.1,
                velocityGain=0.1
            )
        
        p.stepSimulation()
        time.sleep(0.001)  # 1 ms sleep to speed up simulation
        
        # Record data every 10 steps (~0.0417 seconds)
        if step % 10 == 0:
            base_pos, base_orient = p.getBasePositionAndOrientation(robot_id)
            base_euler = p.getEulerFromQuaternion(base_orient)
            
            # Calculate speed (change in x-position over time)
            if previous_x is not None:
                dx = base_pos[0] - previous_x
                speed = dx / (10 * time_step)  # Speed in m/s
            previous_x = base_pos[0]
            
            record = {
                'time': step * time_step,
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

def adjust_camera():
    """
    Adjust the camera to match typical settings used in hello_bullet.py.
    """
    # Typical camera settings for a better side view
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,       # Adjusted yaw for a side perspective
        cameraPitch=-30,    # Adjusted pitch for a better angle
        cameraTargetPosition=[0, 0, 0]
    )

def save_position_orientation_data(data_records, filename, results_dir):
    """
    Save the recorded position and orientation data to a CSV file.

    Args:
        data_records (list): List of dictionaries containing time, position, orientation, and speed data.
        filename (str): Name of the CSV file to save data.
        results_dir (str): Directory where the CSV file will be saved.
    """
    keys = ['time', 'x', 'y', 'z', 'pitch', 'yaw', 'roll', 'speed']
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, filename), 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(data_records)
    print(f"Position, orientation, and speed data saved to '{os.path.join(results_dir, filename)}'.")

def plot_servo_functions(best_run, run_duration, fps, save_dir):
    """
    Plot the target joint angle functions for each servo in separate subplots.

    Args:
        best_run (dict): Dictionary containing the best run's data.
        run_duration (int): Duration of the simulation in seconds.
        fps (int): Frames per second of the simulation.
        save_dir (str): Directory to save the plots.
    """
    gait_params = best_run['gait_parameters_deg']
    omega = best_run['omega']
    total_steps = run_duration * fps
    time_steps = np.linspace(0, run_duration, total_steps)
    
    HIP_JOINTS = [0, 2, 4, 6]
    KNEE_JOINTS = [1, 3, 5, 7]
    ALL_JOINTS = HIP_JOINTS + KNEE_JOINTS
    
    num_joints = len(ALL_JOINTS)
    cols = 2  # Number of columns in subplot grid
    rows = math.ceil(num_joints / cols)
    
    plt.figure(figsize=(14, 3 * rows))
    
    for idx, joint in enumerate(ALL_JOINTS, 1):
        a = gait_params[str(joint)]['a']
        b = gait_params[str(joint)]['b']
        c = gait_params[str(joint)]['c']
        theta = a + b * np.sin(omega * time_steps + c)
        
        plt.subplot(rows, cols, idx)
        plt.plot(time_steps, theta, label=f'θ(t) = {a:.2f} + {b:.2f}sin({omega:.2f}t + {c:.2f})')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Target Angle (degrees)')
        plt.title(f'Joint {joint} Angle Function')
        plt.legend(fontsize='small')
        plt.grid(True)
        
        # Fix axes for consistency
        plt.ylim(-180, 180)  # Example fixed range; adjust based on your robot's joint limits
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'servo_angle_functions_best_run.png'))
    plt.close()
    print(f"Servo angle functions plot saved to '{os.path.join(save_dir, 'servo_angle_functions_best_run.png')}'.")

def plot_robot_motion(data_records, save_dir):
    """
    Plot the robot's x, y, z positions and pitch, yaw, roll orientations over time.
    Also, display average speed as text on the position plot.

    Args:
        data_records (list): List of dictionaries containing time, position, orientation, and speed data.
        save_dir (str): Directory to save the plots.
    """
    # Extract data
    times = [record['time'] for record in data_records]
    x = [record['x'] for record in data_records]
    y = [record['y'] for record in data_records]
    z = [record['z'] for record in data_records]
    pitch = [record['pitch'] for record in data_records]
    yaw = [record['yaw'] for record in data_records]
    roll = [record['roll'] for record in data_records]
    speed = [record['speed'] for record in data_records]
    
    # Calculate average speed
    average_speed = np.mean(speed)
    
    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(14, 10))
    
    # Subplot 1: X, Y, Z positions
    axs[0].plot(times, x, label='X Position', color='blue')
    axs[0].plot(times, y, label='Y Position', color='green')
    axs[0].plot(times, z, label='Z Position', color='red')
    axs[0].set_xlabel('Time (seconds)')
    axs[0].set_ylabel('Position (meters)')
    axs[0].set_title('Robot Positions Over Time (Best Run)')
    axs[0].legend()
    axs[0].grid(True)
    
    # Display average speed as text
    axs[0].text(0.95, 0.01, f'Average Speed: {average_speed:.2f} m/s',
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=axs[0].transAxes,
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5))
    
    # Subplot 2: Pitch, Yaw, Roll orientations
    axs[1].plot(times, pitch, label='Pitch', color='purple')
    axs[1].plot(times, yaw, label='Yaw', color='orange')
    axs[1].plot(times, roll, label='Roll', color='brown')
    axs[1].set_xlabel('Time (seconds)')
    axs[1].set_ylabel('Orientation (degrees)')
    axs[1].set_title('Robot Orientations Over Time (Best Run)')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'robot_motion_best_run.png'))
    plt.close()
    print(f"Robot motion plot saved to '{os.path.join(save_dir, 'robot_motion_best_run.png')}'.")

def plot_servo_functions_each_joint(best_run, run_duration, fps, save_dir):
    """
    Plot the target joint angle functions for each servo with optimal a, b, c values.

    Args:
        best_run (dict): Dictionary containing the best run's data.
        run_duration (int): Duration of the simulation in seconds.
        fps (int): Frames per second of the simulation.
        save_dir (str): Directory to save the plots.
    """
    gait_params = best_run['gait_parameters_deg']
    omega = best_run['omega']
    total_steps = run_duration * fps
    time_steps = np.linspace(0, run_duration, total_steps)
    
    HIP_JOINTS = [0, 2, 4, 6]
    KNEE_JOINTS = [1, 3, 5, 7]
    ALL_JOINTS = HIP_JOINTS + KNEE_JOINTS
    
    num_joints = len(ALL_JOINTS)
    cols = 2  # Number of columns in subplot grid
    rows = math.ceil(num_joints / cols)
    
    plt.figure(figsize=(14, 3 * rows))
    
    for idx, joint in enumerate(ALL_JOINTS, 1):
        a = gait_params[str(joint)]['a']
        b = gait_params[str(joint)]['b']
        c = gait_params[str(joint)]['c']
        theta = a + b * np.sin(omega * time_steps + c)
        
        plt.subplot(rows, cols, idx)
        function_expression = f'θ(t) = {a:.2f} + {b:.2f}sin({omega:.2f}t + {c:.2f})'
        plt.plot(time_steps, theta, label=function_expression)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Target Angle (degrees)')
        plt.title(f'Joint {joint} Angle Function')
        plt.legend(fontsize='small')
        plt.grid(True)
        
        # Fix axes for consistency
        plt.ylim(-180, 180)  # Example fixed range; adjust based on your robot's joint limits
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'servo_angle_functions_each_joint.png'))
    plt.close()
    print(f"Servo angle functions (each joint) plot saved to '{os.path.join(save_dir, 'servo_angle_functions_each_joint.png')}'.")

def visualize_best_run(run_name, results_dir="results/hillclimber"):
    """
    Main function to visualize the best gait run, plot robot motion,
    and plot servo angle functions for each joint.

    Args:
        run_name (str): The unique name of the training run.
        results_dir (str): Base directory where results are stored.
    """
    # Load the best run's data
    best_run = load_best_run(run_name, results_dir=results_dir)
    
    if best_run is None:
        print(f"Failed to load best run data for run '{run_name}'. Exiting.")
        return
    
    gait_params = best_run['gait_parameters_deg']
    omega = best_run['omega']
    best_distance = best_run['best_distance_traveled_m']
    
    print(f"Best Distance Traveled: {best_distance:.3f} meters.")
    print(f"Omega: {omega:.3f} rad/s.")
    
    # Initialize PyBullet in GUI mode
    physicsClient = setup_pybullet(gui=True)
    
    # Define robot URDF path
    robot_urdf = "urdf-assembly.urdf"  # Ensure this path is correct relative to script location
    
    # Define initial position and orientation
    start_pos = [0, 0, 0.5]
    start_orientation = [math.radians(90), 0, 0]  # [roll, pitch, yaw] in radians
    
    # Reset simulation and load robot
    robot_id = reset_simulation(robot_urdf, start_pos, start_orientation)
    
    # Adjust the camera to match hello_bullet.py
    adjust_camera()
    
    # Define simulation duration (e.g., 10 seconds)
    simulation_duration = 10  # seconds
    fps = 240  # Frames per second
    total_steps = int(simulation_duration * fps)  # 240 FPS
    
    # Apply gait parameters and record data
    print("Starting gait simulation and data recording...")
    data_records = apply_gait(robot_id, gait_params, omega, total_steps)
    
    # Define save directory
    save_dir = os.path.join(results_dir, run_name, "plots_best_run")
    
    # Save the recorded data
    save_position_orientation_data(data_records, filename='best_run_data.csv', results_dir=save_dir)
    
    # Plot robot motion
    plot_robot_motion(data_records, save_dir=save_dir)
    
    # Plot servo angle functions (combined)
    plot_servo_functions(best_run, run_duration=simulation_duration, fps=fps, save_dir=save_dir)
    
    # Plot servo angle functions for each joint separately
    plot_servo_functions_each_joint(best_run, run_duration=simulation_duration, fps=fps, save_dir=save_dir)
    
    # Disconnect PyBullet
    p.disconnect()
    print("Simulation ended.")

def main():
    parser = argparse.ArgumentParser(description="Visualize the Best Hill Climber Training Run")
    parser.add_argument('--run', type=str, required=True, help="Name of the training run (e.g., happy_sun)")
    args = parser.parse_args()
    
    run_name = args.run
    visualize_best_run(run_name)

if __name__ == "__main__":
    main()
