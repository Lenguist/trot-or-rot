"""
understand_minitaur.py - A tool to visualize and understand the Minitaur environment

This script provides functionality to:
1. Load and visualize the Minitaur environment
2. Control individual motors to understand their effects
3. Explore the observation and action spaces
4. Collect and save data for later analysis/plotting
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pybullet_envs.minitaur.envs import minitaur_gym_env
import gym

def print_separator(title=None):
    """Print a separator line with optional title."""
    line = "="*80
    if title:
        print(f"\n{line}\n{title.center(80)}\n{line}")
    else:
        print(f"\n{line}\n")

def print_env_info(env):
    """Print detailed information about the environment."""
    print_separator("ENVIRONMENT INFORMATION")
    
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    
    # Display the objective weights
    print(f"\nReward Function Weights:")
    print(f"  - Distance Weight: {env._distance_weight}")
    print(f"  - Energy Weight: {env._energy_weight}")
    print(f"  - Drift Weight: {env._drift_weight}")
    print(f"  - Shake Weight: {env._shake_weight}")
    
    print(f"\nSimulation Parameters:")
    print(f"  - Time Step: {env._time_step}")
    print(f"  - Action Repeat: {env._action_repeat}")
    print(f"  - Control Time Step: {env.control_time_step}")
    print(f"  - Bullet Solver Iterations: {env._num_bullet_solver_iterations}")
    print(f"  - Distance Limit: {env._distance_limit}")
    print(f"  - Observation Noise StdDev: {env._observation_noise_stdev}")
    print(f"  - URDF Version: {env._urdf_version}")
    
    print(f"\nMotor Controls:")
    print(f"  - PD Control Enabled: {env._pd_control_enabled}")
    print(f"  - Leg Model Enabled: {env._leg_model_enabled}")
    print(f"  - Accurate Motor Model: {env._accurate_motor_model_enabled}")
    print(f"  - Motor KP: {env._motor_kp}")
    print(f"  - Motor KD: {env._motor_kd}")
    print(f"  - Torque Control: {env._torque_control_enabled}")
    print(f"  - Motor Velocity Limit: {env._motor_velocity_limit}")

def print_observation_breakdown(observation):
    """Print a detailed breakdown of an observation."""
    print_separator("OBSERVATION BREAKDOWN")
    
    # Motor angles (first 8 values)
    motor_angles = observation[0:8]
    print("Motor Angles (radians):")
    for i, angle in enumerate(motor_angles):
        print(f"  Motor {i}: {angle:.6f}")
    
    # Motor velocities (next 8 values)
    motor_velocities = observation[8:16]
    print("\nMotor Velocities (rad/s):")
    for i, velocity in enumerate(motor_velocities):
        print(f"  Motor {i}: {velocity:.6f}")
    
    # Motor torques (next 8 values)
    motor_torques = observation[16:24]
    print("\nMotor Torques (N·m):")
    for i, torque in enumerate(motor_torques):
        print(f"  Motor {i}: {torque:.6f}")
    
    # Base orientation (last 4 values)
    base_orientation = observation[24:28]
    print("\nBase Orientation (quaternion):")
    print(f"  x: {base_orientation[0]:.6f}")
    print(f"  y: {base_orientation[1]:.6f}")
    print(f"  z: {base_orientation[2]:.6f}")
    print(f"  w: {base_orientation[3]:.6f}")

def create_environment(render=True):
    """Create and return a Minitaur environment."""
    env = minitaur_gym_env.MinitaurGymEnv(
        render=render,
        motor_velocity_limit=np.inf,
        pd_control_enabled=True,
        hard_reset=True,
        # Set a higher camera position to see more of the environment
        # cam_dist=1.5,
        # cam_yaw=0,
        # cam_pitch=-30
    )
    
    # Reset the environment to initialize everything
    observation = env.reset()
    
    return env, observation

def reset_to_standing_pose(env):
    """Reset the environment to a stable standing pose."""
    # This is a neutral standing pose for the minitaur
    standing_pose = [0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5]
    observation = env.reset(initial_motor_angles=standing_pose, reset_duration=2.0)
    
    # Allow the robot to stabilize
    for _ in range(100):
        env.step(standing_pose)
    
    return observation

def motor_sweep_experiment(env, motor_index, amplitude=0.3, cycles=2, steps_per_cycle=100):
    """
    Experiment: Sweep a single motor back and forth while keeping others fixed.
    
    Args:
        env: The Minitaur environment
        motor_index: Which motor to control (0-7)
        amplitude: How far to move the motor (radians)
        cycles: Number of back-and-forth cycles
        steps_per_cycle: How many steps per cycle
    
    Returns:
        DataFrame with collected data
    """
    print_separator(f"MOTOR SWEEP EXPERIMENT: Motor {motor_index}")
    
    # Reset to a stable position
    observation = reset_to_standing_pose(env)
    base_action = [0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5]  # Standing pose
    
    # Data collection
    data = {
        'step': [],
        'motor_angle': [],
        'motor_velocity': [],
        'motor_torque': [],
        'base_pos_x': [],
        'base_pos_y': [],
        'base_pos_z': [],
        'action': [],
        'reward': [],
    }
    
    # Perform the sweep
    total_steps = int(cycles * steps_per_cycle)
    for step in range(total_steps):
        # Calculate the sine wave for smooth motion
        phase = step / steps_per_cycle * 2 * np.pi
        offset = amplitude * np.sin(phase)
        
        # Create action by modifying only the specified motor
        action = base_action.copy()
        action[motor_index] = base_action[motor_index] + offset
        
        # Take a step
        next_obs, reward, done, _ = env.step(action)
        
        # Collect data
        base_pos = env.minitaur.GetBasePosition()
        data['step'].append(step)
        data['motor_angle'].append(next_obs[motor_index])
        data['motor_velocity'].append(next_obs[motor_index + 8])
        data['motor_torque'].append(next_obs[motor_index + 16])
        data['base_pos_x'].append(base_pos[0])
        data['base_pos_y'].append(base_pos[1])
        data['base_pos_z'].append(base_pos[2])
        data['action'].append(action[motor_index])
        data['reward'].append(reward)
        
        # Print progress every 10 steps
        if step % 10 == 0:
            print(f"Step {step}/{total_steps}: Motor {motor_index} = {action[motor_index]:.4f}, Reward = {reward:.4f}")
        
        if done:
            print("Environment terminated early")
            break
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    filename = f"motor_{motor_index}_sweep_data.csv"
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    
    return df

def render_still_frame(env):
    """Render a still frame of the robot in its current state."""
    frame = env.render(mode="rgb_array")
    plt.figure(figsize=(10, 8))
    plt.imshow(frame)
    plt.axis('off')
    plt.title('Minitaur Robot')
    plt.savefig('minitaur_frame.png')
    plt.close()
    print("Saved still frame to minitaur_frame.png")

def map_actions_to_legs():
    """Print information about which motor corresponds to which leg."""
    print_separator("MOTOR TO LEG MAPPING")
    
    # This mapping may need verification with the actual implementation
    # Based on examination of the code and common quadruped designs
    print("Motor-to-Leg Mapping:")
    print("  Motors 0-1: Front Right Leg (0: hip, 1: knee)")
    print("  Motors 2-3: Front Left Leg (2: hip, 3: knee)")
    print("  Motors 4-5: Back Right Leg (4: hip, 5: knee)")
    print("  Motors 6-7: Back Left Leg (6: hip, 7: knee)")
    
    print("\nStanding Pose [0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5] means:")
    print("  - All hip joints at 0 radians (straight)")
    print("  - All knee joints at 0.5 radians (bent)")

def random_action_experiment(env, num_steps=100):
    """
    Experiment: Apply random actions and observe the effects.
    
    Args:
        env: The Minitaur environment
        num_steps: Number of random actions to apply
    
    Returns:
        DataFrame with collected data
    """
    print_separator("RANDOM ACTION EXPERIMENT")
    
    # Reset the environment
    observation = env.reset()
    
    # Data collection
    data = {
        'step': [],
        'action': [[] for _ in range(8)],  # One list per motor
        'motor_angle': [[] for _ in range(8)],
        'motor_velocity': [[] for _ in range(8)],
        'motor_torque': [[] for _ in range(8)],
        'base_pos_x': [],
        'base_pos_y': [],
        'base_pos_z': [],
        'reward': [],
        'is_fallen': []
    }
    
    # Apply random actions
    for step in range(num_steps):
        # Generate random action within the action space
        action = env.action_space.sample()
        
        # Take a step
        next_obs, reward, done, _ = env.step(action)
        
        # Collect data
        base_pos = env.minitaur.GetBasePosition()
        data['step'].append(step)
        for i in range(8):
            data['action'][i].append(action[i])
            data['motor_angle'][i].append(next_obs[i])
            data['motor_velocity'][i].append(next_obs[i + 8])
            data['motor_torque'][i].append(next_obs[i + 16])
        data['base_pos_x'].append(base_pos[0])
        data['base_pos_y'].append(base_pos[1])
        data['base_pos_z'].append(base_pos[2])
        data['reward'].append(reward)
        data['is_fallen'].append(env.is_fallen())
        
        # Print progress every 10 steps
        if step % 10 == 0:
            print(f"Step {step}/{num_steps}: Reward = {reward:.4f}, Fallen = {env.is_fallen()}")
            print(f"  Base Pos: X={base_pos[0]:.2f}, Y={base_pos[1]:.2f}, Z={base_pos[2]:.2f}")
        
        if done:
            print("Environment terminated early at step", step)
            break
    
    # Convert to DataFrame with flattened structure
    flat_data = {
        'step': data['step'],
        'base_pos_x': data['base_pos_x'],
        'base_pos_y': data['base_pos_y'],
        'base_pos_z': data['base_pos_z'],
        'reward': data['reward'],
        'is_fallen': data['is_fallen']
    }
    
    # Add flattened motor data
    for i in range(8):
        flat_data[f'action_motor_{i}'] = data['action'][i]
        flat_data[f'angle_motor_{i}'] = data['motor_angle'][i]
        flat_data[f'velocity_motor_{i}'] = data['motor_velocity'][i]
        flat_data[f'torque_motor_{i}'] = data['motor_torque'][i]
    
    df = pd.DataFrame(flat_data)
    
    # Save to CSV
    filename = "random_action_data.csv"
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    
    return df

def walking_pattern_experiment(env, num_steps=500):
    """
    Experiment: Apply a simple walking pattern and collect data.
    
    Args:
        env: The Minitaur environment
        num_steps: Number of steps to run
    
    Returns:
        DataFrame with collected data
    """
    print_separator("WALKING PATTERN EXPERIMENT")
    
    # Reset the environment
    observation = env.reset()
    
    # Data collection setup
    data = {
        'step': [],
        'base_pos_x': [],
        'base_pos_y': [],
        'base_pos_z': [],
        'base_orientation': [[] for _ in range(4)],
        'reward': [],
        'forward_distance': [],
        'is_fallen': []
    }
    
    # Add columns for each motor
    for i in range(8):
        data[f'action_motor_{i}'] = []
        data[f'angle_motor_{i}'] = []
        data[f'velocity_motor_{i}'] = []
        data[f'torque_motor_{i}'] = []
    
    # Simple walking pattern - a sine wave with phase differences for each leg
    base_pose = [0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5]  # Standing pose
    
    # Store initial position to calculate total distance
    initial_pos = env.minitaur.GetBasePosition()
    last_pos = initial_pos
    
    # Run the pattern
    for step in range(num_steps):
        # Calculate the sine wave for smooth motion
        t = step / 50.0  # Time parameter
        
        # Create walking pattern by applying sine waves with phase differences
        # Front right leg (motors 0-1)
        action = base_pose.copy()
        action[0] = base_pose[0] + 0.2 * np.sin(t * 2 * np.pi)
        action[1] = base_pose[1] + 0.2 * np.sin(t * 2 * np.pi + np.pi/2)
        
        # Front left leg (motors 2-3) - opposite phase
        action[2] = base_pose[2] + 0.2 * np.sin(t * 2 * np.pi + np.pi)
        action[3] = base_pose[3] + 0.2 * np.sin(t * 2 * np.pi + np.pi + np.pi/2)
        
        # Back right leg (motors 4-5) - opposite phase from front left
        action[4] = base_pose[4] + 0.2 * np.sin(t * 2 * np.pi + np.pi)
        action[5] = base_pose[5] + 0.2 * np.sin(t * 2 * np.pi + np.pi + np.pi/2)
        
        # Back left leg (motors 6-7) - opposite phase from front right
        action[6] = base_pose[6] + 0.2 * np.sin(t * 2 * np.pi)
        action[7] = base_pose[7] + 0.2 * np.sin(t * 2 * np.pi + np.pi/2)
        
        # Take a step
        next_obs, reward, done, _ = env.step(action)
        
        # Get positions and orientation
        base_pos = env.minitaur.GetBasePosition()
        base_orientation = env.minitaur.GetBaseOrientation()
        
        # Calculate forward distance from last step
        forward_dist = base_pos[0] - last_pos[0]
        last_pos = base_pos
        
        # Collect data
        data['step'].append(step)
        data['base_pos_x'].append(base_pos[0])
        data['base_pos_y'].append(base_pos[1])
        data['base_pos_z'].append(base_pos[2])
        for i in range(4):
            data['base_orientation'][i].append(base_orientation[i])
        data['reward'].append(reward)
        data['forward_distance'].append(forward_dist)
        data['is_fallen'].append(env.is_fallen())
        
        # Store motor data
        for i in range(8):
            data[f'action_motor_{i}'].append(action[i])
            data[f'angle_motor_{i}'].append(next_obs[i])
            data[f'velocity_motor_{i}'].append(next_obs[i + 8])
            data[f'torque_motor_{i}'].append(next_obs[i + 16])
        
        # Print progress periodically
        if step % 50 == 0:
            total_distance = base_pos[0] - initial_pos[0]
            print(f"Step {step}/{num_steps}: Reward = {reward:.4f}")
            print(f"  Position: X={base_pos[0]:.2f}, Y={base_pos[1]:.2f}, Z={base_pos[2]:.2f}")
            print(f"  Total forward distance: {total_distance:.2f} meters")
        
        if done:
            print("Environment terminated early at step", step)
            break
    
    # Convert to DataFrame with flattened structure
    flat_data = {
        'step': data['step'],
        'base_pos_x': data['base_pos_x'],
        'base_pos_y': data['base_pos_y'],
        'base_pos_z': data['base_pos_z'],
        'reward': data['reward'],
        'forward_distance': data['forward_distance'],
        'is_fallen': data['is_fallen']
    }
    
    # Add orientation data
    for i in range(4):
        flat_data[f'orientation_{i}'] = data['base_orientation'][i]
    
    # Add motor data
    for i in range(8):
        flat_data[f'action_motor_{i}'] = data[f'action_motor_{i}']
        flat_data[f'angle_motor_{i}'] = data[f'angle_motor_{i}']
        flat_data[f'velocity_motor_{i}'] = data[f'velocity_motor_{i}']
        flat_data[f'torque_motor_{i}'] = data[f'torque_motor_{i}']
    
    df = pd.DataFrame(flat_data)
    
    # Save to CSV
    filename = "walking_pattern_data.csv"
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    
    # Calculate and print statistics
    total_distance = data['base_pos_x'][-1] - initial_pos[0]
    print(f"\nExperiment Results:")
    print(f"  Total Steps: {len(data['step'])}")
    print(f"  Total Forward Distance: {total_distance:.2f} meters")
    print(f"  Average Reward per Step: {np.mean(data['reward']):.4f}")
    
    return df

def simple_plot(df, filename, title, x_col, y_cols, legends=None):
    """Create a simple line plot from DataFrame columns."""
    plt.figure(figsize=(12, 6))
    
    if not isinstance(y_cols, list):
        y_cols = [y_cols]
    
    if legends is None:
        legends = y_cols
    
    for y_col, legend in zip(y_cols, legends):
        plt.plot(df[x_col], df[y_col], label=legend)
    
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved to {filename}")

def main():
    """Main function to run experiments."""
    print_separator("MINITAUR ENVIRONMENT EXPLORATION")
    print("Loading environment...")
    
    # Create the environment
    env, initial_observation = create_environment(render=True)
    
    # Print environment information
    print_env_info(env)
    
    # Print observation breakdown
    print_observation_breakdown(initial_observation)
    
    # Print motor-to-leg mapping information
    map_actions_to_legs()
    
    # Allow user to see the initial state
    print("\nDisplaying initial state for 5 seconds...")
    time.sleep(5)
    
    # Render a still frame
    render_still_frame(env)
    
    # Menu of experiments
    while True:
        print_separator("EXPERIMENT MENU")
        print("1. Individual Motor Sweep")
        print("2. Random Action Experiment")
        print("3. Walking Pattern Experiment")
        print("4. Reset to Standing Pose")
        print("5. Render Still Frame")
        print("6. Print Environment Info")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ")
        
        if choice == '1':
            motor_idx = int(input("Enter motor index (0-7): "))
            if 0 <= motor_idx <= 7:
                df = motor_sweep_experiment(env, motor_idx)
                # Create plots
                simple_plot(df, f"motor_{motor_idx}_angle.png", 
                            f"Motor {motor_idx} Angle vs. Action", 
                            'step', ['motor_angle', 'action'],
                            ['Actual Angle', 'Commanded Action'])
                
                simple_plot(df, f"motor_{motor_idx}_velocity.png", 
                            f"Motor {motor_idx} Velocity", 
                            'step', 'motor_velocity')
                
                simple_plot(df, f"motor_{motor_idx}_torque.png", 
                            f"Motor {motor_idx} Torque", 
                            'step', 'motor_torque')
            else:
                print("Invalid motor index! Must be between 0 and 7.")
        
        elif choice == '2':
            steps = int(input("Enter number of steps (default: 100): ") or "100")
            df = random_action_experiment(env, steps)
            # Create plots
            simple_plot(df, "random_action_position.png", 
                        "Robot Position During Random Actions", 
                        'step', ['base_pos_x', 'base_pos_y', 'base_pos_z'],
                        ['X Position', 'Y Position', 'Z Position'])
            
            simple_plot(df, "random_action_reward.png", 
                        "Reward During Random Actions", 
                        'step', 'reward')
        
        elif choice == '3':
            steps = int(input("Enter number of steps (default: 500): ") or "500")
            df = walking_pattern_experiment(env, steps)
            # Create plots
            simple_plot(df, "walking_pattern_position.png", 
                        "Robot Position During Walking Pattern", 
                        'step', ['base_pos_x', 'base_pos_y', 'base_pos_z'],
                        ['X Position', 'Y Position', 'Z Position'])
            
            simple_plot(df, "walking_pattern_reward.png", 
                        "Reward During Walking Pattern", 
                        'step', 'reward')
            
            simple_plot(df, "walking_pattern_distance.png", 
                        "Forward Distance per Step", 
                        'step', 'forward_distance')
        
        elif choice == '4':
            print("Resetting to standing pose...")
            reset_to_standing_pose(env)
            print("Done!")
        
        elif choice == '5':
            print("Rendering still frame...")
            render_still_frame(env)
        
        elif choice == '6':
            print_env_info(env)
            # Get current observation
            observation = env._get_observation()
            print_observation_breakdown(observation)
        
        elif choice == '7':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice! Please enter a number between 1 and 7.")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()