# plot_hill_climber_run.py

import os
import json
import matplotlib.pyplot as plt
import numpy as np

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

def plot_best_distance(run_logs, save_path):
    """
    Plot the best distance achieved over iterations.

    Args:
        run_logs (list): List of run data dictionaries.
        save_path (str): Path to save the plot.
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

    plt.figure(figsize=(8, 4.5))  # Adjusted size to prevent overlapping
    plt.plot(best_distance, label='Best Distance', color='blue')
    plt.xlabel('Run Number')
    plt.ylabel('Distance Traveled (meters)')
    plt.title('Best Distance Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Best distance plot saved to '{save_path}'.")

def plot_parameter_evolution(run_logs, save_dir):
    """
    Plot the evolution of the best gait parameters (a, b, c, omega) over iterations.

    Args:
        run_logs (list): List of run data dictionaries.
        save_dir (str): Directory to save the plots.
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

    # Define subplot grid
    cols = 2  # Number of columns in subplot grid
    total_plots = 1 + len(ALL_JOINTS)  # 1 for omega + 1 per joint
    rows = int(np.ceil(total_plots / cols))  # Calculate number of rows needed

    plt.figure(figsize=(10, 3 * rows))  # Adjusted size to prevent overlapping

    # Plot Omega
    plt.subplot(rows, cols, 1)
    plt.plot(run_numbers, omega_values, label='Omega', color='green')
    plt.xlabel('Run Number')
    plt.ylabel('Omega (rad/s)')
    plt.title('Evolution of Omega Over Iterations')
    plt.legend(fontsize='small')
    plt.grid(True)

    # Plot a, b, c for each joint in separate subplots
    for idx, joint in enumerate(ALL_JOINTS, start=2):
        plt.subplot(rows, cols, idx)
        plt.plot(run_numbers, a_values[joint], label='a', color='blue')
        plt.plot(run_numbers, b_values[joint], label='b', color='orange')
        plt.plot(run_numbers, c_values[joint], label='c', color='red')
        plt.xlabel('Run Number')
        plt.ylabel('Parameter Value')
        plt.title(f'Evolution of Parameters for Joint {joint}')
        plt.legend(fontsize='small')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'best_parameters_evolution_over_runs.png'))
    plt.close()
    print(f"Gait parameters evolution plot saved to '{os.path.join(save_dir, 'best_parameters_evolution_over_runs.png')}'.")

def plot_hyperparameters(run_logs, save_dir):
    """
    Plot training hyperparameters over iterations.

    Args:
        run_logs (list): List of run data dictionaries.
        save_dir (str): Directory to save the plots.
    """
    # Check if hyperparameters are present in logs
    has_hyperparams = all('hyperparameters' in run for run in run_logs)
    if not has_hyperparams:
        print("No hyperparameters found in run logs. Skipping hyperparameter plotting.")
        return

    # Sort runs by run_number
    run_logs_sorted = sorted(run_logs, key=lambda x: x['run_number'])

    run_numbers = [run['run_number'] for run in run_logs_sorted]

    # Assuming hyperparameters is a dictionary of hyperparam_name: value
    # Collect all hyperparameter names
    hyperparam_names = set()
    for run in run_logs_sorted:
        hyperparam_names.update(run['hyperparameters'].keys())

    hyperparam_values = {name: [] for name in hyperparam_names}

    for run in run_logs_sorted:
        for name in hyperparam_names:
            hyperparam_values[name].append(run['hyperparameters'].get(name, np.nan))

    # Plot each hyperparameter
    plt.figure(figsize=(10, 6))  # Adjusted size to prevent overlapping
    for name in hyperparam_names:
        plt.plot(run_numbers, hyperparam_values[name], label=name)
    plt.xlabel('Run Number')
    plt.ylabel('Hyperparameter Value')
    plt.title('Evolution of Training Hyperparameters Over Iterations')
    plt.legend(fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hyperparameters_evolution.png'))
    plt.close()
    print(f"Hyperparameters evolution plot saved to '{os.path.join(save_dir, 'hyperparameters_evolution.png')}'.")

def plot_hill_climber_run(run_logs, save_dir='plots_hill_climber_run'):
    """
    Plot training run metrics including best distance and parameter evolution.

    Args:
        run_logs (list): List of run data dictionaries.
        save_dir (str): Directory to save the plots.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Plot best distance over iterations
    plot_best_distance(run_logs, save_path=os.path.join(save_dir, 'best_distance_over_iterations.png'))

    # Plot parameter evolution
    plot_parameter_evolution(run_logs, save_dir=save_dir)

    # Plot hyperparameters if available
    plot_hyperparameters(run_logs, save_dir=save_dir)

def plot_hill_climber_run_main():
    """
    Main function to visualize hill climber training runs.
    """
    logs_dir = 'logs_hill_climber'

    # Load all run logs
    run_logs = load_run_logs(logs_dir)

    if not run_logs:
        print("No run logs found. Please ensure that the logs are correctly saved in 'logs_hill_climber/run_X/log.json'.")
        return

    # Plot training run metrics
    plot_hill_climber_run(run_logs, save_dir='plots_hill_climber_run')

if __name__ == "__main__":
    plot_hill_climber_run_main()
