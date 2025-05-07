# visualize_hillclimber_training_run.py

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

def load_run_logs(run_name, training_runs_dir="training_runs/hillclimber"):
    """
    Load all run logs for a specific training run from the specified directory.

    Args:
        run_name (str): The unique name of the training run.
        training_runs_dir (str): Base directory where training runs are stored.

    Returns:
        list: List of dictionaries containing run data.
    """
    logs_dir = os.path.join(training_runs_dir, run_name, "logs")
    if not os.path.exists(logs_dir):
        print(f"Logs directory '{logs_dir}' does not exist.")
        return []

    run_logs = []
    for entry in os.listdir(logs_dir):
        if entry.startswith("log_") and entry.endswith(".json"):
            run_path = os.path.join(logs_dir, entry)
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

    plt.figure(figsize=(10, 6))
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

def plot_parameter_evolution(run_logs, save_path):
    """
    Plot the evolution of the best gait parameters (a, b, c, omega) over iterations.

    Args:
        run_logs (list): List of run data dictionaries.
        save_path (str): Path to save the plot.
    """
    # Sort runs by run_number
    run_logs_sorted = sorted(run_logs, key=lambda x: x['run_number'])

    run_numbers = [run['run_number'] for run in run_logs_sorted]

    # Extract gait parameters and omega
    a_values = {joint: [] for joint in run_logs_sorted[0]['gait_parameters_deg'].keys()}
    b_values = {joint: [] for joint in run_logs_sorted[0]['gait_parameters_deg'].keys()}
    c_values = {joint: [] for joint in run_logs_sorted[0]['gait_parameters_deg'].keys()}
    omega_values = []

    for run in run_logs_sorted:
        gait_params = run['gait_parameters_deg']
        omega = run['omega']
        omega_values.append(omega)
        for joint, params in gait_params.items():
            a_values[joint].append(params['a'])
            b_values[joint].append(params['b'])
            c_values[joint].append(params['c'])

    # Define subplot grid
    num_joints = len(a_values)
    cols = 2  # Number of columns in subplot grid
    total_plots = num_joints * 3 + 1  # a, b, c for each joint + 1 for omega
    rows = int(np.ceil(total_plots / cols))  # Calculate number of rows needed

    plt.figure(figsize=(15, 4 * rows))

    plot_idx = 1

    # Plot Omega
    plt.subplot(rows, cols, plot_idx)
    plt.plot(run_numbers, omega_values, label='Omega', color='green')
    plt.xlabel('Run Number')
    plt.ylabel('Omega (rad/s)')
    plt.title('Evolution of Omega Over Iterations')
    plt.legend(fontsize='small')
    plt.grid(True)
    plot_idx += 1

    # Plot a, b, c for each joint
    for joint in sorted(a_values.keys(), key=lambda x: int(x)):
        for param, values, color in zip(['a', 'b', 'c'], 
                                       [a_values[joint], b_values[joint], c_values[joint]], 
                                       ['blue', 'orange', 'red']):
            plt.subplot(rows, cols, plot_idx)
            plt.plot(run_numbers, values, label=param.upper(), color=color)
            plt.xlabel('Run Number')
            plt.ylabel(f'Parameter {param.upper()} (degrees)')
            plt.title(f'Evolution of Parameter {param.upper()} for Joint {joint}')
            plt.legend(fontsize='small')
            plt.grid(True)
            plot_idx += 1

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Gait parameters evolution plot saved to '{save_path}'.")

def plot_training_hyperparameters(run_name, results_dir="results/hillclimber"):
    """
    Plot the training hyperparameters for a specific run.

    Args:
        run_name (str): The unique name of the training run.
        results_dir (str): Base directory where results are stored.
    """
    hyperparams_path = os.path.join(results_dir, run_name, "training_hyperparameters.json")
    if not os.path.exists(hyperparams_path):
        print(f"Hyperparameters file '{hyperparams_path}' does not exist.")
        return

    with open(hyperparams_path, 'r') as f:
        hyperparams = json.load(f)

    # Exclude joint indices from plotting
    hyperparams_to_plot = {k: v for k, v in hyperparams.items() if k != "joint_indices"}

    keys = list(hyperparams_to_plot.keys())
    values = list(hyperparams_to_plot.values())

    plt.figure(figsize=(10, 6))
    plt.barh(keys, values, color='skyblue')
    plt.xlabel('Value')
    plt.title(f'Training Hyperparameters for Run {run_name}')
    plt.tight_layout()
    save_path = os.path.join(results_dir, run_name, "plots_training_run", "training_hyperparameters.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Training hyperparameters plot saved to '{save_path}'.")

def plot_hill_climber_training_run(run_name):
    """
    Plot training run metrics including best distance and parameter evolution.

    Args:
        run_name (str): The unique name of the training run.
    """
    training_runs_dir = "training_runs/hillclimber"
    results_dir = "results/hillclimber"

    # Load all run logs for the specified run
    run_logs = load_run_logs(run_name, training_runs_dir=training_runs_dir)

    if not run_logs:
        print(f"No run logs found for run '{run_name}'. Please ensure that the logs are correctly saved.")
        return

    # Define save directories
    save_dir_best_run = os.path.join(results_dir, run_name, "plots_best_run")
    save_dir_training_run = os.path.join(results_dir, run_name, "plots_training_run")

    os.makedirs(save_dir_best_run, exist_ok=True)
    os.makedirs(save_dir_training_run, exist_ok=True)

    # Plot best distance over iterations
    best_distance_plot_path = os.path.join(save_dir_training_run, 'best_distance_over_iterations.png')
    plot_best_distance(run_logs, best_distance_plot_path)

    # Plot parameter evolution
    parameter_evolution_plot_path = os.path.join(save_dir_training_run, 'gait_parameters_evolution_over_runs.png')
    plot_parameter_evolution(run_logs, parameter_evolution_plot_path)

    # Plot training hyperparameters
    plot_training_hyperparameters(run_name, results_dir=results_dir)

    print(f"All plots for run '{run_name}' have been saved in '{save_dir_training_run}' and '{save_dir_best_run}'.")

def main():
    parser = argparse.ArgumentParser(description="Visualize Hill Climber Training Runs")
    parser.add_argument('--run', type=str, required=True, help="Name of the training run (e.g., happy_sun)")
    args = parser.parse_args()

    run_name = args.run
    plot_hill_climber_training_run(run_name)

if __name__ == "__main__":
    main()
