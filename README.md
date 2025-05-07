# Quadruped Robot Gait Optimization using Hill Climber Algorithm

## Table of Contents
- [Quadruped Robot Gait Optimization using Hill Climber Algorithm](#quadruped-robot-gait-optimization-using-hill-climber-algorithm)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Directory Structure](#directory-structure)
  - [Installation](#installation)
- [Simulation Parameters](#simulation-parameters)
- [Hill Climber Parameters](#hill-climber-parameters)

## Overview

This project focuses on optimizing the gait parameters of a quadruped robot using a Hill Climber optimization algorithm. By iteratively adjusting parameters such as joint angles and angular frequency, the goal is to maximize the distance traveled by the robot in a simulated environment powered by PyBullet.

## Features

- **Hill Climber Optimization**: Iteratively searches for optimal gait parameters to maximize robot movement.
- **Structured Logging**: Saves detailed logs for each run, facilitating analysis and reproducibility.
- **Visualization Tools**: Provides scripts to visualize both training runs and the best-performing run.
- **Organized Directory Structure**: Separates raw training data from analyzed results for better management and version control.

## Directory Structure

your_project_directory/ ├── code_files/ │ ├── hillclimber.py │ ├── visualize_hillclimber_training_run.py │ └── visualize_hillclimber_best_run.py ├── training_runs/ │ └── hillclimber/ │ ├── happy_sun/ │ │ ├── training_hyperparameters.json │ │ ├── best.json │ │ └── logs/ │ │ ├── log_1.json │ │ ├── log_2.json │ │ └── ... (other log files) │ ├── brave_moon/ │ │ ├── training_hyperparameters.json │ │ ├── best.json │ │ └── logs/ │ │ ├── log_1.json │ │ ├── log_2.json │ │ └── ... (other log files) │ └── ... (other training runs) ├── results/ │ └── hillclimber/ │ ├── happy_sun/ │ │ ├── training_hyperparameters.json │ │ ├── values.json │ │ ├── plots_best_run/ │ │ │ ├── robot_motion_best_run.png │ │ │ ├── servo_angle_functions_best_run.png │ │ │ └── servo_angle_functions_each_joint.png │ │ └── plots_training_run/ │ │ ├── best_distance_over_iterations.png │ │ ├── gait_parameters_evolution_over_runs.png │ │ └── training_hyperparameters.png │ ├── brave_moon/ │ │ ├── training_hyperparameters.json │ │ ├── values.json │ │ ├── plots_best_run/ │ │ │ ├── robot_motion_best_run.png │ │ │ ├── servo_angle_functions_best_run.png │ │ │ └── servo_angle_functions_each_joint.png │ │ └── plots_training_run/ │ │ ├── best_distance_over_iterations.png │ │ ├── gait_parameters_evolution_over_runs.png │ │ └── training_hyperparameters.png │ └── ... (other results runs) ├── urdf/ │ └── urdf-assembly.urdf ├── README.md └── requirements.txt

bash
Copy code

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your_username/quadruped-gait-optimization.git
   cd quadruped-gait-optimization
Set Up a Virtual Environment (Optional but Recommended)

bash
Copy code
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Required Packages

bash
Copy code
pip install -r requirements.txt
If requirements.txt is not provided, install the necessary packages manually:

bash
Copy code
pip install pybullet matplotlib numpy argparse
Ensure URDF File is Present

Verify that the urdf-assembly.urdf file is located in the urdf/ directory or adjust the paths in the scripts accordingly.
Usage
Running the Hill Climber Optimization
The hillclimber.py script performs the hill climber optimization to find the best gait parameters.

Navigate to the Code Directory

bash
Copy code
cd code_files
Execute the Script

bash
Copy code
python hillclimber.py
What Happens:
Generates a unique run name (e.g., happy_sun).
Initializes the directory structure for the run.
Saves initial hyperparameters.
Runs the hill climber optimization for the specified number of iterations.
Logs each run's parameters and performance.
Saves the best run's data for visualization.
Launches a PyBullet GUI to visualize the best gait.
Visualizing Training Runs
The visualize_hillclimber_training_run.py script generates plots for a specific training run, showing metrics like best distance over iterations and parameter evolution.

Navigate to the Code Directory

bash
Copy code
cd code_files
Execute the Visualization Script

bash
Copy code
python visualize_hillclimber_training_run.py --run happy_sun
Parameters:

--run: The name of the training run you wish to visualize (e.g., happy_sun).
What Happens:

Loads all log files from training_runs/hillclimber/happy_sun/logs/.
Generates and saves plots in results/hillclimber/happy_sun/plots_training_run/:
best_distance_over_iterations.png
gait_parameters_evolution_over_runs.png
training_hyperparameters.png
Visualizing the Best Run
The visualize_hillclimber_best_run.py script visualizes the best-performing run, providing insights into the robot's movement and joint behaviors.

Navigate to the Code Directory

bash
Copy code
cd code_files
Execute the Visualization Script

bash
Copy code
python visualize_hillclimber_best_run.py --run happy_sun
Parameters:

--run: The name of the training run you wish to visualize (e.g., happy_sun).
What Happens:

Loads the best run's parameters from results/hillclimber/happy_sun/values.json.
Initializes a PyBullet GUI simulation using the best gait parameters.
Records and saves robot motion data.
Generates and saves plots in results/hillclimber/happy_sun/plots_best_run/:
robot_motion_best_run.png
servo_angle_functions_best_run.png
servo_angle_functions_each_joint.png
Configuration
All configuration parameters are defined at the beginning of each script. You can adjust parameters such as simulation duration, mutation rates, and directories as needed.

Example Parameters in hillclimber.py:

python
Copy code
# Simulation Parameters
SIMULATION_FPS = 240
RUN_DURATION = 10
TOTAL_STEPS = SIMULATION_FPS * RUN_DURATION
SETTLING_TIME = 0.5
SETTLING_STEPS = int(SIMULATION_FPS * SETTLING_TIME)

# Hill Climber Parameters
MAX_ITERATIONS = 1000
MUTATION_RATE = 0.9
MUTATION_SCALE = 0.05
Results
After running the optimization and visualization scripts, you can find the results organized in the results/ directory.

Training Runs:

Contains plots related to the training process, such as distance progression and parameter evolution.
Best Runs:

Contains detailed plots and data for the best-performing run, including robot motion and joint angle functions.
Troubleshooting
PyBullet Issues:

Ensure that the URDF file path is correct.
Verify that PyBullet is properly installed.
Missing Logs or Plots:

Confirm that the scripts have permission to read/write in the specified directories.
Ensure that the run names provided match existing directories.
Visualization Errors:

Check that all required Python packages are installed.
Verify that the simulation parameters are set correctly.
Contributing
Contributions are welcome! Please follow these steps to contribute:

Fork the Repository

Create a New Branch

bash
Copy code
git checkout -b feature/YourFeatureName
Make Your Changes

Commit Your Changes

bash
Copy code
git commit -m "Add feature: YourFeatureName"
Push to Your Fork

bash
Copy code
git push origin feature/YourFeatureName
Create a Pull Request

License
This project is licensed under the MIT License.

Acknowledgements
PyBullet for providing a robust physics simulation environment.
Matplotlib for facilitating data visualization.
NumPy for numerical computations.
Inspired by research in robotics and optimization algorithms.
