import os
import json
import pybullet as p
import pybullet_data
import time
import math
import random
from shutil import rmtree
from datetime import datetime
from tqdm import trange  # For progress bar

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

# ------------------------ Hill Climber Parameters ------------------------ #
MAX_ITERATIONS = 1000        # Maximum number of iterations
MUTATION_RATE = 0.5         # Probability of each parameter being mutated
MUTATION_SCALE = 0.1        # 10% mutation scale
LOG_INTERVAL = 100          # How often to log progress

# ------------------------ Directory Paths ------------------------ #
TRAINING_RUNS_DIR = "training_runs/hillclimber"   # Directory to store training runs
RESULTS_DIR = "results/hillclimber"               # Directory to store results

# ------------------------ Other Parameters ------------------------ #
SIMULATION_FPS = 240         # Simulation steps per second
RUN_DURATION = 10           # Duration of each run in seconds
TOTAL_STEPS = SIMULATION_FPS * RUN_DURATION
SETTLING_TIME = 0.5         # Settling time in seconds
SETTLING_STEPS = int(SIMULATION_FPS * SETTLING_TIME)

# ------------------------ Gait Parameters Class ------------------------ #
class GaitParameters:
    def __init__(self, a=None, b=None, c=None, omega=None):
        self.a = a if a is not None else {joint: random.uniform(A_MIN, A_MAX) for joint in JOINT_MAPPING.keys()}
        self.b = b if b is not None else {joint: random.uniform(B_MIN, B_MAX) for joint in JOINT_MAPPING.keys()}
        self.c = c if c is not None else {joint: random.uniform(C_MIN, C_MAX) for joint in JOINT_MAPPING.keys()}
        self.omega = omega if omega is not None else random.uniform(OMEGA_MIN, OMEGA_MAX)

    def mutate(self):
        for joint in JOINT_MAPPING.keys():
            if random.random() < MUTATION_RATE:
                delta_a = self.a[joint] * MUTATION_SCALE
                self.a[joint] += random.uniform(-delta_a, delta_a)
                self.a[joint] = max(A_MIN, min(A_MAX, self.a[joint]))

            if random.random() < MUTATION_RATE:
                delta_b = self.b[joint] * MUTATION_SCALE
                self.b[joint] += random.uniform(-delta_b, delta_b)
                self.b[joint] = max(B_MIN, min(B_MAX, self.b[joint]))

            if random.random() < MUTATION_RATE:
                delta_c = MUTATION_SCALE * math.pi
                self.c[joint] += random.uniform(-delta_c, delta_c)
                self.c[joint] = (self.c[joint] + 2 * math.pi) % (2 * math.pi)

        if random.random() < MUTATION_RATE:
            delta_omega = (OMEGA_MAX - OMEGA_MIN) * MUTATION_SCALE
            self.omega += random.uniform(-delta_omega, delta_omega)
            self.omega = max(OMEGA_MIN, min(OMEGA_MAX, self.omega))

    def copy(self):
        return GaitParameters(a=self.a.copy(), b=self.b.copy(), c=self.c.copy(), omega=self.omega)

    def to_dict(self):
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "omega": self.omega
        }

    def to_visualization_dict(self):
        """Convert parameters to format expected by visualization script"""
        gait_params = {}
        for joint in JOINT_MAPPING.keys():
            gait_params[str(joint)] = {
                "a": self.a[joint],
                "b": self.b[joint],
                "c": self.c[joint]
            }
        return gait_params

    def __str__(self):
        return json.dumps(self.to_dict(), indent=4)

# ------------------------ Initialization Functions ------------------------ #
def initialize_directories(run_name):
    """Initialize all necessary directories for training and results."""
    training_run_dir = os.path.join(TRAINING_RUNS_DIR, run_name)
    results_run_dir = os.path.join(RESULTS_DIR, run_name)

    dirs_to_create = [
        training_run_dir,
        os.path.join(training_run_dir, "logs"),
        results_run_dir,
        os.path.join(results_run_dir, "plots_best_run"),
        os.path.join(results_run_dir, "plots_training_run")
    ]

    for directory in dirs_to_create:
        os.makedirs(directory, exist_ok=True)

    return {
        "training_run": training_run_dir,
        "results_run": results_run_dir
    }

# ------------------------ Gait Evaluation Function ------------------------ #
def evaluate_gait(gait_params):
    """
    Evaluate a set of gait parameters by running a PyBullet simulation.
    Returns the distance traveled in the X-direction.
    """
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load Plane with specified friction
    plane_id = p.loadURDF("plane.urdf")
    p.changeDynamics(plane_id, -1, lateralFriction=FLOOR_FRICTION_COEFFICIENT)

    # Define Robot Orientation
    roll = math.radians(90)
    pitch = 0
    yaw = 0
    start_orientation = p.getQuaternionFromEuler([roll, pitch, yaw])

    # Load Robot URDF
    start_pos = [0, 0, 0.5]
    robot_id = p.loadURDF("urdf-assembly.urdf", start_pos, start_orientation)

    # Set friction for all robot links
    for link in range(p.getNumJoints(robot_id)):
        p.changeDynamics(robot_id, link, lateralFriction=FLOOR_FRICTION_COEFFICIENT)

    # Settling Steps
    for _ in range(SETTLING_STEPS):
        p.stepSimulation()

    # Reset Position and Orientation
    p.resetBasePositionAndOrientation(robot_id, start_pos, start_orientation)

    # Initial Position
    initial_pos, _ = p.getBasePositionAndOrientation(robot_id)
    initial_x = initial_pos[0]

    # Simulation Loop
    for step in range(TOTAL_STEPS):
        t = step / SIMULATION_FPS

        for joint in JOINT_MAPPING.keys():
            theta = gait_params.a[joint] + gait_params.b[joint] * math.sin(gait_params.omega * t + gait_params.c[joint])
            theta_rad = math.radians(theta)

            # Determine if the joint is a hip or knee
            if joint in [0, 2, 4, 6]:  # Upper joints
                max_force = HIP_MAX_FORCE
            else:  # Lower joints
                max_force = KNEE_MAX_FORCE

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

    # Final Position
    final_pos, _ = p.getBasePositionAndOrientation(robot_id)
    final_x = final_pos[0]

    p.disconnect()

    distance_traveled = final_x - initial_x
    return distance_traveled

def save_hyperparameters(directories, timestamp):
    """Save hyperparameters used in the training run."""
    hyperparams = {
        "optimization": {
            "max_iterations": MAX_ITERATIONS,
            "mutation_rate": MUTATION_RATE,
            "mutation_scale": MUTATION_SCALE,
            "log_interval": LOG_INTERVAL
        },
        "simulation": {
            "fps": SIMULATION_FPS,
            "run_duration": RUN_DURATION,
            "settling_time": SETTLING_TIME,
            "floor_friction": FLOOR_FRICTION_COEFFICIENT
        },
        "robot": {
            "hip_max_force": HIP_MAX_FORCE,
            "knee_max_force": KNEE_MAX_FORCE,
            "position_gain": POSITION_GAIN,
            "velocity_gain": VELOCITY_GAIN
        },
        "parameter_ranges": {
            "a_min": A_MIN,
            "a_max": A_MAX,
            "b_min": B_MIN,
            "b_max": B_MAX,
            "c_min": C_MIN,
            "c_max": C_MAX,
            "omega_min": OMEGA_MIN,
            "omega_max": OMEGA_MAX
        }
    }
    
    # Save in both training and results directories
    for dir_type in ["training_run", "results_run"]:
        hyperparams_file = os.path.join(directories[dir_type], f"hyperparameters_{timestamp}.json")
        with open(hyperparams_file, 'w') as f:
            json.dump(hyperparams, f, indent=4)

def hill_climber_optimization(run_name, directories):
    """
    Main hill climber loop with enhanced logging and progress updates.
    Includes hyperparameter logging and full parameter history tracking.
    """
    # Get timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save hyperparameters first
    save_hyperparameters(directories, timestamp)
    
    # Initialize parameters
    current_params = GaitParameters()
    current_distance = evaluate_gait(current_params)

    best_params = current_params.copy()
    best_distance = current_distance

    # Initialize progress logging
    metrics_log_file = os.path.join(directories["training_run"], "logs", f"metrics_{timestamp}.csv")
    params_log_file = os.path.join(directories["training_run"], "logs", f"params_{timestamp}.csv")
    
    # Write header to metrics log file
    with open(metrics_log_file, 'w') as f:
        f.write("iteration,current_distance,best_distance\n")

    # Write header to parameters log file
    param_header = ["iteration"]
    for joint in JOINT_MAPPING.keys():
        param_header.extend([f"a_{joint}", f"b_{joint}", f"c_{joint}"])
    param_header.append("omega")
    
    with open(params_log_file, 'w') as f:
        f.write(",".join(param_header) + "\n")

    # Save initial state
    results = {
        "gait_parameters_deg": best_params.to_visualization_dict(),
        "omega": best_params.omega,
        "best_distance_traveled_m": best_distance,
        "timestamp": timestamp
    }
    
    values_path = os.path.join(directories["results_run"], "values.json")
    with open(values_path, 'w') as f:
        json.dump(results, f, indent=4)

    # Function to log parameters
    def log_parameters(iteration, params):
        param_values = [str(iteration)]
        for joint in JOINT_MAPPING.keys():
            param_values.extend([
                str(params.a[joint]),
                str(params.b[joint]),
                str(params.c[joint])
            ])
        param_values.append(str(params.omega))
        return ",".join(param_values)

    # Main optimization loop
    for iteration in trange(1, MAX_ITERATIONS + 1, desc="Hill Climber Progress"):
        new_params = current_params.copy()
        new_params.mutate()

        new_distance = evaluate_gait(new_params)

        # Log metrics
        with open(metrics_log_file, 'a') as f:
            f.write(f"{iteration},{new_distance},{best_distance}\n")

        # Log parameters
        with open(params_log_file, 'a') as f:
            f.write(log_parameters(iteration, new_params) + "\n")

        if new_distance > best_distance:
            best_params = new_params.copy()
            best_distance = new_distance
            current_params = new_params.copy()

            # Update results file with new best parameters
            results = {
                "gait_parameters_deg": best_params.to_visualization_dict(),
                "omega": best_params.omega,
                "best_distance_traveled_m": best_distance,
                "timestamp": timestamp
            }
            with open(values_path, 'w') as f:
                json.dump(results, f, indent=4)

            # Save best parameters to a separate file
            best_params_file = os.path.join(directories["results_run"], f"best_params_{timestamp}.json")
            with open(best_params_file, 'w') as f:
                json.dump({
                    "iteration": iteration,
                    "distance": best_distance,
                    "parameters": best_params.to_dict(),
                    "timestamp": timestamp
                }, f, indent=4)

        # Print progress at intervals
        if iteration % LOG_INTERVAL == 0:
            print(f"\nIteration {iteration}/{MAX_ITERATIONS}")
            print(f"Current Distance: {new_distance:.3f} m")
            print(f"Best Distance: {best_distance:.3f} m")
            print("Current Parameters:")
            print(f"Omega: {new_params.omega:.3f}")
            print("Joint Parameters Sample:")
            print(f"Joint 0 - a: {new_params.a[0]:.3f}, b: {new_params.b[0]:.3f}, c: {new_params.c[0]:.3f}\n")

    return best_params, best_distance

def main():
    # Generate unique run name using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    
    print(f"Starting new training run: {run_name}")
    
    # Print hyperparameters at start
    print("\nHyperparameters:")
    print("------------------------")
    print(f"Optimization:")
    print(f"  Max Iterations: {MAX_ITERATIONS}")
    print(f"  Mutation Rate: {MUTATION_RATE}")
    print(f"  Mutation Scale: {MUTATION_SCALE}")
    print(f"\nSimulation:")
    print(f"  FPS: {SIMULATION_FPS}")
    print(f"  Run Duration: {RUN_DURATION}s")
    print(f"  Settling Time: {SETTLING_TIME}s")
    print(f"  Floor Friction: {FLOOR_FRICTION_COEFFICIENT}")
    print(f"\nRobot:")
    print(f"  Hip Force: {HIP_MAX_FORCE}")
    print(f"  Knee Force: {KNEE_MAX_FORCE}")
    print(f"  Position Gain: {POSITION_GAIN}")
    print(f"  Velocity Gain: {VELOCITY_GAIN}")
    
    # Initialize directories
    directories = initialize_directories(run_name)

    # Run optimization
    best_params, best_distance = hill_climber_optimization(run_name, directories)
    
    # Print final results
    print("\nOptimization Complete!")
    print(f"Best Distance Traveled: {best_distance:.3f} meters")
    print("Best Gait Parameters:")
    print(best_params)
    print(f"\nResults saved in: {directories['results_run']}")
    print(f"Full logs and hyperparameters saved in: {directories['training_run']}")

if __name__ == "__main__":
    main()