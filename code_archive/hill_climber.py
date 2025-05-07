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

# ------------------------ Explanation of Key Parameters ------------------------ #
# 1. Position Gain (Kp): Determines how strongly the robot tries to reach the target position.
#    - High values can make the robot snap to the target position quickly.
#    - Low values make the motion smoother but less precise.
#
# 2. Velocity Gain (Kd): Determines how much the velocity is controlled to avoid overshooting.
#    - High values can dampen motion and reduce oscillations.
#    - Low values make the robot move more freely but with less stability.
#
# 3. Torque: The maximum force the motor can apply.
#    - Higher torque allows the robot to push harder but can cause instability on low-friction surfaces.
#    - Lower torque results in weaker movements but better control for slippery surfaces.

# ------------------------ Configuration Parameters ------------------------ #

# Simulation Parameters
SIMULATION_FPS = 240          # Simulation steps per second
RUN_DURATION = 1000           # Duration of each run in seconds
TOTAL_STEPS = SIMULATION_FPS * RUN_DURATION
SETTLING_TIME = 0.5           # Settling time in seconds
SETTLING_STEPS = int(SIMULATION_FPS * SETTLING_TIME)

# Hill Climber Parameters
MAX_ITERATIONS = 1000         # Maximum number of iterations (adjust as needed)
MUTATION_RATE = 0.5           # Probability of each parameter being mutated
MUTATION_SCALE = 0.5          # Scale of mutation (5%)

# Logging Parameters
TRAINING_RUNS_DIR = "training_runs/hillclimber"   # Directory to store training runs (not backed up with git)
RESULTS_DIR = "results/hillclimber"               # Directory to store results (backed up with git)

# Gait Parameters Ranges (degrees)
A_MIN, A_MAX = 0, 15             # Range for 'a' parameters
B_MIN, B_MAX = 0, 15             # Range for 'b' parameters
C_MIN, C_MAX = 0, 2 * math.pi    # Range for 'c' parameters
OMEGA_MIN, OMEGA_MAX = 0.5, 10   # Range for 'omega' parameter

# Motor Force Limits (Adjusted for better movement)
HIP_MAX_FORCE = 3    # Reduced for low-friction surfaces
KNEE_MAX_FORCE = 3   # Reduced for low-friction surfaces

# Position and Velocity Gains (Fine-tuned for responsiveness)
POSITION_GAIN = 1  # Lowered to reduce aggressive movements
VELOCITY_GAIN = 1  # Lowered to reduce oscillations

# Friction Coefficients (Simulate low-friction environment)
FRICTION_COEFFICIENT = 0.1  # Very low friction to mimic slipping

# Joint Indices (as per your URDF)
HIP_JOINTS = [0, 2, 4, 6]
KNEE_JOINTS = [1, 3, 5, 7]
ALL_JOINTS = HIP_JOINTS + KNEE_JOINTS

# ------------------------ Gait Parameters Class ------------------------ #

class GaitParameters:
    def __init__(self, a=None, b=None, c=None, omega=None):
        self.a = a if a is not None else {joint: random.uniform(A_MIN, A_MAX) for joint in ALL_JOINTS}
        self.b = b if b is not None else {joint: random.uniform(B_MIN, B_MAX) for joint in ALL_JOINTS}
        self.c = c if c is not None else {joint: random.uniform(C_MIN, C_MAX) for joint in ALL_JOINTS}
        self.omega = omega if omega is not None else random.uniform(OMEGA_MIN, OMEGA_MAX)

    def mutate(self):
        for joint in ALL_JOINTS:
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

    def __str__(self):
        return json.dumps(self.to_dict(), indent=4)

# ------------------------ Initialization Functions ------------------------ #

def initialize_directories(run_name):
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

    plane_id = p.loadURDF("plane.urdf")
    p.changeDynamics(plane_id, -1, lateralFriction=FRICTION_COEFFICIENT)

    roll = math.radians(90)
    pitch = 0
    yaw = 0
    start_orientation = p.getQuaternionFromEuler([roll, pitch, yaw])

    start_pos = [0, 0, 0.5]
    robot_id = p.loadURDF("urdf-assembly.urdf", start_pos, start_orientation)

    for link in range(p.getNumJoints(robot_id)):
        p.changeDynamics(robot_id, link, lateralFriction=FRICTION_COEFFICIENT)

    for _ in range(SETTLING_STEPS):
        p.stepSimulation()

    p.resetBasePositionAndOrientation(robot_id, start_pos, start_orientation)

    initial_pos, _ = p.getBasePositionAndOrientation(robot_id)
    initial_x = initial_pos[0]

    for step in range(TOTAL_STEPS):
        t = step / SIMULATION_FPS

        for joint in ALL_JOINTS:
            theta = gait_params.a[joint] + gait_params.b[joint] * math.sin(gait_params.omega * t + gait_params.c[joint])
            theta_rad = math.radians(theta)
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=theta_rad,
                force=HIP_MAX_FORCE if joint in HIP_JOINTS else KNEE_MAX_FORCE,
                positionGain=POSITION_GAIN,
                velocityGain=VELOCITY_GAIN
            )

        p.stepSimulation()

    final_pos, _ = p.getBasePositionAndOrientation(robot_id)
    final_x = final_pos[0]

    p.disconnect()

    distance_traveled = final_x - initial_x
    return distance_traveled

# ------------------------ Hill Climber Optimization ------------------------ #

def hill_climber_optimization(run_name, directories):
    """
    Main hill climber loop. Initializes random gait parameters, evaluates them,
    then iteratively mutates and evaluates to find improvements.
    """
    current_params = GaitParameters()
    current_distance = evaluate_gait(current_params)

    best_params = current_params.copy()
    best_distance = current_distance

    for iteration in trange(1, MAX_ITERATIONS + 1, desc="Hill Climber Progress"):
        new_params = current_params.copy()
        new_params.mutate()

        new_distance = evaluate_gait(new_params)

        if new_distance > best_distance:
            best_params = new_params.copy()
            best_distance = new_distance
            current_params = new_params.copy()

    return best_params, best_distance

# ------------------------ Main Execution Flow ------------------------ #

def main():
    run_name = "low_friction_test"
    directories = initialize_directories(run_name)

    best_params, best_distance = hill_climber_optimization(run_name, directories)
    print(f"Best Distance Traveled: {best_distance:.3f} meters")

if __name__ == "__main__":
    main()
