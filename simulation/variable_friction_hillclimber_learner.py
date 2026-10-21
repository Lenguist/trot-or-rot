import os
import json
import pybullet as p
import pybullet_data
import time
import math
import random
import argparse
from datetime import datetime
from tqdm import trange
import sys

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
MAX_ITERATIONS = 10000
MUTATION_RATE = 0.9
MUTATION_SCALE = 0.5
LOG_INTERVAL = 1000

LEARNER_NAME = "HillClimberVarFriction"

# ------------------------ Directory Paths ------------------------ #
TRAINING_RUNS_DIR = "training_runs/hillclimber"
RESULTS_DIR = "results/hillclimber"

# ------------------------ Other Parameters ------------------------ #
SIMULATION_FPS = 240
RUN_DURATION = 10
TOTAL_STEPS = SIMULATION_FPS * RUN_DURATION
SETTLING_TIME = 1
SETTLING_STEPS = int(SIMULATION_FPS * SETTLING_TIME)

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

def initialize_directories(run_name):
    training_run_dir = os.path.join(TRAINING_RUNS_DIR, run_name)
    results_run_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(training_run_dir, exist_ok=True)
    os.makedirs(results_run_dir, exist_ok=True)
    return {
        "training_run": training_run_dir,
        "results_run": results_run_dir
    }

def evaluate_gait(gait_params, floor_friction):
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    plane_id = p.loadURDF("plane.urdf")
    p.changeDynamics(plane_id, -1, lateralFriction=floor_friction)

    roll = math.radians(90)
    pitch = 0
    yaw = 0
    start_orientation = p.getQuaternionFromEuler([roll, pitch, yaw])

    start_pos = [0, 0, 0.5]
    robot_id = p.loadURDF("urdf-assembly.urdf", start_pos, start_orientation)

    for link in range(p.getNumJoints(robot_id)):
        p.changeDynamics(robot_id, link, lateralFriction=floor_friction)

    for _ in range(SETTLING_STEPS):
        p.stepSimulation()

    p.resetBasePositionAndOrientation(robot_id, start_pos, start_orientation)

    initial_pos, _ = p.getBasePositionAndOrientation(robot_id)
    initial_x = initial_pos[0]

    for step in range(TOTAL_STEPS):
        t = step / SIMULATION_FPS
        for joint in JOINT_MAPPING.keys():
            theta = gait_params.a[joint] + gait_params.b[joint] * math.sin(gait_params.omega * t + gait_params.c[joint])
            theta_rad = math.radians(theta)
            if joint in [0, 2, 4, 6]:
                max_force = HIP_MAX_FORCE
            else:
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

    final_pos, _ = p.getBasePositionAndOrientation(robot_id)
    final_x = final_pos[0]
    p.disconnect()

    return final_x - initial_x

def save_hyperparameters(directories, timestamp, floor_friction):
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
            "floor_friction": floor_friction
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

    for dir_type in ["training_run", "results_run"]:
        hyperparams_file = os.path.join(directories[dir_type], f"hyperparameters_{timestamp}.json")
        with open(hyperparams_file, 'w') as f:
            json.dump(hyperparams, f, indent=4)

    return hyperparams

def main():
    parser = argparse.ArgumentParser(description="Hill Climber Optimization")
    parser.add_argument("--floor_friction", type=float, default=FLOOR_FRICTION_COEFFICIENT, 
                        help="Floor friction coefficient for the simulation")
    args = parser.parse_args()

    print("Please enter any notes for this run (press Ctrl+D or Ctrl+Z+Enter when done):")
    user_notes = sys.stdin.read().strip()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    
    print(f"\nStarting new training run: {run_name}")
    directories = initialize_directories(run_name)
    
    best_params, best_distance = hill_climber_optimization(run_name, directories, user_notes, args.floor_friction)
    
    print("\nOptimization Complete!")
    print(f"Best Distance Traveled: {best_distance:.3f} meters")
    speed = best_distance / RUN_DURATION
    print(f"Speed: {speed*100:.2f} cm/s")
    print("Check log_card.txt for full details.")
    print(f"Results saved in: {directories['results_run']}")

def hill_climber_optimization(run_name, directories, user_notes, floor_friction):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hyperparams = save_hyperparameters(directories, timestamp, floor_friction)

    current_params = GaitParameters()
    current_distance = evaluate_gait(current_params, floor_friction)

    best_params = current_params.copy()
    best_distance = current_distance

    values_path = os.path.join(directories["results_run"], "values.json")
    with open(values_path, 'w') as f:
        json.dump({
            "gait_parameters_deg": best_params.to_dict(),
            "best_distance_traveled_m": best_distance,
            "timestamp": timestamp
        }, f, indent=4)

    for iteration in trange(1, MAX_ITERATIONS + 1, desc="Hill Climber Progress"):
        new_params = current_params.copy()
        new_params.mutate()
        new_distance = evaluate_gait(new_params, floor_friction)

        if new_distance > best_distance:
            best_params = new_params.copy()
            best_distance = new_distance
            current_params = new_params.copy()

            with open(values_path, 'w') as f:
                json.dump({
                    "gait_parameters_deg": best_params.to_dict(),
                    "best_distance_traveled_m": best_distance,
                    "timestamp": timestamp
                }, f, indent=4)

        if iteration % LOG_INTERVAL == 0:
            print(f"Iteration {iteration}/{MAX_ITERATIONS}")
            print(f"Current Distance: {new_distance:.3f} m")
            print(f"Best Distance: {best_distance:.3f} m")

    return best_params, best_distance

if __name__ == "__main__":
    main()
