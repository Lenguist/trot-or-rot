import os
import json
import pybullet as p
import pybullet_data
import time
import math
import random
from shutil import rmtree

# Configuration Parameters
NUM_RUNS = 100          # Number of simulation runs
RUN_DURATION = 10       # Duration of each run in seconds
SIMULATION_FPS = 240   # Simulation steps per second
LOGS_DIR = "logs"      # Directory to store logs

# Define joint limits (assuming ±30 degrees for hips and ±45 degrees for knees)
HIP_MIN, HIP_MAX = -30, 30
KNEE_MIN, KNEE_MAX = -45, 45

# Define joints
hip_joints = [0, 2, 4, 6]
knee_joints = [1, 3, 5, 7]

# Ensure the logs directory exists
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)
else:
    # Optional: Clear existing logs
    confirmation = input(f"The '{LOGS_DIR}' directory already exists. Do you want to delete existing logs? (y/n): ")
    if confirmation.lower() == 'y':
        rmtree(LOGS_DIR)
        os.makedirs(LOGS_DIR)
    else:
        print("Existing logs will be preserved.")

# Helper function to set joint angles
def set_joint_angle(robot_id, joint_index, angle_deg):
    angle_rad = math.radians(angle_deg)
    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=joint_index,
        controlMode=p.POSITION_CONTROL,
        targetPosition=angle_rad,
        force=100,
        positionGain=0.1,
        velocityGain=0.1
    )

# Function to apply random angles within specified limits
def apply_random_angles(robot_id):
    for joint in hip_joints:
        angle = random.uniform(HIP_MIN, HIP_MAX)
        set_joint_angle(robot_id, joint, angle)
    for joint in knee_joints:
        angle = random.uniform(KNEE_MIN, KNEE_MAX)
        set_joint_angle(robot_id, joint, angle)

# Function to measure forward progress (distance along X-axis)
def get_forward_distance(robot_id):
    pos, _ = p.getBasePositionAndOrientation(robot_id)
    return pos[0]  # Assuming X-axis is forward

# List to store all run data
all_runs = []

# Main Simulation Loop
for run in range(1, NUM_RUNS + 1):
    print(f"Starting Run {run}/{NUM_RUNS}...")
    
    # Connect to PyBullet in DIRECT mode (no GUI)
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Load a plane
    plane_id = p.loadURDF("plane.urdf")
    
    # Rotate 90 degrees about X-axis (red axis)
    roll = math.radians(90)
    pitch = 0
    yaw = 0
    start_orientation = p.getQuaternionFromEuler([roll, pitch, yaw])
    
    # Drop the robot slightly above the ground
    start_pos = [0, 0, 0.2]
    robot_id = p.loadURDF("urdf-assembly.urdf", start_pos, start_orientation)
    
    # Let the robot settle
    for _ in range(int(0.5 * SIMULATION_FPS)):
        p.stepSimulation()
        # No sleep needed in DIRECT mode
    
    # Initialize logging data
    run_data = {
        "run_number": run,
        "distance_traveled": 0.0,
        "final_joint_angles_deg": {}
    }
    
    # Run the simulation for RUN_DURATION seconds
    num_steps = RUN_DURATION * SIMULATION_FPS
    for step in range(num_steps):
        apply_random_angles(robot_id)
        p.stepSimulation()
        # No sleep needed in DIRECT mode
    
    # Record the distance traveled
    distance = get_forward_distance(robot_id)
    run_data["distance_traveled"] = distance
    
    # Record the final joint angles
    for joint in hip_joints + knee_joints:
        joint_state = p.getJointState(robot_id, joint)
        angle_deg = math.degrees(joint_state[0])
        run_data["final_joint_angles_deg"][f"joint_{joint}"] = angle_deg
    
    # Save run data to logs/run_X/log.json
    run_folder = os.path.join(LOGS_DIR, f"run_{run}")
    os.makedirs(run_folder, exist_ok=True)
    log_file = os.path.join(run_folder, "log.json")
    with open(log_file, 'w') as f:
        json.dump(run_data, f, indent=4)
    
    print(f"Run {run} completed. Distance Traveled: {distance:.3f} meters.")
    
    # Store run data
    all_runs.append(run_data)
    
    # Disconnect PyBullet
    p.disconnect()

# Identify the best run (maximum distance traveled)
best_run = max(all_runs, key=lambda x: x["distance_traveled"])
best_run_number = best_run["run_number"]
best_distance = best_run["distance_traveled"]
best_joint_angles = best_run["final_joint_angles_deg"]

print(f"\nBest Run: Run {best_run_number} with Distance Traveled: {best_distance:.3f} meters.")

# Visualization of the Best Run
print("\nStarting visualization of the best run...")

# Connect to PyBullet in GUI mode
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load a plane
plane_id = p.loadURDF("plane.urdf")

# Rotate 90 degrees about X-axis (red axis)
roll = math.radians(90)
pitch = 0
yaw = 0
start_orientation = p.getQuaternionFromEuler([roll, pitch, yaw])

# Drop the robot slightly above the ground
start_pos = [0, 0, 0.5]
robot_id = p.loadURDF("urdf-assembly.urdf", start_pos, start_orientation)

# Let the robot settle
for _ in range(int(2 * SIMULATION_FPS)):
    p.stepSimulation()
    time.sleep(1. / SIMULATION_FPS)

# Apply the best joint angles
for joint in hip_joints + knee_joints:
    angle_deg = best_joint_angles.get(f"joint_{joint}", 0)
    set_joint_angle(robot_id, joint, angle_deg)

# Let the robot adjust to the best configuration
for _ in range(int(1 * SIMULATION_FPS)):
    p.stepSimulation()
    time.sleep(1. / SIMULATION_FPS)

# Adjust the camera: Zoom in, distance=1.5, slight angle
p.resetDebugVisualizerCamera(
    cameraDistance=1.5,
    cameraYaw=0,
    cameraPitch=-30,  # Slight angle from above
    cameraTargetPosition=[0, 0, 0]
)

print("Visualization of the best run is now active. Close the window or interrupt the script to exit.")

try:
    while True:
        p.stepSimulation()
        time.sleep(1. / SIMULATION_FPS)
except KeyboardInterrupt:
    print("Visualization interrupted by user.")
finally:
    p.disconnect()
