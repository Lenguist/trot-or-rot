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
from robot_config import (
    JOINT_MAPPING,
    HIP_MAX_FORCE,
    KNEE_MAX_FORCE,
    POSITION_GAIN,
    VELOCITY_GAIN,
    FLOOR_FRICTION_COEFFICIENT
)

SIMULATION_FPS = 240
RUN_DURATION = 20
SETTLING_TIME = 0.5
ROBOT_URDF = "urdf-assembly.urdf"
START_POS = [0, 0, 0.5]
START_ORIENTATION = [math.radians(90), 0, 0]

HIP_JOINTS = [0, 2, 4, 6]
KNEE_JOINTS = [1, 3, 5, 7]
ALL_JOINTS = HIP_JOINTS + KNEE_JOINTS

TOTAL_STEPS = SIMULATION_FPS * RUN_DURATION
SETTLING_STEPS = int(SIMULATION_FPS * SETTLING_TIME)

def setup_pybullet(gui=False):
    if gui:
        p.connect(p.GUI)
        # Set an initial closer camera view
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5, 
            cameraYaw=0, 
            cameraPitch=-30, 
            cameraTargetPosition=[0, 0, 0]
        )
    else:
        p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

def reset_simulation(robot_urdf, start_pos, start_orientation):
    plane_id = p.loadURDF("plane.urdf")
    p.changeDynamics(plane_id, -1, lateralFriction=FLOOR_FRICTION_COEFFICIENT)
    
    quat = p.getQuaternionFromEuler(start_orientation)
    robot_id = p.loadURDF(robot_urdf, start_pos, quat)
    
    for link in range(p.getNumJoints(robot_id)):
        p.changeDynamics(robot_id, link, lateralFriction=FLOOR_FRICTION_COEFFICIENT)
    
    # Simulate settling
    for _ in range(SETTLING_STEPS):
        p.stepSimulation()

    # Reset base position and orientation to match training code
    p.resetBasePositionAndOrientation(robot_id, start_pos, quat)

    return robot_id

def apply_gait(robot_id, joint_functions, total_steps, gui_mode=False):
    data_records = []
    previous_x = None
    speed = 0.0

    # Safe environment for eval
    safe_globals = {"math": math}

    for step in range(total_steps):
        t = step / SIMULATION_FPS

        for joint in ALL_JOINTS:
            j_str = str(joint)
            expr = joint_functions[j_str]['expression']
            params = joint_functions[j_str]['params']

            local_vars = {"t": t}
            local_vars.update(params)

            theta = eval(expr, safe_globals, local_vars)
            theta_rad = math.radians(theta)
            
            if joint in HIP_JOINTS:
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
        
        # Record data every 10 steps
        if step % 10 == 0:
            base_pos, base_orient = p.getBasePositionAndOrientation(robot_id)
            base_euler = p.getEulerFromQuaternion(base_orient)
            
            if previous_x is not None:
                dx = base_pos[0] - previous_x
                speed = dx / (10 / SIMULATION_FPS)
            previous_x = base_pos[0]
            
            record = {
                'time': t,
                'x': base_pos[0],
                'y': base_pos[1],
                'z': base_pos[2],
                'pitch': math.degrees(base_euler[0]),
                'yaw': math.degrees(base_euler[1]),
                'roll': math.degrees(base_euler[2]),
                'speed_m_s': speed
            }
            data_records.append(record)

            # If GUI is enabled, follow the robot by updating camera target
            if gui_mode:
                p.resetDebugVisualizerCamera(
                    cameraDistance=1.5,
                    cameraYaw=0,
                    cameraPitch=-30,
                    cameraTargetPosition=[base_pos[0], base_pos[1], base_pos[2]]
                )
    return data_records

def save_data_records_csv(data_records, filename):
    if not data_records:
        return
    keys = data_records[0].keys()
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data_records)

def plot_robot_motion(data_records, filename):
    if not data_records:
        return
    times = [r['time'] for r in data_records]
    x = [r['x'] for r in data_records]
    y = [r['y'] for r in data_records]
    z = [r['z'] for r in data_records]
    pitch = [r['pitch'] for r in data_records]
    yaw = [r['yaw'] for r in data_records]
    roll = [r['roll'] for r in data_records]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,10))

    # Positions
    ax1.plot(times, x, label='X (m)')
    ax1.plot(times, y, label='Y (m)')
    ax1.plot(times, z, label='Z (m)')
    ax1.set_title('Position Over Time')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.grid(True)
    ax1.legend()

    # Orientations
    ax2.plot(times, pitch, label='Pitch (deg)')
    ax2.plot(times, yaw, label='Yaw (deg)')
    ax2.plot(times, roll, label='Roll (deg)')
    ax2.set_title('Orientation Over Time')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angle (deg)')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize Best Parameters From a Run")
    parser.add_argument('--run', type=str, required=True, help="Name of the training run (e.g., run_20240101_123000)")
    parser.add_argument('--gui', action='store_true', help="Enable GUI mode in PyBullet")
    args = parser.parse_args()

    run_name = args.run
    gui_mode = args.gui

    results_dir = os.path.join("results/hillclimber", run_name)
    analysis_dir = os.path.join(results_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    values_file = os.path.join(results_dir, "values.json")
    if not os.path.exists(values_file):
        print(f"No values.json found in {results_dir}")
        return

    with open(values_file, 'r') as f:
        values = json.load(f)

    if "joint_functions" not in values:
        print("No 'joint_functions' found in values.json. Cannot visualize.")
        return

    joint_functions = values["joint_functions"]

    setup_pybullet(gui=gui_mode)
    robot_id = reset_simulation(ROBOT_URDF, START_POS, START_ORIENTATION)
    data_records = apply_gait(robot_id, joint_functions, TOTAL_STEPS, gui_mode=gui_mode)
    p.disconnect()

    # Save data records
    simulation_data_csv = os.path.join(analysis_dir, 'simulation_data.csv')
    save_data_records_csv(data_records, simulation_data_csv)

    # Plot motion
    plot_robot_motion(data_records, os.path.join(analysis_dir, 'robot_motion.png'))

    print(f"Visualization complete! Results saved in: {analysis_dir}")

if __name__ == "__main__":
    main()
