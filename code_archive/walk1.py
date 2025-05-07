import pybullet as p
import pybullet_data
import time
import math
import csv
import os

# Connect to PyBullet GUI
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)

# Load a plane
plane_id = p.loadURDF("plane.urdf")

# Rotate 90 degrees about X-axis (red axis)
roll = math.radians(90)
pitch = 0
yaw = 0
start_orientation = p.getQuaternionFromEuler([roll, pitch, yaw])

# Drop the robot slightly above ground
start_pos = [0,0,0.5]
robot_id = p.loadURDF("urdf-assembly.urdf", start_pos, start_orientation)

# Let it settle
for i in range(240*2):
    p.stepSimulation()
    time.sleep(1./240.)

# Setup camera POV
p.resetDebugVisualizerCamera(
    cameraDistance=1.5,
    cameraYaw=0,
    cameraPitch=-30,
    cameraTargetPosition=[0,0,0]
)

# Identify joints
num_joints = p.getNumJoints(robot_id)
for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    print(i, info[1].decode('utf-8'))

hip_joints = [0, 2, 4, 6]
knee_joints = [1, 3, 5, 7]

time_counter = 0.0
time_step = 1.0/240.0

# Higher torque value for more aggressive motion
max_force = 2000  
position_gain = 0.1
velocity_gain = 0.1

def set_joint_angle(robot_id, joint_index, angle_deg):
    angle_rad = math.radians(angle_deg)
    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=joint_index,
        controlMode=p.POSITION_CONTROL,
        targetPosition=angle_rad,
        force=max_force,
        positionGain=position_gain,
        velocityGain=velocity_gain
    )

def compute_robot_com(robot_id):
    mass_total = 0.0
    weighted_pos = [0.0, 0.0, 0.0]
    num_links = p.getNumJoints(robot_id)
    for link_id in range(-1, num_links):
        dyn = p.getDynamicsInfo(robot_id, link_id)
        link_mass = dyn[0]
        mass_total += link_mass
        if link_id == -1:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
        else:
            link_state = p.getLinkState(robot_id, link_id, computeForwardKinematics=True)
            pos = link_state[0]
        weighted_pos[0] += pos[0]*link_mass
        weighted_pos[1] += pos[1]*link_mass
        weighted_pos[2] += pos[2]*link_mass
    return [weighted_pos[0]/mass_total, weighted_pos[1]/mass_total, weighted_pos[2]/mass_total]

if not os.path.exists("logs"):
    os.makedirs("logs")

csv_file = "logs/simulation_data.csv"
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    joint_headers = [f"joint_{i}_angle_deg" for i in range(num_joints)]
    writer.writerow(["time_s", "COM_x", "COM_y", "COM_z"] + joint_headers)

def record_data(writer):
    global time_counter, time_step
    com = compute_robot_com(robot_id)
    joint_angles = []
    for ji in range(num_joints):
        state = p.getJointState(robot_id, ji)
        angle_deg = math.degrees(state[0])
        joint_angles.append(angle_deg)
    writer.writerow([time_counter, com[0], com[1], com[2]] + joint_angles)
    time_counter += time_step

def cycle_angles(joint_list, min_angle, max_angle, step=10):
    global time_counter
    if min_angle < max_angle:
        angle_range = range(min_angle, max_angle+1, step)
    else:
        angle_range = range(min_angle, max_angle-1, -step)
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        for a in angle_range:
            for j in joint_list:
                set_joint_angle(robot_id, j, a)
            
            # Simulate and record
            for _ in range(60):
                p.stepSimulation()
                time.sleep(1./240.)
                
                # Update camera POV to follow robot COM
                com = compute_robot_com(robot_id)
                p.resetDebugVisualizerCamera(
                    cameraDistance=1.5,
                    cameraYaw=0,
                    cameraPitch=-30,
                    cameraTargetPosition=[com[0], com[1], com[2]]
                )
                
                record_data(writer)

# More aggressive gait: -30 to +30 on hips and knees
for _ in range(2):
    cycle_angles(hip_joints, -30, 30, step=10)
    cycle_angles(knee_joints, -30, 30, step=10)
    cycle_angles(hip_joints, 30, -30, step=10)
    cycle_angles(knee_joints, 30, -30, step=10)

p.disconnect()
