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

plane_id = p.loadURDF("plane.urdf")

# Rotate 90 degrees about X-axis (red axis)
roll = math.radians(90)
pitch = 0
yaw = 0
start_orientation = p.getQuaternionFromEuler([roll, pitch, yaw])
start_pos = [0,0,0.3]  # start a bit lower so it doesn't fall too far

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

num_joints = p.getNumJoints(robot_id)
for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    print(i, info[1].decode('utf-8'))

# Joints (assuming same indexing):
# 0: right_back_hip
# 1: right_back_knee
# 2: right_front_hip
# 3: right_front_knee
# 4: left_front_hip
# 5: left_front_knee
# 6: left_back_hip
# 7: left_back_knee

# Let's define naming for clarity:
right_back_hip = 0
right_back_knee = 1
right_front_hip = 2
right_front_knee = 3
left_front_hip = 4
left_front_knee = 5
left_back_hip = 6
left_back_knee = 7

# Group legs by pairs (for a trot gait):
# Diagonal pairs: (front-left & back-right) and (front-right & back-left)
diagonal_pair_1_hips = [left_front_hip, right_back_hip]
diagonal_pair_1_knees = [left_front_knee, right_back_knee]

diagonal_pair_2_hips = [right_front_hip, left_back_hip]
diagonal_pair_2_knees = [right_front_knee, left_back_knee]

# Control parameters
max_force = 2000  
position_gain = 0.1
velocity_gain = 0.1
time_counter = 0.0
time_step = 1.0/240.0

# Attempting a trot gait:
# We'll define a cycle where diagonal_pair_1 swings forward (hip angle positive) and diagonal_pair_2 swings backward (hip angle negative), then switch.
# Knees: slightly bent on stance (negative angle), extended on swing (less bent).
# Angles are guesses: 
#   Hip: +/- 15 degrees for stance/swing
#   Knee: vary from 0 (straight) to -20 (bent)
# We’ll do a cycle:
#   Half cycle:
#     - Pair 1 hips at +15 deg, knees at -20 deg (stance)
#     - Pair 2 hips at -15 deg, knees at 0 deg (swing)
#   Other half cycle:
#     - Pair 1 hips at -15 deg, knees at 0 deg (swing)
#     - Pair 2 hips at +15 deg, knees at -20 deg (stance)

def set_joint_angle(joint_index, angle_deg):
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

def apply_gait(pair1_hips, pair1_knees, pair2_hips, pair2_knees, 
               pair1_hip_angle, pair1_knee_angle, 
               pair2_hip_angle, pair2_knee_angle, 
               duration=1.0):
    # Set angles
    for h in pair1_hips:
        set_joint_angle(h, pair1_hip_angle)
    for k in pair1_knees:
        set_joint_angle(k, pair1_knee_angle)

    for h in pair2_hips:
        set_joint_angle(h, pair2_hip_angle)
    for k in pair2_knees:
        set_joint_angle(k, pair2_knee_angle)

    steps = int(duration/time_step)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        for _ in range(steps):
            p.stepSimulation()
            time.sleep(time_step)
            # Follow COM
            com = compute_robot_com(robot_id)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=0,
                cameraPitch=-30,
                cameraTargetPosition=[com[0], com[1], com[2]]
            )
            record_data(writer)

# Perform a trot cycle for a few seconds.
# We'll do multiple cycles: each half-cycle is 1 second.
# Half cycle 1: pair1 stance, pair2 swing
# Half cycle 2: pair1 swing, pair2 stance

for _ in range(5):  # 5 full cycles
    # Stance: hips +15 deg, knees -20 deg; Swing: hips -15 deg, knees 0 deg
    apply_gait(diagonal_pair_1_hips, diagonal_pair_1_knees, diagonal_pair_2_hips, diagonal_pair_2_knees,
               pair1_hip_angle=15, pair1_knee_angle=-20,
               pair2_hip_angle=-15, pair2_knee_angle=0,
               duration=1.0)
    apply_gait(diagonal_pair_1_hips, diagonal_pair_1_knees, diagonal_pair_2_hips, diagonal_pair_2_knees,
               pair1_hip_angle=-15, pair1_knee_angle=0,
               pair2_hip_angle=15, pair2_knee_angle=-20,
               duration=1.0)

p.disconnect()
