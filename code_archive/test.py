import pybullet as p
import pybullet_data
import time
import math
import os
import csv

# Connect to PyBullet in GUI mode
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Disable gravity
p.setGravity(0, 0, 0)

# Load plane
plane_id = p.loadURDF("plane.urdf")

# Rotate the robot 90 degrees about X-axis
roll = math.radians(90)
pitch = 0
yaw = 0
start_orientation = p.getQuaternionFromEuler([roll, pitch, yaw])

# Start position slightly above the ground
start_pos = [0, 0, 0.5]

# Load the robot URDF
robot_id = p.loadURDF("urdf-assembly.urdf", start_pos, start_orientation)

# Allow the robot to settle with fewer simulation steps
settle_steps = 120  # Reduced from 240
for _ in range(settle_steps):
    p.stepSimulation()

# Get number of joints
num_joints = p.getNumJoints(robot_id)

# Get default joint angles and print them
default_angles = []
print("Default Joint Angles (in degrees):")
for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    joint_name = joint_info[1].decode('utf-8')
    state = p.getJointState(robot_id, i)
    angle_rad = state[0]
    angle_deg = math.degrees(angle_rad)
    default_angles.append(angle_deg)
    print(f"Joint {i} ({joint_name}): {angle_deg:.2f}°")

# Helper function to set joint angle in degrees
def set_joint_angle(joint_index, angle_deg):
    angle_rad = math.radians(angle_deg)
    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=joint_index,
        controlMode=p.POSITION_CONTROL,
        targetPosition=angle_rad,
        force=1000,
        positionGain=0.1,
        velocityGain=0.1
    )

# Adjust the camera to a more intuitive POV (similar to helloworld example)
p.resetDebugVisualizerCamera(
    cameraDistance=1.5,       # Distance from the target
    cameraYaw=50,             # Yaw angle
    cameraPitch=-35,          # Pitch angle
    cameraTargetPosition=[0, 0, 0]  # Look at the origin
)

# Define movement parameters
forward_angle = 15    # Degrees
back_angle = -15      # Degrees
step_size = 5         # Degrees per step
steps_per_movement = 24  # Number of simulation steps per movement phase
# Total movement from -15 to +15 degrees in steps of 5 degrees: 6 steps (from -15, -10, -5, 0, 5, 10, 15)

# Create a directory for logs if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

csv_file = "logs/joint_movement_log.csv"

# Open the CSV file and write headers
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    header = [f"Joint_{i}_Angle_Deg" for i in range(num_joints)]
    writer.writerow(["Phase", "Joint_Index", "Target_Angle_Deg"] + header)

# Function to log joint angles
def log_joint_angles(writer, phase, joint_index, target_angle):
    joint_angles = []
    for ji in range(num_joints):
        state = p.getJointState(robot_id, ji)
        angle_deg = math.degrees(state[0])
        joint_angles.append(round(angle_deg, 2))
    writer.writerow([phase, joint_index, target_angle] + joint_angles)

# Open the CSV file for appending
with open(csv_file, 'a', newline='') as f:
    writer = csv.writer(f)

    # Iterate through each joint
    for j in range(num_joints):
        print(f"\nMoving Joint {j} ({p.getJointInfo(robot_id, j)[1].decode('utf-8')})")

        # Move joint forward in steps
        for angle in range(0, forward_angle + step_size, step_size):
            set_joint_angle(j, angle)
            for _ in range(steps_per_movement):
                p.stepSimulation()
                time.sleep(1./120.)  # Half the original speed (original was 240 Hz)
            log_joint_angles(writer, f"Joint {j} Forward", j, angle)
            print(f"  Set to {angle}°")

        # Move joint back in steps
        for angle in range(forward_angle, back_angle - step_size, -step_size):
            set_joint_angle(j, angle)
            for _ in range(steps_per_movement):
                p.stepSimulation()
                time.sleep(1./120.)
            log_joint_angles(writer, f"Joint {j} Backward", j, angle)
            print(f"  Set to {angle}°")

        # Return joint to default angle in steps
        default = default_angles[j]
        step_direction = step_size if default > angle else -step_size
        steps_to_default = int(abs(default - angle) / step_size)

        for step in range(steps_to_default):
            intermediate_angle = angle + step_direction * (step + 1)
            set_joint_angle(j, intermediate_angle)
            for _ in range(steps_per_movement):
                p.stepSimulation()
                time.sleep(1./120.)
            log_joint_angles(writer, f"Joint {j} Return to Default", j, intermediate_angle)
            print(f"  Returning to {intermediate_angle}°")

        # Ensure the joint is set exactly to the default angle
        set_joint_angle(j, default)
        for _ in range(steps_per_movement):
            p.stepSimulation()
            time.sleep(1./120.)
        log_joint_angles(writer, f"Joint {j} Final", j, default)
        print(f"  Set to default {default}°")

p.disconnect()
print(f"\nJoint movement log saved to {csv_file}")
