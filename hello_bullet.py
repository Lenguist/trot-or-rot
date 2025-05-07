import pybullet as p
import pybullet_data
import time
import math

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

# Drop the robot slightly above the ground
start_pos = [0,0,0.5]
robot_id = p.loadURDF("urdf-assembly.urdf", start_pos, start_orientation)

# Let the robot settle
for i in range(240*2):
    p.stepSimulation()
    time.sleep(1./240.)

# Adjust the camera: Zoom in, distance=1.5, top-down view or a slight angle
p.resetDebugVisualizerCamera(
    cameraDistance=1.5,
    cameraYaw=0,
    cameraPitch=-30,  # Slight angle from above
    cameraTargetPosition=[0,0,0]
)

# Identify joints from the URDF:
# Based on the provided URDF snippet:
# Joints (in order they appear):
# 0: base-combined-v1_Revolute-5       (hip joint back right)
# 1: right_back_leg_Revolute-6         (knee joint back right)
# 2: base-combined-v1_Revolute-7       (hip joint front right)
# 3: right_front_leg_Revolute-8        (knee joint front right)
# 4: base-combined-v1_Revolute-9       (hip joint front left)
# 5: left_front_leg_Revolute-10        (knee joint front left)
# 6: base-combined-v1_Revolute-11      (hip joint back left)
# 7: left_back_leg_Revolute-12         (knee joint back left)

# We'll assume "leg" joints are the "hip" joints (0,2,4,6) and "knee" joints are (1,3,5,7).
hip_joints = [0, 2, 4, 6]
knee_joints = [1, 3, 5, 7]

# A helper function to set joint angles
def set_joint_angle(robot_id, joint_index, angle_deg):
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

# We'll create a simple gait cycle:
# Move joints from -15 to +15 degrees and back.
# For simplicity, move all hip joints together and all knees together in a cycle.

def cycle_angles(joint_list, min_angle, max_angle, step=5):
    # From min to max
    for a in range(min_angle, max_angle+1, step):
        for j in joint_list:
            set_joint_angle(robot_id, j, a)
        # Step simulation
        for _ in range(240//4):  # short pause
            p.stepSimulation()
            time.sleep(1./240.)
    # From max back to min
    for a in range(max_angle, min_angle-1, -step):
        for j in joint_list:
            set_joint_angle(robot_id, j, a)
        # Step simulation
        for _ in range(240//4):
            p.stepSimulation()
            time.sleep(1./240.)

# Run a simple gait loop
# Alternate moving hips and knees:
for _ in range(3):  # 3 cycles for demonstration
    cycle_angles(hip_joints, -15, 15, step=5)
    cycle_angles(knee_joints, -15, 15, step=5)

p.disconnect()
