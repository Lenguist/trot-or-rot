import time
import math
import numpy as np
import pybullet as p
import pybullet_data

def main():
    # Connect to PyBullet (GUI mode to visualize).
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Load plane and a quadruped-like URDF (replace with your robot URDF).
    plane_id = p.loadURDF("plane.urdf")
    startPos = [0, 0, 0.2]
    startOrientation = p.getQuaternionFromEuler([0,0,0])
    robot_id = p.loadURDF("urdf-assembly.urdf", startPos, startOrientation)
    
    # Map servo IDs {1..8} to PyBullet joint indices {0..7} 
    # (Adjust these if your URDF joint ordering is different.)
    servo_to_joint = {
        1: 0,  # servo1 -> joint index 0
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5,
        7: 6,
        8: 7
    }
    
    # Hardcoded target positions (degrees) and amplitudes for each "servo"
    target6 = {"position": 115, "amplitude": 20} 
    target2 = {"position": 219, "amplitude": 20}  
    target4 = {"position": 185, "amplitude": 20}  
    target8 = {"position": 125, "amplitude": 20}  

    target5 = {"position": 155, "amplitude": 40}
    target1 = {"position": 165, "amplitude": 40}
    target3 = {"position": 100, "amplitude": 40}
    target7 = {"position": 40,  "amplitude": 40}

    # Oscillation parameters
    frequency = 4 / (2 * np.pi)  # Frequency in Hz
    period = 1 / frequency
    number_of_periods = 30
    steps = 100                  # Number of steps per period
    step_duration = period / steps  # Time per step

    # Quick helper function: move a servo in PyBullet
    def move_servo(servo_id, angle_deg):
        joint_index = servo_to_joint[servo_id]
        angle_rad = math.radians(angle_deg)
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=angle_rad,
            force=500,
            positionGain=0.1,
            velocityGain=0.1
        )

    # Slowly move the joints to the starting positions using the same phase offsets:
    move_servo(2,  target2["amplitude"]*math.sin(1.6)+target2["position"])
    move_servo(4, -target4["amplitude"]*math.sin(3.14)+target4["position"])
    move_servo(6,  target6["amplitude"]*math.sin(3.14)+target6["position"])
    move_servo(8, -target8["amplitude"]*math.sin(4.8)+target8["position"])
    p.stepSimulation()
    time.sleep(1)

    move_servo(1,  target1["amplitude"]*math.sin(1.6)+target1["position"])
    move_servo(3, -target3["amplitude"]*math.sin(3.14)+target3["position"])
    move_servo(5,  target5["amplitude"]*math.sin(3.14)+target5["position"])
    move_servo(7, -target7["amplitude"]*math.sin(4.8)+target7["position"])
    p.stepSimulation()
    time.sleep(1)

    # Oscillation loop
    for step_i in range(number_of_periods * steps):
        t = step_i / steps  # fraction of a period

        # Calculate target angles
        target_angle2 =  target2["amplitude"]*math.sin(frequency*2*math.pi*t + 1.6) + target2["position"]
        target_angle4 = -target4["amplitude"]*math.sin(frequency*2*math.pi*t + 3.14)+ target4["position"]
        target_angle6 =  target6["amplitude"]*math.sin(frequency*2*math.pi*t + 3.14)+ target6["position"]
        target_angle8 = -target8["amplitude"]*math.sin(frequency*2*math.pi*t + 4.8) + target8["position"]
        
        target_angle1 =  target1["amplitude"]*math.sin(frequency*2*math.pi*t + 1.6) + target1["position"]
        target_angle3 = -target3["amplitude"]*math.sin(frequency*2*math.pi*t + 3.14)+ target3["position"]
        target_angle5 =  target5["amplitude"]*math.sin(frequency*2*math.pi*t + 3.14)+ target5["position"]
        target_angle7 = -target7["amplitude"]*math.sin(frequency*2*math.pi*t + 4.8) + target7["position"]

        # Move the joints in PyBullet
        move_servo(1, target_angle1)
        move_servo(3, target_angle3)
        move_servo(5, target_angle5)
        move_servo(7, target_angle7)

        move_servo(2, target_angle2)
        move_servo(4, target_angle4)
        move_servo(6, target_angle6)
        move_servo(8, target_angle8)

        p.stepSimulation()
        time.sleep(step_duration)

    p.disconnect()

if __name__ == "__main__":
    main()
