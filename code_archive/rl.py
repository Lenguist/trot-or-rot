import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

class QuadrupedEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render=False):
        super(QuadrupedEnv, self).__init__()
        self.render_mode = render
        if self.render_mode:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Load plane and robot
        self.plane = p.loadURDF("plane.urdf")
        start_pos = [0, 0, 0.5]
        start_orientation = p.getQuaternionFromEuler([math.radians(90), 0, 0])
        self.robot = p.loadURDF("urdf-assembly.urdf", start_pos, start_orientation)

        # Define action and observation space
        # Example: Actions are target joint angles for each joint
        num_joints = p.getNumJoints(self.robot)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_joints,), dtype=np.float32)

        # Observation could include joint angles, velocities, robot's base position, etc.
        obs_low = np.array([-np.pi]*num_joints + [-np.inf]*6)  # Example
        obs_high = np.array([np.pi]*num_joints + [np.inf]*6)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Simulation parameters
        self.time_step = 1./240.
        p.setTimeStep(self.time_step)

    def reset(self):
        # Reset robot position and orientation
        start_pos = [0, 0, 0.5]
        start_orientation = p.getQuaternionFromEuler([math.radians(90), 0, 0])
        p.resetBasePositionAndOrientation(self.robot, start_pos, start_orientation)

        # Reset joint states
        num_joints = p.getNumJoints(self.robot)
        for joint in range(num_joints):
            p.resetJointState(self.robot, joint, targetValue=0.0, targetVelocity=0.0)

        # Get initial observation
        obs = self._get_observation()
        return obs

    def step(self, action):
        # Apply actions
        num_joints = p.getNumJoints(self.robot)
        for joint in range(num_joints):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot,
                jointIndex=joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=action[joint],
                force=200  # Adjust force as needed
            )

        # Step simulation
        p.stepSimulation()
        if self.render_mode:
            time.sleep(self.time_step)

        # Get observation
        obs = self._get_observation()

        # Calculate reward
        distance = self._get_distance()
        reward = distance  # Simple reward: distance traveled

        # Check if episode is done
        done = False
        if distance < -10 or distance > 100:  # Example conditions
            done = True

        info = {}

        return obs, reward, done, info

    def _get_observation(self):
        joint_states = [p.getJointState(self.robot, i)[0] for i in range(p.getNumJoints(self.robot))]
        base_pos, base_orient = p.getBasePositionAndOrientation(self.robot)
        base_vel, base_ang_vel = p.getBaseVelocity(self.robot)
        obs = joint_states + list(base_pos) + list(base_orient)
        return np.array(obs, dtype=np.float32)

    def _get_distance(self):
        base_pos, _ = p.getBasePositionAndOrientation(self.robot)
        return base_pos[0]  # Distance along X-axis

    def render(self, mode='human'):
        pass  # Rendering is handled in the simulation loop

    def close(self):
        p.disconnect()
