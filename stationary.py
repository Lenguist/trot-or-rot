import gym
import pybullet_envs
import numpy as np
import time
import pprint

def explore_minitaur_environment():
    """
    Create the Minitaur environment, explore its initial state, and
    print detailed information about observations, actions, and rewards.
    """
    # Create the environment
    print("Creating Minitaur environment...")
    env = gym.make("MinitaurBulletEnv-v0", render=True)  # Set render=True to visualize
    
    # Print basic information about the environment
    print("\n=== ENVIRONMENT INFORMATION ===")
    print(f"Observation space: {env.observation_space}")
    print(f"Observation shape: {env.observation_space.shape}")
    print(f"Observation bounds: \n  Low: {env.observation_space.low}\n  High: {env.observation_space.high}")
    print(f"Action space: {env.action_space}")
    print(f"Action shape: {env.action_space.shape}")
    print(f"Action bounds: \n  Low: {env.action_space.low}\n  High: {env.action_space.high}")
    
    # Reset the environment to get initial observation
    print("\n=== RESETTING ENVIRONMENT ===")
    initial_obs = env.reset()
    
    # Print initial observation details
    print("\n=== INITIAL OBSERVATION ===")
    if hasattr(env, 'get_minitaur_motor_angles'):
        motor_angles = env.get_minitaur_motor_angles()
        motor_velocities = env.get_minitaur_motor_velocities()
        motor_torques = env.get_minitaur_motor_torques()
        base_orientation = env.get_minitaur_base_orientation()
        
        print(f"Motor angles (8 values): {motor_angles}")
        print(f"Motor velocities (8 values): {motor_velocities}")
        print(f"Motor torques (8 values): {motor_torques}")
        print(f"Base orientation (4 values): {base_orientation}")
    else:
        # If the method doesn't exist, print raw observation
        print("Raw observation (28 values):")
        # Split observation into meaningful chunks
        print(f"  Motor angles: {initial_obs[0:8]}")
        print(f"  Motor velocities: {initial_obs[8:16]}")
        print(f"  Motor torques: {initial_obs[16:24]}")
        print(f"  Base orientation: {initial_obs[24:28]}")
    
    # Try zero action and print results
    print("\n=== TAKING ZERO ACTION ===")
    zero_action = np.zeros(env.action_space.shape)
    print(f"Action: {zero_action}")
    
    # Take a step in the environment with zero action
    next_obs, reward, done, info = env.step(zero_action)
    
    # Print result details
    print("\n=== STEP RESULT ===")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info: {info}")
    
    # Print next observation details
    print("\n=== NEXT OBSERVATION ===")
    if hasattr(env, 'get_minitaur_motor_angles'):
        motor_angles = env.get_minitaur_motor_angles()
        motor_velocities = env.get_minitaur_motor_velocities()
        motor_torques = env.get_minitaur_motor_torques()
        base_orientation = env.get_minitaur_base_orientation()
        
        print(f"Motor angles (8 values): {motor_angles}")
        print(f"Motor velocities (8 values): {motor_velocities}")
        print(f"Motor torques (8 values): {motor_torques}")
        print(f"Base orientation (4 values): {base_orientation}")
    else:
        # If the method doesn't exist, print raw observation
        print("Raw observation (28 values):")
        # Split observation into meaningful chunks
        print(f"  Motor angles: {next_obs[0:8]}")
        print(f"  Motor velocities: {next_obs[8:16]}")
        print(f"  Motor torques: {next_obs[16:24]}")
        print(f"  Base orientation: {next_obs[24:28]}")
    
    # Print reward calculation components if available
    print("\n=== REWARD COMPONENTS ===")
    if hasattr(env, 'get_objectives'):
        objectives = env.get_objectives()
        if objectives and len(objectives) > 0:
            latest_obj = objectives[-1]
            print(f"Forward reward: {latest_obj[0]}")
            print(f"Energy reward: {latest_obj[1]}")
            print(f"Drift reward: {latest_obj[2]}")
            print(f"Shake reward: {latest_obj[3]}")
            
            if hasattr(env, 'objective_weights'):
                weights = env.objective_weights
                print(f"Weights: {weights}")
                print(f"Weighted rewards: {[o * w for o, w in zip(latest_obj, weights)]}")
                print(f"Total reward: {sum([o * w for o, w in zip(latest_obj, weights)])}")
    
    # Hold the environment state for a few seconds to observe
    print("\n=== HOLDING STATE FOR OBSERVATION ===")
    print("Keeping the environment in its current state for 5 seconds...")
    for _ in range(50):  # Hold for 5 seconds (assuming 0.1s per iteration)
        env.render()
        time.sleep(0.1)
    
    # Test different actions
    print("\n=== TESTING DIFFERENT ACTIONS ===")
    # Try a few different actions
    test_actions = [
        ("All zeros", np.zeros(env.action_space.shape)),
        ("All ones", np.ones(env.action_space.shape)),
        ("All negative ones", -np.ones(env.action_space.shape)),
        ("Alternating", np.array([1, -1, 1, -1, 1, -1, 1, -1])),
        ("Random within bounds", np.random.uniform(-1, 1, env.action_space.shape))
    ]
    
    for name, action in test_actions:
        print(f"\nTesting action: {name}")
        print(f"Action values: {action}")
        
        next_obs, reward, done, info = env.step(action)
        
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        
        # Hold for 1 second to observe effect
        for _ in range(10):
            env.render()
            time.sleep(0.1)
    
    # Clean up
    env.close()
    print("\n=== EXPLORATION COMPLETE ===")

if __name__ == "__main__":
    explore_minitaur_environment()