# train_minitaur_ppo.py

import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from minitaur_env import MinitaurGymEnv

class ProgressCallback(BaseCallback):
    """
    Custom callback for printing training progress and evaluating policy every `print_freq` timesteps.
    """
    def __init__(self, print_freq=50000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        # create a separate env for evaluation
        self.eval_env = MinitaurGymEnv(render=False)

    def _on_step(self) -> bool:
        # Called at each step: self.num_timesteps is the count of total steps so far
        if self.num_timesteps % self.print_freq == 0:
            # evaluate current policy on eval_env for 1 episode
            mean_reward, std_reward = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=1, render=False, warn=False)
            print(f"=== Step {self.num_timesteps}: eval reward {mean_reward:.2f} +/- {std_reward:.2f} ===")
        return True

def make_env(render=False):
    def _init():
        return MinitaurGymEnv(
            render=render,
            energy_weight=0.004,  # Reduced energy penalty
            distance_weight=2.5,  # Increased reward for forward movement
            drift_weight=0.0,
            shake_weight=0.0,
        )
    return _init

# Function to generate sinusoidal actions based on hillclimber parameters
def get_sinusoidal_action(theta, t, control_dt):
    amp = theta[:8]
    phase = theta[8:16]
    freq = abs(theta[16])  # ensure positive frequency
    # time in seconds
    time_s = t * control_dt
    # sinusoidal action, clipped to [-1, 1]
    actions = amp * np.sin(2 * np.pi * freq * time_s + phase)
    return np.clip(actions, -1.0, 1.0)

# Custom initial weights callback to influence network initialization
class HillClimberWeightInit:
    def __init__(self, theta_path, env):
        self.theta = np.load(theta_path)
        # Create an environment to get control_dt
        self.env = env
        # Keep track of time steps for action generation
        self.t = 0
        
    def init_weights(self, model):
        print("Initializing PPO policy using hill-climber solution...")
        # Calculate initial actions for a few time steps
        actions = []
        for t in range(100):
            action = get_sinusoidal_action(self.theta, t, self.env.control_time_step)
            actions.append(action)
        
        # Create sample observations and actions
        sample_obs = []
        for _ in range(100):
            # Create random observations from the observation space
            obs = self.env.observation_space.sample()
            sample_obs.append(obs)
        
        sample_obs = np.array(sample_obs)
        actions = np.array(actions)
        
        # Set initial actions as the target for the policy network
        # This will make the policy output values similar to the sinusoidal pattern
        for _ in range(1000):  # Pre-train the policy network
            model.policy.optimizer.zero_grad()
            actions_pred, _, _ = model.policy.forward(torch.FloatTensor(sample_obs))
            loss = ((actions_pred - torch.FloatTensor(actions))**2).mean()
            loss.backward()
            model.policy.optimizer.step()
            
        print(f"Policy pre-training complete. Final loss: {loss.item():.4f}")
        return model

if __name__ == "__main__":
    # Path to the saved hill-climber parameters
    theta_path = 'best_theta.npy'
    
    # Make sure the file exists
    if not os.path.exists(theta_path):
        raise FileNotFoundError(f"Could not find hill-climber solution at {theta_path}")
    
    print(f"Found hill-climber solution at {theta_path}")
    
    # Create a single environment for initialization
    env = MinitaurGymEnv(render=False, 
                        energy_weight=0.004,
                        distance_weight=2.5,
                        drift_weight=0.0,
                        shake_weight=0.0)
    
    # Create weight initializer
    weight_init = HillClimberWeightInit(theta_path, env)
    
    n_envs = 25
    print(f"Building {n_envs} parallel Minitaur environments...")
    env_fns = [make_env(render=False) for _ in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)
    
    # Add observation normalization
    print("Adding observation normalization...")
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99
    )

    policy_kwargs = dict(net_arch=dict(pi=[185, 95], vf=[95, 85]))
    print("Initializing PPO model with policy_kwargs=", policy_kwargs)
    
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=1000,
        batch_size=256,
        n_epochs=10,
        clip_range=0.2,
        learning_rate=5e-4,  # Slightly lower learning rate since we're starting from a good solution
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.05,  # Higher entropy to explore variations of the hill-climber solution
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )
    
    # Initialize the policy weights based on hill-climber solution
    model = weight_init.init_weights(model)
    
    # Evaluate the initialized policy
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=3)
    print(f"Initial policy performance: {mean_reward:.2f} +/- {std_reward:.2f}")

    progress_freq = 50000
    checkpoint_freq = 500000
    progress_callback = ProgressCallback(print_freq=progress_freq)
    ckpt_callback = CheckpointCallback(save_freq=checkpoint_freq, save_path="./checkpoints/", verbose=1)
    print(f"Setting up callbacks: progress every {progress_freq} steps (with eval), checkpoints every {checkpoint_freq} steps...")

    total_timesteps = int(3e6)  # Reduced time since we're starting from a good solution
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=[progress_callback, ckpt_callback])
    print("Training complete.")

    # Save final model and VecNormalize stats
    os.makedirs("models", exist_ok=True)
    print("Saving final model and normalization stats...")
    model.save("models/minitaur_hillclimber_ppo")
    vec_env.save("models/vec_normalize.pkl")
    print("Model and normalization stats saved. All checkpoints have been saved to ./checkpoints/")
    
    # Close environments
    vec_env.close()
    env.close()