# train_minitaur_ppo_simple.py

import os
import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from minitaur_env import MinitaurGymEnv
from gym import spaces

class HillClimberActionWrapper(gym.ActionWrapper):
    """
    Wrapper that adds the hill-climber actions as a bias to the PPO actions.
    This allows PPO to learn residual actions on top of the hill-climber solution.
    """
    def __init__(self, env, theta_path, action_scale=0.2):
        super().__init__(env)
        self.theta = np.load(theta_path)
        self.timestep = 0
        self.action_scale = action_scale
        
        # Reduce the action space to be centered around 0 with smaller range
        # This makes PPO learn adjustments to the hill-climber solution
        self.action_space = spaces.Box(
            low=-self.action_scale, 
            high=self.action_scale, 
            shape=self.env.action_space.shape,
            dtype=np.float32
        )
        
    def reset(self, **kwargs):
        self.timestep = 0
        return super().reset(**kwargs)
        
    def action(self, action):
        # Get hill-climber base action
        hc_action = self._get_hill_climber_action()
        
        # Add the PPO adjustment (scaled to small values)
        combined_action = hc_action + action
        
        # Clip to original action space
        combined_action = np.clip(
            combined_action, 
            self.env.action_space.low, 
            self.env.action_space.high
        )
        
        self.timestep += 1
        return combined_action
    
    def _get_hill_climber_action(self):
        # Extract parameters
        amp = self.theta[:8]
        phase = self.theta[8:16]
        freq = abs(self.theta[16])  # ensure positive frequency
        
        # Calculate time in seconds
        time_s = self.timestep * self.env.control_time_step
        
        # Generate sinusoidal action
        actions = amp * np.sin(2 * np.pi * freq * time_s + phase)
        return np.clip(actions, -1.0, 1.0)

class ProgressCallback(BaseCallback):
    """
    Custom callback for printing training progress and evaluating policy
    """
    def __init__(self, print_freq=50000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        # Create a separate env for evaluation
        self.eval_env = MinitaurGymEnv(render=False)
        
        # Load hill-climber parameters and wrap the evaluation environment
        theta_path = 'best_theta.npy'
        self.eval_env = HillClimberActionWrapper(self.eval_env, theta_path)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            mean_reward, std_reward = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=1, render=False, warn=False)
            print(f"=== Step {self.num_timesteps}: eval reward {mean_reward:.2f} +/- {std_reward:.2f} ===")
        return True

def make_env(render=False, theta_path='best_theta.npy'):
    def _init():
        env = MinitaurGymEnv(
            render=render,
            energy_weight=0.008,  # Keep original energy weight
            distance_weight=1.0,  # Keep original distance weight
            drift_weight=0.0,
            shake_weight=0.0,
        )
        # Wrap with hill-climber actions
        env = HillClimberActionWrapper(env, theta_path)
        return env
    return _init

if __name__ == "__main__":
    # Path to the saved hill-climber parameters
    theta_path = 'best_theta.npy'
    
    # Make sure the file exists
    if not os.path.exists(theta_path):
        raise FileNotFoundError(f"Could not find hill-climber solution at {theta_path}")
    
    print(f"Found hill-climber solution at {theta_path}")
    
    # Create environments
    n_envs = 16  # Reduced number of environments for more focused learning
    print(f"Building {n_envs} parallel Minitaur environments with hill-climber action wrapper...")
    env_fns = [make_env(render=False, theta_path=theta_path) for _ in range(n_envs)]
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

    # Smaller network since we're learning adjustments, not full control
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    print("Initializing PPO model with policy_kwargs=", policy_kwargs)
    
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=1000,
        batch_size=64,  # Smaller batch size for more frequent updates
        n_epochs=10,
        clip_range=0.2,
        learning_rate=1e-4,  # Lower learning rate for fine-tuning
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,  # Lower entropy coefficient for more stable behavior
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,  # More detailed logging
    )
    
    # Set up callbacks
    progress_freq = 10000  # More frequent evaluation
    checkpoint_freq = 100000  # More frequent checkpoints
    progress_callback = ProgressCallback(print_freq=progress_freq)
    ckpt_callback = CheckpointCallback(save_freq=checkpoint_freq, save_path="./checkpoints/", verbose=1)
    
    print(f"Setting up callbacks: progress every {progress_freq} steps (with eval), checkpoints every {checkpoint_freq} steps...")

    # Reduced timesteps since we're fine-tuning rather than learning from scratch
    total_timesteps = int(1e6)
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=[progress_callback, ckpt_callback])
    print("Training complete.")

    # Save final model and VecNormalize stats
    os.makedirs("models", exist_ok=True)
    print("Saving final model and normalization stats...")
    model.save("models/minitaur_residual_ppo")
    vec_env.save("models/vec_normalize.pkl")
    print("Model and normalization stats saved. All checkpoints have been saved to ./checkpoints/")
    
    vec_env.close()