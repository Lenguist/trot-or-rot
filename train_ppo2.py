# train_minitaur_ppo.py

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from minitaur_env import MinitaurGymEnv
import numpy as np
from torch.optim.lr_scheduler import LinearLR

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

class LearningRateSchedulerCallback(BaseCallback):
    """
    Callback for scheduling learning rate decay
    """
    def __init__(self, start_lr=1e-3, end_lr=3e-4, total_steps=7e6):
        super().__init__()
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.total_steps = total_steps
        
    def _on_training_start(self):
        # Create linear learning rate scheduler
        self.scheduler = LinearLR(
            optimizer=self.model.policy.optimizer,
            start_factor=1.0,
            end_factor=self.end_lr / self.start_lr,
            total_iters=int(self.total_steps)
        )
        
    def _on_step(self):
        if self.n_calls % 25000 == 0:  # Update every 25k steps to reduce overhead
            self.scheduler.step()
            current_lr = self.model.policy.optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.6f}")
        return True

def make_env(render=False):
    def _init():
        return MinitaurGymEnv(
            render=render,
            energy_weight=0.008,
            distance_weight=1.5,  # Increased reward weight for forward movement
            drift_weight=0.0,
            shake_weight=0.0,
        )
    return _init

if __name__ == "__main__":
    n_envs = 25
    print(f"Building {n_envs} parallel Minitaur environments...")
    env_fns = [make_env(render=False) for _ in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)
    
    # Add observation normalization
    print("Adding observation normalization...")
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,       # Normalize observations
        norm_reward=True,    # Normalize rewards
        clip_obs=10.0,       # Clip observation values
        clip_reward=10.0,    # Clip reward values
        gamma=0.99           # Discount factor for normalizing rewards
    )

    policy_kwargs = dict(net_arch=dict(pi=[185, 95], vf=[95, 85]))
    print("Initializing PPO model with policy_kwargs=", policy_kwargs)
    
    # Start with higher learning rate
    initial_lr = 1e-3
    
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=1000,         # Keep the same steps per rollout
        batch_size=256,       # Reduce batch size for more updates per collection
        n_epochs=10,
        clip_range=0.2,
        learning_rate=initial_lr,  # Higher initial learning rate
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,        # Add entropy coefficient for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )

    progress_freq = 50000  # More frequent feedback
    checkpoint_freq = 500000  # More frequent checkpoints
    total_timesteps = int(7e6)
    
    # Setup callbacks
    progress_callback = ProgressCallback(print_freq=progress_freq)
    ckpt_callback = CheckpointCallback(save_freq=checkpoint_freq, save_path="./checkpoints/", verbose=1)
    lr_callback = LearningRateSchedulerCallback(start_lr=initial_lr, end_lr=3e-4, total_steps=total_timesteps)
    
    print(f"Setting up callbacks: progress every {progress_freq} steps (with eval), checkpoints every {checkpoint_freq} steps...")
    print(f"Learning rate scheduling: starting at {initial_lr} and decaying to 3e-4")
    print(f"Added entropy coefficient of 0.01 to encourage exploration")
    print(f"Batch size reduced to 256 for more gradient updates per collection")

    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=[progress_callback, ckpt_callback, lr_callback])
    print("Training complete.")

    # Save final model and VecNormalize stats (important for using the model later)
    os.makedirs("models", exist_ok=True)
    print("Saving final model and normalization stats...")
    model.save("models/minitaur_gallop_ppo")
    vec_env.save("models/vec_normalize.pkl")
    print("Model and normalization stats saved. All checkpoints have been saved to ./checkpoints/")
    
    vec_env.close()