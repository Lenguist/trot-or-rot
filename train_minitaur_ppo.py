# train_minitaur_ppo.py

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
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
            energy_weight=0.008,
            distance_weight=1.0,
            drift_weight=0.0,
            shake_weight=0.0,
        )
    return _init

if __name__ == "__main__":
    n_envs = 25
    print(f"Building {n_envs} parallel Minitaur environments...")
    env_fns = [make_env(render=False) for _ in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)

    policy_kwargs = dict(net_arch=dict(pi=[185, 95], vf=[95, 85]))
    print("Initializing PPO model with policy_kwargs=", policy_kwargs)
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=1000,
        batch_size=1000,
        n_epochs=10,
        clip_range=0.2,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )

    progress_freq = 100000
    checkpoint_freq = 1000000
    progress_callback = ProgressCallback(print_freq=progress_freq)
    ckpt_callback = CheckpointCallback(save_freq=checkpoint_freq, save_path="./checkpoints/", verbose=1)
    print(f"Setting up callbacks: progress every {progress_freq} steps (with eval), checkpoints every {checkpoint_freq} steps...")

    total_timesteps = int(7e6)
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=[progress_callback, ckpt_callback])
    print("Training complete.")

    os.makedirs("models", exist_ok=True)
    print("Saving final model... and all checkpoints have been saved to ./checkpoints/")
    model.save("models/minitaur_gallop_ppo")
    vec_env.close()
