# train_minitaur_hillclimb.py

import numpy as np
from minitaur_env import MinitaurGymEnv

# Define a simple sinusoidal gait policy
#   - theta[:8]: amplitude for each of 8 motors
#   - theta[8:16]: phase offset for each motor
#   - theta[16]: frequency (Hz)
def policy(theta, t, control_dt):
    amp = theta[:8]
    phase = theta[8:16]
    freq = abs(theta[16])  # ensure positive frequency
    # time in seconds
    time_s = t * control_dt
    # sinusoidal action, clipped to [-1, 1]
    actions = amp * np.sin(2 * np.pi * freq * time_s + phase)
    return np.clip(actions, -1.0, 1.0)

# Evaluate a parameter vector theta on one episode, returning total reward
def evaluate(env, theta, max_steps=1000):
    obs = env.reset()
    total_reward = 0.0
    for t in range(max_steps):
        a = policy(theta, t, env.control_time_step)
        obs, r, done, _ = env.step(a)
        total_reward += r
        if done:
            break
    return total_reward

if __name__ == "__main__":
    # Hyperparameters
    n_iters = 5000     # hill-climbing iterations
    sigma = 0.1       # noise stddev for perturbations
    max_steps = 1000  # steps per evaluation

    # Initialize environment (headless)
    env = MinitaurGymEnv(render=False)

    # Parameter vector: 17 values (8 amplitudes, 8 phases, 1 freq)
    theta = np.random.randn(17) * 0.5
    best_reward = evaluate(env, theta, max_steps)
    print(f"Initial reward: {best_reward:.2f}")

    for itr in range(1, n_iters+1):
        # Propose new candidate
        candidate = theta + sigma * np.random.randn(17)
        reward = evaluate(env, candidate, max_steps)
        # Accept if better
        if reward > best_reward:
            theta, best_reward = candidate, reward
            print(f"Iteration {itr:3d}: New best reward = {best_reward:.2f}")
    
    print("Hill-climbing finished. Best reward:", best_reward)
    # at the end of train_minitaur_hillclimb.py, add:
    np.save('best_theta.npy', theta)
    print("Saved best_theta.npy")

    env.close()
