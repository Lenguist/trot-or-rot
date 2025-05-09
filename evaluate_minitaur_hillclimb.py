# evaluate_minitaur_hillclimb.py

import numpy as np
import time
from minitaur_env import MinitaurGymEnv

# 1) Load the best parameters from hill-climbing
# Make sure you have run train_minitaur_hillclimb.py with np.save('best_theta.npy', theta)
theta = np.load('best_theta.npy')
print("Loaded theta vector shape:", theta.shape)

# 2) Recreate policy and evaluation function
# (must match train_minitaur_hillclimb definitions)
def policy(theta, t, control_dt):
    amp = theta[:8]
    phase = theta[8:16]
    freq = abs(theta[16])
    time_s = t * control_dt
    actions = amp * np.sin(2 * np.pi * freq * time_s + phase)
    return np.clip(actions, -1.0, 1.0)

# 3) Create a renderable environment
env = MinitaurGymEnv(render=True)
obs = env.reset()

# 4) Run one episode and render
total_reward = 0.0
for t in range(1000):
    a = policy(theta, t, env.control_time_step)
    obs, r, done, _ = env.step(a)
    total_reward += r
    time.sleep(env.control_time_step)
    if done:
        break

print(f"Evaluation finished. Total reward = {total_reward:.2f}")
env.close()
