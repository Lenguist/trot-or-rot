# run_minitaur_fixed.py

from minitaur_env import MinitaurGymEnv   # adjust to your filename
import numpy as np

def main():
    # 1. Make the env (headless by default)
    env = MinitaurGymEnv(render=True)

    # 2. Reset to get initial observation (convert list→ndarray)
    obs = np.array(env.reset())
    print("obs dim:", obs.shape)

    # 3. Step through 100 random actions
    total_reward = 0.0
    for t in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        # step() already returns an ndarray, but if you ever swap in a custom env:
        obs = np.array(obs)
        total_reward += reward
        if done:
            print(f"Terminated at step {t}")
            break

    print("Total reward:", total_reward)
    env.close()

if __name__ == "__main__":
    main()
