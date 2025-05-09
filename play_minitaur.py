# play_minitaur.py

import time
from stable_baselines3 import PPO
from minitaur_env import MinitaurGymEnv

def main():
    # 1) Create a renderable env
    env = MinitaurGymEnv(render=True)

    # 2) Load the trained model (automatically links the env for rollouts)
    model = PPO.load("models/minitaur_gallop_ppo", env=env)

    # 3) Run one episode
    obs = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        # Predict action (deterministic for evaluation)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Let the GUI catch up
        time.sleep(env.control_time_step)

    print(f"Episode finished, total reward = {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    main()
