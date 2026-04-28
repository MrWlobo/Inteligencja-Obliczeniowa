import gymnasium as gym
from stable_baselines3 import PPO, DDPG
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy


def get_data(model_type):
    log_dir = f"./logs_{model_type}/"
    import os
    os.makedirs(log_dir, exist_ok=True)
    env = gym.make("Pendulum-v1")
    env = Monitor(env, log_dir)

    if model_type == "PPO":
        model = PPO("MlpPolicy", env, learning_rate=0.003, gamma=0.90, verbose=1).learn(50_000)
    else:
        model = DDPG("MlpPolicy", env, learning_rate=0.003, gamma=0.90, verbose=1).learn(50_000)

    x, y = ts2xy(load_results(log_dir), "timesteps")
    return x, y

x_ppo, y_ppo = get_data("PPO")
x_ddpg, y_ddpg = get_data("DDPG")

window_size = 10

y_ppo_smooth = np.convolve(y_ppo, np.ones(window_size) / window_size, mode="valid")
x_ppo_smooth = x_ppo[window_size - 1:]

y_ddpg_smooth = np.convolve(y_ddpg, np.ones(window_size) / window_size, mode="valid")
x_ddpg_smooth = x_ddpg[window_size - 1:]

plt.figure(figsize=(10, 6))
plt.plot(x_ppo_smooth, y_ppo_smooth, label="PPO")
plt.plot(x_ddpg_smooth, y_ddpg_smooth, label="DDPG")

plt.title("PPO vs DDPG: Pendulum-v1 (Gamma 0.9)")
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig("ppo_ddpg_comparison.png", dpi=300)
plt.show()

