import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from envs.deep_sea_scavenger_env import DeepSeaScavenger
import matplotlib.pyplot as plt
import numpy as np

class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards = []
        self.current = 0

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards")
        done = self.locals.get("dones")

        if reward is not None:
            if hasattr(reward, "__len__"):
                reward = reward[0]
            self.current += float(reward)

        if done is not None:
            if hasattr(done, "__len__"):
                done = done[0]
            if done:
                self.rewards.append(self.current)
                self.current = 0
        return True

env = DeepSeaScavenger()
check_env(env)

models_dir = "models/PPO"
logdir = "logs"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.01,
    tensorboard_log=logdir
)

logger = RewardLogger()

TIMESTEPS = 100
for i in range(1, 101):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO", callback=logger)
    model.save(f"{models_dir}/{TIMESTEPS*i}")

rewards = logger.rewards
window_size = 20

if len(rewards) >= window_size:
    rewards_smoothed = np.convolve(rewards, np.ones(window_size) / window_size, mode="valid")
    plt.figure(figsize=(10, 6))
    plt.plot(range(window_size - 1, len(rewards)), rewards_smoothed, color='red', linewidth=2, label="Reward")
    plt.title("PPO Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{logdir}/learning_curve_ppo.png", dpi=300)
    plt.show()