import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

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


env = gym.make("Pendulum-v1")
gamma = 0.10

model = PPO(
    "MlpPolicy",
    env,

    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=gamma,
    gae_lambda=0.95,

    clip_range=0.2,
    ent_coef=0.01,

    verbose=1
)

logger = RewardLogger()

model.learn(total_timesteps=200_000, callback=logger)

model.save(f"models/ppo_pendulum_gamma_{gamma}")

rewards = logger.rewards

window_size = 20

rewards_smoothed = np.convolve(rewards, np.ones(window_size) / window_size, mode="valid")

plt.figure(figsize=(10, 6))

plt.plot(range(window_size - 1, len(rewards)), rewards_smoothed, color='red', linewidth=2,
         label=f"Reward")

plt.title(f"PPO Learning Curve (Gamma: {gamma})")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig(f"logs/learning_curve_gamma_{gamma}.png", dpi=300)
plt.show()
