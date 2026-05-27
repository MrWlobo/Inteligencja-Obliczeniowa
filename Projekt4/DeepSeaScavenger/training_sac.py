import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import numpy as np

from Projekt4.DeepSeaScavenger.envs.deep_sea_scavenger_env import DeepSeaScavenger

models_dir = "models/SAC_4"
logdir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)


class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards = []
        self.current = 0

    def _on_step(self) -> bool:
        reward = self.locals.get("reward")
        done = self.locals.get("done")

        if reward is None:
            reward = self.locals.get("rewards")

        if reward is not None:
            if isinstance(reward, (np.ndarray, list)):
                reward = reward[0]
            self.current += float(reward)

        if done is None:
            done = self.locals.get("dones")

        if done is not None:
            finished = done[0] if isinstance(done, (np.ndarray, list)) else done
            if finished:
                self.rewards.append(self.current)
                self.current = 0

        return True

env = DeepSeaScavenger()
model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    ent_coef="auto",
    verbose=1
)

logger = RewardLogger()
TOTAL_TRAINING_STEPS = 100000
model.learn(total_timesteps=TOTAL_TRAINING_STEPS, callback=logger, reset_num_timesteps=False)
model.save(f"{models_dir}/{TOTAL_TRAINING_STEPS}")

rewards = logger.rewards
window_size = 20
if len(rewards) >= window_size:
    rewards_smoothed = np.convolve(rewards, np.ones(window_size) / window_size, mode="valid")
    plt.figure(figsize=(10, 6))
    plt.plot(range(window_size - 1, len(rewards)), rewards_smoothed, color='blue', linewidth=2)
    plt.savefig(f"{logdir}/sac_learning_curve_4.png")
    plt.show()