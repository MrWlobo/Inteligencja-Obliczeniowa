import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

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
gamma = 0.99

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

model.learn(total_timesteps=300_000, callback=logger)

model.save(f"models/ppo_pendulum_gamma_{gamma}")

rewards = logger.rewards

plt.plot(rewards)
plt.title("PPO Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.savefig(f"logs/learning_curve_gamma_{gamma}.png", dpi=300)
plt.show()
