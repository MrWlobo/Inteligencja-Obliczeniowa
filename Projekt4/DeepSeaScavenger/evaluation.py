import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3 import SAC

from Projekt4.DeepSeaScavenger.envs.deep_sea_scavenger_env import DeepSeaScavenger

env = DeepSeaScavenger(render_mode="human")
# model_path = "models/PPO/48000.zip"
# model = PPO.load(model_path, env=env)
model_path = "models/SAC_4/100000.zip"

model = SAC.load(model_path, env=env)

episodes = 5
for ep in range(episodes):
    obs, info = env.reset()
    done = False
    score = 0

    while not done:
        env.render()

        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward

    print(f"Episode: {ep + 1}, Score: {score}")

env.close()
