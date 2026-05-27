from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
model_path = 'Projekt5/models/Default/ppo_lunar_Set 1 (Default)_run_0.zip'
print('Loading', model_path)
model = PPO.load(model_path)
print('Model loaded')
env = gym.make('LunarLanderContinuous-v2', continuous=True)
reset_res = env.reset()
obs = reset_res[0] if isinstance(reset_res, tuple) else reset_res
print('obs type:', type(obs))
action, _ = model.predict(obs, deterministic=True)
print('action repr:', action)
print('action type:', type(action))
if isinstance(action, np.ndarray):
    print('ndim, shape, dtype, size:', action.ndim, action.shape, action.dtype, action.size)
if isinstance(action, (np.floating,)):
    print('np floating')
env.close()
print('done')
