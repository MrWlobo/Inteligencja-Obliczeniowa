"""
Zadanie 6 punktów – Porównanie algorytmów w środowisku Simple Tag
Eksperymenty:
  1. PPO v1  – wszyscy agenci, parameter sharing, lr=3e-4
  2. PPO v2  – wszyscy agenci, parameter sharing, lr=1e-3 (inny wariant)
  3. A2C     – wszyscy agenci, parameter sharing
  4. Różne algorytmy: adversary=PPO (bez good agent), good agent=losowy
  5. Różne algorytmy: adversary=PPO (exp 4), good agent=A2C
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

from pettingzoo.mpe import simple_tag_v3
import supersuit as ss
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


# ── Callback ──────────────────────────────────────────────────────────────────

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episodes_rewards = []
        self.timesteps = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episodes_rewards.append(info["episode"]["r"])
                self.timesteps.append(self.num_timesteps)
        return True


# ── Środowisko z parameter sharing (wszyscy agenci) ──────────────────────────

def make_shared_env(num_vec_envs=4):
    env = simple_tag_v3.parallel_env(
        num_good=1, num_adversaries=3, num_obstacles=2,
        max_cycles=250, continuous_actions=False
    )
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=num_vec_envs, num_cpus=1,
                                 base_class="stable_baselines3")
    return VecMonitor(env)


# ── Custom env: tylko adversary ───────────────────────────────────────────────

class AdversaryGymEnv(gym.Env):
    """
    Gym env dla treningu wyłącznie adversary.
    Obserwacja: wekt. adversary_0 (16-dim).
    Akcja wspólna dla wszystkich adversary (parameter sharing w roli).
    Good agent: losowa polityka (lub podany model).
    """

    def __init__(self, good_model=None):
        super().__init__()
        self.good_model = good_model
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)
        self._env = None
        self._adversaries = []
        self._good_agents = []
        self._obs = {}

    def reset(self, seed=None, options=None):
        if self._env is None:
            self._env = simple_tag_v3.parallel_env(
                num_good=1, num_adversaries=3, num_obstacles=2,
                max_cycles=250, continuous_actions=False
            )
        self._obs, _ = self._env.reset(seed=seed)
        self._adversaries = [a for a in self._env.agents if "adversary" in a]
        self._good_agents = [a for a in self._env.agents if "agent_" in a]
        obs = self._obs.get(self._adversaries[0], np.zeros(16, dtype=np.float32))
        return np.asarray(obs, dtype=np.float32), {}

    def step(self, action):
        act_dict = {
            adv: int(action)
            for adv in self._adversaries if adv in self._obs
        }
        for good in self._good_agents:
            if good not in self._obs:
                continue
            if self.good_model is not None:
                g_obs = np.asarray(self._obs[good], dtype=np.float32)
                g_act, _ = self.good_model.predict(g_obs, deterministic=False)
                act_dict[good] = int(g_act)
            else:
                act_dict[good] = self._env.action_space(good).sample()

        self._obs, rewards, terms, truncs, _ = self._env.step(act_dict)

        adv_reward = sum(rewards.get(a, 0.0) for a in self._adversaries)
        terminated = all(terms.get(a, False) for a in self._adversaries)
        truncated = all(truncs.get(a, False) for a in self._adversaries)

        if self._adversaries and self._adversaries[0] in self._obs:
            obs = np.asarray(self._obs[self._adversaries[0]], dtype=np.float32)
        else:
            obs = np.zeros(16, dtype=np.float32)

        return obs, float(adv_reward), terminated, truncated, {}


# ── Custom env: tylko good agent ──────────────────────────────────────────────

class GoodAgentGymEnv(gym.Env):
    """
    Gym env dla treningu wyłącznie good agent.
    Obserwacja: wekt. agent_0 (14-dim).
    Adversary: podany model (lub losowa polityka).
    """

    def __init__(self, adversary_model=None):
        super().__init__()
        self.adversary_model = adversary_model
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)
        self._env = None
        self._adversaries = []
        self._good_agents = []
        self._obs = {}

    def reset(self, seed=None, options=None):
        if self._env is None:
            self._env = simple_tag_v3.parallel_env(
                num_good=1, num_adversaries=3, num_obstacles=2,
                max_cycles=250, continuous_actions=False
            )
        self._obs, _ = self._env.reset(seed=seed)
        self._adversaries = [a for a in self._env.agents if "adversary" in a]
        self._good_agents = [a for a in self._env.agents if "agent_" in a]
        obs = self._obs.get(self._good_agents[0], np.zeros(14, dtype=np.float32))
        return np.asarray(obs, dtype=np.float32), {}

    def step(self, action):
        act_dict = {
            good: int(action)
            for good in self._good_agents if good in self._obs
        }
        for adv in self._adversaries:
            if adv not in self._obs:
                continue
            if self.adversary_model is not None:
                a_obs = np.asarray(self._obs[adv], dtype=np.float32)
                a_act, _ = self.adversary_model.predict(a_obs, deterministic=False)
                act_dict[adv] = int(a_act)
            else:
                act_dict[adv] = self._env.action_space(adv).sample()

        self._obs, rewards, terms, truncs, _ = self._env.step(act_dict)

        good_reward = sum(rewards.get(a, 0.0) for a in self._good_agents)
        terminated = all(terms.get(a, False) for a in self._good_agents)
        truncated = all(truncs.get(a, False) for a in self._good_agents)

        if self._good_agents and self._good_agents[0] in self._obs:
            obs = np.asarray(self._obs[self._good_agents[0]], dtype=np.float32)
        else:
            obs = np.zeros(14, dtype=np.float32)

        return obs, float(good_reward), terminated, truncated, {}


# ── Narzędzia ─────────────────────────────────────────────────────────────────

def smooth(y, window=20):
    if len(y) < window:
        return np.array(y), np.arange(len(y))
    s = np.convolve(y, np.ones(window) / window, mode="valid")
    return s, np.arange(window - 1, len(y))


def plot_curve(ax, x, y, color, label):
    if len(y) < 2:
        return
    ax.plot(x, y, color=color, alpha=0.18, linewidth=0.8)
    w = max(5, len(y) // 8)
    y_s, idx = smooth(y, w)
    ax.plot(x[idx], y_s, color=color, linewidth=2.2, label=label)


# ── Eksperymenty ──────────────────────────────────────────────────────────────

def main():
    TIMESTEPS = 100_000
    results = {}

    # 1. PPO v1 – wszyscy agenci, parameter sharing
    print("\n" + "="*55)
    print("[1/5] PPO v1 – wszyscy agenci, lr=3e-4, n_steps=2048")
    print("="*55)
    env = make_shared_env()
    m_ppo_v1 = PPO("MlpPolicy", env, verbose=1,
                   learning_rate=3e-4, n_steps=2048, batch_size=64, gamma=0.99)
    cb1 = RewardLoggerCallback()
    m_ppo_v1.learn(TIMESTEPS, callback=cb1)
    m_ppo_v1.save("model_ppo_v1")
    env.close()
    results["PPO v1 (wszyscy, lr=3e-4)"] = (np.array(cb1.timesteps), np.array(cb1.episodes_rewards))

    # 2. PPO v2 – wszyscy agenci, inne hiperparametry
    print("\n" + "="*55)
    print("[2/5] PPO v2 – wszyscy agenci, lr=1e-3, n_steps=1024")
    print("="*55)
    env = make_shared_env()
    m_ppo_v2 = PPO("MlpPolicy", env, verbose=1,
                   learning_rate=1e-3, n_steps=1024, batch_size=128, gamma=0.99)
    cb2 = RewardLoggerCallback()
    m_ppo_v2.learn(TIMESTEPS, callback=cb2)
    m_ppo_v2.save("model_ppo_v2")
    env.close()
    results["PPO v2 (wszyscy, lr=1e-3)"] = (np.array(cb2.timesteps), np.array(cb2.episodes_rewards))

    # 3. A2C – wszyscy agenci, parameter sharing
    print("\n" + "="*55)
    print("[3/5] A2C – wszyscy agenci (parameter sharing)")
    print("="*55)
    env = make_shared_env()
    m_a2c = A2C("MlpPolicy", env, verbose=1, learning_rate=7e-4, gamma=0.99)
    cb3 = RewardLoggerCallback()
    m_a2c.learn(TIMESTEPS, callback=cb3)
    m_a2c.save("model_a2c_shared")
    env.close()
    results["A2C (wszyscy)"] = (np.array(cb3.timesteps), np.array(cb3.episodes_rewards))

    # 4. Różne algorytmy – PPO tylko dla adversary, good agent = losowy
    print("\n" + "="*55)
    print("[4/5] PPO – tylko adversary, good agent = losowy")
    print("="*55)
    adv_env = DummyVecEnv([AdversaryGymEnv])
    adv_env = VecMonitor(adv_env)
    m_adv_ppo = PPO("MlpPolicy", adv_env, verbose=1,
                    learning_rate=3e-4, n_steps=2048, batch_size=64, gamma=0.99)
    cb4 = RewardLoggerCallback()
    m_adv_ppo.learn(TIMESTEPS, callback=cb4)
    m_adv_ppo.save("model_adversary_ppo")
    adv_env.close()
    results["PPO adversary (good=losowy)"] = (np.array(cb4.timesteps), np.array(cb4.episodes_rewards))

    # 5. Różne algorytmy – A2C dla good agent vs wytrenowane PPO adversary
    print("\n" + "="*55)
    print("[5/5] A2C – good agent vs wytrenowane PPO adversary")
    print("="*55)

    def make_good_env():
        return GoodAgentGymEnv(adversary_model=m_adv_ppo)

    good_env = DummyVecEnv([make_good_env])
    good_env = VecMonitor(good_env)
    m_good_a2c = A2C("MlpPolicy", good_env, verbose=1, learning_rate=7e-4, gamma=0.99)
    cb5 = RewardLoggerCallback()
    m_good_a2c.learn(TIMESTEPS, callback=cb5)
    m_good_a2c.save("model_good_agent_a2c")
    good_env.close()
    results["A2C good agent (vs PPO adv.)"] = (np.array(cb5.timesteps), np.array(cb5.episodes_rewards))

    # ── Wykresy ───────────────────────────────────────────────────────────────

    COLORS = {
        "PPO v1 (wszyscy, lr=3e-4)":      "steelblue",
        "PPO v2 (wszyscy, lr=1e-3)":      "darkorange",
        "A2C (wszyscy)":                   "seagreen",
        "PPO adversary (good=losowy)":     "crimson",
        "A2C good agent (vs PPO adv.)":   "purple",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel lewy – ten sam algorytm (parameter sharing)
    ax1.set_title("Ten sam algorytm – parameter sharing\n(PPO v1 vs PPO v2 vs A2C)", fontsize=12)
    for lbl in ["PPO v1 (wszyscy, lr=3e-4)", "PPO v2 (wszyscy, lr=1e-3)", "A2C (wszyscy)"]:
        x, y = results[lbl]
        plot_curve(ax1, x, y, COLORS[lbl], lbl)
    ax1.set_xlabel("Kroki (Timesteps)")
    ax1.set_ylabel("Nagroda za epizod")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel prawy – różne algorytmy
    ax2.set_title("Różne algorytmy (PPO adversary + A2C good agent)", fontsize=12)
    for lbl in ["PPO adversary (good=losowy)", "A2C good agent (vs PPO adv.)"]:
        x, y = results[lbl]
        plot_curve(ax2, x, y, COLORS[lbl], lbl)
    ax2.set_xlabel("Kroki (Timesteps)")
    ax2.set_ylabel("Nagroda za epizod")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Porównanie algorytmów – Simple Tag (PettingZoo)", fontsize=14)
    plt.tight_layout()
    plt.savefig("comparison_curves.png", dpi=150)
    plt.show()
    print("\nWykres zapisany jako 'comparison_curves.png'")

    np.save("results.npy", results)
    print("Wyniki zapisane jako 'results.npy'")


if __name__ == "__main__":
    main()
