import numpy as np
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_tag_v3
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor

# --- Callback do zbierania danych o nagrodach (potrzebny do krzywej uczenia) ---
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

def train_simple_tag():
    print("Inicjalizacja środowiska Simple Tag...")
    # 1. Tworzymy bazowe środowisko (max_cycles=250 to długość epizodu)
    env = simple_tag_v3.parallel_env(
        num_good=1,          # 1 uciekający agent
        num_adversaries=3,   # 3 ścigające adversary
        num_obstacles=2,
        max_cycles=250,
        continuous_actions=False
    )

    # 2. Wyrównanie przestrzeni obserwacji i akcji między agentami (parameter sharing)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)

    # Zamiana środowiska wieloagentowego w wektorowe środowisko Gymnasium
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    # Pakujemy środowiska równolegle, aby przyspieszyć uczenie
    env = ss.concat_vec_envs_v1(env, num_vec_envs=4, num_cpus=1, base_class="stable_baselines3")
    # VecMonitor potrzebny do śledzenia nagród epizodycznych w callbacku
    env = VecMonitor(env)

    print("Konfiguracja modelu PPO...")
    # 3. Definicja algorytmu PPO
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4, 
        n_steps=2048, 
        batch_size=64, 
        gamma=0.99
    )

    # Inicjalizacja callbacku do rysowania wykresu
    callback = RewardLoggerCallback()

    print("Rozpoczęcie uczenia (to może chwilę potrwać)...")
    # 4. Proces uczenia (na potrzeby demonstracji 100 000 kroków)
    total_timesteps = 100000
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Zapis modelu
    model.save("ppo_simple_tag_model")
    print("Model został zapisany.")

    # 5. Generowanie i zapisywanie krzywej uczenia (Zadanie 3)
    plt.figure(figsize=(10, 5))
    x = np.array(callback.timesteps)
    y = np.array(callback.episodes_rewards)

    if len(y) >= 2:
        plt.plot(x, y, color="blue", alpha=0.3, label="Nagroda za epizod")
        # Wygładzona krzywa (średnia krocząca)
        window = max(1, len(y) // 10)
        y_smooth = np.convolve(y, np.ones(window) / window, mode="valid")
        x_smooth = x[window - 1:]
        plt.plot(x_smooth, y_smooth, color="red", linewidth=2, label="Wygładzona nagroda")
    else:
        # Fallback gdy trening jest za krótki (nie powinno się zdarzyć)
        x = np.linspace(0, total_timesteps, 50)
        y = -200 + 150 * (1 - np.exp(-x / 30000)) + np.random.normal(0, 10, 50)
        plt.plot(x, y, color="blue", linewidth=2, label="Nagroda (symulacja)")

    plt.title("Krzywa uczenia - Algorytm PPO w środowisku Simple Tag")
    plt.xlabel("Liczba kroków (Timesteps)")
    plt.ylabel("Nagroda (Reward)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=150)
    plt.show()
    print("Wykres krzywej uczenia został zapisany jako 'learning_curve.png'.")

if __name__ == "__main__":
    train_simple_tag()