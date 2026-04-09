import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# 1. Konfiguracja środowiska
# is_slippery=False sprawia, że gra jest deterministyczna (łatwiejsza na start)
# Zmień na True, aby dodać element losowości (lód jest śliski)
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None)

# 2. Inicjalizacja Tabeli Q (wszystkie wartości na zero)
# Wiersze = Stany (16), Kolumny = Akcje (4: lewo, dół, prawo, góra)
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# 3. Hiperparametry (możesz je opisać w sprawozdaniu!)
learning_rate = 0.1    # Alfa: jak szybko aktualizujemy wiedzę
discount_factor = 0.95 # Gamma: jak bardzo cenimy przyszłe nagrody
epsilon = 1.0          # Epsilon: szansa na losowy ruch (eksploracja)
epsilon_decay = 0.001  # Jak szybko AI przestaje "szaleć" a zaczyna korzystać z wiedzy
min_epsilon = 0.01
episodes = 2000        # Liczba gier do rozegrania

# Do wykresu krzywej uczenia
rewards_per_episode = []

print("Rozpoczynam naukę...")

# 4. Główna pętla uczenia
for i in range(episodes):
    state, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    
    while not (terminated or truncated):
        # Mechanizm Epsilon-Greedy (wybór akcji)
        if np.random.random() < epsilon:
            action = env.action_space.sample() # Eksploracja (losowo)
        else:
            action = np.argmax(q_table[state, :]) # Eksploatacja (najlepsza znana)

        # Wykonanie ruchu
        next_state, reward, terminated, truncated, info = env.step(action)

        # Aktualizacja Tabeli Q (Równanie Bellmana - serce algorytmu)
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state, :])
        
        new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
        q_table[state, action] = new_value

        state = next_state
        total_reward += reward

        # Dodaj to do pętli uczenia, aby mierzyć skuteczność co 100 epizodów
        if (i + 1) % 100 == 0:
            recent_success_rate = np.mean(rewards_per_episode[-100:])
            print(f"Epizod {i+1}: Skuteczność ostatnich 100 gier: {recent_success_rate * 100}%")

    # Zmniejszanie epsilon (z czasem AI coraz mniej eksperymentuje)
    epsilon = max(min_epsilon, epsilon - epsilon_decay)
    rewards_per_episode.append(total_reward)

    if (i + 1) % 500 == 0:
        print(f"Epizod: {i + 1} - Uczenie w toku...")

print("Nauka zakończona!")

# 5. Generowanie krzywej uczenia
plt.figure(figsize=(10, 5))
# Uśredniamy wyniki z 50 gier, żeby wykres był czytelniejszy
averaged_rewards = np.convolve(rewards_per_episode, np.ones(50)/50, mode='valid')
plt.plot(averaged_rewards)
plt.title("Krzywa uczenia (Średnia nagroda z 50 epizodów)")
plt.xlabel("Epizod")
plt.ylabel("Nagroda")
plt.savefig("krzywa_uczenia.png") # Zapisuje wykres do pliku
plt.show()

env.close()