import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


env = gym.make("FrozenLake-v1", is_slippery=False, render_mode=None)

# Inicjalizacja Tabeli Q 
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hiperparametry 
learning_rate = 0.1    # Alfa: jak szybko aktualizujemy wiedzę
discount_factor = 0.95 # Gamma: jak bardzo cenimy przyszłe nagrody
epsilon = 1.0          # Epsilon: szansa na losowy ruch (eksploracja)
epsilon_decay = 0.001  # Jak szybko AI przestaje "szaleć" a zaczyna korzystać z wiedzy
min_epsilon = 0.01
episodes = 2000      

rewards_per_episode = []

print("Rozpoczynam naukę...")

# Główna pętla uczenia
for i in range(episodes):
    state, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    
    while not (terminated or truncated):
        if np.random.random() < epsilon:
            action = env.action_space.sample() 
        else:
            action = np.argmax(q_table[state, :])

        # Wykonanie ruchu
        next_state, reward, terminated, truncated, info = env.step(action)

        # Aktualizacja Tabeli Q
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state, :])
        
        new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
        q_table[state, action] = new_value

        state = next_state
        total_reward += reward

        
        if (i + 1) % 100 == 0:
            recent_success_rate = np.mean(rewards_per_episode[-100:])
            print(f"Epizod {i+1}: Skuteczność ostatnich 100 gier: {recent_success_rate * 100}%")

    # Zmniejszanie epsilon 
    epsilon = max(min_epsilon, epsilon - epsilon_decay)
    rewards_per_episode.append(total_reward)

    if (i + 1) % 500 == 0:
        print(f"Epizod: {i + 1} - Uczenie w toku...")

print("Nauka zakończona!")

# Generowanie krzywej uczenia
plt.figure(figsize=(10, 5))
averaged_rewards = np.convolve(rewards_per_episode, np.ones(50)/50, mode='valid')
plt.plot(averaged_rewards)
plt.title("Krzywa uczenia (Średnia nagroda z 50 epizodów)")
plt.xlabel("Epizod")
plt.ylabel("Nagroda")
plt.savefig("krzywa_uczenia.png")
plt.show()

env.close()