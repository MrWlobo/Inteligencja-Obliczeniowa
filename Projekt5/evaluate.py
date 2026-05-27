import os
import json
import numpy as np
import matplotlib.pyplot as plt

try:
    import gym
except Exception:
    gym = None

try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None


def make_env(env_id, continuous=False):
    if gym is None:
        raise RuntimeError("gym is not installed")

    if env_id.endswith("-v3"):
        alt_env_id = env_id[:-2] + "v2"
        print(f"Gym environment {env_id} not available; switching to {alt_env_id} for compatibility.")
        env_id = alt_env_id

    kwargs = {}
    if continuous and env_id.startswith("LunarLander"):
        kwargs["continuous"] = True
        print(f"Creating LunarLander environment in continuous mode: {env_id}")

    try:
        return gym.make(env_id, **kwargs)
    except gym.error.Error as err:
        raise RuntimeError(f"Failed to create environment {env_id}: {err}") from err


def eval_model(model, env, n_episodes=50):
    rewards = []
    for ep in range(n_episodes):
        reset_res = env.reset()
        obs = reset_res[0] if isinstance(reset_res, tuple) else reset_res
        done = False
        ep_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # convert numpy types to native Python types for compatibility
            if isinstance(action, (np.ndarray,)):
                try:
                    if action.shape == ():
                        action = float(action)
                    elif action.size == 1:
                        action = float(action.flatten()[0])
                    else:
                        action = action.tolist()
                except Exception:
                    action = action.tolist()
            elif isinstance(action, (np.floating,)):
                action = float(action)
            # now step
            step_res = env.step(action)
            if len(step_res) == 5:
                obs, reward, terminated, truncated, _ = step_res
                done = terminated or truncated
            else:
                obs, reward, done, _ = step_res
            ep_r += float(reward)
        rewards.append(ep_r)
    return np.array(rewards)


def main(models_dir="models/Default", env_id="LunarLander-v2", n_episodes=50, continuous=True):
    out_dir = os.path.join(models_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)

    if PPO is None:
        print("stable_baselines3 is not installed. Install it to run evaluation.")
        return
    if gym is None:
        print("gym is not installed. Install it to run evaluation.")
        return

    model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip') or f.endswith('.pkl')]
    if not model_files:
        print(f"No model files found in {models_dir}")
        return

    summary = {}
    env = make_env(env_id, continuous=continuous)

    for mf in model_files:
        path = os.path.join(models_dir, mf)
        print(f"Loading {path} ...")
        model = PPO.load(path)
        rewards = eval_model(model, env, n_episodes=n_episodes)
        summary[mf] = {"mean": float(np.mean(rewards)), "std": float(np.std(rewards)), "rewards": rewards.tolist()}
        # save per-model rewards
        np.savetxt(os.path.join(out_dir, mf + "_rewards.csv"), rewards, delimiter=',')

    env.close()

    # pick best model
    best = max(summary.items(), key=lambda kv: kv[1]["mean"])
    best_name, best_stats = best[0], best[1]
    print(f"Best model: {best_name} mean reward={best_stats['mean']:.2f} std={best_stats['std']:.2f}")

    # save summary
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # plot evaluation rewards for best model
    rewards = np.array(best_stats['rewards'])
    plt.figure(figsize=(10, 4))
    plt.plot(rewards, marker='o')
    plt.axhline(np.mean(rewards), color='r', linestyle='--', label=f"mean={np.mean(rewards):.2f}")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Evaluation rewards (deterministic) - {best_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "best_eval_rewards.png"))
    plt.close()

    # also save a small textual report
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(f"Best model: {best_name}\n")
        f.write(f"Mean reward: {best_stats['mean']:.2f}\n")
        f.write(f"Std reward: {best_stats['std']:.2f}\n")

    print(f"Evaluation finished. Results saved in {out_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', default='models/Default')
    parser.add_argument('--env', default='LunarLander-v2')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--continuous', dest='continuous', action='store_true', default=True,
                        help='Use continuous LunarLander mode (gym.make(..., continuous=True))')
    parser.add_argument('--discrete', dest='continuous', action='store_false',
                        help='Use discrete LunarLander mode (gym.make(..., continuous=False))')
    args = parser.parse_args()
    main(models_dir=args.models_dir, env_id=args.env, n_episodes=args.episodes, continuous=args.continuous)
