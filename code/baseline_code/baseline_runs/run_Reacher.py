# pip install "gymnasium[mujoco]" 
import gymnasium as gym
import pandas as pd
from pathlib import Path

from stable_baselines3 import PPO, DDPG, SAC, TD3
from sb3_contrib import TRPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


methods = {
    "PPO": {
        "model": PPO,
        "timeSteps": 50_000,
        "args": {
            "learning_rate": 3e-4,
            "gamma": 0.99,
        },
    },

    "DDPG": {
        "model": DDPG,
        "timeSteps": 100_000,
        "args": {},
    },

    "SAC": {
        "model": SAC,
        "timeSteps": 100_000,
        "args": {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "buffer_size": 100_000,
            "learning_starts": 1_000,
            "batch_size": 256,
            "tau": 0.005,
            "train_freq": 1,
            "gradient_steps": 1,
        },
    },

    "TD3": {
        "model": TD3,
        "timeSteps": 100_000,
        "args": {
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "buffer_size": 100_000,
            "learning_starts": 1_000,
            "batch_size": 256,
            "tau": 0.005,
            "train_freq": 1,
            "gradient_steps": 1,
            "policy_delay": 2,
            "target_policy_noise": 0.2,
            "target_noise_clip": 0.5,
        },
    },

    "TRPO": {
        "model": TRPO,
        "timeSteps": 50_000,
        "args": {
            "gamma": 0.99,
        },
    },
}


env_id = "Reacher-v5"

total_timesteps = 2_000

results = []

for method_name, spec in methods.items():
    algo_model = spec["model"]
    algo_args = spec["args"]

    # Training the model
    train_env = make_vec_env(env_id, n_envs=1)

    model = algo_model("MlpPolicy", train_env, **algo_args, verbose=0)
    model.learn(total_timesteps=total_timesteps)

    # Evaluating the model
    eval_env = make_vec_env(env_id, n_envs=1)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)

    # Saving the model
    base_dir = Path(__file__).resolve().parents[1] 
    model_dir = base_dir / "baseline_models" / "reacher"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / f"{method_name}_reacher")

    print(
        f"Env ID: {env_id}\n",
        f"Method: {method_name}\n",
        f"Mean reward: {mean_reward}\n",
        f"Std reward: {std_reward}\n"
    )

    results.append({
    "Env ID": env_id,
    "Method": method_name,
    "Training timesteps": total_timesteps,
    "Mean reward": mean_reward,
    "Std reward": std_reward,
    })


    train_env.close()
    eval_env.close()

# Saving results
results_df = pd.DataFrame(results)
csv_path = model_dir / "results_reacher.csv"
results_df.to_csv(csv_path, index=False)
