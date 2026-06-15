# pip install "gymnasium[mujoco]" 
import gymnasium as gym
import pandas as pd
from pathlib import Path
import numpy as np

from stable_baselines3 import PPO, DDPG, SAC, TD3
from sb3_contrib import TRPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise

env_id = "Swimmer-v5"
train_env = make_vec_env(env_id, n_envs=1)
n_actions = train_env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))


methods = {
    "PPO": {
        "model": PPO,
        "timeSteps": 1_000_000,
        "args": {
            "learning_rate": 3e-4,
            "gamma": 0.9999,
        },
    },

    "DDPG": {
        "model": DDPG,
        "timeSteps": 1_000_000,
        "args": {
            "action_noise": action_noise
        },
    },

    "SAC": {
        "model": SAC,
        "timeSteps": 1_000_000,
        "args": {
            "learning_rate": 3e-4,
            "gamma": 0.9999,
            "buffer_size": 100_000,
            "learning_starts": 10_000,
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
            "gamma": 0.9999,
            #"buffer_size": 100_000,
            "learning_starts": 10_000,
            #"batch_size": 256,
            #"tau": 0.005,
            #"train_freq": 1,
            #"gradient_steps": 1,
            #"policy_delay": 2,
            "target_policy_noise": 0.2,
            "target_noise_clip": 0.5,
            "action_noise" : action_noise
        },
    },

    "TRPO": {
        "model": TRPO,
        "timeSteps": 1_000_000,
        "args": {
            "gamma": 0.9999,
            "normalize":True
        },
    },
}

results = []
base_dir = Path(__file__).resolve().parents[1]
model_dir = base_dir / "baseline_models" / "swimmer"
model_dir.mkdir(parents=True, exist_ok=True)
csv_path = model_dir / "results_swimmer.csv"

for method_name, spec in methods.items():
    algo_model = spec["model"]
    algo_args = spec["args"]

    # Training the model
    train_env = make_vec_env(env_id, n_envs=1)

    model = algo_model("MlpPolicy", train_env, **algo_args, verbose=0)
    model.learn(total_timesteps=spec["timeSteps"])

    # Evaluating the model
    eval_env = make_vec_env(env_id, n_envs=1)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=100, deterministic=True
    )

    # Saving the model
    model.save(model_dir / f"{method_name}_swimmer")

    row = {
        "Env ID": env_id,
        "Method": method_name,
        "Training timesteps": spec["timeSteps"],
        "Mean reward": mean_reward,
        "Std reward": std_reward,
    }

    print(
        f"Env ID: {env_id}\n"
        f"Method: {method_name}\n"
        f"Mean reward: {mean_reward}\n"
        f"Std reward: {std_reward}\n"
    )

    row_df = pd.DataFrame([row])
    row_df.to_csv(
        csv_path,
        mode="a",
        header=not csv_path.exists(),
        index=False
    )

    train_env.close()
    eval_env.close()
