import pandas as pd
from stable_baselines3 import PPO, DDPG, SAC, TD3
from sb3_contrib import TRPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize
import sys
import os
from pathlib import Path
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise

# Added Vec normalization
sys.path.append(os.path.abspath(".."))

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from baseline_enviroments.MountainCar import make_continuous_mountaincar

model_saves_folder = PROJECT_ROOT / "baseline_models" / "mountaincar"
model_saves_folder.mkdir(parents=True, exist_ok=True)

# Added action noise for DDPG and TD3 to avoid collapses to near-zero actions
def make_action_noise(n_actions: int, sigma: float = 0.2):
    return NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=sigma * np.ones(n_actions),
    )

methods = {
    "PPO": {
        "model": PPO,
        "n_envs": 16,
        "args": { # args inspired by https://arxiv.org/pdf/1707.06347
            "learning_rate": 3e-4,      # Gradient step size (smaller = more stable learning)
            "n_steps": 1024,            # Rollout length before each update (long horizon credit)
            "batch_size": 64,           # Minibatch size for optimization (adds useful gradient noise)
            "n_epochs": 4,
            "gamma": 0.99,             # Discount factor (keeps distant +100 reward relevant)
            "gae_lambda": 0.98,         # Bias-variance tradeoff in advantage estimation
            "clip_range": 0.2,          # PPO trust-region style update constraint
            "ent_coef": 0.05,           # Encourages exploration (prevents zero-action collapse)
            "vf_coef": 0.5,             # Weight of value function loss
            "max_grad_norm": 0.5,       # Gradient clipping for stability
            },
    },
    # "DDPG": {
    #     "model": DDPG,
    #     "n_envs": 16,
    #     "args": {
    #         "learning_rate": 1e-3,
    #         "buffer_size": 1_000_000,
    #         "learning_starts": 10_000,
    #         "batch_size": 256,
    #         "tau": 0.005,
    #         "gamma": 0.99,
    #         "train_freq": (1, "step"),
    #         "gradient_steps": 1,
    #     },
    #     "needs_action_noise": True,
    #     "noise_sigma": 0.2,
    # },
    # "TRPO": {
    #     "model": TRPO,
    #     "n_envs": 4,
    #     "args": {"gamma": 0.99,
    #              "n_steps":1024,
    #              "gae_lambda": 0.98,},
    # },
    # DID NOT RUN TO PERFECTION:
    # "SAC": {
    #     "model": SAC,
    #     "n_envs": 16,
    #     "args": {
    #         "learning_rate": 3e-4,
    #         "buffer_size": 1_000_000,
    #         "batch_size": 256,
    #         "tau": 0.01,
    #         "gamma": 0.9999,
    #         "train_freq": (1, "step"),         # can be int or (32, "step")
    #         "gradient_steps": 1,
    #         "learning_starts": 10_000,
    #         "ent_coef": 0.1,
    #     },
    # },
    # "TD3": {
    #     "model": TD3,
    #     "n_envs": 4,
    #     "args": {
    #         "learning_rate": 1e-3,
    #         "gamma": 0.99,
    #         "buffer_size": 200_000,
    #         "learning_starts": 5_000,
    #         "batch_size": 256,
    #         "tau": 0.005,
    #         "train_freq": (1, "step"),
    #         "gradient_steps": 1,
    #     },
    #     "needs_action_noise": True,
    #     "noise_sigma": 0.2,
    # }
}

ENV_FACTORY = make_continuous_mountaincar
ENV_NAME = "MountainCar"  

TOTAL_TIMESTEPS = 300_000
results = []

for method_name, spec in methods.items():
    print(f"\nRunning {method_name} on {ENV_NAME}")

    n_envs = spec.get("n_envs", 1)
    print(n_envs)
    train_env = make_vec_env(ENV_FACTORY, n_envs=n_envs, seed=42)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    Algo = spec["model"]
    model = Algo("MlpPolicy", train_env, verbose=0, **spec["args"])
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    eval_env = make_vec_env(ENV_FACTORY, n_envs=1, seed=0)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)
    eval_env.obs_rms = train_env.obs_rms
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

    model_path = model_saves_folder / f"{method_name}_mountaincar"
    model.save(str(model_path))

    vecnorm_path = model_saves_folder / f"{method_name}_vecnormalize.pkl"
    train_env.save(str(vecnorm_path))
    print(
        "==============\n"
        f"Average reward: {mean_reward}\n"
        f"Std reward: {std_reward}\n"
        "=============="
    )

    results.append(
        {
            "algorithm": method_name,
            "env": ENV_NAME,
            "mean reward": mean_reward,
            "std reward": std_reward,
        }
    )

    train_env.close()
    eval_env.close()


results = pd.DataFrame(results)
print(results)
