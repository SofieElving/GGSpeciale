# https://gymnasium.farama.org/environments/classic_control/cart_pole/
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from stable_baselines3 import PPO, DDPG, SAC, TD3
from sb3_contrib import TRPO

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from baseline_code.baseline_enviroments.cartpole_env import make_continuous_cartpole
from pathlib import Path

model_saves_folder = Path(r"code\baseline_code\baseline_models\cartpole")
model_settings_path = 
model_saves_folder.mkdir(parents=True, exist_ok=True)

algos = ["PPO", "DDPG", "SAC", "TD3", "TRPO"]

methods = {
    "PPO": {
        "model": PPO,
        "usesDiscreteAction": True,
        "timeSteps": 15_000,
        "args": {
            "learning_rate": 1e-3,
            "gamma": 0.99,
        },
    },

    "DDPG": {
        "model": DDPG,
        "usesDiscreteAction": False,
        "timeSteps": 50_000,
        "args": {
            # You can add DDPG-specific hyperparams here later
        },
    },
    "SAC": {
        "model": SAC,
        "usesDiscreteAction": False,
        "timeSteps": 50_000,
        "args": {
            # Reasonable defaults for a simple continuous env wrapper
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "buffer_size": 100_000,
            "learning_starts": 1_000,
            "batch_size": 256,
            "tau": 0.005,
            "train_freq": 1,        # or (1, "step")
            "gradient_steps": 1,
        },
    },
    "TD3": {
        "model": TD3,
        "usesDiscreteAction": False,
        "timeSteps": 50_000,
        "args": {
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "buffer_size": 1_000,
            "learning_starts": 1_000,
            "batch_size": 256,
            "tau": 0.005,
            "train_freq": 1,        # or (1, "step")
            "gradient_steps": 1,
            # TD3-specific knobs (defaults are fine too)
            "policy_delay": 2,
            "target_policy_noise": 0.2,
            "target_noise_clip": 0.5,
        },
    },

    "TRPO": {
        "model": TRPO,
        "usesDiscreteAction": True,
        "timeSteps": 25_000,
        "args": {
            "gamma": 0.99,
        },
    },
}

for methodName, spec in methods.items():
    algo = spec["model"]
    args = spec["args"]
    usesDiscrete = spec["usesDiscreteAction"]

    # Train/eval envs must match action space type
    if usesDiscrete:
        train_env = make_vec_env("CartPole-v1", n_envs=1)
        eval_env  = make_vec_env("CartPole-v1", n_envs=1)
    else:
        train_env = make_vec_env(make_continuous_cartpole, n_envs=1)
        eval_env  = make_vec_env(make_continuous_cartpole, n_envs=1)

    model = algo("MlpPolicy", train_env, verbose=0, **args)
    model.learn(total_timesteps=spec["timeSteps"])

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=10,
        deterministic=True
    )

    model_path = model_saves_folder / f"{methodName}_cartpole"
    model.save(str(model_path))

    print(
        f"==============\n"
        f"Method: {methodName}\n"
        f"Average reward: {mean_reward}\n"
        f"Std reward: {std_reward}\n"
        f"=============="
    )

    train_env.close()
    eval_env.close()


