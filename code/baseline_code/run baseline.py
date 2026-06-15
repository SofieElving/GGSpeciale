import gymnasium as gym

import pandas as pd

from stable_baselines3 import PPO, DDPG, SAC, TD3
from sb3_contrib import TRPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from code.baseline_code.baseline_enviroments.cartpole_env import make_continuous_cartpole


methods = {
    "PPO": {
        "model": PPO,
        "args": {
            "learning_rate": 1e-3,
            "gamma": 0.99,
        },
    },
    "DDPG": {
        "model": DDPG,
        "args": {
            "learning_rate": 1e-3,
            "buffer_size": 200_000,
            "learning_starts": 5_000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": (1, "step"),
            "gradient_steps": 1,
            # action_noise can be injected here later
        },
    },
    "TRPO": {
        "model": TRPO,
        "args": {
            "gamma": 0.99,
        },
    },
}


environments = ['CartPole-v1', 
            #    'MountainCar-v0', 
            #    'Pendulum-v1', 
            #    'Acrobot-v1', 
            #    'Swimmer-v5',
            #    'Reacher-v5'
               ]

TOTAL_TIMESTEPS = 25_000

results = []

for method_name, spec in methods.items():
    for env_id in environments:
        # Determine if env has discrete actions
        check_env = gym.make(env_id)
        is_discrete = hasattr(check_env.action_space, "n")
        check_env.close()

        # Decide how to build envs for this (algo, env) combo
        env_factory = env_id  # default: standard gym env by id

        if method_name == "DDPG":
            if is_discrete:
                if env_id == "CartPole-v1":
                    # Special-case: wrap CartPole into continuous-action variant
                    env_factory = make_continuous_cartpole
                else:
                    print(f"Skipping {method_name} on {env_id} (discrete action space)")
                    continue

        print(f"\nRunning {method_name} on {env_id}")

        # Train
        train_env = make_vec_env(env_factory, n_envs=1)

        Algo = spec["model"]
        algo_args = spec["args"]
        model = Algo(
            "MlpPolicy",
            train_env,
            verbose=0,
            **algo_args
        )
        model.learn(total_timesteps=TOTAL_TIMESTEPS)

        # Evaluate (use same env factory type as training)
        eval_env = make_vec_env(env_factory, n_envs=1)
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

        print(
            "==============\n"
            f"Average reward: {mean_reward}\n"
            f"Std reward: {std_reward}\n"
            "=============="
        )

        results.append({"algorithm" : method_name, 
                        "env" : env_id, 
                        "mean reward" : mean_reward, 
                        "std reward" : std_reward})

        train_env.close()
        eval_env.close()

results = pd.DataFrame(results)

print(results)

