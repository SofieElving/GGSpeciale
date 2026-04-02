import argparse
import json
import os
import warnings
import joblib
import matplotlib.pyplot as plt


print("1: starting script", flush=True)

from SPID_code.gmDAGGER2 import train_spid
import SPID_code.gmDAGGER2 as mod
print("module file:", mod.__file__, flush=True)
print("teacher_path exists:", os.path.exists(args.teacher_path), flush=True)
print("environment arg:", args.environment, flush=True)
print("2: imported train_spid", flush=True)

def get_algo(name):
    print(f"3: resolving algo {name}", flush=True)
    if name == "PPO":
        from stable_baselines3 import PPO
        return PPO
    if name == "DDPG":
        from stable_baselines3 import DDPG
        return DDPG
    if name == "SAC":
        from stable_baselines3 import SAC
        return SAC
    if name == "TD3":
        from stable_baselines3 import TD3
        return TD3
    if name == "DQN":
        from stable_baselines3 import DQN
        return DQN
    if name == "TRPO":
        from sb3_contrib import TRPO
        return TRPO
    raise ValueError(f"Unknown algorithm: {name}")

def main():
    print("4: entering main", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_path", type=str, required=True)
    parser.add_argument("--teacher_algo", type=str, required=True)
    parser.add_argument("--environment", type=str, required=True)
    parser.add_argument("--n_iter", type=int, default=4)
    parser.add_argument("--total_timesteps", type=int, default=120)
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    print("5: parsed args", flush=True)
    print(args, flush=True)

    warnings.filterwarnings("ignore", message="You are trying to run PPO on the GPU")
    warnings.filterwarnings("ignore", message="Note: it looks like you are running in Jupyter")

    os.makedirs(args.output_dir, exist_ok=True)
    print("6: output dir created", flush=True)

    teacher_algo = get_algo(args.teacher_algo)
    print("7: got teacher algo", flush=True)

    print("8: calling train_spid", flush=True)
    rewards, best_policy, wrapper = train_spid(
        args.teacher_path,
        teacher_algo,
        "",
        args.environment,
        n_iter=args.n_iter,
        total_timesteps=args.total_timesteps,
        verbose=args.verbose,
    )
    print("9: train_spid returned", flush=True)

    rewards_py = [float(x) for x in rewards]

    # save rewards
    with open(os.path.join(args.output_dir, "rewards.json"), "w") as f:
        json.dump(rewards_py, f, indent=2)

    # save best PySR model
    joblib.dump(best_policy, os.path.join(args.output_dir, "best_policy.joblib"))

    # save wrapper
    joblib.dump(wrapper, os.path.join(args.output_dir, "wrapper.joblib"))

    # save equations if present
    if hasattr(best_policy, "equations_") and best_policy.equations_ is not None:
        best_policy.equations_.to_csv(
            os.path.join(args.output_dir, "equations.csv"),
            index=False
        )

    # save a reward plot
    plt.figure()
    plt.plot(range(len(rewards_py)), rewards_py, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title("SPID rewards")
    plt.savefig(os.path.join(args.output_dir, "rewards.png"), bbox_inches="tight")
    plt.close()

    print("Saved files:", os.listdir(args.output_dir), flush=True)
    print("10: saved rewards", flush=True)
    print(rewards, flush=True)

if __name__ == "__main__":
    main()