

import os
import sys
import argparse

root = os.path.abspath(os.path.join(os.getcwd(), "..", "..", ".."))
sys.path.insert(0, os.path.join(root, "gym-pybullet-drones"))

import socket
print("Host:", socket.gethostname())
print("SLURM job:", os.environ.get("SLURM_JOB_ID"))

import warnings
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

from gmDAGGER_drone import train_spid
from PySRWrapper_drone import PySRPolicy

warnings.filterwarnings("ignore", message="You are trying to run PPO on the GPU")
warnings.filterwarnings("ignore", message="Note: it looks like you are running in Jupyter")

def parse_args():
    parser = argparse.ArgumentParser(description="Run SPID distillation")
    parser.add_argument("--teacher_path",        type=str, required=True)
    parser.add_argument("--save_folder",         type=str, required=True)
    parser.add_argument("--n_iter",              type=int, default=10)
    parser.add_argument("--total_timesteps",     type=int, default=20000)
    parser.add_argument("--n_eval_episodes",     type=int, default=10)
    parser.add_argument("--maxsize",             type=int, default=20)
    parser.add_argument("--maxdepth",            type=int, default=None)
    parser.add_argument("--verbose",             type=int, default=2)
    parser.add_argument("--binary_operators",    type=str, nargs="+", default=["+", "*", "-", "/"])
    parser.add_argument("--unary_operators",     type=str, nargs="+", default=None)
    parser.add_argument("--nested_constraints",  type=str, default=None,
                        help='JSON string e.g. \'{"sin": {"sin": 0, "cos": 0}, "cos": {"sin": 0, "cos": 0}}\'')
    return parser.parse_args()

def main():
    args = parse_args()

    import json
    nested_constraints = json.loads(args.nested_constraints) if args.nested_constraints else None


    DEFAULT_OBS = ObservationType('kin')
    DEFAULT_ACT = ActionType('one_d_rpm')

    class HoverActionShapeWrapper(gym.Wrapper):
        def step(self, action):
            action = np.asarray(action, dtype=np.float32)
            if action.ndim == 1:
                action = action.reshape(1, -1)
            return self.env.step(action)

    environment = lambda: HoverActionShapeWrapper(
        HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, gui=False)
    )

    rewards, best_policy, wrapper, run_dir = train_spid(
        teacher_path=args.teacher_path,
        teacher_model=PPO,
        save_folder_path=args.save_folder,
        save_results=True,
        environment=environment,
        n_iter=args.n_iter,
        total_timesteps=args.total_timesteps,
        n_eval_episodes=args.n_eval_episodes,
        verbose=args.verbose,
        maxsize=args.maxsize,
        maxdepth=args.maxdepth,
        binary_operators=args.binary_operators,
        unary_operators=args.unary_operators,
        nested_constraints=nested_constraints
    )

if __name__ == "__main__":
    main()