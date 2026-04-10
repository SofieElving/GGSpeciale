# GGSpeciale
Symbolic policy distillation for interpretable and trustworthy Reinforcement Learning in real-world applications

```
code/
├── baseline_code/
│   ├── baseline_environments/
│   │   ├── wrappers
│   │   └── ...
│   ├── baseline_runs/
│   │   ├── run_CartPole.py
│   │   ├── run_MountainCar.py
│   │   ├── run_Pendulum.py
│   │   ├── run_Acrobot.py
│   │   ├── run_Swimmer.py
│   │   └── run_Reacher.py
│   └── run_baselines.py
└── SPID/
```

To create environment, do: 
```
conda create -n thesis-env -c conda-forge python=3.10 pip numpy pandas tqdm ipykernel gymnasium shimmy -y
conda activate thesis-env
pip install torch stable-baselines3 sb3-contrib
pip install pysr
pip install "gymnasium[mujoco]"
```


# Installing the PyBullet Drones env.

To do *locally*, simply navigate to where you want the repository to be in the terminal, and do: 
```
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones
pip install -e .
```
I have already cloned the repository to git, so simply navigate to the repo and do:
```
cd gym-pybullet-drones
pip install -e .
```
Lastly, you will need the following dependencies to run the environment 
```
pip install pybullet
pip install matplotlib opencv-python
pip install setuptools
```

