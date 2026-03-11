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
```
