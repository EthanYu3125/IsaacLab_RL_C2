# Miislab_humanoid_gait_RL

Train a Unitree G1 humanoid to walk over rough terrain with reinforcement learning (PPO), built on Isaac Lab 3.0.

The registered task is:
- `Miislab-Humanoid-Gait-Rl-G1-Direct-v1`

## Installation

### 0. Prerequisites

- Linux (x86_64) with an NVIDIA GPU and driver
- [uv](https://docs.astral.sh/uv/getting-started/installation/):
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### 1. Clone Isaac Lab 3.0 (clone only, do not install)

```bash
git clone --branch v3.0.0 https://github.com/isaac-sim/IsaacLab.git ~/isaac/IsaacLab-3.0.0
```

> - The directory must be named `IsaacLab-3.0.0` and the version must be `v3.0.0`.
> - Do **not** run `./isaaclab.sh -i` or `uv sync` inside the Isaac Lab directory. Every package is installed into
>   this project's own environment in step 3.

### 2. Clone this project

```bash
mkdir -p ~/codes
git clone https://github.com/Frank0613/Miislab_humanoid_gait_RL.git ~/codes/Miislab_humanoid_gait_RL
```

This project finds Isaac Lab through the relative path `../../isaac/IsaacLab-3.0.0`, so the directories must be laid
out like this:

```
~/
├── isaac/
│   └── IsaacLab-3.0.0/            ← step 1
└── codes/
    └── Miislab_humanoid_gait_RL/   ← step 2 (codes can have any name, but must be exactly one level deep)
```

### 3. Install

```bash
cd ~/codes/Miislab_humanoid_gait_RL
uv sync
```

The first run downloads Isaac Sim, PyTorch and the other dependencies (over 10 GB), so it takes a while. Everything is
installed into the project's `.venv`.

> A warning like `VIRTUAL_ENV=... does not match the project environment path` means another virtual environment is
> active in your shell. It is safe to ignore, or run `deactivate` first.

### 4. Check that the task is found

```bash
export OMNI_KIT_ACCEPT_EULA=YES   # accept the Isaac Sim EULA; without it, the first launch asks and you answer Yes
uv run isaaclab list_envs
```

The table must list `Miislab-Humanoid-Gait-Rl-G1-Direct-v1`.

### 5. Test training

```bash
uv run isaaclab train --rl_library skrl --task Miislab-Humanoid-Gait-Rl-G1-Direct-v1 \
    --num_envs 64 --max_iterations 10
```

It finishes in about a minute with the progress bar at `240/240`. The run creates
`logs/skrl/humanoid_gait_rl_g1/<date-time>_ppo_torch/checkpoints/` containing `best_agent.pt`, which means the
installation works.

## Train and play

```bash
# Full training (4096 environments by default)
uv run isaaclab train --rl_library skrl --task Miislab-Humanoid-Gait-Rl-G1-Direct-v1

# Play the latest run
uv run isaaclab play --rl_library skrl --task Miislab-Humanoid-Gait-Rl-G1-Direct-v1 \
    --num_envs 16 --viz kit --checkpoint latest

# Play a specific checkpoint
uv run isaaclab play --rl_library skrl --task Miislab-Humanoid-Gait-Rl-G1-Direct-v1 \
    --num_envs 16 --viz kit --checkpoint logs/skrl/humanoid_gait_rl_g1/<run>/checkpoints/best_agent.pt

# Training curves
uv run tensorboard --logdir logs/skrl
```

Each run is saved under `logs/skrl/humanoid_gait_rl_g1/<run>/`:

```
<run>/
├── checkpoints/best_agent.pt   ← the checkpoint to submit
├── params/                     ← env and agent configuration used by this run
└── events.out.tfevents.*       ← TensorBoard log
```

## What you can edit

Only edit the blocks framed by `TODO: EDIT BELOW` and `TODO: END` in these two files; leave everything else unchanged.

| File | What you can change |
|---|---|
| `src/Miislab_humanoid_gait_RL/tasks/humanoid_gait_rl_direct/config/g1/rewards.py` | Reward weights and parameters, and how each reward term is computed |
| `src/Miislab_humanoid_gait_RL/tasks/humanoid_gait_rl_direct/config/g1/agents/skrl_ppo_cfg.yaml` | PPO hyperparameters and the training length `timesteps` |

Reward weights can also be overridden from the command line without editing any file, for example:

```bash
uv run isaaclab train --rl_library skrl --task Miislab-Humanoid-Gait-Rl-G1-Direct-v1 \
    env.rewards.flat_orientation_l2=-3.0
```
