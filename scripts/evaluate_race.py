"""Evaluate a checkpoint on exactly N race episodes; requires the remote Isaac Lab environment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path


def summarize(records: list[dict]) -> dict:
    """Aggregate episodes equally, rather than weighting small reset batches too heavily."""
    count = len(records)
    if count == 0:
        raise ValueError("Cannot summarize an empty evaluation")
    finish_times = [r["seconds"] for r in records if r["success"]]
    return {
        "episodes": count,
        "success_rate": sum(r["success"] for r in records) / count,
        "fall_rate": sum(r["fell"] for r in records) / count,
        "fence_contact_rate": sum(r["fence_contact"] for r in records) / count,
        "timeout_rate": sum(r["timeout"] for r in records) / count,
        "finish_time_mean_s": statistics.mean(finish_times) if finish_times else None,
        "finish_time_median_s": statistics.median(finish_times) if finish_times else None,
        "forward_distance_mean_m": statistics.mean(r["distance"] for r in records),
        "average_speed_mean_m_s": statistics.mean(r["distance"] / r["seconds"] for r in records),
        **{
            key + "_mean": statistics.mean(r[key] for r in records)
            for key in (
                "completion_fraction",
                "mean_lateral_deviation",
                "max_lateral_deviation",
                "mean_heading_error",
                "fall_seconds",
                "fence_seconds",
                "unsafe_seconds",
            )
        },
    }


def collect_records(batch: dict, records: list[dict], counts: list[int], quotas: list[int]) -> None:
    """Limit each environment's contribution so faster environments cannot dominate the sample."""
    values = {key: value.detach().cpu().tolist() for key, value in batch.items()}
    for i, env_id in enumerate(values["env_ids"]):
        if counts[env_id] >= quotas[env_id]:
            continue
        record = {key: value[i] for key, value in values.items() if key != "env_ids"}
        record["env_id"] = env_id
        records.append(record)
        counts[env_id] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("logs/race_evaluation.json"))
    options, remaining = parser.parse_known_args()
    if options.episodes <= 0:
        parser.error("--episodes must be positive")

    # Mirror Isaac Lab 3.0's official skrl playback pipeline and CLI arguments.
    import torch

    from isaaclab.app import launch_simulation
    from isaaclab.utils.seed import configure_seed

    from isaaclab_rl.entrypoints.backends.play_skrl import _parse_args
    from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env, create_isaaclab_env
    from isaaclab_rl.skrl import check_skrl_version, import_skrl_runner, resolve_skrl_agent_cfg_entry_point

    from isaaclab_tasks.utils import resolve_task_config

    import Miislab_humanoid_gait_RL.tasks  # noqa: F401

    args = _parse_args(remaining)
    if args.ml_framework != "torch":
        parser.error("Race evaluation supports the project's PyTorch checkpoints only")
    if not args.checkpoint or not Path(args.checkpoint).is_file():
        parser.error("Pass --checkpoint with an explicit existing local checkpoint path")
    check_skrl_version()
    cfg, agent_cfg = resolve_task_config(
        args.task, resolve_skrl_agent_cfg_entry_point(args.agent, args.algorithm), play_mode=True
    )
    cfg.seed = agent_cfg["seed"] = args.seed if args.seed is not None else 42
    options.output = options.output.resolve()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.log_dir = str(options.output.parent)
    records = []
    with launch_simulation(cfg, args):
        from isaaclab_rl.skrl import SkrlVecEnvWrapper

        apply_env_overrides(args, cfg)
        if cfg.terrain_mode != "race":
            parser.error("Evaluation requires env.terrain_mode=race")
        raw_env = create_isaaclab_env(args.task, cfg, args)
        try:
            task = raw_env.unwrapped
            env = SkrlVecEnvWrapper(raw_env, ml_framework="torch")
            agent_cfg["trainer"]["close_environment_at_exit"] = False
            agent_cfg["agent"]["experiment"]["write_interval"] = 0
            agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
            runner = import_skrl_runner("torch")(env, agent_cfg)
            if args.deterministic:
                configure_seed(cfg.seed, torch_deterministic=True)
            runner.agent.load(str(Path(args.checkpoint).resolve()))
            runner.agent.enable_training_mode(False, apply_to_models=True)
            obs, _ = env.reset()
            states = env.state()
            quotas = [options.episodes // task.num_envs + (i < options.episodes % task.num_envs) for i in range(task.num_envs)]
            counts = [0] * task.num_envs
            max_steps = (max(quotas) + 1) * math.ceil(cfg.race_episode_length_s / task.step_dt)
            with torch.inference_mode():
                for step in range(max_steps):
                    outputs = runner.agent.act(obs, states, timestep=step, timesteps=max_steps)
                    actions = outputs[-1].get("mean_actions", outputs[0])
                    obs, _, _, _, info = env.step(actions)
                    states = env.state()
                    if "race_episode" in info:
                        collect_records(info["race_episode"], records, counts, quotas)
                    if len(records) == options.episodes:
                        break
            if len(records) != options.episodes:
                raise RuntimeError(f"Expected {options.episodes} episodes, received {len(records)}; inspect race resets")
            report = {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "seed": cfg.seed,
                "num_envs": task.num_envs,
                "track_length_m": cfg.race_track_length,
                "track_width_m": cfg.race_track_width,
                "episode_limit_s": cfg.race_episode_length_s,
                "course": "randomized" if cfg.race_randomization_enabled else "fixed_nominal",
                "domains": cfg.race_domains,
                "floor_readback": task._race_floor_readback,
                "policy": "mean_actions",
                "use_mujoco_contacts": cfg.sim.physics.solver_cfg.use_mujoco_contacts,
                **summarize(records),
            }
        finally:
            close_env(raw_env)
    options.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with options.output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(report, indent=2))
    print(f"Saved {options.output} and {options.output.with_suffix('.csv')}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    main()
