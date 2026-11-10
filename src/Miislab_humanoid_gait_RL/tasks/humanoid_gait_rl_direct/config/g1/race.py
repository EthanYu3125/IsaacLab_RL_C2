"""Vectorized race lifecycle; state and domain parameters never enter policy observations."""

from __future__ import annotations

import math

import numpy as np
import torch


class RaceLifecycleMixin:
    """Race state, safe finish detection, and nonterminal accident accounting."""

    @property
    def _race_mode(self) -> bool:
        return self.cfg.terrain_mode == "race"

    @property
    def _legacy_stage1(self) -> bool:
        return self.cfg.stage1 and not self._race_mode

    def _init_race_buffers(self) -> None:
        domains = self.cfg.race_domains
        if len(domains) != self.num_envs:
            raise ValueError("Race domain count must match the constructed environment count")
        self._race_lengths, self._race_widths, self._race_frictions = [
            torch.tensor([d[key] for d in domains], device=self.device) for key in ("length", "width", "friction")
        ]
        self._race_track_origins = self._terrain.env_origins[:, :2].clone()
        self._race_previous_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._race_reset_counts = np.zeros(self.num_envs, dtype=np.int64)
        self._race_initial_y = torch.zeros(self.num_envs, device=self.device)
        self._race_initial_yaw = torch.zeros_like(self._race_initial_y)
        for name in ("finished", "fell", "fence", "terminal_paid", "ever_fell", "ever_fence"):
            setattr(self, "_race_" + name, torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        self._race_fall_causes = {
            name: torch.zeros_like(self._race_finished) for name in ("torso_contact", "excessive_tilt", "low_height")
        }
        self._race_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in ("fall_seconds", "fence_seconds", "unsafe_seconds", "lateral", "heading", "steps")
        }
        self._race_max_lateral = torch.zeros(self.num_envs, device=self.device)

    def _race_position(self) -> torch.Tensor:
        return self._robot.data.root_pos_w.torch[:, :2] - self._race_track_origins

    def _sample_race_spawn(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-env reset streams: resetting another world cannot change this world's future samples."""
        samples = np.zeros((len(env_ids), 2), dtype=np.float32)
        if self.cfg.race_randomization_enabled:
            for row, env_id in enumerate(env_ids.cpu().tolist()):
                seed = np.random.SeedSequence([self.cfg.race_seed, 711, env_id, int(self._race_reset_counts[env_id])])
                samples[row] = np.random.default_rng(seed).uniform(-1.0, 1.0, 2)
                self._race_reset_counts[env_id] += 1
        values = torch.as_tensor(samples, device=self.device)
        return (
            values[:, 0] * self._race_widths[env_ids] * self.cfg.race_spawn_y_fraction,
            values[:, 1] * math.radians(self.cfg.race_spawn_yaw_deg),
        )

    def _reset_race_state(self, env_ids: torch.Tensor, root_pose: torch.Tensor | None = None) -> None:
        self._race_track_origins[env_ids] = self._terrain.env_origins[env_ids, :2]
        if root_pose is None:
            position = self._robot.data.root_pos_w.torch[env_ids, :2]
            quat = self._robot.data.root_quat_w.torch[env_ids]
        else:
            position, quat = root_pose[:, :2], root_pose[:, 3:7]
        relative = position - self._race_track_origins[env_ids]
        self._race_previous_xy[env_ids] = relative
        self._race_initial_y[env_ids] = relative[:, 1]
        x, y, z, w = quat.unbind(dim=1)
        self._race_initial_yaw[env_ids] = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
        self._target_heading[env_ids] = 0.0
        for name in ("finished", "fell", "fence", "terminal_paid", "ever_fell", "ever_fence"):
            getattr(self, "_race_" + name)[env_ids] = False
        for value in (*self._race_fall_causes.values(), *self._race_sums.values(), self._race_max_lateral):
            value[env_ids] = 0

    def _filtered_contact(self, sensor, threshold: float, label: str) -> torch.Tensor:
        matrix = sensor.data.normal_force_matrix_w_history
        if matrix is None:
            raise RuntimeError(f"Race {label} contact filter returned no force matrix; refusing unfiltered fallback")
        forces = matrix.torch
        if (
            forces.ndim != 5
            or forces.shape[0] != self.num_envs
            or min(forces.shape[1:4]) == 0
            or forces.shape[-1] != 3
            or (label == "ground" and forces.shape[2] != 1)
        ):
            raise RuntimeError(f"Unexpected {label}-filtered contact shape: {tuple(forces.shape)}")
        return (forces.norm(dim=-1) > threshold).flatten(1).any(dim=1)

    def _race_get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Terminate an episode when the agent falls; fence contact remains a penalty only."""
        torso = self._filtered_contact(self._torso_ground_sensor, self.cfg.torso_contact_threshold, "ground")
        fence = self._filtered_contact(
            self._fence_contact_sensor, self.cfg.rewards.race_fence_contact_force_threshold, "fence"
        )
        data = self._robot.data
        tilted = -data.projected_gravity_b.torch[:, 2] < math.cos(math.radians(self.cfg.stage1_max_tilt_deg))
        height = data.root_pos_w.torch[:, 2] - self._terrain.env_origins[:, 2]
        low = height < data.default_root_pose.torch[:, 2] * self.cfg.stage1_min_height_ratio
        for name, value in (("torso_contact", torso), ("excessive_tilt", tilted), ("low_height", low)):
            self._race_fall_causes[name][:] = value
        self._torso_contact_termination[:] = torso
        self._race_fell[:] = torso | tilted | low
        self._race_fence[:] = fence

        xy, previous = self._race_position(), self._race_previous_xy
        dx = xy[:, 0] - previous[:, 0]
        crossed = (previous[:, 0] < self._race_lengths) & (xy[:, 0] >= self._race_lengths) & (dx > 0)
        alpha = (self._race_lengths - previous[:, 0]) / torch.where(dx > 0, dx, torch.ones_like(dx))
        crossing_y = previous[:, 1] + alpha * (xy[:, 1] - previous[:, 1])
        crossed &= crossing_y.abs() <= self._race_widths / 2
        crossed &= ~self._race_fell & ~fence & (self.episode_length_buf <= self.max_episode_length)
        self._race_finished |= crossed
        terminated = self._race_finished | self._race_fell
        truncated = (self.episode_length_buf >= self.max_episode_length) & ~terminated
        return terminated, truncated

    def _race_record_step(self) -> None:
        active = ~self._race_terminal_paid
        self._race_ever_fell |= self._race_fell & active
        self._race_ever_fence |= self._race_fence & active
        lateral = self._race_position()[:, 1].abs()
        heading = self._robot.data.heading_w.torch
        heading = torch.atan2(heading.sin(), heading.cos()).abs()
        for key, value in (
            ("fall_seconds", self._race_fell.float() * self.step_dt),
            ("fence_seconds", self._race_fence.float() * self.step_dt),
            ("unsafe_seconds", (self._race_fell | self._race_fence).float() * self.step_dt),
            ("lateral", lateral),
            ("heading", heading),
            ("steps", torch.ones_like(lateral)),
        ):
            self._race_sums[key] += value * active
        self._race_max_lateral[:] = torch.maximum(self._race_max_lateral, lateral * active)
        self._race_previous_xy[:] = self._race_position()
        self._race_terminal_paid |= self._race_finished  # Falling never consumes the finish reward.

    def _race_episode_logs(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        completed = (self._race_finished | self._race_fell | self.reset_time_outs) & (self.episode_length_buf > 0)
        ids = env_ids[completed[env_ids]]
        if len(ids) == 0:
            return {}
        seconds = self.episode_length_buf[ids].float() * self.step_dt
        distance = torch.minimum(self._race_position()[ids, 0].clamp_min(0), self._race_lengths[ids])
        success = self._race_finished[ids]
        steps = self._race_sums["steps"][ids].clamp_min(1)
        records = {
            "env_ids": ids,
            "success": success,
            "fell": self._race_ever_fell[ids],
            "fence_contact": self._race_ever_fence[ids],
            "timeout": self.reset_time_outs[ids] & ~success,
            "seconds": seconds,
            "distance": distance,
            "completion_fraction": distance / self._race_lengths[ids],
            "track_length": self._race_lengths[ids],
            "track_width": self._race_widths[ids],
            "floor_friction": self._race_frictions[ids],
            "initial_y": self._race_initial_y[ids],
            "initial_yaw": self._race_initial_yaw[ids],
            "fall_seconds": self._race_sums["fall_seconds"][ids],
            "fence_seconds": self._race_sums["fence_seconds"][ids],
            "unsafe_seconds": self._race_sums["unsafe_seconds"][ids],
            "mean_lateral_deviation": self._race_sums["lateral"][ids] / steps,
            "mean_heading_error": self._race_sums["heading"][ids] / steps,
            "max_lateral_deviation": self._race_max_lateral[ids],
        }
        self.extras["race_episode"] = {key: value.clone() for key, value in records.items()}
        logs = {
            "Episode_Race/episodes": torch.tensor(len(ids), device=self.device),
            "Episode_Race/success_rate": success.float().mean(),
            "Episode_Race/fall_rate": records["fell"].float().mean(),
            "Episode_Race/fence_contact_rate": records["fence_contact"].float().mean(),
            "Episode_Race/timeout_rate": records["timeout"].float().mean(),
            "Episode_Race/length_s": seconds.mean(),
            "Metrics/race_forward_distance": distance.mean(),
            "Metrics/race_average_speed": (distance / seconds).mean(),
            "Episode_Termination/finish": success.sum(),
        }
        for key in (
            "completion_fraction",
            "fall_seconds",
            "fence_seconds",
            "unsafe_seconds",
            "mean_lateral_deviation",
            "mean_heading_error",
            "max_lateral_deviation",
        ):
            logs["Metrics/race_" + key] = records[key].mean()
        if success.any():
            logs["Episode_Race/finish_time_s"] = seconds[success].mean()
        return logs
