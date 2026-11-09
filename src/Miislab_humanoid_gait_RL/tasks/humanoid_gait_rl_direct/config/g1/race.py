"""Vectorized race lifecycle; all state is private to the task, never policy observations."""

from __future__ import annotations

import math

import torch


class RaceLifecycleMixin:
    """Race-specific bookkeeping shared by termination, reward accounting and reset."""

    @property
    def _race_mode(self) -> bool:
        return self.cfg.terrain_mode == "race"

    @property
    def _legacy_stage1(self) -> bool:
        return self.cfg.stage1 and not self._race_mode

    def _init_race_buffers(self) -> None:
        self._race_previous_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._race_finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._race_fell = torch.zeros_like(self._race_finished)
        self._race_terminal_paid = torch.zeros_like(self._race_finished)
        self._race_fall_causes = {
            name: torch.zeros_like(self._race_finished) for name in ("torso_contact", "excessive_tilt", "low_height")
        }

    def _race_position(self) -> torch.Tensor:
        return self._robot.data.root_pos_w.torch[:, :2] - self._reset_line_start_pos

    def _reset_race_state(self, env_ids: torch.Tensor) -> None:
        self._race_previous_xy[env_ids] = 0.0
        self._target_heading[env_ids] = 0.0
        for value in (self._race_finished, self._race_fell, self._race_terminal_paid, *self._race_fall_causes.values()):
            value[env_ids] = False

    def _race_get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Ground-only torso contact and pose failures take precedence over a finish crossing."""
        matrix = self._torso_ground_sensor.data.normal_force_matrix_w_history
        if matrix is None:
            raise RuntimeError("Race ground contact filter returned no force matrix; refusing unfiltered fallback")
        forces = matrix.torch
        if forces.ndim != 5 or forces.shape[0] != self.num_envs or forces.shape[2] != 1 or forces.shape[3] == 0:
            raise RuntimeError(f"Unexpected ground-filtered contact shape: {tuple(forces.shape)}")
        torso = (forces.norm(dim=-1) > self.cfg.torso_contact_threshold).flatten(1).any(dim=1)
        data = self._robot.data
        tilted = -data.projected_gravity_b.torch[:, 2] < math.cos(math.radians(self.cfg.stage1_max_tilt_deg))
        height = data.root_pos_w.torch[:, 2] - self._terrain.env_origins[:, 2]
        low = height < data.default_root_pose.torch[:, 2] * self.cfg.stage1_min_height_ratio
        for name, value in (("torso_contact", torso), ("excessive_tilt", tilted), ("low_height", low)):
            self._race_fall_causes[name][:] = value
        self._torso_contact_termination[:] = torso
        self._race_fell[:] = torso | tilted | low

        xy = self._race_position()
        previous = self._race_previous_xy
        dx = xy[:, 0] - previous[:, 0]
        length = self.cfg.race_track_length
        crossed = (previous[:, 0] < length) & (xy[:, 0] >= length) & (dx > 0)
        # A segment crossing skips neither the line nor its lateral bounds at high speed.
        alpha = (length - previous[:, 0]) / torch.where(dx > 0, dx, torch.ones_like(dx))
        crossing_y = previous[:, 1] + alpha * (xy[:, 1] - previous[:, 1])
        crossed &= crossing_y.abs() <= self.cfg.race_track_width / 2
        self._race_finished[:] = (self._race_finished | crossed) & ~self._race_fell
        terminated = self._race_fell | self._race_finished
        truncated = (self.episode_length_buf >= self.max_episode_length) & ~terminated
        return terminated, truncated

    def _race_episode_logs(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Expose exact per-episode records before teleport, plus scalar TensorBoard aggregates."""
        completed = (self._race_finished | self._race_fell | self.reset_time_outs) & (self.episode_length_buf > 0)
        ids = env_ids[completed[env_ids]]
        if len(ids) == 0:
            return {}
        seconds = self.episode_length_buf[ids].float() * self.step_dt
        distance = self._race_position()[ids, 0].clamp(0, self.cfg.race_track_length)
        success = self._race_finished[ids]
        failed = self._race_fell[ids]
        timed_out = self.reset_time_outs[ids] & ~success & ~failed
        # A remote evaluator can count exactly N episodes without averaging per-reset batches equally.
        self.extras["race_episode"] = {
            "env_ids": ids.clone(),
            "success": success.clone(),
            "fell": failed.clone(),
            "timeout": timed_out.clone(),
            "seconds": seconds.clone(),
            "distance": distance.clone(),
        }
        logs = {
            "Episode_Race/episodes": torch.tensor(len(ids), device=self.device),
            "Episode_Race/success_rate": success.float().mean(),
            "Episode_Race/fall_rate": failed.float().mean(),
            "Episode_Race/timeout_rate": timed_out.float().mean(),
            "Episode_Race/length_s": seconds.mean(),
            "Metrics/race_forward_distance": distance.mean(),
            "Metrics/race_average_speed": (distance / seconds).mean(),
            "Episode_Termination/finish": success.sum(),
        }
        if success.any():
            logs["Episode_Race/finish_time_s"] = seconds[success].mean()
        for name, value in self._race_fall_causes.items():
            logs["Episode_Termination/" + name] = value[ids].sum()
        return logs
