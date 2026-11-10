# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.envs import DirectRLEnv
from isaaclab.utils import index_fill_
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, quat_mul, sample_uniform, wrap_to_pi, yaw_quat

from .race import RaceLifecycleMixin
from .race_domain import apply_race_floor_friction
from .rewards import compute_rewards

if TYPE_CHECKING:
    from .env_cfg import HumanoidGaitRlEnvCfg


class HumanoidGaitRlEnv(RaceLifecycleMixin, DirectRLEnv):
    """Unitree G1 learning to follow a forward-velocity and heading command on rough terrain."""

    cfg: HumanoidGaitRlEnvCfg

    def __init__(self, cfg: HumanoidGaitRlEnvCfg, render_mode: str | None = None, **kwargs):
        # Resolve random/unspecified seeds before drawing construction-time domains.
        if cfg.terrain_mode == "race" and (cfg.seed is None or cfg.seed < 0):
            cfg.seed = self.seed(42 if cfg.seed is None else cfg.seed)
        cfg.configure_stage1()
        super().__init__(cfg, render_mode, **kwargs)

        self._robot, self._contact_sensor, self._height_scanner, self._terrain = [
            self.scene[name] for name in ("robot", "contact_forces", "height_scanner", "terrain")
        ]

        # actions
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)

        # commands: [lin_vel_x, lin_vel_y, ang_vel_z]; ang_vel_z is derived from the heading error every step
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_heading = torch.zeros(self.num_envs, device=self.device)
        self._command_time_left = torch.zeros(self.num_envs, device=self.device)

        # joint groups used by the rewards
        self._leg_torque_joint_ids, _ = self._robot.find_joints([".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])
        self._leg_acc_joint_ids, _ = self._robot.find_joints([".*_hip_.*", ".*_knee_joint"])
        self._ankle_joint_ids, _ = self._robot.find_joints([".*_ankle_pitch_joint", ".*_ankle_roll_joint"])
        self._hip_joint_ids, _ = self._robot.find_joints([".*_hip_yaw_joint", ".*_hip_roll_joint"])
        self._arm_joint_ids, _ = self._robot.find_joints(
            [
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_roll_joint",
            ]
        )
        self._finger_joint_ids, _ = self._robot.find_joints([".*_(zero|one|two|three|four|five|six)_joint"])
        self._torso_joint_ids, _ = self._robot.find_joints(["torso_joint"])

        # bodies: contact sensor indices and the matching articulation body indices, feet ordered [left, right]
        self._feet_sensor_ids, _ = self._contact_sensor.find_sensors(".*_ankle_roll_link")
        self._torso_sensor_ids, _ = self._contact_sensor.find_sensors("torso_link")
        self._feet_body_ids, _ = self._robot.find_bodies(".*_ankle_roll_link")

        # logging: per-episode sum of each reward term, created the first time a term is seen
        self._episode_sums: dict[str, torch.Tensor] = {}
        # per-episode velocity tracking error, summed over steps and averaged at reset
        self._error_vel_xy_sum = torch.zeros(self.num_envs, device=self.device)
        self._error_vel_yaw_sum = torch.zeros(self.num_envs, device=self.device)

        self._init_reset_line_buffers()
        if self._race_mode:
            self._torso_ground_sensor = self.scene["torso_ground_contact"]
            self._fence_contact_sensor = self.scene["fence_contact"]
            self._init_race_buffers()
            plan = self.sim.get_clone_plan()
            if plan is None:
                raise RuntimeError("Race domain allocation requires an Isaac Lab clone plan")
            layout = plan.topology.world_prototype_layout
            layout = layout.numpy() if hasattr(layout, "numpy") else layout
            expected = list(range(self.num_envs)) if cfg.race_randomization_enabled else [0] * self.num_envs
            if list(layout) != expected:
                raise RuntimeError("Race asset clone order does not match sampled domains")
            self._race_floor_readback = apply_race_floor_friction(self.sim.physics_manager, self._race_frictions)
            # Upstream dumps its config before gym.make; preserve the effective late-applied race settings too.
            if getattr(self.cfg, "log_dir", None):
                log_dir = Path(self.cfg.log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                settings = {
                    "terrain_mode": "race",
                    "race_track_length": self.cfg.race_track_length,
                    "race_track_width": self.cfg.race_track_width,
                    "episode_length_s": self.cfg.episode_length_s,
                    "env_spacing": self.cfg.scene.env_spacing,
                    "use_mujoco_contacts": self.cfg.sim.physics.solver_cfg.use_mujoco_contacts,
                    "track_usd": self.cfg.scene.race_track.spawn.usd_path,
                    "race_weights": self.cfg.rewards.race_weights,
                    "finish_time_discount": self.cfg.rewards.race_finish_time_discount,
                    "seed": self.cfg.race_seed,
                    "randomization_enabled": self.cfg.race_randomization_enabled,
                    "spawn_y_fraction": self.cfg.race_spawn_y_fraction,
                    "spawn_yaw_deg": self.cfg.race_spawn_yaw_deg,
                    "size_ratio": self.cfg.race_track_size_ratio,
                    "friction_ratio": self.cfg.race_friction_ratio,
                    "domains": self.cfg.race_domains,
                    "asset_assignment": expected,
                    "floor_readback": self._race_floor_readback,
                }
                (log_dir / "race_config.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        elif self.cfg.stage1:
            self._init_stage1_buffers()

    def _init_reset_line_buffers(self) -> None:
        """Keep the spawn line independent of commands that may change during an episode."""
        self._reset_line_start_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self._reset_line_heading = torch.zeros(self.num_envs, device=self.device)
        self._lateral_out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._torso_contact_termination = torch.zeros_like(self._lateral_out_of_bounds)
        self._height_sample_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._invalid_height_ray_sum = torch.zeros(self.num_envs, device=self.device)
        self._missing_local_height_sum = torch.zeros(self.num_envs, device=self.device)

    def _init_stage1_buffers(self) -> None:
        """Allocate per-environment state; these buffers never enter the policy observation."""
        self._stage1_start_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._stage1_target_height = self._robot.data.default_root_pose.torch[:, 2].clone()
        self._stage1_last_touchdown = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._stage1_alternating_touchdown = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stage1_episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._stage1_steady_steps = torch.zeros_like(self._stage1_episode_steps)
        self._stage1_alternating_steps = torch.zeros(self.num_envs, device=self.device)
        self._stage1_max_lateral_deviation = torch.zeros(self.num_envs, device=self.device)
        self._stage1_metric_sums = {
            key: torch.zeros(self.num_envs, device=self.device)
            for key in ("forward_speed", "lateral_deviation", "heading_error", "foot_slide_speed")
        }
        self._stage1_steady_metric_sums = {
            key: torch.zeros_like(value) for key, value in self._stage1_metric_sums.items()
        }
        self._stage1_termination_causes = {
            key: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for key in ("torso_contact", "excessive_tilt", "low_height", "lateral_out_of_bounds")
        }

    def _resample_commands(self, env_ids: torch.Tensor):
        if self.cfg.stage1 or self._race_mode:
            # The target heading was captured from the reset pose and stays fixed for the episode.
            self._commands[env_ids, 0] = self.cfg.stage1_lin_vel_x
            self._commands[env_ids, 1] = 0.0
            self._commands[env_ids, 2] = 0.0
            return
        num = len(env_ids)
        if self.cfg.is_play:
            self._commands[env_ids, 0] = self.cfg.play_lin_vel_x
        else:
            self._commands[env_ids, 0] = sample_uniform(*self.cfg.lin_vel_x_range, (num,), self.device)
        self._commands[env_ids, 1] = 0.0
        self._target_heading[env_ids] = sample_uniform(-math.pi, math.pi, (num,), self.device)
        self._command_time_left[env_ids] = self.cfg.resampling_command_time
        self._update_yaw_rate_command()

    def _update_yaw_rate_command(self):
        heading_error = wrap_to_pi(self._target_heading - self._robot.data.heading_w.torch)
        straight = self.cfg.stage1 or self._race_mode
        stiffness = self.cfg.stage1_heading_control_stiffness if straight else self.cfg.heading_control_stiffness
        max_rate = self.cfg.stage1_max_yaw_rate if straight else 1.0
        self._commands[:, 2] = torch.clamp(heading_error * stiffness, min=-max_rate, max=max_rate)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.extras.pop("race_episode", None)
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos.torch

        # Stage 1 holds the reset heading; the original mode periodically chooses a new target.
        if not (self.cfg.stage1 or self._race_mode):
            self._command_time_left -= self.step_dt
            resample_env_ids = (self._command_time_left <= 0.0).nonzero(as_tuple=False).flatten()
            if len(resample_env_ids) > 0:
                self._resample_commands(resample_env_ids)
        self._update_yaw_rate_command()

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target_index(target=self._processed_actions)

    def _height_measurements(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return finite observations, nearest local ground z, its validity, and invalid-ray fractions.

        The scanner's pos_w is its attachment position; the 20 m offset is baked into ray starts.
        Keep all sensor tensors read-only. A missing local hit must not become a false low-height fall.
        """
        hits = self._height_scanner.data.ray_hits_w.torch
        scan_z = self._height_scanner.data.pos_w.torch[:, 2]
        finite_hits = torch.isfinite(hits).all(dim=-1)
        valid_scan = finite_hits & torch.isfinite(scan_z).unsqueeze(1)
        raw_scan = scan_z.unsqueeze(1) - hits[..., 2] - 0.5
        height_scan = torch.where(valid_scan, raw_scan, -1.0).clamp(-1.0, 1.0)
        root_xy = self._robot.data.root_pos_w.torch[:, :2]
        distance_sq = (hits[..., :2] - root_xy.unsqueeze(1)).square().sum(dim=-1)
        distance_sq = torch.where(finite_hits & torch.isfinite(distance_sq), distance_sq, torch.inf)
        nearest_distance, nearest_index = distance_sq.min(dim=1)
        local_valid = nearest_distance.sqrt() <= 0.2
        nearest_z = hits[..., 2].gather(1, nearest_index.unsqueeze(1)).squeeze(1)
        ground_z = torch.where(local_valid, nearest_z, 0.0)
        return height_scan, ground_z, local_valid, (~valid_scan).float().mean(dim=1)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        data = self._robot.data
        height_scan, _, _, _ = self._height_measurements()
        obs = torch.cat(
            (
                data.root_com_lin_vel_b.torch,  # 3
                data.root_com_ang_vel_b.torch,  # 3
                data.projected_gravity_b.torch,  # 3
                self._commands,  # 3
                data.joint_pos.torch - data.default_joint_pos.torch,  # 37
                data.joint_vel.torch - data.default_joint_vel.torch,  # 37
                self._actions,  # 37
                height_scan,  # 187
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        if self._legacy_stage1:
            touchdown_foot, valid_touchdown = self._prepare_stage1_touchdown()
        reward = torch.zeros(self.num_envs, device=self.device)
        for name, value in compute_rewards(self, self.cfg.rewards).items():
            if self._race_mode:
                weight = self.cfg.rewards.race_weights.get(name)
            elif self.cfg.stage1:
                weight = self.cfg.rewards.stage1_weight(name, self.cfg.terrain_mode)
            else:
                weight = getattr(self.cfg.rewards, name, None)
            if weight is None:
                location = "RewardsCfg.stage1_weights" if self.cfg.stage1 else "RewardsCfg"
                if self._race_mode:
                    location = "RewardsCfg.race_weights"
                raise KeyError(f"Reward term '{name}' has no weight in {location} in rewards.py")
            value = value * weight * self.step_dt
            if name not in self._episode_sums:
                self._episode_sums[name] = torch.zeros(self.num_envs, device=self.device)
            self._episode_sums[name] += value
            reward += value

        # tracking errors for logging only, they do not enter the reward
        data = self._robot.data
        vel_yaw = quat_apply_inverse(yaw_quat(data.root_quat_w.torch), data.root_lin_vel_w.torch[:, :3])
        self._error_vel_xy_sum += torch.linalg.norm(self._commands[:, :2] - vel_yaw[:, :2], dim=1)
        self._error_vel_yaw_sum += torch.abs(self._commands[:, 2] - data.root_ang_vel_w.torch[:, 2])
        _, _, local_valid, invalid_rays = self._height_measurements()
        self._height_sample_steps += 1
        self._invalid_height_ray_sum += invalid_rays
        if self.cfg.terrain_mode == "rough":
            self._missing_local_height_sum += (~local_valid).float()
        if self._race_mode:
            self._race_record_step()
        elif self.cfg.stage1:
            self._accumulate_stage1_metrics()
            # Update history only after the reward has compared this event with the previous landing.
            self._stage1_last_touchdown[valid_touchdown] = touchdown_foot[valid_touchdown]
        return reward

    def _prepare_stage1_touchdown(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Recognize a single supported landing, rejecting chatter, hops and simultaneous landings."""
        contact = self._contact_sensor.data
        first = self._contact_sensor.compute_first_contact(self.step_dt).torch[:, self._feet_sensor_ids]
        in_contact = contact.current_contact_time.torch[:, self._feet_sensor_ids] > 0.0
        last_air_time = contact.last_air_time.torch[:, self._feet_sensor_ids]
        foot = first.long().argmax(dim=1)
        air_time = last_air_time.gather(1, foot.unsqueeze(1)).squeeze(1)
        other_support = in_contact.gather(1, (1 - foot).unsqueeze(1)).squeeze(1)
        valid = (
            (first.sum(dim=1) == 1)
            & other_support
            & (air_time >= self.cfg.rewards.stage1_min_air_time)
            & (air_time <= self.cfg.rewards.stage1_max_air_time)
            & ~self.reset_terminated
        )
        self._stage1_alternating_touchdown[:] = (
            valid & (self._stage1_last_touchdown >= 0) & (foot != self._stage1_last_touchdown)
        )
        return foot, valid

    def _accumulate_stage1_metrics(self) -> None:
        """Track full-episode metrics and a second set excluding the first two seconds."""
        data = self._robot.data
        offset = data.root_pos_w.torch[:, :2] - self._stage1_start_pos[:, :2]
        heading = self._target_heading
        in_contact = self._contact_sensor.data.current_contact_time.torch[:, self._feet_sensor_ids] > 0.0
        feet_speed = data.body_lin_vel_w.torch[:, self._feet_body_ids, :2].norm(dim=-1)
        values = {
            "forward_speed": heading.cos() * data.root_lin_vel_w.torch[:, 0]
            + heading.sin() * data.root_lin_vel_w.torch[:, 1],
            "lateral_deviation": (-heading.sin() * offset[:, 0] + heading.cos() * offset[:, 1]).abs(),
            "heading_error": wrap_to_pi(heading - data.heading_w.torch).abs(),
            "foot_slide_speed": (feet_speed * in_contact).sum(dim=1) / in_contact.sum(dim=1).clamp_min(1),
        }
        self._stage1_episode_steps += 1
        steady = self._stage1_episode_steps > round(self.cfg.stage1_metrics_warmup_s / self.step_dt)
        self._stage1_steady_steps += steady.long()
        self._stage1_alternating_steps += self._stage1_alternating_touchdown.float()
        self._stage1_max_lateral_deviation[:] = torch.maximum(
            self._stage1_max_lateral_deviation, values["lateral_deviation"]
        )
        for name, value in values.items():
            self._stage1_metric_sums[name] += value
            self._stage1_steady_metric_sums[name] += value * steady

    def _stage1_episode_logs(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Summarize completed episodes before reset; heading errors are in radians."""
        ids = env_ids[self._stage1_episode_steps[env_ids] > 0]
        if len(ids) == 0:
            return {}
        steps = self._stage1_episode_steps[ids].float()
        offset = self._robot.data.root_pos_w.torch[ids, :2] - self._stage1_start_pos[ids, :2]
        heading = self._target_heading[ids]
        forward_displacement = heading.cos() * offset[:, 0] + heading.sin() * offset[:, 1]
        logs = {
            "Episode_Stage1/length_s": (steps * self.step_dt).mean(),
            "Episode_Stage1/survival_rate": (~self.reset_terminated[ids]).float().mean(),
            "Episode_Stage1/alternating_steps": self._stage1_alternating_steps[ids].mean(),
            "Episode_Stage1/lateral_out_of_bounds_rate": self._stage1_termination_causes["lateral_out_of_bounds"][ids]
            .float()
            .mean(),
            "Metrics/stage1_forward_displacement": forward_displacement.mean(),
            "Metrics/stage1_max_lateral_deviation": self._stage1_max_lateral_deviation[ids].mean(),
        }
        for name, value in self._stage1_termination_causes.items():
            logs["Episode_Termination/" + name] = torch.count_nonzero(value[ids])
        for name, value in self._stage1_metric_sums.items():
            logs["Metrics/stage1_" + name] = (value[ids] / steps).mean()
        if self.cfg.terrain_mode == "rough":
            logs.update(self._terrain_episode_logs(ids, forward_displacement))
        steady_ids = ids[self._stage1_steady_steps[ids] > 0]
        if len(steady_ids) > 0:
            for name, value in self._stage1_steady_metric_sums.items():
                logs["Metrics/stage1_steady_" + name] = (
                    value[steady_ids] / self._stage1_steady_steps[steady_ids]
                ).mean()
        return logs

    def _terrain_episode_logs(self, ids: torch.Tensor, displacement: torch.Tensor) -> dict[str, torch.Tensor]:
        """Group finished episodes by sub-terrain, before the curriculum changes their origins/levels."""
        generator = self.cfg.scene.terrain.terrain_generator
        # Mirror TerrainGenerator's sorted-column assignment (including its boundary epsilon).
        column_fraction = self._terrain.terrain_types[ids].double() / generator.num_cols + 0.001
        total = sum(sub.proportion for sub in generator.sub_terrains.values())
        lower, logs = 0.0, {}
        for name, sub in generator.sub_terrains.items():
            upper = lower + sub.proportion / total
            selected = (column_fraction >= lower) & (column_fraction < upper)
            lower = upper
            if not selected.any():
                continue
            group = ids[selected]
            steps = self._stage1_episode_steps[group].float()
            prefix = f"Terrain/{name}/"
            logs[prefix + "episodes"] = selected.sum()
            logs[prefix + "forward_displacement"] = displacement[selected].mean()
            logs[prefix + "length_s"] = (steps * self.step_dt).mean()
            logs[prefix + "lateral_deviation"] = (self._stage1_metric_sums["lateral_deviation"][group] / steps).mean()
            logs[prefix + "max_lateral_deviation"] = self._stage1_max_lateral_deviation[group].mean()
            logs[prefix + "failure_rate"] = self.reset_terminated[group].float().mean()
            logs[prefix + "terrain_level"] = self._terrain.terrain_levels[group].float().mean()
        return logs

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._race_mode:
            return self._race_get_dones()
        time_out = self.episode_length_buf >= self.max_episode_length - (0 if self.cfg.stage1 else 1)
        # the torso touching anything counts as a fall
        torso_forces = self._contact_sensor.data.net_normal_forces_w_history.torch[:, :, self._torso_sensor_ids]
        died = torch.any(
            torch.max(torch.norm(torso_forces, dim=-1), dim=1)[0] > self.cfg.torso_contact_threshold, dim=1
        )
        self._torso_contact_termination[:] = died
        data = self._robot.data
        start_pos = self._stage1_start_pos[:, :2] if self.cfg.stage1 else self._reset_line_start_pos
        heading = self._target_heading if self.cfg.stage1 else self._reset_line_heading
        offset = data.root_pos_w.torch[:, :2] - start_pos
        lateral_offset = -heading.sin() * offset[:, 0] + heading.cos() * offset[:, 1]
        self._lateral_out_of_bounds[:] = (
            lateral_offset.abs() > self.cfg.stage1_max_lateral_deviation
        ) & self.cfg.stage1_enable_lateral_termination
        if self.cfg.stage1:
            tilted = -data.projected_gravity_b.torch[:, 2] < math.cos(math.radians(self.cfg.stage1_max_tilt_deg))
            if self.cfg.terrain_mode == "rough":
                _, ground_z, local_valid, _ = self._height_measurements()
                height = data.root_pos_w.torch[:, 2] - ground_z
                low = local_valid & (height < self._stage1_target_height * self.cfg.stage1_min_height_ratio)
            else:
                height = data.root_pos_w.torch[:, 2] - self._terrain.env_origins[:, 2]
                low = height < self._stage1_target_height * self.cfg.stage1_min_height_ratio
            self._stage1_termination_causes["torso_contact"][:] = died
            self._stage1_termination_causes["excessive_tilt"][:] = tilted
            self._stage1_termination_causes["low_height"][:] = low
            self._stage1_termination_causes["lateral_out_of_bounds"][:] = self._lateral_out_of_bounds
            died = died | tilted | low
        return died | self._lateral_out_of_bounds, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = wp.to_torch(self._robot._ALL_INDICES)
        # the curriculum reads how far each robot walked, so it must run before the root pose is reset
        step_buffer = self._stage1_episode_steps if self._legacy_stage1 else self.episode_length_buf
        episode_steps = step_buffer[env_ids].float().clamp_min(1.0)
        stage1_logs = self._stage1_episode_logs(env_ids) if self._legacy_stage1 else {}
        race_logs = self._race_episode_logs(env_ids) if self._race_mode else {}
        torso_failures = torch.count_nonzero(self._torso_contact_termination[env_ids])
        lateral_failures = torch.count_nonzero(self._lateral_out_of_bounds[env_ids])
        terrain_level = (
            self._terrain.terrain_levels[env_ids].float().mean()
            if self.cfg.terrain_mode == "rough"
            else torch.zeros((), device=self.device)
        )
        sampled_ids = env_ids[self._height_sample_steps[env_ids] > 0]
        height_logs = {}
        if len(sampled_ids) > 0:
            samples = self._height_sample_steps[sampled_ids].float()
            height_logs = {
                "Metrics/invalid_height_ray_fraction": (self._invalid_height_ray_sum[sampled_ids] / samples).mean(),
                "Metrics/missing_local_height_fraction": (self._missing_local_height_sum[sampled_ids] / samples).mean(),
            }
        if self.cfg.terrain_curriculum:
            self._update_terrain_curriculum(env_ids)
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs and not (self.cfg.stage1 or self._race_mode):
            # spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        index_fill_(self._actions, env_ids, 0.0)
        index_fill_(self._previous_actions, env_ids, 0.0)

        # root pose: random x/y offset, fixed yaw by default, zero velocity
        num = len(env_ids)
        pose_range = self.cfg.reset_pose_range
        root_pose = self._robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self._terrain.env_origins[env_ids]
        root_pose[:, 0] += sample_uniform(*pose_range["x"], (num,), self.device)
        root_pose[:, 1] += sample_uniform(*pose_range["y"], (num,), self.device)
        yaw = sample_uniform(*pose_range["yaw"], (num,), self.device)
        zeros = torch.zeros_like(yaw)
        root_pose[:, 3:7] = quat_mul(root_pose[:, 3:7], quat_from_euler_xyz(zeros, zeros, yaw))
        root_vel = self._robot.data.default_root_vel.torch[env_ids].clone()
        if self._race_mode:
            spawn_y, spawn_yaw = self._sample_race_spawn(env_ids)
            root_pose[:, :2] = self._terrain.env_origins[env_ids, :2]
            root_pose[:, 1] += spawn_y
            root_pose[:, 3:7] = quat_from_euler_xyz(zeros, zeros, spawn_yaw)
            root_vel.zero_()
        # joints: default pose at rest
        joint_pos = self._robot.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self._robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        self._robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        # commands depend on the new heading, so sample them after the root pose is written
        self._reset_line_state(env_ids, root_pose)
        if self._race_mode:
            self._reset_race_state(env_ids, root_pose)
        elif self.cfg.stage1:
            self._reset_stage1_state(env_ids, root_pose)
        self._resample_commands(env_ids)
        if self._race_mode:
            # Use the composed reset pose, not a potentially stale cached simulator heading.
            self._commands[env_ids, 2] = (
                -self._race_initial_yaw[env_ids] * self.cfg.stage1_heading_control_stiffness
            ).clamp(-self.cfg.stage1_max_yaw_rate, self.cfg.stage1_max_yaw_rate)

        # logging
        extras = dict()
        for key in self._episode_sums:
            extras["Episode_Reward/" + key] = torch.mean(self._episode_sums[key][env_ids]) / self.max_episode_length_s
            index_fill_(self._episode_sums[key], env_ids, 0.0)
        if not self._race_mode:
            extras["Episode_Termination/torso_contact"] = torso_failures
            extras["Episode_Termination/lateral_out_of_bounds"] = lateral_failures
        # Simultaneous failure and time limit is a failed episode, not a successful completion.
        completed = self.reset_time_outs[env_ids] & ~self.reset_terminated[env_ids]
        extras["Episode_Termination/time_out"] = torch.count_nonzero(completed)
        extras["Metrics/error_vel_xy"] = torch.mean(self._error_vel_xy_sum[env_ids] / episode_steps)
        extras["Metrics/error_vel_yaw"] = torch.mean(self._error_vel_yaw_sum[env_ids] / episode_steps)
        extras["Metrics/terrain_level"] = terrain_level
        extras.update(stage1_logs)
        extras.update(race_logs)
        extras.update(height_logs)
        index_fill_(self._error_vel_xy_sum, env_ids, 0.0)
        index_fill_(self._error_vel_yaw_sum, env_ids, 0.0)
        self.extras["log"] = extras

    def _reset_line_state(self, env_ids: torch.Tensor, root_pose: torch.Tensor) -> None:
        self._reset_line_start_pos[env_ids] = root_pose[:, :2]
        x, y, z, w = root_pose[:, 3:7].unbind(dim=1)
        self._reset_line_heading[env_ids] = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
        self._lateral_out_of_bounds[env_ids] = False
        self._torso_contact_termination[env_ids] = False
        self._height_sample_steps[env_ids] = 0
        self._invalid_height_ray_sum[env_ids] = 0.0
        self._missing_local_height_sum[env_ids] = 0.0

    def _reset_stage1_state(self, env_ids: torch.Tensor, root_pose: torch.Tensor) -> None:
        """Capture the composed reset quaternion directly, without reading a stale simulator pose."""
        self._stage1_start_pos[env_ids] = root_pose[:, :3]
        self._stage1_target_height[env_ids] = self._robot.data.default_root_pose.torch[env_ids, 2]
        x, y, z, w = root_pose[:, 3:7].unbind(dim=1)  # Isaac Lab quaternions use XYZW.
        self._target_heading[env_ids] = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
        self._stage1_last_touchdown[env_ids] = -1
        self._stage1_alternating_touchdown[env_ids] = False
        self._stage1_episode_steps[env_ids] = 0
        self._stage1_steady_steps[env_ids] = 0
        self._stage1_alternating_steps[env_ids] = 0.0
        self._stage1_max_lateral_deviation[env_ids] = 0.0
        for buffers in (self._stage1_metric_sums, self._stage1_steady_metric_sums, self._stage1_termination_causes):
            for value in buffers.values():
                value[env_ids] = 0

    def _update_terrain_curriculum(self, env_ids: torch.Tensor):
        """Move robots that crossed their terrain tile to harder terrain, and those that barely moved to easier."""
        if self.cfg.terrain_mode != "rough" or not self.cfg.terrain_curriculum or self.cfg.is_play:
            return
        if self.cfg.stage1:
            # Only completed episodes count; explicit/manual resets and initial setup do not change difficulty.
            finished = (self.reset_terminated[env_ids] | self.reset_time_outs[env_ids]) & (
                self._stage1_episode_steps[env_ids] > 0
            )
            env_ids = env_ids[finished]
            if len(env_ids) == 0:
                return
            offset = self._robot.data.root_pos_w.torch[env_ids, :2] - self._stage1_start_pos[env_ids, :2]
            heading = self._target_heading[env_ids]
            distance = heading.cos() * offset[:, 0] + heading.sin() * offset[:, 1]
            failed = self.reset_terminated[env_ids]
            completed = self.reset_time_outs[env_ids] & ~failed
            move_up = completed & (distance > 4.0)
            move_down = failed | (completed & (distance < 2.0))
            self._terrain.update_env_origins(env_ids, move_up, move_down)
            return
        # skip envs that have not run an episode yet (e.g. the initial reset)
        env_ids = env_ids[self.episode_length_buf[env_ids] > 0]
        if len(env_ids) == 0:
            return
        distance = torch.linalg.norm(
            self._robot.data.root_pos_w.torch[env_ids, :2] - self._terrain.env_origins[env_ids, :2], dim=1
        )
        move_up = distance > self._terrain.cfg.terrain_generator.size[0] / 2
        move_down = distance < torch.linalg.norm(self._commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        move_down *= ~move_up
        self._terrain.update_env_origins(env_ids, move_up, move_down)
