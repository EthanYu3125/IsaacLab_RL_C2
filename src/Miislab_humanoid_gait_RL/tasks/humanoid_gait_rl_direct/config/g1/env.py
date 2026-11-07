# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.envs import DirectRLEnv
from isaaclab.utils import index_fill_
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, quat_mul, sample_uniform, wrap_to_pi, yaw_quat

from .rewards import compute_rewards

if TYPE_CHECKING:
    from .env_cfg import HumanoidGaitRlEnvCfg


class HumanoidGaitRlEnv(DirectRLEnv):
    """Unitree G1 learning to follow a forward-velocity and heading command on rough terrain."""

    cfg: HumanoidGaitRlEnvCfg

    def __init__(self, cfg: HumanoidGaitRlEnvCfg, render_mode: str | None = None, **kwargs):
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
        if self.cfg.stage1:
            self._init_stage1_buffers()

    def _init_reset_line_buffers(self) -> None:
        """Keep the spawn line independent of commands that may change during an episode."""
        self._reset_line_start_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self._reset_line_heading = torch.zeros(self.num_envs, device=self.device)
        self._lateral_out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._torso_contact_termination = torch.zeros_like(self._lateral_out_of_bounds)

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
        if self.cfg.stage1:
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
        stiffness = self.cfg.stage1_heading_control_stiffness if self.cfg.stage1 else self.cfg.heading_control_stiffness
        max_rate = self.cfg.stage1_max_yaw_rate if self.cfg.stage1 else 1.0
        self._commands[:, 2] = torch.clamp(heading_error * stiffness, min=-max_rate, max=max_rate)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos.torch

        # Stage 1 holds the reset heading; the original mode periodically chooses a new target.
        if not self.cfg.stage1:
            self._command_time_left -= self.step_dt
            resample_env_ids = (self._command_time_left <= 0.0).nonzero(as_tuple=False).flatten()
            if len(resample_env_ids) > 0:
                self._resample_commands(resample_env_ids)
        self._update_yaw_rate_command()

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target_index(target=self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        data = self._robot.data
        height_scan = (
            self._height_scanner.data.pos_w.torch[:, 2].unsqueeze(1)
            - self._height_scanner.data.ray_hits_w.torch[..., 2]
            - 0.5
        ).clip(-1.0, 1.0)
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
        if self.cfg.stage1:
            touchdown_foot, valid_touchdown = self._prepare_stage1_touchdown()
        reward = torch.zeros(self.num_envs, device=self.device)
        for name, value in compute_rewards(self, self.cfg.rewards).items():
            weight = (
                self.cfg.rewards.stage1_weights.get(name) if self.cfg.stage1 else getattr(self.cfg.rewards, name, None)
            )
            if weight is None:
                location = "RewardsCfg.stage1_weights" if self.cfg.stage1 else "RewardsCfg"
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
        if self.cfg.stage1:
            self._accumulate_stage1_metrics(vel_yaw)
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

    def _accumulate_stage1_metrics(self, vel_yaw: torch.Tensor) -> None:
        """Track full-episode metrics and a second set excluding the first two seconds."""
        data = self._robot.data
        offset = data.root_pos_w.torch[:, :2] - self._stage1_start_pos[:, :2]
        heading = self._target_heading
        in_contact = self._contact_sensor.data.current_contact_time.torch[:, self._feet_sensor_ids] > 0.0
        feet_speed = data.body_lin_vel_w.torch[:, self._feet_body_ids, :2].norm(dim=-1)
        values = {
            "forward_speed": vel_yaw[:, 0],
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
        steady_ids = ids[self._stage1_steady_steps[ids] > 0]
        if len(steady_ids) > 0:
            for name, value in self._stage1_steady_metric_sums.items():
                logs["Metrics/stage1_steady_" + name] = (
                    value[steady_ids] / self._stage1_steady_steps[steady_ids]
                ).mean()
        return logs

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
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
            height = data.root_pos_w.torch[:, 2] - self._terrain.env_origins[:, 2]
            tilted = -data.projected_gravity_b.torch[:, 2] < math.cos(math.radians(self.cfg.stage1_max_tilt_deg))
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
        step_buffer = self._stage1_episode_steps if self.cfg.stage1 else self.episode_length_buf
        episode_steps = step_buffer[env_ids].float().clamp_min(1.0)
        stage1_logs = self._stage1_episode_logs(env_ids) if self.cfg.stage1 else {}
        torso_failures = torch.count_nonzero(self._torso_contact_termination[env_ids])
        lateral_failures = torch.count_nonzero(self._lateral_out_of_bounds[env_ids])
        if self.cfg.terrain_curriculum:
            self._update_terrain_curriculum(env_ids)
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs and not self.cfg.stage1:
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
        # joints: default pose at rest
        joint_pos = self._robot.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self._robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        self._robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        # commands depend on the new heading, so sample them after the root pose is written
        self._reset_line_state(env_ids, root_pose)
        if self.cfg.stage1:
            self._reset_stage1_state(env_ids, root_pose)
        self._resample_commands(env_ids)

        # logging
        extras = dict()
        for key in self._episode_sums:
            extras["Episode_Reward/" + key] = torch.mean(self._episode_sums[key][env_ids]) / self.max_episode_length_s
            index_fill_(self._episode_sums[key], env_ids, 0.0)
        extras["Episode_Termination/torso_contact"] = torso_failures
        extras["Episode_Termination/lateral_out_of_bounds"] = lateral_failures
        # Simultaneous failure and time limit is a failed episode, not a successful completion.
        completed = self.reset_time_outs[env_ids] & ~self.reset_terminated[env_ids]
        extras["Episode_Termination/time_out"] = torch.count_nonzero(completed)
        extras["Metrics/error_vel_xy"] = torch.mean(self._error_vel_xy_sum[env_ids] / episode_steps)
        extras["Metrics/error_vel_yaw"] = torch.mean(self._error_vel_yaw_sum[env_ids] / episode_steps)
        extras["Metrics/terrain_level"] = (
            torch.zeros((), device=self.device) if self.cfg.stage1 else torch.mean(self._terrain.terrain_levels.float())
        )
        extras.update(stage1_logs)
        index_fill_(self._error_vel_xy_sum, env_ids, 0.0)
        index_fill_(self._error_vel_yaw_sum, env_ids, 0.0)
        self.extras["log"] = extras

    def _reset_line_state(self, env_ids: torch.Tensor, root_pose: torch.Tensor) -> None:
        self._reset_line_start_pos[env_ids] = root_pose[:, :2]
        x, y, z, w = root_pose[:, 3:7].unbind(dim=1)
        self._reset_line_heading[env_ids] = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
        self._lateral_out_of_bounds[env_ids] = False
        self._torso_contact_termination[env_ids] = False

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
