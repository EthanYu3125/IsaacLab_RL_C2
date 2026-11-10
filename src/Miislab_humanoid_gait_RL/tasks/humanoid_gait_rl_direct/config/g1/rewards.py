# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward design for the G1 gait task: weights and computation of every reward term live in this file.

Each step, :func:`compute_rewards` returns one raw value per term. The environment multiplies each value
by the weight of the same name in :class:`RewardsCfg` and by ``step_dt``, then sums them into the reward.

Edit only the blocks framed by ``TODO: EDIT BELOW`` and ``TODO: END``.

The default terms and weights follow Isaac Lab's G1 rough-terrain velocity task.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from .env import HumanoidGaitRlEnv


@configclass
class RewardsCfg:
    """Reward weights, one per term name returned by :func:`compute_rewards`, and the term parameters."""

    # ==================== TODO: EDIT BELOW (reward weights and parameters) ====================
    # weights: positive encourages the behavior, negative penalizes it, zero disables the term
    termination = -200.0
    track_lin_vel_xy_exp = 1.0
    track_ang_vel_z_exp = 2.0
    ang_vel_xy_l2 = -0.05
    flat_orientation_l2 = -1.0
    dof_torques_l2 = -1.5e-7
    dof_acc_l2 = -1.25e-7
    action_rate_l2 = -0.005
    dof_pos_limits = -1.0
    joint_deviation_hip = -0.1
    joint_deviation_arms = -0.1
    joint_deviation_fingers = -0.05
    joint_deviation_torso = -0.1
    feet_air_time = 0.25
    feet_slide = -0.1

    # term parameters
    tracking_std = 0.5  # velocity error at which the tracking rewards drop to exp(-1)
    feet_air_time_max = 0.4  # [s], single-stance time rewarded per step
    feet_contact_force_threshold = 1.0  # [N], a foot pushing harder than this counts as touching the ground

    # Stage 1 maximizes signed distance along the reset heading. Keep all 15 default
    # terms available, but disable command tracking and in-place stepping bonuses.
    # The scalar weights above preserve the original stage1=False task.
    stage1_weights: dict[str, float] = {
        "termination": -400.0,
        "track_lin_vel_xy_exp": 0.0,
        "track_ang_vel_z_exp": 0.0,
        "ang_vel_xy_l2": -0.2,
        "flat_orientation_l2": -4.0,
        "dof_torques_l2": -1.5e-7,
        "dof_acc_l2": -1.25e-7,
        "action_rate_l2": -0.005,
        "dof_pos_limits": -1.0,
        "joint_deviation_hip": -0.1,
        "joint_deviation_arms": -0.02,
        "joint_deviation_fingers": -0.005,
        "joint_deviation_torso": -0.1,
        "feet_air_time": 0.0,
        "feet_slide": -0.5,
        "forward_progress": 5.0,
        "lateral_drift_l2": -2.4,  # modestly tighten the path for continuation training
        "heading_error_l2": -2.2,
        "lateral_velocity_l2": -2.2,
        "yaw_rate_error_l2": -0.2,
        "lin_vel_z_l2": -0.5,
    }
    # Retained only for the environment's landing statistics; they do not affect rewards.
    stage1_min_air_time = 0.12  # [s], reject contact chatter
    stage1_max_air_time = 0.60  # [s], reject prolonged one-legged balancing

    # Uphill/downhill motion is necessary on rough terrain. Raw terms remain unchanged.
    stage1_rough_weight_overrides: dict[str, float] = {"lin_vel_z_l2": 0.0}

    # Race keeps the stable straight-walking costs, with recoverable accidents and a timed finish.
    race_weights: dict[str, float] = {
        **stage1_weights,
        "termination": 0.0,
        "lateral_drift_l2": -3.0,
        "heading_error_l2": -3.0,
        "unsafe_state": -25.0,
        "time_cost": -0.1,
        "finish_bonus": 500.0,
    }
    race_fence_contact_force_threshold = 1.0  # [N]
    race_finish_time_discount = 0.8  # 500 * (1 - 0.8*t/T): 500 at t=0, 100 at the deadline

    def stage1_weight(self, name: str, terrain_mode: str) -> float | None:
        """Resolve the effective weight without mutating the configured plane weights."""
        weight = self.stage1_weights.get(name)
        if weight is not None and terrain_mode == "rough":
            return self.stage1_rough_weight_overrides.get(name, weight)
        return weight

    # ==================== TODO: END ====================


def compute_rewards(env: HumanoidGaitRlEnv, cfg: RewardsCfg) -> dict[str, torch.Tensor]:
    """Raw value of every reward term.

    Returns ``{term name: tensor of shape (num_envs,)}``; every name needs a weight in :class:`RewardsCfg`.
    Only read from ``env``: writing to it changes the task itself.
    """
    # ==================== TODO: EDIT BELOW (reward terms) ====================
    data = env._robot.data
    contact = env._contact_sensor.data
    commands = env._commands
    joint_pos = data.joint_pos.torch
    default_joint_pos = data.default_joint_pos.torch
    moving = torch.linalg.norm(commands[:, :2], dim=1) > 0.1

    # velocity tracking: linear in the gravity-aligned (yaw) frame, yaw rate in world frame
    vel_yaw = quat_apply_inverse(yaw_quat(data.root_quat_w.torch), data.root_lin_vel_w.torch[:, :3])
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - vel_yaw[:, :2]), dim=1)
    yaw_rate_error = torch.square(commands[:, 2] - data.root_ang_vel_w.torch[:, 2])

    # base stability
    ang_vel_xy = torch.sum(torch.square(data.root_ang_vel_b.torch[:, :2]), dim=1)
    flat_orientation = torch.sum(torch.square(data.projected_gravity_b.torch[:, :2]), dim=1)

    # joint effort / smoothness
    joint_torques = torch.sum(
        torch.square(env._robot.actuators.applied_effort.torch[:, env._leg_torque_joint_ids]), dim=1
    )
    joint_accel = torch.sum(torch.square(data.joint_acc.torch[:, env._leg_acc_joint_ids]), dim=1)
    action_rate = torch.sum(torch.square(env._actions - env._previous_actions), dim=1)

    # ankle soft limits
    ankle_pos = joint_pos[:, env._ankle_joint_ids]
    ankle_limits = data.soft_joint_pos_limits.torch[:, env._ankle_joint_ids]
    joint_pos_limits = torch.sum(
        -(ankle_pos - ankle_limits[..., 0]).clip(max=0.0) + (ankle_pos - ankle_limits[..., 1]).clip(min=0.0), dim=1
    )

    # deviation from the default pose of joints not essential for walking
    def joint_deviation(joint_ids):
        return torch.sum(torch.abs(joint_pos[:, joint_ids] - default_joint_pos[:, joint_ids]), dim=1)

    # feet: reward single-stance steps
    air_time = contact.current_air_time.torch[:, env._feet_sensor_ids]
    contact_time = contact.current_contact_time.torch[:, env._feet_sensor_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    feet_air_time = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    feet_air_time = torch.clamp(feet_air_time, max=cfg.feet_air_time_max) * moving

    # feet sliding while pressed on the ground
    feet_forces = contact.net_normal_forces_w_history.torch[:, :, env._feet_sensor_ids]
    feet_contact = feet_forces.norm(dim=-1).max(dim=1)[0] > cfg.feet_contact_force_threshold
    feet_vel = data.body_lin_vel_w.torch[:, env._feet_body_ids, :2]
    feet_slide = torch.sum(feet_vel.norm(dim=-1) * feet_contact, dim=1)

    rewards = {
        "termination": env.reset_terminated.float(),
        "track_lin_vel_xy_exp": torch.exp(-lin_vel_error / cfg.tracking_std**2),
        "track_ang_vel_z_exp": torch.exp(-yaw_rate_error / cfg.tracking_std**2),
        "ang_vel_xy_l2": ang_vel_xy,
        "flat_orientation_l2": flat_orientation,
        "dof_torques_l2": joint_torques,
        "dof_acc_l2": joint_accel,
        "action_rate_l2": action_rate,
        "dof_pos_limits": joint_pos_limits,
        "joint_deviation_hip": joint_deviation(env._hip_joint_ids),
        "joint_deviation_arms": joint_deviation(env._arm_joint_ids),
        "joint_deviation_fingers": joint_deviation(env._finger_joint_ids),
        "joint_deviation_torso": joint_deviation(env._torso_joint_ids),
        "feet_air_time": feet_air_time,
        "feet_slide": feet_slide,
    }

    if env.cfg.terrain_mode == "race":
        position = env._race_position()
        progress = torch.minimum(position[:, 0].clamp_min(0), env._race_lengths) - torch.minimum(
            env._race_previous_xy[:, 0].clamp_min(0), env._race_lengths
        )
        elapsed_fraction = (env.episode_length_buf.float() * env.step_dt / env.cfg.race_episode_length_s).clamp(0, 1)
        heading = data.heading_w.torch
        heading_error = torch.atan2(heading.sin(), heading.cos())
        rewards.update(
            {
                "termination": torch.zeros_like(progress),
                "forward_progress": progress / env.step_dt,
                "lateral_drift_l2": position[:, 1].square(),
                "heading_error_l2": heading_error.square(),
                "lateral_velocity_l2": data.root_lin_vel_w.torch[:, 1].square(),
                "yaw_rate_error_l2": yaw_rate_error,
                "lin_vel_z_l2": data.root_lin_vel_b.torch[:, 2].square(),
                "unsafe_state": (env._race_fell | env._race_fence).float(),
                "time_cost": torch.ones_like(progress),
                "finish_bonus": (env._race_finished & ~env._race_terminal_paid).float()
                * (1.0 - cfg.race_finish_time_discount * elapsed_fraction)
                / env.step_dt,
                "joint_deviation_arms": joint_deviation(env._arm_joint_ids) / max(len(env._arm_joint_ids), 1),
                "joint_deviation_fingers": joint_deviation(env._finger_joint_ids) / max(len(env._finger_joint_ids), 1),
            }
        )
    elif env.cfg.stage1:
        # The direction and origin are captured at reset; turning does not rotate this line.
        heading = env._target_heading
        offset = data.root_pos_w.torch[:, :2] - env._stage1_start_pos[:, :2]
        lateral_offset = -heading.sin() * offset[:, 0] + heading.cos() * offset[:, 1]
        heading_error = torch.remainder(heading - data.heading_w.torch + torch.pi, 2.0 * torch.pi) - torch.pi
        world_vel = data.root_lin_vel_w.torch
        # [m/s] * the environment's step_dt approximates signed forward displacement.
        # No speed cap, command normalization, or posture/contact gate: safety costs are additive.
        forward_vel = heading.cos() * world_vel[:, 0] + heading.sin() * world_vel[:, 1]
        # Penalize sideways motion before it accumulates into path deviation.
        lateral_vel = -heading.sin() * world_vel[:, 0] + heading.cos() * world_vel[:, 1]
        rewards.update(
            {
                "forward_progress": forward_vel,
                "lateral_drift_l2": lateral_offset.square(),
                "heading_error_l2": heading_error.square(),
                "lateral_velocity_l2": lateral_vel.square(),
                "yaw_rate_error_l2": yaw_rate_error,
                "lin_vel_z_l2": data.root_lin_vel_b.torch[:, 2].square(),
                "joint_deviation_arms": joint_deviation(env._arm_joint_ids) / max(len(env._arm_joint_ids), 1),
                "joint_deviation_fingers": joint_deviation(env._finger_joint_ids) / max(len(env._finger_joint_ids), 1),
            }
        )
    return rewards
    # ==================== TODO: END ====================
