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

    # Stage 1 weights are selected by env.stage1 (True by default).
    # Keep the original scalar weights above intact for stage1=False.
    stage1_weights: dict[str, float] = {
        "termination": -200.0,
        "track_lin_vel_xy_exp": 3.0,
        "track_ang_vel_z_exp": 0.5,
        "heading_error_l2": -2.0,
        "lateral_drift_l2": -0.5,
        "lateral_velocity_l2": -0.5,
        "idle": -0.25,
        "flat_orientation_l2": -2.0,
        "lin_vel_z_l2": -0.5,
        "ang_vel_xy_l2": -0.1,
        "base_height_l2": -0.5,
        "feet_air_time": 0.25,
        "feet_alternation": 5.0,
        "no_ground_contact": -1.0,
        "feet_slide": -0.5,
        "action_rate_l2": -0.01,
        "dof_torques_l2": -1.5e-7,
        "dof_acc_l2": -1.25e-7,
        "dof_pos_limits": -1.0,
        "joint_deviation_hip": -0.05,
        "joint_deviation_arms": -0.05,
        "joint_deviation_fingers": -0.05,
        "joint_deviation_torso": -0.1,
    }
    stage1_tracking_std = 0.35  # [m/s]
    stage1_yaw_tracking_std = 0.5  # [rad/s]
    stage1_lateral_scale = 0.3  # [m]
    stage1_lateral_penalty_max = 4.0
    stage1_idle_speed = 0.2  # [m/s]
    stage1_height_deadband = 0.03  # [m]
    stage1_height_scale = 0.1  # [m]
    stage1_swing_time = 0.25  # [s]
    stage1_swing_time_std = 0.15  # [s]
    stage1_min_air_time = 0.12  # [s], reject contact chatter
    stage1_max_air_time = 0.60  # [s], reject prolonged one-legged balancing
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

    if env.cfg.stage1:
        # Geometry is relative to the actual reset pose, not the terrain origin.
        heading = env._target_heading
        heading_error = torch.remainder(heading - data.heading_w.torch + torch.pi, 2.0 * torch.pi) - torch.pi
        offset = data.root_pos_w.torch[:, :2] - env._stage1_start_pos[:, :2]
        lateral_offset = -torch.sin(heading) * offset[:, 0] + torch.cos(heading) * offset[:, 1]
        height = data.root_pos_w.torch[:, 2] - env._terrain.env_origins[:, 2]

        # Only tracking remains positive at rest; auxiliary bonuses require actual forward motion.
        upright = (-data.projected_gravity_b.torch[:, 2]).clamp(0.0, 1.0).square()
        upright = upright * (~env.reset_terminated).float()
        forward = (vel_yaw[:, 0] / env.cfg.stage1_lin_vel_x).clamp(0.0, 1.0)
        gait_gate = upright * forward
        swing_time = torch.where(~in_contact, air_time, 0.0).sum(dim=1)
        swing_reward = torch.exp(-((swing_time - cfg.stage1_swing_time) / cfg.stage1_swing_time_std).square())
        height_error = (height - env._stage1_target_height).abs().sub(cfg.stage1_height_deadband).clamp_min(0.0)

        rewards.update(
            {
                "track_lin_vel_xy_exp": torch.exp(-lin_vel_error / cfg.stage1_tracking_std**2) * upright,
                "track_ang_vel_z_exp": torch.exp(-yaw_rate_error / cfg.stage1_yaw_tracking_std**2) * gait_gate,
                "heading_error_l2": (heading_error / torch.pi).square(),
                "lateral_drift_l2": (lateral_offset / cfg.stage1_lateral_scale)
                .square()
                .clamp(max=cfg.stage1_lateral_penalty_max),
                "lateral_velocity_l2": vel_yaw[:, 1].square(),
                "idle": ((cfg.stage1_idle_speed - vel_yaw[:, 0]) / cfg.stage1_idle_speed).clamp(0.0, 1.0).square(),
                "flat_orientation_l2": flat_orientation + data.projected_gravity_b.torch[:, 2].clamp_min(0.0).square(),
                "lin_vel_z_l2": data.root_lin_vel_w.torch[:, 2].square(),
                "base_height_l2": (height_error / cfg.stage1_height_scale).square(),
                "feet_air_time": swing_reward * single_stance * gait_gate,
                "feet_alternation": env._stage1_alternating_touchdown.float() * gait_gate,
                "no_ground_contact": (~in_contact.any(dim=1)).float(),
                # Contact timers use the current sensor state and the configured force threshold.
                "feet_slide": (feet_vel.square().sum(dim=-1) * in_contact).sum(dim=1),
            }
        )
    return rewards
    # ==================== TODO: END ====================
