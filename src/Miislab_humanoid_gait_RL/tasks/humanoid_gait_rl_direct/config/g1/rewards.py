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
    stage1_reward_phase = "step"  # "step": learn alternating contacts; "walk": track velocity and heading
    stage1_weights: dict[str, float] = {
        "termination": -200.0,
        "track_lin_vel_xy_exp": 2.0,
        "track_ang_vel_z_exp": 0.2,
        "forward_progress": 2.0,
        "straight_progress": 3.0,
        "backward_motion": -2.0,
        "heading_error_l2": -0.25,
        "lateral_drift_l2": -0.5,
        "lateral_velocity_l2": -0.5,
        "idle": -0.6,
        "flat_orientation_l2": -4.0,
        "tilt_growth_l2": -0.5,
        "lin_vel_z_l2": -0.5,
        "ang_vel_xy_l2": -0.2,
        "base_height_l2": -0.5,
        "feet_air_time": 0.0,  # retained log key; replaced by contact-history-driven step_clearance
        "step_clearance": 1.0,
        "first_touchdown": 0.2,  # per event: raw values divide by step_dt before the environment multiplies it
        "feet_alternation": 0.3,  # per event, independent of control frequency
        "no_ground_contact": -2.0,
        "feet_slide": -1.0,
        "upright_support": 0.5,
        "feet_flat": -4.0,
        "feet_stride_l2": -0.5,
        "feet_reach_l2": -0.5,
        "feet_clearance_l2": -0.5,
        "prolonged_swing": -0.5,
        "action_rate_l2": -0.005,
        "dof_torques_l2": -1.5e-7,
        "dof_acc_l2": -1.25e-7,
        "dof_pos_limits": -1.0,
        "joint_deviation_hip": -0.1,
        "joint_deviation_arms": -0.02,
        "joint_deviation_fingers": -0.005,
        "joint_deviation_torso": -0.5,
    }
    stage1_tracking_std = 0.35  # [m/s]
    stage1_yaw_tracking_std = 0.5  # [rad/s]
    stage1_heading_scale = 0.5  # [rad], absolute heading matters before the robot turns sideways
    stage1_heading_penalty_max = 4.0
    stage1_posture_std = 0.35  # projected-gravity XY norm; suppress motion bonuses well before the 60-degree reset
    stage1_startup_time = 0.60  # [s], ramp speed reward / idle penalty while first establishing support
    stage1_step_height = 0.04  # [m], full clearance reward; excess lift is penalized separately
    stage1_walk_step_gain = 0.25  # reduce clearance assistance after switching to walking
    stage1_step_min_up_z = 0.7660444431  # cos(40 degrees): broader safety gate while discovering the first step
    stage1_support_time_limit = 0.60  # [s], decay assistance if the same foot keeps supporting
    stage1_support_time_stop = 1.20  # [s], invalid contact chatter cannot refresh the supporting foot's timer
    stage1_straight_lateral_std = 0.20  # [m/s], lateral drift reduces the highest-tier straight-walking bonus
    stage1_lateral_scale = 0.3  # [m]
    stage1_lateral_penalty_max = 4.0
    stage1_idle_speed = 0.2  # [m/s]
    stage1_height_deadband = 0.03  # [m]
    stage1_height_scale = 0.1  # [m]
    stage1_swing_time_std = 0.15  # [s]
    stage1_min_air_time = 0.12  # [s], reject contact chatter
    stage1_max_air_time = 0.60  # [s], reject prolonged one-legged balancing
    # Flat-ground Stage 1 only: ankle-roll link +Z is the sole normal (Isaac Lab XYZW quaternions).
    # Tolerate about 10 degrees of foot roll/pitch, including a small heel/toe transition.
    stage1_foot_tilt_deadband = 0.0152  # 1 - cos(10 degrees)
    stage1_max_step_length = 0.35  # [m], fore/aft separation of the two feet, not a full stride cycle
    stage1_step_length_scale = 0.15  # [m], excess separation penalty scale
    stage1_max_foot_forward = 0.25  # [m], each ankle's fore/aft reach relative to the pelvis
    stage1_max_foot_lateral = 0.22  # [m], allows normal stance width but discourages side kicks
    stage1_foot_reach_scale = 0.15  # [m]
    stage1_max_foot_height = 0.08  # [m], no extra clearance bonus; start penalizing excess lift here
    stage1_foot_height_scale = 0.10  # [m]
    stage1_swing_time_limit = 0.40  # [s], begin penalizing a foot held in the air
    # ==================== TODO: END ====================


def compute_rewards(env: HumanoidGaitRlEnv, cfg: RewardsCfg) -> dict[str, torch.Tensor]:
    """Raw value of every reward term.

    Returns ``{term name: tensor of shape (num_envs,)}``; every name needs a weight in :class:`RewardsCfg`.
    Only read from ``env``: writing to it changes the task itself.
    """
    # ==================== TODO: EDIT BELOW (reward terms) ====================
    if env.cfg.stage1:
        if cfg.stage1_reward_phase not in ("step", "walk"):
            raise ValueError(f"stage1_reward_phase must be 'step' or 'walk', got {cfg.stage1_reward_phase!r}")
        if not 0.0 < env.step_dt < float("inf"):
            raise ValueError("step_dt must be finite and positive for touchdown rewards")
        if not 0.0 < cfg.stage1_step_height <= cfg.stage1_max_foot_height < float("inf"):
            raise ValueError("Step heights must satisfy 0 < stage1_step_height <= stage1_max_foot_height")
        if not 0.0 <= cfg.stage1_swing_time_limit < cfg.stage1_max_air_time < float("inf"):
            raise ValueError("Swing decay must satisfy 0 <= stage1_swing_time_limit < stage1_max_air_time")
        if not 0.0 <= cfg.stage1_support_time_limit < cfg.stage1_support_time_stop < float("inf"):
            raise ValueError("Support decay must satisfy 0 <= stage1_support_time_limit < stage1_support_time_stop")
        if not 0.0 <= cfg.stage1_step_min_up_z < 1.0 or not 0.0 <= cfg.stage1_walk_step_gain <= 1.0:
            raise ValueError("Require 0 <= stage1_step_min_up_z < 1 and 0 <= stage1_walk_step_gain <= 1")
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
        walk = cfg.stage1_reward_phase == "walk"
        # Path coordinates follow the existing environment target. Velocity tracking below uses
        # the body yaw frame, matching the commands observed by the policy.
        heading = env._target_heading
        heading_error = torch.remainder(heading - data.heading_w.torch + torch.pi, 2.0 * torch.pi) - torch.pi
        offset = data.root_pos_w.torch[:, :2] - env._stage1_start_pos[:, :2]
        lateral_offset = -heading.sin() * offset[:, 0] + heading.cos() * offset[:, 1]
        world_vel = data.root_lin_vel_w.torch
        forward_vel = heading.cos() * world_vel[:, 0] + heading.sin() * world_vel[:, 1]
        lateral_vel = -heading.sin() * world_vel[:, 0] + heading.cos() * world_vel[:, 1]
        heading_cost = (heading_error / cfg.stage1_heading_scale).square()
        height = data.root_pos_w.torch[:, 2] - env._terrain.env_origins[:, 2]
        height_error = (height - env._stage1_target_height).abs().sub(cfg.stage1_height_deadband).clamp_min(0.0)

        # Flat-sole orientation is a proxy, not contact area. Keep these quality costs additive
        # so an imperfect first step still earns feedback (Isaac Lab quaternions are XYZW).
        feet_quat = data.body_quat_w.torch[:, env._feet_body_ids]
        sole_up_z = 1.0 - 2.0 * feet_quat[..., :2].square().sum(dim=-1)
        foot_tilt = (1.0 - sole_up_z.clamp(-1.0, 1.0) - cfg.stage1_foot_tilt_deadband).clamp_min(0.0)
        feet_flat = (foot_tilt * in_contact).sum(dim=1) / in_contact.sum(dim=1).clamp_min(1)
        feet_pos = data.body_pos_w.torch[:, env._feet_body_ids]
        feet_offset = feet_pos[..., :2] - data.root_pos_w.torch[:, None, :2]
        body_heading = data.heading_w.torch
        feet_forward = body_heading.cos()[:, None] * feet_offset[..., 0]
        feet_forward = feet_forward + body_heading.sin()[:, None] * feet_offset[..., 1]
        feet_lateral = -body_heading.sin()[:, None] * feet_offset[..., 0]
        feet_lateral = feet_lateral + body_heading.cos()[:, None] * feet_offset[..., 1]
        step_length = (feet_forward[:, 0] - feet_forward[:, 1]).abs()
        feet_stride = (
            (step_length - cfg.stage1_max_step_length).clamp_min(0.0) / cfg.stage1_step_length_scale
        ).square()
        forward_reach = (feet_forward.abs() - cfg.stage1_max_foot_forward).clamp_min(0.0)
        lateral_reach = (feet_lateral.abs() - cfg.stage1_max_foot_lateral).clamp_min(0.0)
        feet_reach = (forward_reach.square() + lateral_reach.square()).sum(dim=1) / cfg.stage1_foot_reach_scale**2
        feet_clearance = (
            ((feet_pos[:, 0, 2] - feet_pos[:, 1, 2]).abs() - cfg.stage1_max_foot_height).clamp_min(0.0)
            / cfg.stage1_foot_height_scale
        ).square() * single_stance

        alive = (~env.reset_terminated).float()
        up_z = (-data.projected_gravity_b.torch[:, 2]).clamp(0.0, 1.0)
        upright = up_z.square() * alive
        stable_support = upright * in_contact.any(dim=1)
        motion_gate = stable_support * torch.exp(-flat_orientation / cfg.stage1_posture_std**2)
        step_pose_gate = ((up_z - cfg.stage1_step_min_up_z) / (1.0 - cfg.stage1_step_min_up_z)).clamp(0.0, 1.0)
        step_pose_gate = step_pose_gate * alive
        startup = (env._stage1_episode_steps.float() * env.step_dt / cfg.stage1_startup_time).clamp(0.0, 1.0)
        gravity = data.projected_gravity_b.torch
        omega = data.root_ang_vel_b.torch
        tilt_growth = (omega[:, 1] * gravity[:, 0] - omega[:, 0] * gravity[:, 1]).clamp_min(0.0).square()

        # Read history BEFORE env._get_rewards updates it. Start with the left foot (index 0);
        # after a valid landing, the opposite foot must swing. No clock phase or new state.
        last_foot = env._stage1_last_touchdown
        swing_foot = torch.where(last_foot < 0, 0, 1 - last_foot).unsqueeze(1)
        support_foot = 1 - swing_foot
        swing_air = air_time.gather(1, swing_foot).squeeze(1)
        support_time = contact_time.gather(1, support_foot).squeeze(1)
        supported_swing = ~in_contact.gather(1, swing_foot).squeeze(1)
        supported_swing = supported_swing & in_contact.gather(1, support_foot).squeeze(1)
        clearance = (feet_pos[..., 2].gather(1, swing_foot) - feet_pos[..., 2].gather(1, support_foot)).squeeze(1)
        clearance_score = (clearance / cfg.stage1_step_height).clamp(0.0, 1.0)
        air_window = (cfg.stage1_max_air_time - swing_air) / (cfg.stage1_max_air_time - cfg.stage1_swing_time_limit)
        air_window = air_window.clamp(0.0, 1.0)
        support_window = (cfg.stage1_support_time_stop - support_time) / (
            cfg.stage1_support_time_stop - cfg.stage1_support_time_limit
        )
        support_window = support_window.clamp(0.0, 1.0)
        step_activity = clearance_score * supported_swing * air_window * support_window
        step_clearance = step_activity * step_pose_gate

        # Same landing validity as the environment: one new contact, opposite support, valid
        # air time and no fall. Repeated same-foot landings cannot collect alternation rewards.
        first_contact = env._contact_sensor.compute_first_contact(env.step_dt).torch[:, env._feet_sensor_ids]
        landing_foot = first_contact.long().argmax(dim=1, keepdim=True)
        last_air = contact.last_air_time.torch[:, env._feet_sensor_ids].gather(1, landing_foot).squeeze(1)
        valid_landing = (
            (first_contact.sum(dim=1) == 1)
            & in_contact.gather(1, 1 - landing_foot).squeeze(1)
            & (last_air >= cfg.stage1_min_air_time)
            & (last_air <= cfg.stage1_max_air_time)
            & ~env.reset_terminated
        )
        first_landing = valid_landing & (last_foot < 0)
        alternating = valid_landing & (last_foot >= 0) & (landing_foot.squeeze(1) != last_foot)
        # Dividing events by step_dt cancels the environment's time scaling exactly once.
        first_bonus = first_landing.float() * step_pose_gate / env.step_dt
        alternating_bonus = alternating.float() * step_pose_gate / env.step_dt
        activity = torch.maximum(step_activity, (first_landing | alternating).float())

        target_forward = (forward_vel / env.cfg.stage1_lin_vel_x).clamp(max=1.0)
        body_forward = (vel_yaw[:, 0] / env.cfg.stage1_lin_vel_x).clamp(0.0, 1.0)
        body_backward = (-vel_yaw[:, 0] / env.cfg.stage1_lin_vel_x).clamp_min(0.0)
        straight_quality = torch.exp(-heading_cost - (lateral_vel / cfg.stage1_straight_lateral_std).square())
        turning = (data.root_ang_vel_w.torch[:, 2].abs() / env.cfg.stage1_max_yaw_rate).clamp(max=1.0)
        yaw_activity = torch.maximum(activity, torch.maximum(body_forward, turning))
        idle = (1.0 - activity) * startup
        if walk:
            idle = idle * ((cfg.stage1_idle_speed - vel_yaw[:, 0]) / cfg.stage1_idle_speed).clamp(0.0, 1.0).square()

        rewards.update(
            {
                "track_lin_vel_xy_exp": torch.exp(-lin_vel_error / cfg.stage1_tracking_std**2)
                * motion_gate
                * body_forward
                * startup
                * walk,
                "track_ang_vel_z_exp": torch.exp(-yaw_rate_error / cfg.stage1_yaw_tracking_std**2)
                * motion_gate
                * yaw_activity
                * walk,
                "forward_progress": target_forward * motion_gate * startup * walk,
                "straight_progress": target_forward.clamp_min(0.0) * straight_quality * motion_gate * startup * walk,
                "backward_motion": body_backward * upright * startup * walk,
                "heading_error_l2": heading_cost.clamp(max=cfg.stage1_heading_penalty_max)
                * body_forward
                * motion_gate
                * startup
                * walk,
                "lateral_drift_l2": (lateral_offset / cfg.stage1_lateral_scale)
                .square()
                .clamp(max=cfg.stage1_lateral_penalty_max)
                * walk,
                "lateral_velocity_l2": lateral_vel.square() * walk,
                "idle": idle,
                "flat_orientation_l2": flat_orientation + data.projected_gravity_b.torch[:, 2].clamp_min(0.0).square(),
                "tilt_growth_l2": tilt_growth,
                "lin_vel_z_l2": world_vel[:, 2].square(),
                "base_height_l2": (height_error / cfg.stage1_height_scale).square(),
                "feet_air_time": torch.zeros_like(upright),
                "step_clearance": step_clearance * (cfg.stage1_walk_step_gain if walk else 1.0),
                "first_touchdown": first_bonus,
                "feet_alternation": alternating_bonus,
                "no_ground_contact": (~in_contact.any(dim=1)).float(),
                "upright_support": stable_support,
                "feet_flat": feet_flat,
                "feet_stride_l2": feet_stride,
                "feet_reach_l2": feet_reach,
                "feet_clearance_l2": feet_clearance,
                "prolonged_swing": ((air_time - cfg.stage1_swing_time_limit).clamp_min(0.0) / cfg.stage1_swing_time_std)
                .square()
                .sum(dim=1),
                "feet_slide": (feet_vel.square().sum(dim=-1) * in_contact).sum(dim=1),
                "joint_deviation_arms": joint_deviation(env._arm_joint_ids) / max(len(env._arm_joint_ids), 1),
                "joint_deviation_fingers": joint_deviation(env._finger_joint_ids) / max(len(env._finger_joint_ids), 1),
            }
        )
    return rewards
    # ==================== TODO: END ====================
