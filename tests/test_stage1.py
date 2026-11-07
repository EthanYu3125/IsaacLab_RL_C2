"""Stage 1 task logic on CPU; simulator boundaries are replaced by explicit test doubles."""

from __future__ import annotations

import math

import pytest
import torch
from stage1_test_support import load_task_modules, make_env, quat_from_euler_xyz

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


@pytest.fixture(scope="module")
def modules():
    with pytest.MonkeyPatch.context() as patch:
        yield load_task_modules(patch)


@pytest.fixture
def env(modules):
    return make_env(modules)


def tensors_in(value, prefix=""):
    """Snapshot tensor state recursively, including tensors accessed through .torch proxies."""
    if isinstance(value, torch.Tensor):
        return {prefix: value.clone()}
    if isinstance(value, dict):
        children = value.items()
    elif isinstance(value, (list, tuple)):
        children = enumerate(value)
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        children = vars(value).items()
    else:
        return {}
    result = {}
    for key, child in children:
        result.update(tensors_in(child, f"{prefix}/{key}"))
    return result


def total_reward(env, terms):
    return (
        sum(env.cfg.rewards.stage1_weight(name, env.cfg.terrain_mode) * term for name, term in terms.items())
        * env.step_dt
    )


def set_landing(env, foot, air_time=0.25, other_support=True):
    contact = env._contact_sensor.data
    contact.current_contact_time.torch[:, :2] = 0.2
    if foot is not None:
        contact.current_contact_time.torch[:, foot] = 0.01
        contact.last_air_time.torch[:, foot] = air_time
        if not other_support:
            contact.current_contact_time.torch[:, 1 - foot] = 0.0


@pytest.mark.parametrize("num_envs", [1, 4, 64])
@pytest.mark.parametrize("terrain_mode", ["plane", "rough"])
def test_reward_contract_and_input_immutability(modules, num_envs, terrain_mode):
    env = make_env(modules, num_envs=num_envs, terrain_mode=terrain_mode)
    before = tensors_in(env)
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    assert len(terms) == 21
    assert terms.keys() == env.cfg.rewards.stage1_weights.keys()
    for value in terms.values():
        assert value.shape == (env.num_envs,)
        assert value.device.type == env.device
        assert value.dtype == torch.float32
        assert torch.isfinite(value).all()
    after = tensors_in(env)
    assert after.keys() == before.keys()
    for name in before:
        torch.testing.assert_close(after[name], before[name], rtol=0, atol=0)


@pytest.mark.parametrize("speed", [-1.0, 0.0, 0.001, 0.5, 1.5, 3.0])
def test_distance_reward_is_signed_uncapped_and_not_command_tracking(modules, env, speed):
    env._robot.data.root_lin_vel_w.torch[:, 0] = speed
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    torch.testing.assert_close(terms["forward_progress"], torch.full((env.num_envs,), speed))
    torch.testing.assert_close(total_reward(env, terms), torch.full((env.num_envs,), 5.0 * speed * env.step_dt))
    env._commands[:, 0] = 2.0
    changed = modules.rewards.compute_rewards(env, env.cfg.rewards)
    torch.testing.assert_close(total_reward(env, changed), total_reward(env, terms))


@pytest.mark.parametrize("kind", ["drift", "heading", "sliding", "tilt", "vertical", "roll_rate", "sideways", "yaw"])
def test_safety_costs_are_additive(modules, env, kind):
    baseline = modules.rewards.compute_rewards(env, env.cfg.rewards)
    data = env._robot.data
    if kind == "drift":
        data.root_pos_w.torch[:, 1] = 0.3
    elif kind == "heading":
        data.heading_w.torch[:] = 0.5
    elif kind == "sliding":
        data.body_lin_vel_w.torch[:, 0, 0] = 0.5
    elif kind == "tilt":
        data.projected_gravity_b.torch[:] = torch.tensor([0.5, 0.0, -math.sqrt(0.75)])
    elif kind == "vertical":
        data.root_lin_vel_b.torch[:, 2] = 0.5
    elif kind == "sideways":
        data.root_lin_vel_w.torch[:, 1] = 0.3
    elif kind == "yaw":
        data.root_ang_vel_w.torch[:, 2] = 0.5
    else:
        data.root_ang_vel_b.torch[:, 0] = 0.5
    changed = modules.rewards.compute_rewards(env, env.cfg.rewards)
    torch.testing.assert_close(changed["forward_progress"], baseline["forward_progress"])
    assert torch.all(total_reward(env, changed) < total_reward(env, baseline))


@pytest.mark.parametrize("condition", ["standing", "stepping", "turning", "far_away"])
def test_stationary_agent_cannot_collect_positive_bonuses(modules, env, condition):
    env._robot.data.root_lin_vel_w.torch[:] = 0.0
    if condition == "stepping":
        env._contact_sensor.data.current_contact_time.torch[:, 1] = 0.0
        env._contact_sensor.data.current_air_time.torch[:, 1] = 0.25
    elif condition == "turning":
        env._robot.data.root_ang_vel_w.torch[:, 2] = 0.5
    elif condition == "far_away":
        env._robot.data.root_pos_w.torch[:, 0] += 100.0
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    assert torch.count_nonzero(terms["forward_progress"]) == 0
    assert torch.all(total_reward(env, terms) <= 0.0)
    for name in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp", "feet_air_time"):
        assert env.cfg.rewards.stage1_weights[name] == 0.0


@pytest.mark.parametrize("condition", ["flight", "terminated", "tilt", "startup"])
def test_progress_has_no_contact_posture_or_startup_gate(modules, env, condition):
    if condition == "flight":
        env._contact_sensor.data.current_contact_time.torch[:] = 0.0
    elif condition == "terminated":
        env.reset_terminated[:] = True
    elif condition == "tilt":
        env._robot.data.projected_gravity_b.torch[:] = torch.tensor([0.8, 0.0, -0.6])
    else:
        env._stage1_episode_steps[:] = 0
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    torch.testing.assert_close(terms["forward_progress"], torch.full((env.num_envs,), 0.5))


def test_vertical_cost_uses_body_frame(modules, env):
    env._robot.data.root_lin_vel_w.torch[:, 2] = 2.0
    env._robot.data.root_lin_vel_b.torch[:, 2] = 0.3
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    torch.testing.assert_close(terms["lin_vel_z_l2"], torch.full((env.num_envs,), 0.09))


def test_default_contact_terms_keep_original_formulas(modules, env):
    contact = env._contact_sensor.data
    contact.current_contact_time.torch[:, 1] = 0.0
    contact.current_air_time.torch[:, 1] = 0.25
    env._robot.data.body_lin_vel_w.torch[:, 1, 0] = 4.0
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    # Original sliding uses force history; air time uses single-stance mode duration.
    torch.testing.assert_close(terms["feet_slide"], torch.full((env.num_envs,), 4.0))
    torch.testing.assert_close(terms["feet_air_time"], torch.full((env.num_envs,), 0.2))
    contact.net_normal_forces_w_history.torch[:, :, 1] = 0.0
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    assert torch.count_nonzero(terms["feet_slide"]) == 0


def test_turning_does_not_rotate_the_goal_line(modules, env):
    env._robot.data.heading_w.torch[:] = math.pi / 2
    env._robot.data.root_lin_vel_w.torch[:] = torch.tensor([0.0, 0.5, 0.0])
    env._robot.data.root_pos_w.torch[:, 1] = 1.0
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    assert torch.count_nonzero(terms["forward_progress"]) == 0
    torch.testing.assert_close(terms["heading_error_l2"], torch.full((env.num_envs,), (math.pi / 2) ** 2))
    torch.testing.assert_close(terms["lateral_drift_l2"], torch.ones(env.num_envs))
    env._target_heading[:] = -math.pi + 0.1
    env._robot.data.heading_w.torch[:] = math.pi - 0.1
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    torch.testing.assert_close(terms["heading_error_l2"], torch.full((env.num_envs,), 0.04))


def test_arm_and_finger_costs_are_joint_means(modules, env):
    data = env._robot.data
    data.joint_pos.torch[:, env._arm_joint_ids] += 0.2
    data.joint_pos.torch[:, env._finger_joint_ids] += 0.3
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    torch.testing.assert_close(terms["joint_deviation_arms"], torch.full((env.num_envs,), 0.2))
    torch.testing.assert_close(terms["joint_deviation_fingers"], torch.full((env.num_envs,), 0.3))


def test_translation_and_heading_invariance(modules, env):
    baseline = modules.rewards.compute_rewards(env, env.cfg.rewards)
    data = env._robot.data
    yaw = torch.full((env.num_envs,), math.pi / 2)
    data.root_quat_w.torch[:] = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    data.heading_w.torch[:] = yaw
    env._target_heading[:] = yaw
    env._stage1_start_pos[:, :2] += torch.tensor([10.0, -7.0])
    data.root_pos_w.torch[:, :2] = env._stage1_start_pos[:, :2] + torch.tensor([0.0, 2.0])
    data.root_lin_vel_w.torch[:] = torch.tensor([0.0, 0.5, 0.0])
    rotated = modules.rewards.compute_rewards(env, env.cfg.rewards)
    for name in baseline:
        torch.testing.assert_close(rotated[name], baseline[name], atol=1e-6, rtol=1e-5)


def test_alternation_is_logged_but_not_rewarded(modules):
    env = make_env(modules, num_envs=1)
    for foot in (0, 0, 1, None, 0):
        set_landing(env, foot)
        assert env._get_rewards().item() == pytest.approx(0.05)
        assert "feet_alternation" not in env._episode_sums
    assert env._stage1_alternating_steps.item() == 2


@pytest.mark.parametrize("kind", ["chatter", "long_air", "both_feet", "no_support", "terminated"])
def test_invalid_landings_do_not_advance_history(modules, kind):
    env = make_env(modules, num_envs=1)
    env._stage1_last_touchdown[:] = 0
    set_landing(env, 1)
    if kind == "chatter":
        env._contact_sensor.data.last_air_time.torch[:, 1] = 0.05
    elif kind == "long_air":
        env._contact_sensor.data.last_air_time.torch[:, 1] = 0.7
    elif kind == "both_feet":
        env._contact_sensor.data.current_contact_time.torch[:, :2] = 0.01
    elif kind == "no_support":
        env._contact_sensor.data.current_contact_time.torch[:, 0] = 0.0
    else:
        env.reset_terminated[:] = True
    env._get_rewards()
    assert env._stage1_last_touchdown.item() == 0
    assert env._stage1_alternating_steps.item() == 0.0
    assert "feet_alternation" not in env._episode_sums


def test_commands_hold_heading_beyond_resampling_period(env):
    initial = torch.tensor([-3.1, 3.1, 0.0, 0.0])
    env._target_heading[:] = initial
    env._robot.data.heading_w.torch[:] = torch.tensor([3.1, -3.1, -2.0, 2.0])
    for _ in range(501):
        env._pre_physics_step(torch.zeros(env.num_envs, 37))
    torch.testing.assert_close(env._target_heading, initial)
    torch.testing.assert_close(env._commands[:, 0], torch.full((env.num_envs,), 0.5))
    torch.testing.assert_close(env._commands[:, 1], torch.zeros(env.num_envs))
    torch.testing.assert_close(env._commands[:, 2], torch.tensor([2 * math.pi - 6.2, 6.2 - 2 * math.pi, 0.5, -0.5]))


def test_stage1_play_uses_same_speed(env):
    env.cfg.is_play = True
    env.cfg.play_lin_vel_x = 9.0
    env._resample_commands(torch.arange(env.num_envs))
    assert torch.all(env._commands[:, 0] == 0.5)


@pytest.mark.parametrize("play", [False, True])
def test_default_reset_faces_positive_x_without_turn_command(modules, play):
    env = make_env(modules)
    if play:
        env.cfg.play_mode()
    assert env.cfg.reset_pose_range == {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (0.0, 0.0)}
    for _ in range(3):
        env._reset_idx(None)
        torch.testing.assert_close(env._target_heading, torch.zeros(env.num_envs))
        torch.testing.assert_close(env._robot.data.heading_w.torch, env._target_heading)
        torch.testing.assert_close(
            env._robot.data.root_quat_w.torch, torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(env.num_envs, 1)
        )
        env._pre_physics_step(torch.zeros(env.num_envs, 37))
        assert torch.count_nonzero(env._commands[:, 2]) == 0


def test_reset_teleport_is_not_distance_progress(modules, env):
    env._robot.data.root_pos_w.torch[:, 0] = 100.0
    env._terrain.env_origins[:] = torch.tensor([200.0, -300.0, 0.0])
    env._reset_idx(None)
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    assert torch.count_nonzero(terms["forward_progress"]) == 0
    assert torch.count_nonzero(terms["lateral_drift_l2"]) == 0
    assert torch.count_nonzero(terms["heading_error_l2"]) == 0


@pytest.mark.parametrize("dt", [0.01, 0.02, 0.04])
def test_full_trajectory_returns_reward_net_distance_and_stability(modules, dt):
    # Prescribed CPU trajectories, not simulated physics: straight, diagonal, return, stand, fall.
    env = make_env(modules, num_envs=5)
    env.step_dt = dt
    totals = torch.zeros(5)
    progress = torch.zeros(5)
    steps = round(20.0 / dt)
    fall_step = round(0.4 / dt) - 1
    data = env._robot.data
    for tick in range(steps):
        active = torch.tensor([True, True, True, True, tick <= fall_step])
        data.root_lin_vel_w.torch[:] = 0.0
        data.root_lin_vel_w.torch[:, 0] = torch.tensor([0.5, 0.5, 0.5 if tick < steps // 2 else -0.5, 0.0, 1.0])
        data.root_lin_vel_w.torch[1, 1] = 0.05
        data.root_lin_vel_w.torch[~active] = 0.0
        data.root_lin_vel_b.torch[:] = data.root_lin_vel_w.torch
        data.root_pos_w.torch[:, :2] += data.root_lin_vel_w.torch[:, :2] * dt
        if tick == fall_step:
            data.projected_gravity_b.torch[4] = torch.tensor([math.sin(1.1), 0.0, -math.cos(1.1)])
        env.reset_terminated[:], env.reset_time_outs[:] = env._get_dones()
        if tick == fall_step:
            assert env.reset_terminated.tolist() == [False, False, False, False, True]
        reward = env._get_rewards()
        totals += reward * active
        progress += data.root_lin_vel_w.torch[:, 0] * 5.0 * dt
    torch.testing.assert_close(progress, torch.tensor([50.0, 50.0, 0.0, 0.0, 2.0]), atol=2e-3, rtol=1e-5)
    displacement = data.root_pos_w.torch[:, 0] - env._stage1_start_pos[:, 0]
    torch.testing.assert_close(progress, displacement * 5.0, atol=2e-3, rtol=1e-5)
    assert totals[0] > totals[1] > totals[4]
    assert totals[4] < totals[3]  # Short forward dive is worse than standing in this trajectory.
    torch.testing.assert_close(totals[2:4], torch.zeros(2), atol=2e-3, rtol=0.0)
    assert totals[4].item() == pytest.approx(2.0 - (400.0 + 4.0 * math.sin(1.1) ** 2) * dt, abs=2e-5)


def test_partial_reset_captures_composed_pose_and_preserves_other_envs(env):
    env.cfg.reset_pose_range = {"x": (0.2, 0.2), "y": (-0.3, -0.3), "yaw": (0.7, 0.7)}
    q = quat_from_euler_xyz(torch.zeros(1), torch.zeros(1), torch.tensor([0.4]))
    env._robot.data.default_root_pose.torch[1, 3:] = q[0]
    env._terrain.env_origins[1] = torch.tensor([10.0, 20.0, 0.0])
    env._stage1_episode_steps[:] = 7
    env._stage1_steady_steps[:] = 2
    env._stage1_last_touchdown[:] = 1
    env._stage1_alternating_steps[:] = 3
    env._stage1_max_lateral_deviation[:] = 2.5
    env._stage1_alternating_touchdown[:] = True
    for buffers in (env._stage1_metric_sums, env._stage1_steady_metric_sums):
        for value in buffers.values():
            value[:] = 2.0
    before = tensors_in(env)
    env._reset_idx(torch.tensor([1]))
    after = tensors_in(env)
    # Every per-environment tensor outside the reset subset is unchanged.
    for name, value in before.items():
        if value.ndim and value.shape[0] == env.num_envs:
            torch.testing.assert_close(after[name][[0, 2, 3]], value[[0, 2, 3]])
    assert env._target_heading[1].item() == pytest.approx(1.1)
    assert env._robot.data.heading_w.torch[1].item() == pytest.approx(1.1)
    assert env._commands[1, 2].item() == pytest.approx(0.0, abs=1e-6)
    torch.testing.assert_close(env._stage1_start_pos[1], torch.tensor([10.2, 19.7, 0.74]))
    assert env._stage1_episode_steps[1] == 0
    assert env._stage1_steady_steps[1] == 0
    assert env._stage1_last_touchdown[1] == -1
    assert env._stage1_max_lateral_deviation[1] == 0.0
    assert not env._stage1_alternating_touchdown[1]
    assert env.extras["log"]["Episode_Stage1/length_s"].item() == pytest.approx(0.14)


def test_flat_reset_needs_no_terrain_levels_and_keeps_full_episode(env):
    del env._terrain.terrain_levels
    env._reset_idx(None)
    assert env.extras["log"]["Metrics/terrain_level"].item() == 0
    assert torch.count_nonzero(env.episode_length_buf) == 0
    env.episode_length_buf[:] = env.max_episode_length - 1
    assert not env._get_dones()[1].any()
    env.episode_length_buf[:] += 1
    terminated, timed_out = env._get_dones()
    assert not terminated.any()
    assert timed_out.all()


@pytest.mark.parametrize("cause", ["torso_contact", "excessive_tilt", "low_height"])
def test_fall_conditions_are_distinct_from_timeout(env, cause):
    if cause == "torso_contact":
        env._contact_sensor.data.net_normal_forces_w_history.torch[0, :, 2, 2] = 5.0
    elif cause == "excessive_tilt":
        env._robot.data.projected_gravity_b.torch[0] = torch.tensor([math.sin(1.1), 0, -math.cos(1.1)])
    else:
        env._robot.data.root_pos_w.torch[0, 2] = 0.3
    terminated, timed_out = env._get_dones()
    assert terminated.tolist() == [True, False, False, False]
    assert not timed_out.any()
    assert env._stage1_termination_causes[cause][0]


def test_termination_weight_and_dt_applied_once(env):
    env.cfg.rewards.stage1_weights = {name: 0.0 for name in env.cfg.rewards.stage1_weights}
    env.cfg.rewards.stage1_weights["termination"] = -400.0
    env.reset_terminated[0] = True
    env.reset_time_outs[1] = True
    torch.testing.assert_close(env._get_rewards(), torch.tensor([-8.0, 0.0, 0.0, 0.0]))
    env.step_dt = 0.01
    torch.testing.assert_close(env._get_rewards(), torch.tensor([-4.0, 0.0, 0.0, 0.0]))


def test_metrics_use_real_steps_and_exclude_startup(env):
    env.episode_length_buf[:] = 700  # unrelated framework counters must not dilute these averages
    env._robot.data.root_lin_vel_w.torch[:, 0] = 0.0
    for _ in range(100):
        env._get_rewards()
    env._robot.data.root_lin_vel_w.torch[:, 0] = 0.5
    env._get_rewards()
    env._get_rewards()
    logs = env._stage1_episode_logs(torch.arange(env.num_envs))
    assert logs["Episode_Stage1/length_s"].item() == pytest.approx(2.04)
    assert logs["Metrics/stage1_forward_speed"].item() == pytest.approx(1 / 102)
    assert logs["Metrics/stage1_steady_forward_speed"].item() == pytest.approx(0.5)
    assert torch.all(env._stage1_steady_steps == 2)


def test_observation_and_action_contract(env):
    assert env._get_observations()["policy"].shape == (env.num_envs, 310)
    assert env.cfg.action_space == 37
    assert env.cfg.observation_space == 310


@pytest.mark.parametrize("play_before", [False, True])
def test_mode_applied_after_overrides_and_flat_play_is_safe(modules, play_before):
    cfg = modules.env_cfg.HumanoidGaitRlEnvCfg()
    cfg.stage1 = True
    cfg.terrain_mode = "plane"
    assert cfg.scene.terrain.terrain_type == "generator"  # not prematurely converted in __post_init__
    if play_before:
        cfg.play_mode()
    cfg.configure_stage1()
    cfg.play_mode()
    assert cfg.scene.terrain.terrain_type == "plane"
    assert cfg.scene.terrain.terrain_generator is None
    assert cfg.scene.terrain.env_spacing == cfg.scene.env_spacing
    assert not cfg.terrain_curriculum
    assert cfg.scene.height_scanner.pattern_cfg.size == [1.6, 1.0]
    assert cfg.scene.contact_forces.force_threshold == 1.0
    rough = modules.env_cfg.HumanoidGaitRlEnvCfg()
    rough.stage1 = False  # emulate a late Hydra scalar override
    rough.configure_stage1()
    assert rough.scene.terrain.terrain_type == "generator"
    assert rough.terrain_curriculum
    rough.play_mode()
    assert rough.scene.terrain.terrain_generator.num_rows == 5


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_tracking_scale_rejected(modules, invalid):
    cfg = modules.env_cfg.HumanoidGaitRlEnvCfg()
    cfg.stage1 = True
    cfg.rewards.tracking_std = invalid
    with pytest.raises(ValueError, match="tracking_std"):
        cfg.configure_stage1()


def test_weight_overrides_and_missing_weight(env):
    original = env._get_rewards()
    env.cfg.rewards.stage1_weights["track_lin_vel_xy_exp"] = 6.0
    torch.testing.assert_close(env._get_rewards() - original, torch.full((env.num_envs,), 0.12))
    del env.cfg.rewards.stage1_weights["feet_slide"]
    with pytest.raises(KeyError, match="stage1_weights"):
        env._get_rewards()


@pytest.mark.parametrize("terrain_mode", ["plane", "rough"])
def test_original_reward_values_are_preserved(modules, terrain_mode):
    env = make_env(modules, num_envs=1, stage1=False, terrain_mode=terrain_mode)
    data, contact = env._robot.data, env._contact_sensor.data
    data.root_lin_vel_w.torch[:] = torch.tensor([0.1, 0.2, 0.0])
    data.root_ang_vel_w.torch[:, 2] = 0.3
    data.root_ang_vel_b.torch[:, :2] = torch.tensor([0.2, 0.3])
    data.projected_gravity_b.torch[:, :2] = torch.tensor([0.1, 0.2])
    data.joint_pos.torch[:, 0] = 0.3
    data.joint_pos.torch[:, 10] = 1.2
    env._robot.actuators.applied_effort.torch[:, 0] = 2.0
    data.joint_acc.torch[:, 0] = 3.0
    env._actions[:, 0] = 0.5
    contact.current_contact_time.torch[:, :2] = torch.tensor([0.3, 0.0])
    contact.current_air_time.torch[:, 1] = 0.2
    data.body_lin_vel_w.torch[:, 0, :2] = torch.tensor([3.0, 4.0])
    data.body_lin_vel_w.torch[:, 1, 0] = 6.0
    expected = {
        "termination": 0.0,
        "track_lin_vel_xy_exp": math.exp(-0.8),
        "track_ang_vel_z_exp": math.exp(-0.36),
        "ang_vel_xy_l2": 0.13,
        "flat_orientation_l2": 0.05,
        "dof_torques_l2": 4.0,
        "dof_acc_l2": 9.0,
        "action_rate_l2": 0.25,
        "dof_pos_limits": 0.2,
        "joint_deviation_hip": 0.3,
        "joint_deviation_arms": 0.0,
        "joint_deviation_fingers": 0.0,
        "joint_deviation_torso": 0.0,
        "feet_air_time": 0.2,
        "feet_slide": 11.0,
    }
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    assert terms.keys() == expected.keys()
    for name, value in expected.items():
        assert terms[name].item() == pytest.approx(value)
    weighted = sum(getattr(env.cfg.rewards, name) * value for name, value in expected.items()) * env.step_dt
    assert env._get_rewards().item() == pytest.approx(weighted)


def test_original_command_resampling_and_torso_only_termination(modules):
    env = make_env(modules, stage1=False)
    env.cfg.is_play = True
    env._pre_physics_step(torch.zeros(env.num_envs, 37))
    assert torch.all(env._commands[:, 0] == 1.0)
    assert torch.all(env._commands[:, 1] == 0.0)
    assert torch.all(env._commands[:, 2].abs() <= 1.0)
    assert torch.all(env._command_time_left == env.cfg.resampling_command_time)
    env._robot.data.root_pos_w.torch[:, 2] = 0.1
    env._robot.data.projected_gravity_b.torch[:, 2] = 1.0
    assert not env._get_dones()[0].any()


def test_configuration_instances_do_not_share_weights(modules):
    first = modules.rewards.RewardsCfg()
    second = modules.rewards.RewardsCfg()
    first.stage1_weights["feet_slide"] = -99.0
    assert second.stage1_weights["feet_slide"] == -0.5


@pytest.mark.parametrize("side", [-1.0, 1.0])
@pytest.mark.parametrize("distance,failed", [(1.499, False), (1.5, False), (1.501, True)])
@pytest.mark.parametrize("play", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("stage1", [False, True])
def test_lateral_failure_boundary(modules, side, distance, failed, play, enabled, stage1):
    env = make_env(modules, stage1=stage1)
    env.cfg.stage1_enable_lateral_termination = enabled
    failed = failed and enabled
    if play:
        env.cfg.play_mode()
    env._robot.data.root_pos_w.torch[:, 1] = side * distance
    terminated, timed_out = env._get_dones()
    assert terminated.tolist() == [failed] * env.num_envs
    assert not timed_out.any()
    assert env._lateral_out_of_bounds.tolist() == [failed] * env.num_envs
    if stage1:
        assert env._stage1_termination_causes["lateral_out_of_bounds"].tolist() == [failed] * env.num_envs


@pytest.mark.parametrize("heading", [0.0, math.pi / 2, -1.2])
def test_lateral_failure_uses_reset_line_not_world_axes_or_current_heading(env, heading):
    env.cfg.stage1_enable_lateral_termination = True
    env._target_heading[:] = heading
    env._stage1_start_pos[:, :2] = torch.tensor([10.0, -7.0])
    env._robot.data.heading_w.torch[:] = heading + 0.8
    forward = torch.tensor([math.cos(heading), math.sin(heading)])
    normal = torch.tensor([-math.sin(heading), math.cos(heading)])
    offsets = torch.tensor([1.499, -1.499, 1.501, -1.501])
    env._robot.data.root_pos_w.torch[:, :2] = env._stage1_start_pos[:, :2] + 100.0 * forward + offsets[:, None] * normal
    terminated, _ = env._get_dones()
    assert terminated.tolist() == [False, False, True, True]
    env._robot.data.root_pos_w.torch[:, :2] = env._stage1_start_pos[:, :2] + 100.0 * forward
    assert not env._get_dones()[0].any()


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("stage1", [False, True])
def test_invalid_lateral_limit_rejected(modules, invalid, stage1):
    cfg = modules.env_cfg.HumanoidGaitRlEnvCfg()
    cfg.stage1 = stage1
    cfg.stage1_max_lateral_deviation = invalid
    with pytest.raises(ValueError, match="stage1_max_lateral_deviation"):
        cfg.configure_stage1()


def test_lateral_limit_override_and_original_mode(modules, env):
    env.cfg.stage1_enable_lateral_termination = True
    env.cfg.stage1_max_lateral_deviation = 1.0
    env._robot.data.root_pos_w.torch[:, 1] = 1.001
    assert env._get_dones()[0].all()
    rough = make_env(modules, stage1=False)
    rough._robot.data.root_pos_w.torch[:, 1] = 100.0
    assert rough._get_dones()[0].all()
    rough.cfg.stage1_enable_lateral_termination = False
    assert not rough._get_dones()[0].any()


@pytest.mark.parametrize("stage1", [False, True])
def test_spawn_line_failure_and_partial_reset_in_both_modes(modules, stage1):
    env = make_env(modules, stage1=stage1)
    env.cfg.terrain_curriculum = False  # The fixture has no terrain simulation.
    env.cfg.reset_pose_range = {"x": (0.2, 0.2), "y": (-0.3, -0.3), "yaw": (0.7, 0.7)}
    q = quat_from_euler_xyz(torch.zeros(1), torch.zeros(1), torch.tensor([0.4]))
    env._robot.data.default_root_pose.torch[1, 3:] = q[0]
    env._terrain.env_origins[1] = torch.tensor([10.0, 20.0, 0.0])
    before = env._reset_line_start_pos.clone()
    env._reset_idx(torch.tensor([1]))
    torch.testing.assert_close(env._reset_line_start_pos[1], torch.tensor([10.2, 19.7]))
    torch.testing.assert_close(env._reset_line_start_pos[[0, 2, 3]], before[[0, 2, 3]])
    assert env._reset_line_heading[1].item() == pytest.approx(1.1)
    if not stage1:
        # Command resampling must not rotate the spawn line in the original mode.
        env._target_heading[1] = -2.0
    forward = torch.tensor([math.cos(1.1), math.sin(1.1)])
    normal = torch.tensor([-math.sin(1.1), math.cos(1.1)])
    env._robot.data.root_pos_w.torch[1, :2] = env._reset_line_start_pos[1] + 20.0 * forward + 1.501 * normal
    env.episode_length_buf[1] = env.max_episode_length
    env.reset_terminated[:], env.reset_time_outs[:] = env._get_dones()
    assert env.reset_terminated.tolist() == [False, True, False, False]
    assert env.reset_time_outs[1]
    env._get_rewards()
    # The simulator base class selects terminated rows for this same reset method.
    env._reset_idx(env.reset_terminated.nonzero(as_tuple=False).squeeze(-1))
    assert env.extras["log"]["Episode_Termination/lateral_out_of_bounds"].item() == 1
    assert env.extras["log"]["Episode_Termination/torso_contact"].item() == 0
    assert env.extras["log"]["Episode_Termination/time_out"].item() == 0
    assert not env._lateral_out_of_bounds.any()
    assert not env._get_dones()[0].any()


def test_overlapping_failures_pay_once_and_timeout_is_not_success(env):
    env.cfg.stage1_enable_lateral_termination = True
    env.cfg.rewards.stage1_weights = {key: 0.0 for key in env.cfg.rewards.stage1_weights}
    env.cfg.rewards.stage1_weights["termination"] = -400.0
    env.episode_length_buf[:] = env.max_episode_length
    env._robot.data.root_pos_w.torch[0, 1] = 1.501
    env._robot.data.root_pos_w.torch[0, 2] = 0.3
    env._robot.data.projected_gravity_b.torch[0] = torch.tensor([1.0, 0.0, 0.0])
    env._contact_sensor.data.net_normal_forces_w_history.torch[0, :, 2, 2] = 5.0
    env.reset_terminated[:], env.reset_time_outs[:] = env._get_dones()
    assert env.reset_terminated.tolist() == [True, False, False, False]
    assert env.reset_time_outs.all()
    assert all(value[0] for value in env._stage1_termination_causes.values())
    torch.testing.assert_close(env._get_rewards(), torch.tensor([-8.0, 0.0, 0.0, 0.0]))
    env._reset_idx(None)
    logs = env.extras["log"]
    assert logs["Episode_Stage1/survival_rate"].item() == 0.75
    assert logs["Episode_Stage1/lateral_out_of_bounds_rate"].item() == 0.25
    assert logs["Episode_Termination/time_out"].item() == 3
    assert logs["Episode_Termination/lateral_out_of_bounds"].item() == 1
    assert not env._stage1_termination_causes["lateral_out_of_bounds"].any()


def test_final_displacement_and_peak_deviation_logged_before_partial_reset(env):
    # An episode first moves away, returns closer to the line, then resets.
    env._robot.data.root_pos_w.torch[:, 1] = torch.tensor([2.0, -1.5, 1.0, 0.5])
    env._get_rewards()
    env._robot.data.root_pos_w.torch[:, 0] = torch.tensor([8.0, -2.0, 4.0, 3.0])
    env._robot.data.root_pos_w.torch[:, 1] = 0.1
    env._get_rewards()
    env.reset_time_outs[:2] = True
    env._reset_idx(torch.tensor([0, 1]))
    logs = env.extras["log"]
    assert logs["Metrics/stage1_forward_displacement"].item() == pytest.approx(3.0)
    assert logs["Metrics/stage1_max_lateral_deviation"].item() == pytest.approx(1.75)
    torch.testing.assert_close(env._stage1_max_lateral_deviation, torch.tensor([0.0, 0.0, 1.0, 0.5]))
    assert env._stage1_episode_logs(torch.tensor([0, 1])) == {}
    # No old episode peak leaks into the next one.
    env._get_rewards()
    assert env._stage1_episode_logs(torch.tensor([0]))["Metrics/stage1_max_lateral_deviation"].item() == 0.0


def test_displacement_logs_project_onto_nonzero_reset_heading(env):
    env._target_heading[:] = math.pi / 2
    env._stage1_start_pos[:, :2] = torch.tensor([10.0, -7.0])
    env._robot.data.root_pos_w.torch[:, :2] = torch.tensor([9.0, -2.0])
    env._get_rewards()
    logs = env._stage1_episode_logs(torch.arange(env.num_envs))
    assert logs["Metrics/stage1_forward_displacement"].item() == pytest.approx(5.0)
    assert logs["Metrics/stage1_max_lateral_deviation"].item() == pytest.approx(1.0)


@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_lateral_motion_cost_precedes_position_drift_and_uses_world_frame(modules, env, side):
    env._target_heading[:] = math.pi / 2
    env._robot.data.heading_w.torch[:] = math.pi / 2
    env._robot.data.root_lin_vel_w.torch[:] = torch.tensor([0.0, 1.0, 0.0])
    baseline = modules.rewards.compute_rewards(env, env.cfg.rewards)
    env._robot.data.root_lin_vel_w.torch[:, 0] = side * 0.3
    changed = modules.rewards.compute_rewards(env, env.cfg.rewards)
    torch.testing.assert_close(changed["lateral_velocity_l2"], torch.full((env.num_envs,), 0.09))
    torch.testing.assert_close(changed["forward_progress"], baseline["forward_progress"])
    assert torch.count_nonzero(changed["lateral_drift_l2"]) == 0
    assert torch.all(total_reward(env, changed) < total_reward(env, baseline))


@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_corrective_turning_beats_rotation_away_from_target(modules, env, side):
    env._robot.data.heading_w.torch[:] = side * 0.4
    env._update_yaw_rate_command()
    env._robot.data.root_ang_vel_w.torch[:, 2] = env._commands[:, 2]
    corrective = modules.rewards.compute_rewards(env, env.cfg.rewards)
    env._robot.data.root_ang_vel_w.torch[:, 2] *= -1.0
    wrong = modules.rewards.compute_rewards(env, env.cfg.rewards)
    assert torch.count_nonzero(corrective["yaw_rate_error_l2"]) == 0
    torch.testing.assert_close(wrong["yaw_rate_error_l2"], torch.full((env.num_envs,), 0.64))
    assert torch.all(total_reward(env, corrective) > total_reward(env, wrong))


def test_complete_path_returns_prefer_straight_and_recovery_over_drift(modules):
    # Prescribed 20-second trajectories: straight, drifting, rotating, recovering, falling.
    env = make_env(modules, num_envs=5)
    env.cfg.stage1_enable_lateral_termination = True
    data = env._robot.data
    totals = torch.zeros(5)
    active = torch.ones(5, dtype=torch.bool)
    for tick in range(1000):
        data.root_lin_vel_w.torch[:] = torch.tensor([1.0, 0.0, 0.0])
        data.root_lin_vel_w.torch[1, 1] = 0.2
        data.root_lin_vel_w.torch[3, 1] = 0.2 if tick < 250 else (-0.2 if tick < 500 else 0.0)
        data.heading_w.torch[2] = 0.05 * (tick + 1) * env.step_dt
        data.root_ang_vel_w.torch[2, 2] = 0.05
        data.root_lin_vel_w.torch[~active] = 0.0
        data.root_lin_vel_b.torch[:] = data.root_lin_vel_w.torch
        data.root_pos_w.torch[:, :2] += data.root_lin_vel_w.torch[:, :2] * env.step_dt
        env._update_yaw_rate_command()
        if tick == 49:
            data.projected_gravity_b.torch[4] = torch.tensor([math.sin(1.1), 0.0, -math.cos(1.1)])
        env.episode_length_buf[:] += 1
        env.reset_terminated[:], env.reset_time_outs[:] = env._get_dones()
        totals += env._get_rewards() * active
        active &= ~(env.reset_terminated | env.reset_time_outs)
    assert totals[0] > totals[3] > totals[1]
    assert totals[0] > totals[2] > totals[4]
    assert env._stage1_termination_causes["lateral_out_of_bounds"].tolist() == [False, True, False, False, False]
    assert data.root_pos_w.torch[3, 1].abs() < 1e-5
