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
    return sum(env.cfg.rewards.stage1_weights[name] * term for name, term in terms.items()) * env.step_dt


def set_landing(env, foot, air_time=0.25, other_support=True):
    contact = env._contact_sensor.data
    contact.current_contact_time.torch[:, :2] = 0.2
    if foot is not None:
        contact.current_contact_time.torch[:, foot] = 0.01
        contact.last_air_time.torch[:, foot] = air_time
        if not other_support:
            contact.current_contact_time.torch[:, 1 - foot] = 0.0


def test_reward_contract_and_input_immutability(modules, env):
    before = tensors_in(env)
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
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


@pytest.mark.parametrize("speed", [0.0, -0.5, 1.5])
def test_correct_speed_beats_standing_backward_and_overspeed(modules, env, speed):
    correct = total_reward(env, modules.rewards.compute_rewards(env, env.cfg.rewards))
    env._robot.data.root_lin_vel_w.torch[:, 0] = speed
    changed = total_reward(env, modules.rewards.compute_rewards(env, env.cfg.rewards))
    assert torch.all(correct > changed)


@pytest.mark.parametrize("kind", ["sideways", "drift", "heading", "sliding", "height", "vertical", "roll_rate"])
def test_unwanted_motion_reduces_reward(modules, env, kind):
    baseline = total_reward(env, modules.rewards.compute_rewards(env, env.cfg.rewards))
    data = env._robot.data
    if kind == "sideways":
        data.root_lin_vel_w.torch[:, 1] = 0.3
    elif kind == "drift":
        data.root_pos_w.torch[:, 1] = 0.3
    elif kind == "heading":
        data.heading_w.torch[:] = 0.5
    elif kind == "sliding":
        data.body_lin_vel_w.torch[:, 0, 0] = 0.5
    elif kind == "height":
        data.root_pos_w.torch[:, 2] -= 0.15
    elif kind == "vertical":
        data.root_lin_vel_w.torch[:, 2] = 0.5
    else:
        data.root_ang_vel_b.torch[:, 0] = 0.5
    changed = total_reward(env, modules.rewards.compute_rewards(env, env.cfg.rewards))
    assert torch.all(changed < baseline)


@pytest.mark.parametrize("condition", ["stationary", "backward", "inverted", "terminated"])
def test_auxiliary_bonuses_require_upright_forward_movement(modules, env, condition):
    env._contact_sensor.data.current_contact_time.torch[:, 1] = 0.0
    env._contact_sensor.data.current_air_time.torch[:, 1] = 0.25
    env._stage1_alternating_touchdown[:] = True
    if condition == "stationary":
        env._robot.data.root_lin_vel_w.torch[:, 0] = 0.0
    elif condition == "backward":
        env._robot.data.root_lin_vel_w.torch[:, 0] = -0.5
    elif condition == "inverted":
        env._robot.data.projected_gravity_b.torch[:, 2] = 1.0
    else:
        env.reset_terminated[:] = True
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    for name in ("track_ang_vel_z_exp", "feet_air_time", "feet_alternation"):
        assert torch.count_nonzero(terms[name]) == 0
    if condition in ("inverted", "terminated"):
        assert torch.count_nonzero(terms["track_lin_vel_xy_exp"]) == 0
    if condition == "inverted":
        assert torch.all(terms["flat_orientation_l2"] > 0)


def test_current_contact_does_not_penalize_swing_foot(modules, env):
    # The history still contains force on both feet, but the right foot is now swinging.
    env._contact_sensor.data.current_contact_time.torch[:, 1] = 0.0
    env._contact_sensor.data.current_air_time.torch[:, 1] = 0.25
    env._robot.data.body_lin_vel_w.torch[:, 1, 0] = 4.0
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    torch.testing.assert_close(terms["feet_slide"], torch.zeros(env.num_envs))
    torch.testing.assert_close(terms["feet_air_time"], torch.ones(env.num_envs))
    env._robot.data.body_lin_vel_w.torch[:, 0, 0] = 0.5
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    torch.testing.assert_close(terms["feet_slide"], torch.full((env.num_envs,), 0.25))


def test_flight_and_prolonged_single_support(modules, env):
    contact = env._contact_sensor.data
    contact.current_contact_time.torch[:, :2] = 0.0
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    assert torch.all(terms["no_ground_contact"] == 1)
    assert torch.all(terms["feet_air_time"] == 0)
    contact.current_contact_time.torch[:, 0] = 0.2
    contact.current_air_time.torch[:, 1] = 1.0
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    assert torch.all(terms["feet_air_time"] < 1e-6)


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


def test_alternation_rewards_once_and_only_after_opposite_landing(modules):
    env = make_env(modules, num_envs=1)
    previous_sum = 0.0
    for foot, expected in [(0, 0.0), (0, 0.0), (1, 0.1), (None, 0.0), (0, 0.1)]:
        set_landing(env, foot)
        env._get_rewards()
        current_sum = env._episode_sums["feet_alternation"].item()
        assert current_sum - previous_sum == pytest.approx(expected)
        previous_sum = current_sum
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
    assert env._episode_sums["feet_alternation"].item() == 0.0


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


def test_partial_reset_captures_composed_pose_and_preserves_other_envs(env):
    env.cfg.reset_pose_range = {"x": (0.2, 0.2), "y": (-0.3, -0.3), "yaw": (0.7, 0.7)}
    q = quat_from_euler_xyz(torch.zeros(1), torch.zeros(1), torch.tensor([0.4]))
    env._robot.data.default_root_pose.torch[1, 3:] = q[0]
    env._terrain.env_origins[1] = torch.tensor([10.0, 20.0, 0.0])
    env._stage1_episode_steps[:] = 7
    env._stage1_steady_steps[:] = 2
    env._stage1_last_touchdown[:] = 1
    env._stage1_alternating_steps[:] = 3
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
    torch.testing.assert_close(env._stage1_start_pos[1], torch.tensor([10.2, 19.7, 0.74]))
    assert env._stage1_episode_steps[1] == 0
    assert env._stage1_steady_steps[1] == 0
    assert env._stage1_last_touchdown[1] == -1
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
    env.cfg.rewards.stage1_weights["termination"] = -200.0
    env.reset_terminated[0] = True
    env.reset_time_outs[1] = True
    torch.testing.assert_close(env._get_rewards(), torch.tensor([-4.0, 0.0, 0.0, 0.0]))
    env.step_dt = 0.01
    torch.testing.assert_close(env._get_rewards(), torch.tensor([-2.0, 0.0, 0.0, 0.0]))


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
    assert cfg.stage1
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


@pytest.mark.parametrize("name", ["stage1_tracking_std", "stage1_height_scale", "stage1_swing_time_std"])
def test_invalid_reward_scales_rejected(modules, name):
    cfg = modules.env_cfg.HumanoidGaitRlEnvCfg()
    setattr(cfg.rewards, name, 0.0)
    with pytest.raises(ValueError, match=name):
        cfg.configure_stage1()


def test_weight_overrides_and_missing_weight(env):
    original = env._get_rewards()
    env.cfg.rewards.stage1_weights["track_lin_vel_xy_exp"] = 6.0
    torch.testing.assert_close(env._get_rewards() - original, torch.full((env.num_envs,), 0.06))
    del env.cfg.rewards.stage1_weights["feet_slide"]
    with pytest.raises(KeyError, match="stage1_weights"):
        env._get_rewards()


def test_original_reward_values_are_preserved(modules):
    env = make_env(modules, num_envs=1, stage1=False)
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
