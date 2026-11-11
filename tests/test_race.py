"""Race accounting and lifecycle on CPU; no claims about simulated contact physics."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest
import torch
from stage1_test_support import load_task_modules, make_env
from test_stage1 import tensors_in

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


@pytest.fixture(scope="module")
def modules():
    with pytest.MonkeyPatch.context() as patch:
        yield load_task_modules(patch)


@pytest.fixture
def env(modules):
    return make_env(modules, terrain_mode="race")


def advance(env, xy, steps=1):
    """Run real post-physics hooks against a synthetic world position."""
    env.episode_length_buf += steps
    env._robot.data.root_pos_w.torch[:, :2] = torch.as_tensor(xy) + env._race_track_origins
    env.reset_terminated[:], env.reset_time_outs[:] = env._get_dones()
    return env._get_rewards()


def finish_bonus(env, seconds):
    return env.cfg.rewards.race_weights["finish_bonus"] / seconds


@pytest.mark.parametrize("stage1", [True, False])
@pytest.mark.parametrize("play_order", ["before", "after", None])
def test_configuration_and_legacy_isolation(modules, stage1, play_order):
    cfg = modules.env_cfg.HumanoidGaitRlEnvCfg()
    assert cfg.terrain_mode == "race"
    cfg.stage1 = stage1
    cfg.race_domain_randomization = False
    cfg.race_track_length, cfg.race_track_width = 40.0, 4.0
    cfg.race_episode_length_s = 120
    if play_order == "before":
        cfg.play_mode()
    cfg.configure_stage1()
    if play_order == "after":
        cfg.play_mode()
    assert cfg.episode_length_s == 120 and not cfg.terrain_curriculum
    assert cfg.scene.terrain.terrain_type == "plane"
    assert cfg.scene.terrain.terrain_generator is None
    assert cfg.scene.env_spacing == pytest.approx(46.12)
    assert cfg.scene.terrain.env_spacing == cfg.scene.env_spacing
    assert cfg.scene.race_track.prim_path == "{ENV_REGEX_NS}/RaceTrack"
    assert Path(cfg.scene.race_track.spawn.usd_path).is_absolute()
    assert cfg.sim.physics.solver_cfg.use_mujoco_contacts
    assert cfg.scene.fence_contact.prim_path == "{ENV_REGEX_NS}/Robot/.*"
    pattern = cfg.scene.fence_contact.filter_shape_prim_expr[0].replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
    for side in ("Left", "Right"):
        for kind in ("Post", "Rail"):
            assert re.fullmatch(pattern, f"/World/envs/env_0/RaceTrack/{side}_{kind}_0")
    assert not re.fullmatch(pattern, "/World/ground/terrain/mesh")
    assert not re.fullmatch(pattern, "/World/envs/env_0/RaceTrack/Finish_0_0")
    assert cfg.scene.torso_ground_contact.filter_shape_prim_expr == ["/World/ground/terrain/.*"]
    assert cfg.scene.height_scanner.global_world_only
    assert cfg.scene.height_scanner.mesh_prim_paths == ["/World/ground"]
    assert cfg.action_space == 37 and cfg.observation_space == 310
    cfg.configure_stage1()  # repeated application must not fail after terrain conversion
    legacy = modules.env_cfg.HumanoidGaitRlEnvCfg()
    legacy.terrain_mode = "rough"
    legacy.configure_stage1()
    assert legacy.episode_length_s == 20 and legacy.scene.env_spacing == 2.5
    assert legacy.scene.race_track is None and legacy.scene.torso_ground_contact is None
    assert legacy.scene.fence_contact is None
    assert not legacy.sim.physics.default.solver_cfg.use_mujoco_contacts


@pytest.mark.parametrize("key", ["race_track_length", "race_track_width", "race_episode_length_s"])
@pytest.mark.parametrize("bad", [0, -1, math.nan, math.inf])
def test_invalid_race_sizes_and_time(modules, key, bad):
    cfg = modules.env_cfg.HumanoidGaitRlEnvCfg()
    setattr(cfg, key, bad)
    with pytest.raises(ValueError, match="positive"):
        cfg.configure_stage1()


@pytest.mark.parametrize("stage1", [True, False])
def test_spawn_at_each_start_center_and_partial_reset(modules, stage1):
    env = make_env(modules, terrain_mode="race", stage1=stage1)
    env._terrain.env_origins[:] = torch.tensor([[0, 0, 0], [36.12, 0, 0], [0, 36.12, 0], [36.12, 36.12, 0]])
    env._robot.data.default_root_pose.torch[:, :2] = 0.8
    env._robot.data.default_root_vel.torch[:] = 1
    env._reset_idx(None)
    torch.testing.assert_close(env._robot.data.root_pos_w.torch[:, :2], env._terrain.env_origins[:, :2])
    assert (env._robot.data.heading_w.torch == 0).all()
    assert (env._robot.data.root_lin_vel_w.torch == 0).all()
    assert (env._commands == torch.tensor([0.5, 0, 0])).all()
    assert (env.episode_length_buf == 0).all()
    advance(env, [[51, 0], [4, 0], [5, 0], [6, 0]])
    old_pos = env._robot.data.root_pos_w.torch.clone()
    old_prev = env._race_previous_xy.clone()
    env._reset_idx(torch.tensor([0]))
    torch.testing.assert_close(env._robot.data.root_pos_w.torch[1:], old_pos[1:])
    torch.testing.assert_close(env._race_previous_xy[1:], old_prev[1:])
    assert (env._race_previous_xy[0] == 0).all()
    assert env.extras["race_episode"]["env_ids"].tolist() == [0]
    assert env.extras["race_episode"]["distance"].item() == 50
    assert not env._race_finished[0] and not env._race_terminal_paid[0]
    assert not env._terrain.updates
    env._pre_physics_step(torch.zeros_like(env._actions))
    assert "race_episode" not in env.extras
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    assert terms["forward_progress"][0] == 0


def test_crossing_interpolates_width_and_requires_forward_direction(env):
    env._race_previous_xy[:] = torch.tensor([[49, 0], [49, 4], [50.1, 0], [49, 0]])
    advance(env, [[51, 2], [51, 0], [49, 0], [55, 0]])
    assert env._race_finished.tolist() == [True, False, False, True]
    assert env.reset_terminated.tolist() == [True, False, False, True]
    assert not env.reset_time_outs.any()


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("outside,success", [(0.0, True), (0.001, False)])
def test_finish_width_boundary(env, side, outside, success):
    xy = torch.stack((env._race_lengths, side * (env._race_widths / 2 + outside)), dim=1)
    advance(env, xy)
    assert env._race_finished.tolist() == [success] * 4


def test_finish_fall_timeout_precedence_and_event_rewards(env):
    env.episode_length_buf[:] = env.max_episode_length - 1
    env._race_previous_xy[:, 0] = 49
    env._torso_ground_sensor.data.normal_force_matrix_w_history.torch[1, :, 0, 0, 2] = 5
    env._robot.data.projected_gravity_b.torch[3, 2] = 0
    advance(env, [[50, 0], [50, 0], [49, 0], [49, 0]])
    assert env._race_finished.tolist() == [True, False, False, False]
    assert env.reset_terminated.tolist() == [True, True, False, True]
    assert env.reset_time_outs.tolist() == [False, False, True, False]
    torch.testing.assert_close(env._episode_sums["finish_bonus"], torch.tensor([finish_bonus(env, 30), 0, 0, 0]))
    torch.testing.assert_close(
        env._episode_sums["termination"],
        torch.tensor([0, 1, 0, 1]) * env.cfg.rewards.race_weights["termination"] * env.step_dt,
    )
    env._reset_idx(None)
    logs = env.extras["log"]
    assert logs["Episode_Race/success_rate"] == 0.25
    assert logs["Episode_Race/fall_rate"] == 0.5
    assert logs["Episode_Race/timeout_rate"] == 0.25
    assert logs["Episode_Race/finish_time_s"] == 30


def test_unfiltered_contact_is_ignored_and_ground_contact_or_fall_terminates(env):
    env._contact_sensor.data.net_normal_forces_w_history.torch[:, :, 2, :] = 1000
    advance(env, [10, 2.5])
    assert not env.reset_terminated.any()
    assert not env._torso_contact_termination.any()
    env._torso_ground_sensor.data.normal_force_matrix_w_history.torch[0, :, 0, 0, 2] = 2
    env._robot.data.projected_gravity_b.torch[1, 2] = 0
    env._robot.data.root_pos_w.torch[2, 2] = 0.1
    env.reset_terminated[:], env.reset_time_outs[:] = env._get_dones()
    assert env.reset_terminated.tolist() == [True, True, True, False]
    assert env._race_fell.tolist() == [True, True, True, False]


@pytest.mark.parametrize("bad", [None, torch.zeros(4, 3, 1, 0, 3), torch.zeros(4, 3)])
def test_missing_ground_filter_fails_loudly(env, bad):
    if bad is None:
        env._torso_ground_sensor.data.normal_force_matrix_w_history = None
    else:
        env._torso_ground_sensor.data.normal_force_matrix_w_history.torch = bad
    with pytest.raises(RuntimeError, match="contact|filter"):
        env._get_dones()


@pytest.mark.parametrize("dt", [0.01, 0.02, 0.04])
def test_forward_reward_uses_signed_uncapped_velocity_and_stage1_time_scale(modules, dt):
    env = make_env(modules, num_envs=1, terrain_mode="race")
    env.step_dt = dt
    env.max_episode_length = round(30 / dt)
    for x, speed in [(1, 1.0), (1, 0.0), (0, -1.0), (-2, -2.0), (0, 2.0), (10, 3.0)]:
        env._robot.data.root_lin_vel_w.torch[:, 0] = speed
        assert advance(env, [x, 0]).item() == pytest.approx(5 * speed * dt)
    # The finish event bonus is added to the unchanged velocity-based motion reward.
    env.episode_length_buf[:] = round(30 / dt) - 1
    reward = advance(env, [55, 0]).item()
    assert reward == pytest.approx(15 * dt + finish_bonus(env, 30))


def test_faster_finish_earns_larger_bonus_and_failure_is_penalized(modules):
    env = make_env(modules, terrain_mode="race")
    env.episode_length_buf[:] = torch.tensor([499, 999, 1499, 1])
    env._robot.data.projected_gravity_b.torch[3, 2] = 0
    reward = advance(env, [50, 0])
    torch.testing.assert_close(
        reward[:3], torch.tensor([finish_bonus(env, t) for t in (10, 20, 30)]) + 5 * 0.5 * env.step_dt
    )
    assert reward[3] < reward[0]
    torch.testing.assert_close(
        env._episode_sums["termination"],
        torch.tensor([0, 0, 0, 1]) * env.cfg.rewards.race_weights["termination"] * env.step_dt,
    )


@pytest.mark.parametrize("dt", [0.01, 0.02, 0.04])
@pytest.mark.parametrize("seconds", [1.0, 10.0, 30.0])
def test_finish_bonus_is_once_per_episode_and_independent_of_control_dt(modules, dt, seconds):
    env = make_env(modules, num_envs=1, terrain_mode="race")
    env.step_dt = dt
    env.max_episode_length = round(30 / dt)
    env.episode_length_buf[:] = round(seconds / dt) - 1
    advance(env, [env._race_lengths.item(), 0])
    expected = torch.tensor([finish_bonus(env, seconds)])
    torch.testing.assert_close(env._episode_sums["finish_bonus"], expected)
    # Re-evaluating the same terminal state cannot pay again, even if done detection is repeated.
    for _ in range(2):
        env._get_dones()
        env._get_rewards()
    torch.testing.assert_close(env._episode_sums["finish_bonus"], expected)
    env._reset_idx(None)
    assert not env._race_terminal_paid.any()
    assert not env._episode_sums["finish_bonus"].any()
    env.episode_length_buf[:] = round(seconds / dt) - 1
    advance(env, [env._race_lengths.item(), 0])
    torch.testing.assert_close(env._episode_sums["finish_bonus"], expected)


def test_finish_bonus_is_isolated_across_partial_resets(env):
    env.episode_length_buf[:] = 499  # first finish at 10 seconds
    advance(env, [[50, 0], [1, 0], [1, 0], [1, 0]])
    torch.testing.assert_close(env._episode_sums["finish_bonus"], torch.tensor([finish_bonus(env, 10), 0, 0, 0]))
    env._reset_idx(torch.tensor([0]))
    assert env._race_terminal_paid.tolist() == [False] * 4
    advance(env, [[1, 0], [50, 0], [1, 0], [1, 0]], steps=500)
    torch.testing.assert_close(env._episode_sums["finish_bonus"], torch.tensor([0, finish_bonus(env, 20), 0, 0]))
    env._get_rewards()
    torch.testing.assert_close(env._episode_sums["finish_bonus"], torch.tensor([0, finish_bonus(env, 20), 0, 0]))


@pytest.mark.parametrize("width", [2.0, 3.0, 4.0])
@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("outside,eligible", [(-0.001, True), (0.0, True), (0.001, False)])
def test_finish_bonus_central_sixty_percent_boundary(modules, width, side, outside, eligible):
    env = make_env(modules, num_envs=1, terrain_mode="race")
    env._race_widths[:] = width
    env.episode_length_buf[:] = 499
    y = side * (0.3 * env._race_widths + outside)
    advance(env, torch.stack((env._race_lengths, y), dim=1))
    assert env._race_finished.item() and env.reset_terminated.item()
    expected = finish_bonus(env, 10) if eligible else 0
    assert env._episode_sums["finish_bonus"].item() == pytest.approx(expected)
    # Outside the bonus zone still ends the episode; moving inward afterward cannot collect a bonus.
    advance(env, [env._race_lengths.item() + 1, 0])
    assert env._episode_sums["finish_bonus"].item() == pytest.approx(expected)


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("previous_fraction,current_fraction,eligible", [(0, 0.4, True), (0.8, 0, False)])
def test_finish_bonus_uses_interpolated_crossing_not_final_position(
    modules, side, previous_fraction, current_fraction, eligible
):
    env = make_env(modules, num_envs=1, terrain_mode="race")
    env._race_widths[:] = 3
    env.episode_length_buf[:] = 499
    env._race_previous_xy[:] = torch.tensor([[49, side * previous_fraction * 3]])
    advance(env, [51, side * current_fraction * 3])
    assert env._race_finished.item()
    assert env._episode_sums["finish_bonus"].item() == pytest.approx(finish_bonus(env, 10) if eligible else 0)


def test_finish_bonus_uses_each_lane_center_width_and_length(env):
    env._race_lengths[:] = torch.tensor([20, 30, 40, 50])
    env._race_widths[:] = torch.tensor([2, 3, 4, 5])
    env._race_track_origins[:] = torch.tensor([[128, 128], [-128, -128], [256, -256], [-256, 256]])
    # The reset reference line can differ from the lane center and must not define the bonus zone.
    env._reset_line_start_pos[:] = env._race_track_origins + torch.tensor([0, 0.5])
    env._race_previous_xy[:, 0] = env._race_lengths - 1
    env._race_previous_xy[:, 1] = env._race_widths * torch.tensor([0.25, 0.35, -0.25, -0.35])
    env.episode_length_buf[:] = 499
    xy = env._race_previous_xy.clone()
    xy[:, 0] = env._race_lengths
    advance(env, xy)
    assert env._race_finished.all() and env.reset_terminated.all()
    torch.testing.assert_close(
        env._episode_sums["finish_bonus"], torch.tensor([finish_bonus(env, 10), 0, finish_bonus(env, 10), 0])
    )


@pytest.mark.parametrize("stage1", [False, True])
def test_observations_commands_and_reward_immutability(modules, stage1):
    env = make_env(modules, terrain_mode="race", stage1=stage1)
    env._resample_commands(torch.arange(4))
    for _ in range(550):
        env._pre_physics_step(torch.zeros(4, 37))
    assert (env._commands == torch.tensor([0.5, 0, 0])).all()
    obs = env._get_observations()["policy"]
    assert obs.shape == (4, 310)
    torch.testing.assert_close(obs[:, 9:12], env._commands)
    torch.testing.assert_close(obs[:, 86:123], env._actions)
    before = tensors_in(env)
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    assert len(terms) == 22 and terms.keys() == env.cfg.rewards.race_weights.keys()
    for value in terms.values():
        assert value.shape == (4,) and torch.isfinite(value).all()
    after = tensors_in(env)
    assert before.keys() == after.keys()
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    assert "lateral_drift_l2" in terms and "lin_vel_z_l2" in terms


def test_missing_race_weight_is_rejected(env):
    del env.cfg.rewards.race_weights["forward_progress"]
    with pytest.raises(KeyError, match="race_weights"):
        env._get_rewards()


@pytest.mark.parametrize("force,expected", [(0.0, 0.0), (0.99, 0.0), (1.0, 0.0), (1.01, 1.0), (-5.0, 1.0)])
def test_fence_contact_threshold_and_per_environment_isolation(env, modules, force, expected):
    matrix = env._fence_contact_sensor.data.normal_force_matrix_w_history.torch
    matrix[1, 0, 2, 1, 1] = force
    before = matrix.clone()
    env._get_dones()
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    torch.testing.assert_close(terms["termination"], torch.tensor([0.0, expected, 0.0, 0.0]))
    torch.testing.assert_close(matrix, before)


def test_fence_penalty_is_time_scaled_and_not_multiplied_by_contacts(env, modules):
    baseline = env._get_rewards()
    matrix = env._fence_contact_sensor.data.normal_force_matrix_w_history.torch
    matrix[0, 0, 0, 0, 1] = 10
    matrix[1, :, :, :, 1] = -10
    env._get_dones()
    torch.testing.assert_close(
        env._get_rewards() - baseline,
        torch.tensor([1, 1, 0, 0]) * env.cfg.rewards.race_weights["termination"] * env.step_dt,
    )
    assert env._get_dones()[0].tolist() == [True, True, False, False]
    env.step_dt = 0.01
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    penalty = terms["termination"] * env.cfg.rewards.race_weights["termination"] * env.step_dt
    torch.testing.assert_close(
        penalty, torch.tensor([1, 1, 0, 0]) * env.cfg.rewards.race_weights["termination"] * env.step_dt
    )
    matrix.zero_()
    env._get_dones()
    assert not modules.rewards.compute_rewards(env, env.cfg.rewards)["termination"].any()


def test_ground_contacts_do_not_create_fence_penalty(env, modules):
    env._contact_sensor.data.net_normal_forces_w_history.torch[:] = 100
    env._get_dones()
    assert not modules.rewards.compute_rewards(env, env.cfg.rewards)["termination"].any()


def test_missing_fence_matrix_is_not_silently_ignored(env, modules):
    env._fence_contact_sensor.data.normal_force_matrix_w_history = None
    with pytest.raises(RuntimeError, match="fence contact filter"):
        env._get_dones()


def test_unsafe_crossing_terminates_and_reset_starts_new_episode(env):
    env._race_previous_xy[:, 0] = 49
    env._torso_ground_sensor.data.normal_force_matrix_w_history.torch[0, :, 0, 0, 2] = 5
    env._fence_contact_sensor.data.normal_force_matrix_w_history.torch[1:3, :, 0, 0, 1] = 5
    env._robot.data.projected_gravity_b.torch[2, 2] = 0
    advance(env, [51, 0])
    assert env._race_finished.tolist() == [False, False, False, True]
    assert env.reset_terminated.all() and not env.reset_time_outs.any()
    torch.testing.assert_close(
        env._episode_sums["termination"],
        torch.tensor([1, 1, 1, 0]) * env.cfg.rewards.race_weights["termination"] * env.step_dt,
    )
    torch.testing.assert_close(
        env._episode_sums["finish_bonus"], torch.tensor([0, 0, 0, finish_bonus(env, env.step_dt)])
    )
    env._reset_idx(None)
    records = env.extras["race_episode"]
    assert records["success"].tolist() == [False, False, False, True]
    assert records["fell"].tolist() == [True, False, True, False]
    assert records["fence_contact"].tolist() == [False, True, True, False]
    assert not env._race_finished.any() and not env._race_terminal_paid.any()
    assert not env.episode_length_buf.any()
    env._torso_ground_sensor.data.normal_force_matrix_w_history.torch.zero_()
    env._fence_contact_sensor.data.normal_force_matrix_w_history.torch.zero_()
    env._robot.data.projected_gravity_b.torch[:, 2] = -1
    env._robot.data.root_lin_vel_w.torch[:, 0] = 1.0
    advance(env, [1, 0])
    assert not env.reset_terminated.any() and not env.reset_time_outs.any()
    assert not env._race_ever_fell.any() and not env._race_ever_fence.any()
    assert not env._episode_sums["finish_bonus"].any()
    assert not env._episode_sums["termination"].any()
    torch.testing.assert_close(env._episode_sums["forward_progress"], torch.full((4,), 5 * env.step_dt))


@pytest.mark.parametrize("dt", [0.01, 0.02, 0.04])
def test_accidents_end_episode_with_time_scaled_penalty_and_partial_reset(dt, modules):
    env = make_env(modules, terrain_mode="race")
    env.step_dt = dt
    # World 0 falls, 1 hits the fence, 2 does both, 3 remains safe.
    env._robot.data.projected_gravity_b.torch[[0, 2], 2] = 0
    env._fence_contact_sensor.data.normal_force_matrix_w_history.torch[[1, 2], :, 0, 0, 0] = 5
    advance(env, [0, 0])
    assert env.reset_terminated.tolist() == [True, True, True, False]
    assert not env.reset_time_outs.any()
    torch.testing.assert_close(
        env._episode_sums["termination"], torch.tensor([1, 1, 1, 0]) * env.cfg.rewards.race_weights["termination"] * dt
    )
    env._reset_idx(torch.tensor([0, 1, 2]))
    assert env.extras["race_episode"]["env_ids"].tolist() == [0, 1, 2]
    torch.testing.assert_close(env.extras["race_episode"]["unsafe_seconds"], torch.full((3,), dt))
    assert env.episode_length_buf.tolist() == [0, 0, 0, 1]
    assert not env._race_sums["unsafe_seconds"].any()


def test_fence_collision_beats_finish_and_timeout_and_logs_collision(env):
    env.episode_length_buf[:] = env.max_episode_length - 1
    env._race_previous_xy[:, 0] = 49
    env._fence_contact_sensor.data.normal_force_matrix_w_history.torch[1, 0, 2, 1, 1] = 5
    advance(env, [[50, 0], [50, 0], [49, 0], [49, 0]])
    assert env.reset_terminated.tolist() == [True, True, False, False]
    assert env.reset_time_outs.tolist() == [False, False, True, True]
    assert env._race_finished.tolist() == [True, False, False, False]
    torch.testing.assert_close(env._episode_sums["finish_bonus"], torch.tensor([finish_bonus(env, 30), 0, 0, 0]))
    assert env._episode_sums["termination"][1] == pytest.approx(
        env.cfg.rewards.race_weights["termination"] * env.step_dt
    )
    env._reset_idx(torch.tensor([1]))
    record = env.extras["race_episode"]
    assert record["env_ids"].tolist() == [1] and record["fence_contact"].item()
    assert not record["success"].item() and not record["fell"].item() and not record["timeout"].item()
    assert env.extras["log"]["Episode_Race/fence_contact_rate"] == 1


def test_complete_trajectories_favor_straight_and_corrective_yaw(modules):
    env = make_env(modules, terrain_mode="race")
    total = torch.zeros(4)
    env._robot.data.root_lin_vel_w.torch[:, 0] = torch.tensor([1.0, 1.0, 1.0, 0.0])
    for step in range(1, 101):
        t = step * env.step_dt
        xy = torch.tensor([[t, 0], [t, 0.2 * t], [t, 0], [0, 0]])
        env._robot.data.root_lin_vel_w.torch[1, 1] = 0.2
        env._robot.data.heading_w.torch[2] = 0.2 * t
        total += advance(env, xy)
    assert total[0] > total[1] and total[0] > total[2] and total[0] > total[3]
    torch.testing.assert_close(env._episode_sums["forward_progress"], torch.tensor([10, 10, 10, 0.0]))
    env._commands[:, 2] = -0.5
    env._robot.data.root_ang_vel_w.torch[:, 2] = torch.tensor([-0.5, 0.0, 0.5, -0.5])
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    assert terms["yaw_rate_error_l2"][0] < terms["yaw_rate_error_l2"][1] < terms["yaw_rate_error_l2"][2]


def test_late_crossing_after_deadline_is_timeout(env):
    env.episode_length_buf[:] = env.max_episode_length
    advance(env, [50, 0])
    assert not env.reset_terminated.any() and env.reset_time_outs.all()
    assert not env._episode_sums["finish_bonus"].any()
    assert not env._episode_sums["termination"].any()


def test_stage1_profile_penalizes_light_oscillations_without_removing_forward_incentive(modules):
    # Equal prescribed forward motion: straight, lateral sway, yaw sway, roll sway, action chatter, idle.
    # Synthetic kinematics verify reward preferences, not whether a policy can realize these trajectories.
    env = make_env(modules, num_envs=6, terrain_mode="race")
    total = torch.zeros(6)
    data = env._robot.data
    data.root_lin_vel_w.torch[:, 0] = torch.tensor([1, 1, 1, 1, 1, 0.0])
    omega = 2 * math.pi
    for step in range(1, 201):
        t = step * env.step_dt
        sine, cosine = math.sin(omega * t), math.cos(omega * t)
        xy = torch.zeros(6, 2)
        xy[:5, 0] = t
        xy[1, 1] = 0.05 * sine
        data.root_lin_vel_w.torch[1, 1] = 0.05 * omega * cosine
        data.heading_w.torch[2] = 0.06 * sine
        data.root_ang_vel_w.torch[2, 2] = 0.06 * omega * cosine
        roll = 0.04 * sine
        data.projected_gravity_b.torch[3] = torch.tensor([0, math.sin(roll), -math.cos(roll)])
        data.root_ang_vel_b.torch[3, 0] = 0.04 * omega * cosine
        env._previous_actions = env._actions.clone()
        env._actions[4] = 0.1 * (-1) ** step
        env._update_yaw_rate_command()
        total += advance(env, xy)
    assert (total[0] > total[1:5]).all()
    assert (total[1:5] > 0.9 * total[0]).all()  # mild sway still favors moving over stopping
    assert (total[1:5] > total[5]).all()
    torch.testing.assert_close(
        env._episode_sums["forward_progress"], torch.tensor([20, 20, 20, 20, 20, 0.0]), atol=1e-4, rtol=1e-5
    )


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("dt", [0.01, 0.02, 0.04])
def test_stage1_profile_preserves_incentive_to_recover_toward_reset_line(modules, side, dt):
    # Same initial lateral/heading errors: recover, hold the offset, or continue drifting away.
    env = make_env(modules, num_envs=3, terrain_mode="race")
    env.step_dt = dt
    total = torch.zeros(3)
    data = env._robot.data
    data.root_lin_vel_w.torch[:, 0] = 1.0
    for step in range(1, round(4 / dt) + 1):
        t = step * dt
        decay = math.exp(-t)
        y = side * torch.tensor([0.3 * decay, 0.3, 0.3 + 0.05 * t])
        data.root_lin_vel_w.torch[:, 1] = side * torch.tensor([-0.3 * decay, 0, 0.05])
        data.heading_w.torch[:] = side * torch.tensor([0.1 * decay, 0.1, 0.1 + 0.02 * t])
        data.root_ang_vel_w.torch[:, 2] = side * torch.tensor([-0.1 * decay, 0, 0.02])
        env._update_yaw_rate_command()
        total += advance(env, torch.stack((torch.full((3,), t), y), dim=1))
    assert total[0] > total[1] > total[2]
    torch.testing.assert_close(env._episode_sums["forward_progress"], torch.full((3,), 20.0), atol=1e-4, rtol=1e-5)
    assert env._episode_sums["yaw_rate_error_l2"][0].item() == pytest.approx(0, abs=1e-12)
    assert not env.reset_terminated.any()
