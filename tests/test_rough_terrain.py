"""Rough-terrain task contracts on CPU; synthetic sensor hits do not validate simulation physics."""

from __future__ import annotations

import math

import pytest
import torch
from stage1_test_support import load_task_modules, make_env

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


@pytest.fixture(scope="module")
def modules():
    with pytest.MonkeyPatch.context() as patch:
        yield load_task_modules(patch)


def ground_hits(env, ground_z=0.0):
    """Place a yaw-aligned scan over each root; emulate terrain hits, not the elevated ray starts."""
    x, y = torch.meshgrid(torch.linspace(-0.8, 0.8, 17), torch.linspace(-0.5, 0.5, 11), indexing="ij")
    hits = env._height_scanner.data.ray_hits_w.torch
    hits[..., 0] = env._robot.data.root_pos_w.torch[:, 0, None] + x.flatten()
    hits[..., 1] = env._robot.data.root_pos_w.torch[:, 1, None] + y.flatten()
    hits[..., 2] = torch.as_tensor(ground_z).expand(env.num_envs).unsqueeze(1)
    env._height_scanner.data.pos_w.torch[:] = env._robot.data.root_pos_w.torch


@pytest.mark.parametrize("stage1", [False, True])
@pytest.mark.parametrize("terrain_mode", ["plane", "rough"])
@pytest.mark.parametrize("play_order", [None, "before", "after"])
def test_terrain_and_objective_are_independent(modules, stage1, terrain_mode, play_order):
    cfg = modules.env_cfg.HumanoidGaitRlEnvCfg()
    assert cfg.stage1 and cfg.terrain_mode == "rough"
    cfg.stage1, cfg.terrain_mode = stage1, terrain_mode
    if play_order == "before":
        cfg.play_mode()
    cfg.configure_stage1()
    if play_order == "after":
        cfg.play_mode()
    assert cfg.stage1 == stage1
    assert cfg.reset_pose_range["yaw"] == (0.0, 0.0)
    assert cfg.action_space == 37 and cfg.observation_space == 310
    if terrain_mode == "plane":
        assert cfg.scene.terrain.terrain_type == "plane"
        assert cfg.scene.terrain.terrain_generator is None
        assert not cfg.terrain_curriculum
    else:
        gen = cfg.scene.terrain.terrain_generator
        assert cfg.scene.terrain.terrain_type == "generator"
        assert gen.curriculum
        assert gen.num_rows == (5 if play_order else 10)
        assert gen.num_cols == 20 and gen.size == (8.0, 8.0)
        assert len(gen.sub_terrains) == 6
        assert cfg.scene.terrain.max_init_terrain_level == (None if play_order else 5)
        assert cfg.terrain_curriculum == (play_order is None)


def test_terrain_configuration_and_weights_do_not_leak(modules):
    first = modules.env_cfg.HumanoidGaitRlEnvCfg()
    first.configure_stage1()
    first.play_mode()
    first.rewards.stage1_rough_weight_overrides["lin_vel_z_l2"] = -9.0
    second = modules.env_cfg.HumanoidGaitRlEnvCfg()
    second.configure_stage1()
    assert second.scene.terrain.terrain_generator.num_rows == 10
    assert second.rewards.stage1_weight("lin_vel_z_l2", "rough") == 0.0
    assert second.rewards.stage1_weight("lin_vel_z_l2", "plane") == -0.5


def test_invalid_terrain_mode_is_rejected(modules):
    cfg = modules.env_cfg.HumanoidGaitRlEnvCfg()
    cfg.terrain_mode = "typo"
    with pytest.raises(ValueError, match="terrain_mode"):
        cfg.configure_stage1()


@pytest.mark.parametrize("ground_z", [-3.0, -0.8, 0.0, 0.23, 4.0])
def test_local_height_handles_uphill_downhill_and_stairs(modules, ground_z):
    env = make_env(modules, terrain_mode="rough")
    env._terrain.env_origins[:, 2] = 5.0  # unrelated height of the spawn platform
    env._robot.data.root_pos_w.torch[:, 2] = ground_z + 0.74
    ground_hits(env, ground_z)
    assert not env._get_dones()[0].any()
    env._robot.data.root_pos_w.torch[0, 2] = ground_z + 0.3
    assert env._get_dones()[0].tolist() == [True, False, False, False]
    assert env._stage1_termination_causes["low_height"].tolist() == [True, False, False, False]


def test_nearest_height_uses_horizontal_distance_not_scan_maximum(modules):
    env = make_env(modules, num_envs=1, terrain_mode="rough")
    ground_hits(env)
    hits = env._height_scanner.data.ray_hits_w.torch
    # An upcoming raised step must not be used as the ground directly under the pelvis.
    hits[0, hits[0, :, 0] > 0.3, 2] = 1.0
    before = hits.clone()
    scan, z, valid, invalid = env._height_measurements()
    assert z.item() == 0.0 and valid.item() and invalid.item() == 0.0
    assert scan.max().item() == pytest.approx(0.24)
    assert scan.min().item() == pytest.approx(-0.76)
    assert not env._get_dones()[0].any()
    torch.testing.assert_close(hits, before)


@pytest.mark.parametrize("distance,valid", [(0.199, True), (0.2, True), (0.201, False)])
def test_local_height_radius_and_missing_hit_do_not_false_terminate(modules, distance, valid):
    env = make_env(modules, num_envs=1, terrain_mode="rough")
    hits = env._height_scanner.data.ray_hits_w.torch
    hits[:] = torch.inf
    hits[0, 0] = torch.tensor([distance, 0.0, 0.6])
    assert env._height_measurements()[2].item() == valid
    assert env._get_dones()[0].item() == valid


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_bad_scan_is_finite_and_other_failures_remain_active(modules, bad):
    env = make_env(modules, terrain_mode="rough")
    env._height_scanner.data.ray_hits_w.torch[:] = bad
    observations = env._get_observations()["policy"]
    assert observations.shape == (4, 310)
    assert torch.isfinite(observations).all()
    assert (observations[:, -187:] == -1.0).all()
    env._robot.data.root_pos_w.torch[:, 2] = 0.1
    env._robot.data.projected_gravity_b.torch[1, 2] = 0.0
    env._robot.data.root_pos_w.torch[2, 1] = 1.501
    env._contact_sensor.data.net_normal_forces_w_history.torch[3, :, 2, 2] = 10.0
    assert env._get_dones()[0].tolist() == [False, True, True, True]
    assert not env._stage1_termination_causes["low_height"].any()
    env._get_rewards()
    env._reset_idx(torch.tensor([0, 2]))
    logs = env.extras["log"]
    assert logs["Metrics/invalid_height_ray_fraction"] == 1
    assert logs["Metrics/missing_local_height_fraction"] == 1
    assert env._height_sample_steps.tolist() == [0, 1, 0, 1]
    assert env._invalid_height_ray_sum.tolist() == [0, 1, 0, 1]


def test_nonfinite_scan_position_and_xy_hits_use_lower_bound(modules):
    env = make_env(modules, terrain_mode="rough")
    ground_hits(env)
    env._height_scanner.data.pos_w.torch[0, 2] = torch.nan
    env._height_scanner.data.ray_hits_w.torch[1, 0, 0] = torch.inf
    scan, _, _, invalid = env._height_measurements()
    assert torch.isfinite(scan).all()
    assert (scan[0] == -1).all() and scan[1, 0] == -1
    torch.testing.assert_close(invalid, torch.tensor([1.0, 1 / 187, 0.0, 0.0]))


@pytest.mark.parametrize("terrain_mode", ["plane", "rough"])
def test_vertical_weight_is_terrain_specific_and_other_terms_are_preserved(modules, terrain_mode):
    env = make_env(modules, terrain_mode=terrain_mode)
    baseline = env._get_rewards()
    env._robot.data.root_lin_vel_b.torch[:, 2] = 2.0
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    assert len(terms) == 21
    torch.testing.assert_close(terms["lin_vel_z_l2"], torch.full((4,), 4.0))
    expected = -0.04 if terrain_mode == "plane" else 0.0
    torch.testing.assert_close(env._get_rewards() - baseline, torch.full((4,), expected))
    for name, weight in env.cfg.rewards.stage1_weights.items():
        if name != "lin_vel_z_l2":
            assert env.cfg.rewards.stage1_weight(name, terrain_mode) == weight


def test_rough_rewards_and_observations_are_invariant_to_world_translation(modules):
    env = make_env(modules, terrain_mode="rough")
    env._robot.data.root_pos_w.torch[:, :2] = torch.tensor([2.0, 0.4])
    ground_hits(env)
    old_obs = env._get_observations()["policy"]
    old_terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    offset = torch.tensor([25.0, -13.0, 7.0])
    env._robot.data.root_pos_w.torch[:] += offset
    env._stage1_start_pos[:] += offset
    env._reset_line_start_pos[:] += offset[:2]
    env._terrain.env_origins[:] += offset
    env._height_scanner.data.ray_hits_w.torch[:] += offset
    env._height_scanner.data.pos_w.torch[:] += offset
    torch.testing.assert_close(env._get_observations()["policy"], old_obs)
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    for name in old_terms:
        torch.testing.assert_close(terms[name], old_terms[name])
    assert not env._get_dones()[0].any()


def test_curriculum_uses_finished_signed_distance_and_failure_precedence(modules):
    env = make_env(modules, num_envs=10, terrain_mode="rough")
    env._stage1_episode_steps[:] = 20
    env._robot.data.root_pos_w.torch[:, 0] = torch.tensor([4.001, 4.0, 2.0, 1.999, -8, 10, 10, 10, 10, 10])
    env.reset_time_outs[:8] = True
    env.reset_terminated[5:7] = True  # failure beats timeout even after travelling far
    env.reset_time_outs[6] = False
    env._stage1_episode_steps[7] = 0  # stale flags at initialization must not promote
    env._update_terrain_curriculum(torch.arange(10))
    ids, up, down = env._terrain.updates[0]
    assert ids.tolist() == list(range(7))
    assert up.tolist() == [True, False, False, False, False, False, False]
    assert down.tolist() == [False, False, False, True, True, True, True]


def test_curriculum_uses_reset_heading_and_position_not_tile_origin(modules):
    env = make_env(modules, num_envs=1, terrain_mode="rough")
    env._stage1_start_pos[0, :2] = torch.tensor([100.0, 200.0])
    env._target_heading[:] = math.pi / 2
    env._robot.data.root_pos_w.torch[0, :2] = torch.tensor([100.0, 204.1])
    env._stage1_episode_steps[:] = 1
    env.reset_time_outs[:] = True
    env._update_terrain_curriculum(torch.tensor([0]))
    assert env._terrain.updates[0][1].item()


@pytest.mark.parametrize("mode", ["initial", "manual", "play", "disabled", "plane"])
def test_curriculum_skips_ineligible_resets(modules, mode):
    env = make_env(modules, terrain_mode="plane" if mode == "plane" else "rough")
    env._stage1_episode_steps[:] = 10
    env.reset_time_outs[:] = True
    env._robot.data.root_pos_w.torch[:, 0] = 5.0
    if mode == "initial":
        env._stage1_episode_steps[:] = 0
    elif mode == "manual":
        env.reset_time_outs[:] = False
    elif mode == "play":
        env.cfg.is_play = True
    elif mode == "disabled":
        env.cfg.terrain_curriculum = False
    env._update_terrain_curriculum(torch.arange(4))
    assert not env._terrain.updates


def test_reset_logs_old_tile_then_spawns_at_new_tile_center(modules):
    env = make_env(modules, num_envs=20, terrain_mode="rough")
    env._terrain.terrain_levels[:] = 2
    env._terrain.env_origins[:] = env._terrain.terrain_origins[2]
    env._reset_idx(None)
    assert not env._terrain.updates
    root = env._robot.data.root_pos_w.torch
    delta = root - env._terrain.env_origins
    assert (delta[:, :2].abs() <= 0.5).all()
    torch.testing.assert_close(delta[:, 2], torch.full((20,), 0.74))
    assert (env._target_heading == 0).all() and (env._commands[:, 2] == 0).all()
    assert (env.episode_length_buf == 0).all()
    env._stage1_episode_steps[:] = 1000
    env._stage1_max_lateral_deviation[:] = 0.4
    env._stage1_metric_sums["lateral_deviation"][:] = 300
    root[:, 0] += 5.0
    env.reset_time_outs[:] = True
    env.reset_terminated[1] = True
    old_positions, old_starts = root.clone(), env._stage1_start_pos.clone()
    env._reset_idx(torch.tensor([0, 1]))
    assert env._terrain.terrain_levels[:2].tolist() == [3, 1]
    logs = env.extras["log"]
    assert logs["Metrics/terrain_level"] == 2
    assert logs["Terrain/pyramid_stairs/terrain_level"] == 2
    assert logs["Terrain/pyramid_stairs/forward_displacement"].item() == pytest.approx(5.0)
    assert logs["Terrain/pyramid_stairs/failure_rate"] == 0.5
    assert logs["Terrain/pyramid_stairs/length_s"] == 20
    torch.testing.assert_close(root[2:], old_positions[2:])
    torch.testing.assert_close(env._stage1_start_pos[2:], old_starts[2:])
    torch.testing.assert_close(env._stage1_start_pos[:2], root[:2])
    torch.testing.assert_close(env._reset_line_start_pos[:2], root[:2, :2])
    assert (root[:2, :2] - env._terrain.env_origins[:2, :2]).abs().max() <= 0.5
    torch.testing.assert_close(root[:2, 2] - env._terrain.env_origins[:2, 2], torch.full((2,), 0.74))
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    assert (terms["forward_progress"][:2] == 0).all()
    assert (terms["lateral_drift_l2"][:2] == 0).all()


def test_terrain_logs_cover_all_six_types_and_progress_matches_reward(modules):
    env = make_env(modules, num_envs=20, terrain_mode="rough")
    env._robot.data.heading_w.torch[:] = math.pi / 2  # the body turned, but the target did not
    env._robot.data.root_lin_vel_w.torch[:] = torch.tensor([0.6, 0.2, 0.3])
    env._robot.data.root_pos_w.torch[:, 0] = torch.arange(20)
    env._get_rewards()
    logs = env._stage1_episode_logs(torch.arange(20))
    assert logs["Metrics/stage1_forward_speed"].item() == pytest.approx(0.6)
    for name in env.cfg.scene.terrain.terrain_generator.sub_terrains:
        assert f"Terrain/{name}/failure_rate" in logs
    assert logs["Terrain/pyramid_stairs/episodes"] == 4
    assert logs["Terrain/hf_pyramid_slope/episodes"] == 2
    assert logs["Terrain/hf_pyramid_slope_inv/forward_displacement"] == 18.5


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("distance,failed", [(1.499, False), (1.5, False), (1.501, True)])
def test_rough_lateral_boundary_is_independent_of_terrain_height(modules, side, distance, failed):
    env = make_env(modules, num_envs=1, terrain_mode="rough")
    env._robot.data.root_pos_w.torch[:] = torch.tensor([10.0, side * distance, 3.74])
    ground_hits(env, 3.0)
    assert env._get_dones()[0].item() == failed


@pytest.mark.parametrize("stage1", [False, True])
def test_rough_timeout_and_overlapping_failures_pay_once(modules, stage1):
    env = make_env(modules, num_envs=1, stage1=stage1, terrain_mode="rough")
    env._robot.data.root_pos_w.torch[0, 1] = 2.0
    env._contact_sensor.data.net_normal_forces_w_history.torch[0, :, 2, 2] = 10.0
    env.episode_length_buf[:] = env.max_episode_length
    env.reset_terminated[:], env.reset_time_outs[:] = env._get_dones()
    assert env.reset_terminated.item() and env.reset_time_outs.item()
    env._get_rewards()
    assert env._episode_sums["termination"].item() == (-8 if stage1 else -4)
    env._reset_idx(None)
    assert env.extras["log"]["Episode_Termination/time_out"] == 0


def test_complete_rough_paths_reward_horizontal_distance_and_survival(modules):
    # Flat, stairs, drifting on stairs, and an early fall. Synthetic paths test reward accounting only.
    env = make_env(modules, terrain_mode="rough")
    root = env._robot.data.root_pos_w.torch
    velocity = env._robot.data.root_lin_vel_w.torch
    velocity[:] = torch.tensor([0.5, 0.0, 0.0])
    velocity[2, 1] = 0.1
    active = torch.ones(4, dtype=torch.bool)
    returns, distance_returns = torch.zeros(4), torch.zeros(4)
    for step in range(200):
        root[active, :2] += velocity[active, :2] * env.step_dt
        # Ascend five steps, then descend. The spawn origin remains at z=0 throughout.
        stair_z = 0.23 * (min(step, 199 - step) // 20)
        ground = torch.tensor([0.0, stair_z, stair_z, 0.0])
        root[active, 2] = ground[active] + 0.74
        env._robot.data.root_lin_vel_b.torch[1:3, 2] = 0.3
        ground_hits(env, ground)
        if step >= 20:
            env._contact_sensor.data.net_normal_forces_w_history.torch[3, :, 2, 2] = 5.0
        env.reset_terminated[:], env.reset_time_outs[:] = env._get_dones()
        raw = modules.rewards.compute_rewards(env, env.cfg.rewards)
        distance_returns += raw["forward_progress"] * 5.0 * env.step_dt * active
        returns += env._get_rewards() * active
        active &= ~env.reset_terminated
    torch.testing.assert_close(distance_returns, 5.0 * root[:, 0], atol=5e-5, rtol=1e-5)
    assert active.tolist() == [True, True, True, False]
    assert returns[0].item() == pytest.approx(returns[1].item())
    assert returns[1] > returns[2] > returns[3]
