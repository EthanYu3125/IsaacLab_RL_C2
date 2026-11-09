"""Race accounting and lifecycle on CPU; no claims about simulated contact physics."""

from __future__ import annotations

import math
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
    env._robot.data.root_pos_w.torch[:, :2] = torch.as_tensor(xy) + env._reset_line_start_pos
    env.reset_terminated[:], env.reset_time_outs[:] = env._get_dones()
    return env._get_rewards()


@pytest.mark.parametrize("stage1", [True, False])
@pytest.mark.parametrize("play_order", ["before", "after", None])
def test_configuration_and_legacy_isolation(modules, stage1, play_order):
    cfg = modules.env_cfg.HumanoidGaitRlEnvCfg()
    assert cfg.terrain_mode == "race"
    cfg.stage1 = stage1
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
    advance(env, [[31, 0], [4, 0], [5, 0], [6, 0]])
    old_pos = env._robot.data.root_pos_w.torch.clone()
    old_prev = env._race_previous_xy.clone()
    env._reset_idx(torch.tensor([0]))
    torch.testing.assert_close(env._robot.data.root_pos_w.torch[1:], old_pos[1:])
    torch.testing.assert_close(env._race_previous_xy[1:], old_prev[1:])
    assert (env._race_previous_xy[0] == 0).all()
    assert env.extras["race_episode"]["env_ids"].tolist() == [0]
    assert env.extras["race_episode"]["distance"].item() == 30
    assert not env._race_finished[0] and not env._race_terminal_paid[0]
    assert not env._terrain.updates
    env._pre_physics_step(torch.zeros_like(env._actions))
    assert "race_episode" not in env.extras
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    assert terms["forward_progress"][0] == 0


def test_crossing_interpolates_width_and_requires_forward_direction(env):
    env._race_previous_xy[:] = torch.tensor([[29, 0], [29, 4], [30.1, 0], [29, 0]])
    advance(env, [[31, 2], [31, 0], [29, 0], [35, 0]])
    assert env._race_finished.tolist() == [True, False, False, True]
    assert env.reset_terminated.tolist() == [True, False, False, True]
    assert not env.reset_time_outs.any()


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("y,success", [(1.5, True), (1.501, False)])
def test_finish_width_boundary(env, side, y, success):
    advance(env, [30, side * y])
    assert env._race_finished.tolist() == [success] * 4


def test_finish_fall_timeout_precedence_and_event_rewards(env):
    env.episode_length_buf[:] = env.max_episode_length - 1
    env._race_previous_xy[:, 0] = 29
    env._torso_ground_sensor.data.normal_force_matrix_w_history.torch[1, :, 0, 0, 2] = 5
    env._robot.data.projected_gravity_b.torch[3, 2] = 0
    advance(env, [[30, 0], [30, 0], [29, 0], [29, 0]])
    assert env._race_finished.tolist() == [True, False, False, False]
    assert env.reset_terminated.tolist() == [True, True, False, True]
    assert env.reset_time_outs.tolist() == [False, False, True, False]
    torch.testing.assert_close(env._episode_sums["finish_bonus"], torch.tensor([100, 0, 0, 0.0]))
    torch.testing.assert_close(env._episode_sums["termination"], torch.tensor([0, -25, 0, -25.0]))
    env._get_rewards()  # repeated accounting cannot pay a terminal event again
    torch.testing.assert_close(env._episode_sums["finish_bonus"], torch.tensor([100, 0, 0, 0.0]))
    torch.testing.assert_close(env._episode_sums["termination"], torch.tensor([0, -25, 0, -25.0]))
    env._reset_idx(None)
    logs = env.extras["log"]
    assert logs["Episode_Race/success_rate"] == 0.25
    assert logs["Episode_Race/fall_rate"] == 0.5
    assert logs["Episode_Race/timeout_rate"] == 0.25
    assert logs["Episode_Race/finish_time_s"] == 90


def test_fence_contact_and_lateral_drift_do_not_terminate(env):
    env._contact_sensor.data.net_normal_forces_w_history.torch[:, :, 2, :] = 1000
    advance(env, [10, 2.5])
    assert not env.reset_terminated.any()
    assert not env._torso_contact_termination.any()
    env._torso_ground_sensor.data.normal_force_matrix_w_history.torch[0, :, 0, 0, 2] = 2
    env._robot.data.projected_gravity_b.torch[1, 2] = 0
    env._robot.data.root_pos_w.torch[2, 2] = 0.1
    env.reset_terminated[:], env.reset_time_outs[:] = env._get_dones()
    assert env.reset_terminated.tolist() == [True, True, True, False]


@pytest.mark.parametrize("bad", [None, torch.zeros(4, 3, 1, 0, 3), torch.zeros(4, 3)])
def test_missing_ground_filter_fails_loudly(env, bad):
    if bad is None:
        env._torso_ground_sensor.data.normal_force_matrix_w_history = None
    else:
        env._torso_ground_sensor.data.normal_force_matrix_w_history.torch = bad
    with pytest.raises(RuntimeError, match="contact|filter"):
        env._get_dones()


@pytest.mark.parametrize("dt", [0.01, 0.02, 0.04])
def test_distance_rewards_net_progress_and_event_scale(modules, dt):
    env = make_env(modules, num_envs=1, terrain_mode="race")
    env.step_dt = dt
    total = 0
    for x in [1, 1, 0, -2, 0, 10]:
        total += advance(env, [x, 0]).item()
    assert total == pytest.approx(50 - 0.1 * dt * 6)
    # Full overshoot earns exactly the remaining 20 m, and the bonus is independent of dt.
    env.episode_length_buf[:] = round(30 / dt) - 1
    reward = advance(env, [35, 0]).item()
    assert reward == pytest.approx(100 + 100 + 50 * (1 - 30 / 90) - 0.1 * dt, abs=3e-5)


def test_faster_finish_beats_slow_and_falling(modules):
    env = make_env(modules, terrain_mode="race")
    env.episode_length_buf[:] = torch.tensor([1500, 3000, 4499, 1])
    env._robot.data.projected_gravity_b.torch[3, 2] = 0
    reward = advance(env, [30, 0])
    assert reward[0] > reward[1] > reward[2] > reward[3]
    # Waiting at the starting line until timeout costs less than intentionally falling at once.
    assert -0.1 * env.cfg.race_episode_length_s > env.cfg.rewards.race_weights["termination"]


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
    assert len(terms) == 20 and terms.keys() == env.cfg.rewards.race_weights.keys()
    for value in terms.values():
        assert value.shape == (4,) and torch.isfinite(value).all()
    after = tensors_in(env)
    assert before.keys() == after.keys()
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    assert "lateral_drift_l2" not in terms and "lin_vel_z_l2" not in terms


def test_missing_race_weight_is_rejected(env):
    del env.cfg.rewards.race_weights["time_cost"]
    with pytest.raises(KeyError, match="race_weights"):
        env._get_rewards()
