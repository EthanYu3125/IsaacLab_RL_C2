"""Reproducible race domains, reset isolation, and solver material adapter contracts on CPU."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from stage1_test_support import load_task_modules, make_env

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


@pytest.fixture(scope="module")
def modules():
    with pytest.MonkeyPatch.context() as patch:
        yield load_task_modules(patch)


def test_domain_sampling_ranges_independence_seed_and_idempotence(modules):
    cfg = modules.env_cfg.HumanoidGaitRlEnvCfg()
    cfg.seed = 123
    cfg.scene.num_envs = 64
    cfg.configure_stage1()
    domains = cfg.race_domains
    values = np.array([[d[k] for k in ("length", "width", "friction")] for d in domains])
    assert np.all(values >= [40, 2, 0.9]) and np.all(values <= [60, 3, 1.1])
    assert all(len(np.unique(values[:, i])) == 64 for i in range(3))
    assert np.abs(np.corrcoef(values.T) - np.eye(3)).max() < 0.4
    assert cfg.scene.env_spacing == pytest.approx(66.12)
    assert cfg.scene.replicate_physics
    np.testing.assert_array_equal(cfg.scene.clone_cfg.clone_strategy(np.ones(64), 64), np.arange(64))
    assert len(cfg.scene.race_track.spawn.usd_path) == 64
    cfg.configure_stage1()
    assert cfg.race_domains is domains  # do not even redraw identical samples
    fresh = modules.env_cfg.HumanoidGaitRlEnvCfg()
    fresh.scene.num_envs, fresh.seed = 64, 123
    fresh.configure_stage1()
    assert fresh.race_domains == domains
    fresh.seed = 42
    fresh.configure_stage1()
    assert fresh.race_domains != domains


@pytest.mark.parametrize("stage1", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_play_defaults_nominal_with_explicit_randomization(modules, stage1, explicit):
    cfg = modules.env_cfg.HumanoidGaitRlEnvCfg()
    cfg.stage1 = stage1
    cfg.configure_stage1()
    cfg.play_mode()
    cfg.race_randomize_in_play = explicit
    cfg.configure_stage1()
    assert cfg.race_randomization_enabled == explicit
    assert cfg.episode_length_s == 30
    if not explicit:
        assert all(d == dict(length=50.0, width=2.5, friction=1.0) for d in cfg.race_domains)
        assert cfg.scene.env_spacing == pytest.approx(56.12)
    else:
        assert len(set(d["length"] for d in cfg.race_domains)) == cfg.scene.num_envs


@pytest.mark.parametrize("key", ["race_track_size_ratio", "race_friction_ratio", "race_spawn_y_fraction"])
@pytest.mark.parametrize("value", [-0.1, 1.0, float("nan"), float("inf")])
def test_invalid_randomization_ranges(modules, key, value):
    cfg = modules.env_cfg.HumanoidGaitRlEnvCfg()
    setattr(cfg, key, value)
    with pytest.raises(ValueError, match=key):
        cfg.configure_stage1()


def test_random_resets_keep_centerline_domains_and_other_worlds_unchanged(modules):
    a, b = [make_env(modules, terrain_mode="race", randomize=True) for _ in range(2)]
    origins = torch.tensor([[200.0, -100, 0], [0, 100, 0], [-100, 0, 0], [500, 500, 0]])
    for env in (a, b):
        env._terrain.env_origins[:] = origins
        env._reset_idx(None)
        torch.testing.assert_close(env._race_track_origins, origins[:, :2])
        assert (env._target_heading == 0).all()
        assert (env._race_position()[:, 0] == 0).all()
        assert (env._race_initial_y.abs() <= env._race_widths * 0.1 + 1e-5).all()
        assert (env._race_initial_yaw.abs() <= math.radians(5)).all()
        assert (env._commands[:, 2] * env._race_initial_yaw < 0).all()
        terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
        assert (terms["forward_progress"] == 0).all()
        torch.testing.assert_close(terms["lateral_drift_l2"], env._race_initial_y.square())
        torch.testing.assert_close(terms["heading_error_l2"], env._race_initial_yaw.square())
    torch.testing.assert_close(a._race_previous_xy, b._race_previous_xy)
    params = [v.clone() for v in (a._race_lengths, a._race_widths, a._race_frictions)]
    poses = a._robot.data.root_pos_w.torch.clone()
    a._reset_idx(torch.tensor([1]))
    torch.testing.assert_close(a._robot.data.root_pos_w.torch[[0, 2, 3]], poses[[0, 2, 3]])
    a._reset_idx(torch.tensor([0]))
    b._reset_idx(torch.tensor([0]))  # resetting world 1 must not perturb world 0's stream
    torch.testing.assert_close(a._race_initial_yaw[0], b._race_initial_yaw[0])
    torch.testing.assert_close(a._race_initial_y[0], b._race_initial_y[0])
    for actual, before in zip((a._race_lengths, a._race_widths, a._race_frictions), params, strict=True):
        torch.testing.assert_close(actual, before)


def test_random_lane_finish_uses_each_actual_length_width(modules):
    env = make_env(modules, terrain_mode="race", randomize=True)
    env._reset_idx(None)
    env._race_previous_xy[:, 0] = env._race_lengths - 0.1
    env._robot.data.root_pos_w.torch[:, 0] = env._race_track_origins[:, 0] + env._race_lengths
    env._robot.data.root_pos_w.torch[:, 1] = env._race_track_origins[:, 1] + env._race_widths / 2
    env._robot.data.root_pos_w.torch[1, 1] += 0.001
    env.episode_length_buf[:] = 100
    assert env._get_dones()[0].tolist() == [True, False, True, True]


def fake_manager():
    """Same array layout as installed MJWarp; stub Warp conversion is identity in this fixture."""
    mapping = torch.tensor([[0, 1, 4], [2, 3, 4]], dtype=torch.int32)
    solver = SimpleNamespace(
        _use_mujoco_contacts=True,
        use_mujoco_cpu=False,
        mjc_geom_to_newton_shape=mapping,
        mjw_model=SimpleNamespace(geom_friction=torch.ones(2, 3, 3), geom_priority=torch.tensor([2, 0, 0])),
        mj_model=SimpleNamespace(geom_priority=np.array([2, 0, 0])),
    )
    model = SimpleNamespace(shape_label=["robot0", "fence0", "robot1", "fence1", "/World/ground/terrain/mesh"])
    return SimpleNamespace(_solver=solver, get_model=lambda: model)


def test_floor_adapter_changes_only_floor_sliding_friction_and_priority(modules):
    manager = fake_manager()
    solver = manager._solver
    before = solver.mjw_model.geom_friction.clone()
    result = modules.race_domain.apply_race_floor_friction(manager, torch.tensor([0.9, 1.1]))
    assert result["geom_ids"] == [2] and result["priority"] == 3
    torch.testing.assert_close(solver.mjw_model.geom_friction[:, 2, 0], torch.tensor([0.9, 1.1]))
    torch.testing.assert_close(solver.mjw_model.geom_friction[:, :2], before[:, :2])
    torch.testing.assert_close(solver.mjw_model.geom_friction[:, :, 1:], before[:, :, 1:])
    assert solver.mjw_model.geom_priority.tolist() == solver.mj_model.geom_priority.tolist() == [2, 0, 3]


@pytest.mark.parametrize("problem", ["missing_floor", "different_mapping", "unbatched", "cpu", "invalid"])
def test_unsupported_floor_layout_fails_explicitly(modules, problem):
    manager = fake_manager()
    coefficients = torch.tensor([0.9, 1.1])
    if problem == "missing_floor":
        manager.get_model().shape_label[-1] = "unknown"
    elif problem == "different_mapping":
        manager._solver.mjc_geom_to_newton_shape[1] = torch.tensor([4, 2, 3])
    elif problem == "unbatched":
        manager._solver.mjw_model.geom_friction = torch.ones(1, 3, 3)
    elif problem == "cpu":
        manager._solver.use_mujoco_cpu = True
    else:
        coefficients[0] = math.nan
    with pytest.raises(RuntimeError):
        modules.race_domain.apply_race_floor_friction(manager, coefficients)
