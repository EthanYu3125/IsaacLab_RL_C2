"""Term-by-term Stage 1 equivalence for race, including its different failure and reset state."""

from __future__ import annotations

import math

import pytest
import torch
from stage1_test_support import load_task_modules, make_env, quat_apply_inverse, quat_from_euler_xyz
from test_stage1 import tensors_in

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


@pytest.fixture(scope="module")
def modules():
    with pytest.MonkeyPatch.context() as patch:
        yield load_task_modules(patch)


def test_complete_attachment_weight_profile_is_copied_independently(modules):
    expected = {
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
        "lateral_drift_l2": -2.4,
        "heading_error_l2": -2.2,
        "lateral_velocity_l2": -2.2,
        "yaw_rate_error_l2": -0.2,
        "lin_vel_z_l2": -0.5,
    }
    first, second = modules.rewards.RewardsCfg(), modules.rewards.RewardsCfg()
    assert first.race_weights == first.stage1_weights == expected
    assert first.race_weights is not first.stage1_weights
    first.race_weights["forward_progress"] = 17
    first.stage1_weights["feet_slide"] = -9
    assert first.stage1_weights["forward_progress"] == 5
    assert first.race_weights["feet_slide"] == -0.5
    assert second.race_weights == expected
    assert not hasattr(first, "race_finish_time_discount")


@pytest.mark.parametrize("stage1", [False, True])
@pytest.mark.parametrize("heading", [0.0, math.pi / 2, -1.2])
@pytest.mark.parametrize("translation", [(0, 0), (128, -256)])
@pytest.mark.parametrize("dt", [0.01, 0.02, 0.04])
def test_all_terms_and_aggregates_match_stage1_for_equivalent_states(modules, stage1, heading, translation, dt):
    legacy = make_env(modules, terrain_mode="plane")
    race = make_env(modules, terrain_mode="race", stage1=stage1)
    generator = torch.Generator().manual_seed(711)
    data = legacy._robot.data
    data.joint_pos.torch[:] = 0.3 * torch.randn(4, 37, generator=generator)
    data.joint_pos.torch[0, legacy._ankle_joint_ids] = 1.25
    data.joint_acc.torch[:] = 2 * torch.randn(4, 37, generator=generator)
    legacy._robot.actuators.applied_effort.torch[:] = 10 * torch.randn(4, 37, generator=generator)
    data.heading_w.torch[:] = heading + torch.tensor([0.0, 0.1, -0.2, 0.3])
    zeros = torch.zeros(4)
    data.root_quat_w.torch[:] = quat_from_euler_xyz(zeros, zeros, data.heading_w.torch)
    data.root_lin_vel_w.torch[:] = torch.tensor([[-1, 0.25, 0.1], [0, -0.5, 0], [0.5, 0.125, -0.2], [3, 0, 0.3]])
    data.root_lin_vel_b.torch[:] = quat_apply_inverse(data.root_quat_w.torch, data.root_lin_vel_w.torch)
    data.root_ang_vel_w.torch[:, 2] = torch.tensor([0.0, 0.1, -0.2, 0.4])
    data.root_ang_vel_b.torch[:, :2] = torch.tensor([[0, 0], [0.1, -0.2], [0.3, 0.4], [-0.5, 0.2]])
    data.projected_gravity_b.torch[1:, 0] = torch.tensor([0.1, -0.2, 0.3])
    data.body_lin_vel_w.torch[:, :2, :2] = torch.tensor([0.2, -0.3])
    legacy._contact_sensor.data.current_contact_time.torch[:, 0] = 0
    legacy._contact_sensor.data.current_air_time.torch[:, 0] = 0.25
    legacy._actions[:] = torch.randn(4, 37, generator=generator)
    legacy._previous_actions[:] = torch.randn(4, 37, generator=generator)
    legacy._commands[:] = torch.tensor([0.5, 0, -0.1])

    # The simulators may use different world origins; only reset-relative position must match.
    for name, value in vars(data).items():
        getattr(race._robot.data, name).torch.copy_(value.torch)
    race._robot.actuators.applied_effort.torch.copy_(legacy._robot.actuators.applied_effort.torch)
    for name, value in vars(legacy._contact_sensor.data).items():
        getattr(race._contact_sensor.data, name).torch.copy_(value.torch)
    for name in ("_actions", "_previous_actions", "_commands"):
        getattr(race, name).copy_(getattr(legacy, name))
    origin = torch.tensor([[0, 0], [32, 64], [-64, 128], [128, -32.0]])
    offset = torch.tensor([[0.25, 0.125], [0.5, -0.25], [-0.125, 0.375], [-0.25, -0.5]])
    legacy._stage1_start_pos[:, :2] = origin
    legacy._robot.data.root_pos_w.torch[:, :2] = origin + offset
    race._reset_line_start_pos[:] = origin + torch.tensor(translation)
    race._robot.data.root_pos_w.torch[:, :2] = race._reset_line_start_pos + offset
    race._race_track_origins[:] = race._reset_line_start_pos - torch.tensor([0.5, 0.25])
    for env in (race, legacy):
        env.step_dt = dt
        env._target_heading[:] = heading
    legacy.reset_terminated[:] = torch.tensor([False, True, True, False])
    race._race_fell[1], race._race_fence[2], race._race_finished[0] = True, True, True
    race.reset_terminated[:] = race._race_fell | race._race_fence | race._race_finished
    race.reset_time_outs[3] = True

    before = tensors_in(race)
    expected = modules.rewards.compute_rewards(legacy, legacy.cfg.rewards)
    actual = modules.rewards.compute_rewards(race, race.cfg.rewards)
    assert len(actual) == 21 and actual.keys() == expected.keys() == race.cfg.rewards.race_weights.keys()
    for name in actual:
        assert actual[name].shape == (4,) and actual[name].dtype == torch.float32
        assert torch.isfinite(actual[name]).all()
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, msg=name)
    after = tensors_in(race)
    assert before.keys() == after.keys()
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0, msg=name)
    torch.testing.assert_close(race._get_rewards(), legacy._get_rewards(), rtol=0, atol=0)
    # Plain race floor keeps the full vertical-motion cost.
    assert race.cfg.rewards.race_weights["lin_vel_z_l2"] == -0.5


def test_reward_reference_is_reset_line_but_evaluation_keeps_lane_centerline(modules):
    env = make_env(modules, terrain_mode="race", randomize=True)
    env._reset_idx(None)
    env._robot.data.root_pos_w.torch[:, 1] = env._race_track_origins[:, 1] + 0.25
    env._reset_line_start_pos[:, 1] = env._robot.data.root_pos_w.torch[:, 1]
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    assert not terms["lateral_drift_l2"].any()
    torch.testing.assert_close(env._race_position()[:, 1], torch.full((4,), 0.25))
    env._robot.data.root_pos_w.torch[:, 1] += 0.125
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    torch.testing.assert_close(terms["lateral_drift_l2"], torch.full((4,), 0.125**2))


def test_reset_teleport_and_finish_location_cannot_generate_progress_reward(modules):
    env = make_env(modules, terrain_mode="race")
    env._robot.data.root_pos_w.torch[:, 0] = 10000
    env._race_previous_xy[:] = -10000
    env._reset_idx(torch.tensor([0, 1]))
    env._robot.data.root_lin_vel_w.torch[:] = 0
    terms = modules.rewards.compute_rewards(env, env.cfg.rewards)
    assert not terms["forward_progress"].any()  # both the reset teleport and standing far away yield zero
    assert not {"unsafe_state", "time_cost", "finish_bonus"} & terms.keys()


@pytest.mark.parametrize("mode", ["finish", "timeout", "fall", "fence", "fall_and_fence"])
def test_failure_penalty_is_only_for_failure_and_excludes_successful_termination(modules, mode):
    env = make_env(modules, num_envs=1, terrain_mode="race")
    if mode in ("fall", "fall_and_fence"):
        env._torso_ground_sensor.data.normal_force_matrix_w_history.torch[:] = 2
    if mode in ("fence", "fall_and_fence"):
        env._fence_contact_sensor.data.normal_force_matrix_w_history.torch[:] = 2
    env.episode_length_buf[:] = env.max_episode_length
    if mode == "finish":
        env._robot.data.root_pos_w.torch[:, 0] = env._race_lengths
    env.reset_terminated[:], env.reset_time_outs[:] = env._get_dones()
    env._get_rewards()
    failed = mode in ("fall", "fence", "fall_and_fence")
    assert env._episode_sums["termination"].item() == (-8 if failed else 0)
    assert env.reset_terminated.item() == (failed or mode == "finish")
    assert env.reset_time_outs.item() == (mode == "timeout")
