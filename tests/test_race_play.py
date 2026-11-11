"""Race playback recovery and shutdown statistics on CPU, without simulator physics."""

from contextlib import ExitStack

import pytest
import torch
from stage1_test_support import load_task_modules, make_env
from test_race import advance

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


@pytest.fixture(scope="module")
def modules():
    with pytest.MonkeyPatch.context() as patch:
        yield load_task_modules(patch)


@pytest.fixture
def env(modules):
    result = make_env(modules, terrain_mode="race")
    result.cfg.is_play = True
    return result


def stub_parent_close(env, monkeypatch):
    calls = []
    monkeypatch.setattr(type(env).__mro__[2], "close", lambda self: calls.append(self), raising=False)
    return calls


def test_play_configuration_keeps_thirty_second_limit(modules):
    cfg = modules.env_cfg.HumanoidGaitRlEnvCfg()
    cfg.play_mode()
    cfg.configure_stage1()
    assert cfg.is_play and cfg.episode_length_s == cfg.race_episode_length_s == 30


def test_accidents_continue_until_timeout_and_remain_in_metrics(env):
    env._torso_ground_sensor.data.normal_force_matrix_w_history.torch[0, :, 0, 0, 2] = 5
    env._fence_contact_sensor.data.normal_force_matrix_w_history.torch[1, :, 0, 0, 1] = 5
    env._robot.data.projected_gravity_b.torch[2, 2] = 0
    env._robot.data.root_pos_w.torch[3, 2] = 0.1
    advance(env, [1, 0], steps=100)
    assert not env.reset_terminated.any() and not env.reset_time_outs.any()
    assert not env._race_terminal_paid.any()
    assert env._race_play_episodes == 0
    assert env._race_episode_logs(torch.arange(4)) == {}
    advance(env, [2, 0], steps=env.max_episode_length - 100)
    assert not env.reset_terminated.any() and env.reset_time_outs.all()
    assert env._race_play_episodes == 4 and env._race_play_successes == 0
    env._get_rewards()  # repeated terminal evaluation must not count again
    assert env._race_play_episodes == 4
    env._reset_idx(None)
    records = env.extras["race_episode"]
    assert records["fell"].tolist() == [True, False, True, True]
    assert records["fence_contact"].tolist() == [False, True, False, False]
    assert records["timeout"].all() and not records["success"].any()
    assert not env._race_play_counted.any()
    assert env._race_play_episodes == 4


def test_recovered_agent_can_finish_after_fall_and_fence(env):
    env._robot.data.projected_gravity_b.torch[0, 2] = 0
    env._fence_contact_sensor.data.normal_force_matrix_w_history.torch[0, :, 0, 0, 1] = 5
    advance(env, [1, 0], steps=100)
    assert not env.reset_terminated.any()
    env._robot.data.projected_gravity_b.torch[0, 2] = -1
    env._fence_contact_sensor.data.normal_force_matrix_w_history.torch.zero_()
    advance(env, [[50, 0], [2, 0], [2, 0], [2, 0]], steps=400)
    assert env.reset_terminated.tolist() == [True, False, False, False]
    assert env._race_play_episodes == env._race_play_successes == 1
    assert env._race_play_finish_seconds == pytest.approx(10)
    env._reset_idx(torch.tensor([0]))
    records = env.extras["race_episode"]
    assert records["success"].item() and records["fell"].item() and records["fence_contact"].item()
    assert not env._race_play_counted.any()


def test_summary_aggregates_episodes_and_success_times_across_partial_resets(env, monkeypatch, capsys):
    stub_parent_close(env, monkeypatch)
    advance(env, [[50, 0], [1, 0], [50, 0], [1, 0]], steps=500)
    assert env._race_play_episodes == env._race_play_successes == 2
    env._reset_idx(torch.tensor([0, 2]))
    advance(env, [[1, 0], [50, 0], [1, 0], [2, 0]], steps=500)
    assert env._race_play_episodes == env._race_play_successes == 3
    env._reset_idx(torch.tensor([1]))
    advance(env, [[2, 0], [1, 0], [2, 0], [3, 0]], steps=500)
    assert env._race_play_episodes == 4 and env._race_play_successes == 3
    assert env._race_play_finish_seconds == pytest.approx(40)
    env.close()
    output = capsys.readouterr().out
    assert "Completed episodes: 4; successful finishes: 3" in output
    assert "75.00%" in output and "13.333 s" in output
    env.close()
    assert not capsys.readouterr().out


@pytest.mark.parametrize("timeout", [False, True])
def test_interrupt_cleanup_prints_summary_with_no_successes(env, monkeypatch, capsys, timeout):
    closed = stub_parent_close(env, monkeypatch)
    if timeout:
        advance(env, [0, 0], steps=env.max_episode_length)
    else:
        advance(env, [1, 0], steps=10)  # unfinished episodes do not enter the denominator
    with pytest.raises(KeyboardInterrupt), ExitStack() as cleanup:
        cleanup.callback(env.close)
        raise KeyboardInterrupt
    output = capsys.readouterr().out
    assert "N/A (no successful finishes)" in output
    assert ("0.00%" if timeout else "N/A (no completed episodes)") in output
    assert closed == [env]


def test_training_close_is_silent_and_accidents_still_terminate(modules, monkeypatch, capsys):
    env = make_env(modules, terrain_mode="race")
    closed = stub_parent_close(env, monkeypatch)
    env._fence_contact_sensor.data.normal_force_matrix_w_history.torch[0, :, 0, 0, 1] = 5
    env._robot.data.projected_gravity_b.torch[1, 2] = 0
    advance(env, [1, 0])
    assert env.reset_terminated.tolist() == [True, True, False, False]
    assert env._race_play_episodes == 0
    env.close()
    assert not capsys.readouterr().out and closed == [env]
