"""CPU checkpoint loading checks for fine-tuning without overwriting model weights or Adam moments."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.kitless]
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_finetune_checkpoint.py"


@pytest.fixture
def prepare_module():
    spec = importlib.util.spec_from_file_location("prepare_finetune_checkpoint", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checkpoint(tmp_path):
    policy, value = torch.nn.Linear(3, 2), torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam(
        [{"params": policy.parameters(), "lr": 1e-3, "initial_lr": 1e-3}, {"params": value.parameters(), "lr": 2e-3}]
    )
    inputs = torch.ones(4, 3)
    (policy(inputs).square().mean() + value(inputs).square().mean()).backward()
    optimizer.step()
    modules = {"policy": policy.state_dict(), "value": value.state_dict(), "optimizer": optimizer.state_dict()}
    path = tmp_path / "trained.pt"
    torch.save(modules, path)
    return path, modules


def test_prepared_checkpoint_preserves_predictions_and_optimizer_moments(prepare_module, checkpoint, tmp_path):
    source, original = checkpoint
    before = source.read_bytes()
    output = tmp_path / "finetune.pt"
    assert prepare_module.prepare_checkpoint(source, output, 5e-5) == [1e-3, 2e-3]
    assert source.read_bytes() == before
    prepared = torch.load(output, weights_only=True)
    assert prepared.keys() == original.keys()
    for role, outputs in (("policy", 2), ("value", 1)):
        baseline, restored = torch.nn.Linear(3, outputs), torch.nn.Linear(3, outputs)
        baseline.load_state_dict(original[role])
        restored.load_state_dict(prepared[role])
        inputs = torch.randn(5, 3)
        torch.testing.assert_close(baseline(inputs), restored(inputs), rtol=0, atol=0)
    for key, state in original["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, prepared["optimizer"]["state"][key][name], rtol=0, atol=0)
    for group in prepared["optimizer"]["param_groups"]:
        assert group["lr"] == 5e-5
        if "initial_lr" in group:
            assert group["initial_lr"] == 5e-5
    # Actual optimizer.load_state_dict must retain the new YAML LR from the very first update.
    policy, value = torch.nn.Linear(3, 2), torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam([{"params": policy.parameters()}, {"params": value.parameters()}], lr=1e-3)
    optimizer.load_state_dict(prepared["optimizer"])
    assert [group["lr"] for group in optimizer.param_groups] == [5e-5, 5e-5]
    assert optimizer.state_dict()["state"]


def test_original_and_existing_output_cannot_be_overwritten(prepare_module, checkpoint, tmp_path):
    source, _ = checkpoint
    before = source.read_bytes()
    with pytest.raises(ValueError, match="different path"):
        prepare_module.prepare_checkpoint(source, source, 5e-5)
    assert source.read_bytes() == before
    output = tmp_path / "existing.pt"
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        prepare_module.prepare_checkpoint(source, output, 5e-5)
    assert output.read_bytes() == b"keep"


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_invalid_rate_creates_no_output(prepare_module, checkpoint, tmp_path, rate):
    source, _ = checkpoint
    output = tmp_path / "invalid.pt"
    with pytest.raises(ValueError, match="finite and positive"):
        prepare_module.prepare_checkpoint(source, output, rate)
    assert not output.exists()


def test_cli_uses_the_project_yaml_independently_of_working_directory(checkpoint, tmp_path):
    yaml = pytest.importorskip("yaml")
    source, _ = checkpoint
    output = tmp_path / "cli_finetune.pt"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--checkpoint", str(source), "--output", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    config = (
        SCRIPT.parents[1]
        / "src/Miislab_humanoid_gait_RL/tasks/humanoid_gait_rl_direct/config/g1/agents/skrl_ppo_cfg.yaml"
    )
    settings = yaml.safe_load(config.read_text(encoding="utf-8"))["agent"]
    groups = torch.load(output, weights_only=True)["optimizer"]["param_groups"]
    assert all(group["lr"] == float(settings["learning_rate"]) for group in groups)
