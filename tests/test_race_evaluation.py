"""Offline verification of race evaluation aggregation, not model inference."""

import importlib.util
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


def test_evaluation_uses_per_environment_quotas_and_equal_episode_weights():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_race.py"
    spec = importlib.util.spec_from_file_location("race_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records, counts, quotas = [], [0, 0], [1, 1]
    batch = {
        "env_ids": torch.tensor([0]),
        "success": torch.tensor([True]),
        "fell": torch.tensor([False]),
        "timeout": torch.tensor([False]),
        "seconds": torch.tensor([20.0]),
        "distance": torch.tensor([30.0]),
    }
    module.collect_records(batch, records, counts, quotas)
    module.collect_records(batch, records, counts, quotas)  # fast agent must not crowd out slow one
    assert len(records) == 1
    batch.update(env_ids=torch.tensor([1]), success=torch.tensor([False]), timeout=torch.tensor([True]))
    batch.update(seconds=torch.tensor([90.0]), distance=torch.tensor([15.0]))
    module.collect_records(batch, records, counts, quotas)
    report = module.summarize(records)
    assert report["episodes"] == 2 and report["success_rate"] == 0.5
    assert report["finish_time_mean_s"] == report["finish_time_median_s"] == 20
    assert report["forward_distance_mean_m"] == 22.5
    assert module.summarize(records[1:])["finish_time_mean_s"] is None
