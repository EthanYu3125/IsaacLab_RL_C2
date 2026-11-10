"""Race domain sampling and a narrow adapter for installed MuJoCo Warp material arrays."""

from __future__ import annotations

import numpy as np
import torch


def race_clone_assignment(weights: np.ndarray, num_clones: int) -> np.ndarray:
    """Keep sampled domain i paired with lane variant i; never silently repeat/reorder lanes."""
    if len(weights) == 1:
        return np.zeros(num_clones, dtype=np.int32)
    if len(weights) != num_clones or not np.all(weights == weights[0]):
        raise ValueError("Race requires one equally weighted lane variant per environment")
    return np.arange(num_clones, dtype=np.int32)


def sample_race_domains(seed, count, length, width, friction, size_ratio, friction_ratio, enabled):
    """Draw independent columns from a dedicated stream without consuming policy/reset randomness."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 710]))
    samples = rng.uniform(-1.0, 1.0, size=(count, 3)) if enabled else np.zeros((count, 3))
    values = np.array([length, width, friction]) * (1 + samples * [size_ratio, size_ratio, friction_ratio])
    return [dict(length=float(v[0]), width=float(v[1]), friction=float(v[2])) for v in values]


def apply_race_floor_friction(manager, coefficients: torch.Tensor) -> dict:
    """Write only floor sliding friction and its priority; fail on unsupported/mismatched backend layouts.

    The global plane has one Newton material but MuJoCo Warp expands geom_friction per world.
    Updating those arrays once after solver construction preserves the global-only height scanner.
    Normal episode resets do not rebuild these material arrays.
    """
    import warp as wp

    solver = manager._solver
    model = manager.get_model()
    if not getattr(solver, "_use_mujoco_contacts", False) or getattr(solver, "use_mujoco_cpu", False):
        raise RuntimeError("Race friction randomization requires native MuJoCo Warp contacts")
    shape_ids = [i for i, label in enumerate(model.shape_label) if label.startswith("/World/ground/")]
    if not shape_ids:
        raise RuntimeError("Cannot identify race ground shapes")
    mapping = wp.to_torch(solver.mjc_geom_to_newton_shape)
    friction = wp.to_torch(solver.mjw_model.geom_friction)
    priority = wp.to_torch(solver.mjw_model.geom_priority)
    count = coefficients.numel()
    if mapping.ndim != 2 or mapping.shape[0] != count or friction.shape != (*mapping.shape, 3):
        raise RuntimeError(f"Unsupported per-world MuJoCo material layout: {mapping.shape}, {friction.shape}")
    if priority.shape != (mapping.shape[1],) or not torch.isfinite(coefficients).all() or (coefficients <= 0).any():
        raise RuntimeError("Invalid race friction coefficients or MuJoCo priority layout")
    floor_mask = torch.isin(mapping, torch.tensor(shape_ids, device=mapping.device))
    if not floor_mask.any(dim=1).all() or not (floor_mask == floor_mask[0]).all():
        raise RuntimeError("Race floor geom mapping differs across worlds or is missing")
    geom_ids = floor_mask[0].nonzero(as_tuple=False).flatten()
    # Only the sliding component changes; torsional/rolling friction and other materials are preserved.
    friction[:, geom_ids, 0] = coefficients.to(friction.device).unsqueeze(1)
    floor_priority = int(priority.max().item()) + 1
    priority[geom_ids] = floor_priority
    solver.mj_model.geom_priority[geom_ids.cpu().numpy()] = floor_priority
    actual = friction[:, geom_ids, 0]
    if not torch.allclose(actual, coefficients.to(actual.device).unsqueeze(1).expand_as(actual)):
        raise RuntimeError("Race floor friction readback failed")
    if not (priority[geom_ids] == floor_priority).all():
        raise RuntimeError("Race floor priority readback failed")
    return {"geom_ids": geom_ids.cpu().tolist(), "priority": floor_priority, "friction": actual.cpu().tolist()}
