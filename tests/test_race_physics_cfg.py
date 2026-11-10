"""Exercise the installed Newton configuration validation without starting a simulator."""

from copy import deepcopy

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


@pytest.mark.parametrize("resolved", [False, True])
@pytest.mark.parametrize("play", [False, True])
def test_race_resolves_newton_class_without_mutating_source(monkeypatch, tmp_path, resolved, play):
    # Import real config classes: the lightweight race fixtures do not run NewtonCfg.__post_init__.
    from Miislab_humanoid_gait_RL.tasks.humanoid_gait_rl_direct.config.g1 import env_cfg

    monkeypatch.setattr(env_cfg, "ensure_track_asset", lambda spec: tmp_path / "race.usda")
    cfg = env_cfg.HumanoidGaitRlEnvCfg()
    cfg.terrain_mode = "race"
    original = cfg.sim.physics.default
    original.num_substeps = 3
    original.solver_cfg.iterations = 77
    before = original.to_dict()
    if resolved:
        cfg.sim.physics = original  # CLI resolves the preset before environment construction.
    if play:
        cfg.play_mode()

    expected = deepcopy(before)
    expected["solver_cfg"]["use_mujoco_contacts"] = True
    expected["collision_cfg"] = None
    for _ in range(2):
        cfg.configure_stage1()
        assert cfg.sim.physics.to_dict() == expected
        assert cfg.sim.physics.class_type == cfg.sim.physics.solver_cfg.class_type
        assert cfg.sim.physics is not original
        assert cfg.sim.physics.solver_cfg is not original.solver_cfg
        assert original.to_dict() == before

    legacy = env_cfg.HumanoidGaitRlEnvCfg()
    legacy.terrain_mode = "rough"
    legacy.configure_stage1()
    assert not legacy.sim.physics.default.solver_cfg.use_mujoco_contacts
    assert legacy.sim.physics.default.collision_cfg is not None


def test_race_passes_actual_solver_contact_configuration_check(monkeypatch, tmp_path):
    from isaaclab.physics import PhysicsManager
    from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager
    from isaaclab_newton.physics.newton_manager import NewtonManager

    from Miislab_humanoid_gait_RL.tasks.humanoid_gait_rl_direct.config.g1 import env_cfg

    monkeypatch.setattr(env_cfg, "ensure_track_asset", lambda spec: tmp_path / "race.usda")
    cfg = env_cfg.HumanoidGaitRlEnvCfg()
    cfg.terrain_mode = "race"
    cfg.configure_stage1()
    monkeypatch.setattr(PhysicsManager, "_cfg", cfg.sim.physics)
    # Only GPU solver allocation is replaced. Run the installed backend's cross-config validation.
    solver = object()
    monkeypatch.setattr(NewtonMJWarpManager, "_create_solver", classmethod(lambda cls, model, solver_cfg: solver))
    for name in ("_solver", "_use_single_state", "_needs_collision_pipeline", "_supports_rigid_body_force_input"):
        monkeypatch.setattr(NewtonManager, name, getattr(NewtonManager, name))
    NewtonMJWarpManager._build_solver(object(), cfg.sim.physics.solver_cfg)
    assert NewtonManager._solver is solver
    assert not NewtonManager._needs_collision_pipeline
