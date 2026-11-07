# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass, replace
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.visualizers import VisualizerCfg
from isaaclab_newton.physics import (
    KaminoPADMMSolverCfg,
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets import G1_MINIMAL_CFG

from .rewards import RewardsCfg

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


@configclass
class HumanoidGaitRlPhysicsCfg(PresetCfg):
    """Physics presets, matching Isaac Lab's G1 rough-terrain locomotion task."""

    isaacsim_physx: PhysxCfg = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    ovphysx: OvPhysxCfg = OvPhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=300,
            nconmax=300,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        num_substeps=2,
        debug_mode=False,
        default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
    )
    newton_kamino: NewtonCfg = NewtonCfg(solver_cfg=KaminoPADMMSolverCfg(max_contacts_per_world=64))
    default: NewtonCfg = newton_mjwarp


@configclass
class HumanoidGaitRlSceneCfg(InteractiveSceneCfg):
    """Rough terrain with a Unitree G1, a height scanner and contact sensors."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = replace(G1_MINIMAL_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    # 17 x 11 = 187 rays cast down from 20 m above the torso.
    # ``global_world_only`` keeps Newton rays from hitting the robot itself; other backends ignore it.
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class HumanoidGaitRlEnvCfg(DirectRLEnvCfg):
    """Velocity-tracking locomotion for the Unitree G1 on rough terrain."""

    # env
    decimation = 4
    episode_length_s = 20.0
    # - spaces definition (fixed contract: do not change)
    action_space = 37
    observation_space = 310
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        physics=HumanoidGaitRlPhysicsCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # scene
    scene: HumanoidGaitRlSceneCfg = HumanoidGaitRlSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # action: joint position target = default + action_scale * action
    action_scale = 0.5

    # commands
    resampling_command_time = 10.0  # [s]
    lin_vel_x_range = (0.0, 1.0)  # [m/s]
    heading_control_stiffness = 0.5
    play_lin_vel_x = 1.0  # [m/s], fixed forward command in play mode
    is_play = False

    # Stage 1 commands are identical during training and play.
    stage1 = False
    stage1_lin_vel_x = 0.5  # [m/s]
    stage1_heading_control_stiffness = 1.0
    stage1_max_yaw_rate = 0.5  # [rad/s], corrective turns only
    stage1_max_tilt_deg = 60.0
    stage1_min_height_ratio = 0.55
    stage1_enable_lateral_termination = True  # fail when the distance from the reset line exceeds the limit
    stage1_max_lateral_deviation = 1.5  # [m], fail strictly beyond either side of the reset line
    stage1_metrics_warmup_s = 2.0  # exclude startup from steady-walking metrics

    # reset: random x/y offset, fixed default robot heading (+X)
    reset_pose_range = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)}

    # termination: torso contact force above this is a fall [N]
    torso_contact_threshold = 1.0

    # terrain curriculum: robots that cross their tile move to harder terrain, robots that stall move to easier
    terrain_curriculum = True

    # rewards: weights and parameters of every term, see rewards.py
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self) -> None:
        """Configure the default visualizer camera."""
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(8.0, 0.0, 5.0))

    def play_mode(self):
        """Fewer robots, a smaller terrain and a fixed forward command."""
        super().play_mode()
        self.is_play = True
        self.terrain_curriculum = False
        self.scene.terrain.max_init_terrain_level = None
        if not self.stage1 and self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

    def configure_stage1(self) -> None:
        """Apply the mode after CLI overrides, immediately before constructing the scene.

        Keep __post_init__ mode-neutral so a late ``env.stage1=False`` override
        retains the original generator and curriculum. Mode changes require a new environment.
        """
        if not math.isfinite(self.stage1_max_lateral_deviation) or self.stage1_max_lateral_deviation <= 0.0:
            raise ValueError("stage1_max_lateral_deviation must be finite and positive")
        if not self.stage1:
            return
        positive = {
            "stage1_lin_vel_x": self.stage1_lin_vel_x,
            "stage1_max_yaw_rate": self.stage1_max_yaw_rate,
            "scene.env_spacing": self.scene.env_spacing,
            "tracking_std": self.rewards.tracking_std,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value}")
        if not 0.0 < self.stage1_min_height_ratio < 1.0:
            raise ValueError("stage1_min_height_ratio must be between 0 and 1")
        if not 0.0 < self.stage1_max_tilt_deg < 180.0:
            raise ValueError("stage1_max_tilt_deg must be between 0 and 180 degrees")
        if not 0.0 < self.rewards.stage1_min_air_time < self.rewards.stage1_max_air_time:
            raise ValueError("Stage 1 air-time bounds must satisfy 0 < min < max")

        self.scene.terrain = replace(
            self.scene.terrain,
            terrain_type="plane",
            terrain_generator=None,
            max_init_terrain_level=None,
            env_spacing=self.scene.env_spacing,
        )
        self.scene.contact_forces.force_threshold = self.rewards.feet_contact_force_threshold
        self.terrain_curriculum = False
