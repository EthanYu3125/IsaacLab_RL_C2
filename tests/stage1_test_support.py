"""CPU fixtures for real task modules, replacing only Isaac Lab/simulator boundaries.

These fixtures do not validate physics, Isaac Lab imports, or backend ProxyArray behavior.
The task's reward, command, reset, logging and configuration methods run unchanged.
"""

from __future__ import annotations

import copy
import importlib
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

TASK_DIR = Path(__file__).resolve().parents[1] / "src/Miislab_humanoid_gait_RL/tasks/humanoid_gait_rl_direct/config/g1"


class Config:
    """Minimal independently copied settings, without loading simulation plugins."""

    def __init__(self, **kwargs):
        for cls in reversed(type(self).__mro__):
            for name, value in vars(cls).items():
                if not name.startswith("_") and not callable(value) and not isinstance(value, property):
                    setattr(self, name, copy.deepcopy(value))
        self.__dict__.update(kwargs)
        if hasattr(self, "__post_init__"):
            self.__post_init__()

    def play_mode(self):
        self.scene.num_envs = 16


def configclass(cls):
    cls.__init__ = Config.__init__
    return cls


def replace(cfg, **changes):
    result = copy.deepcopy(cfg)
    result.__dict__.update(changes)
    return result


def wrap_to_pi(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def quat_from_euler_xyz(roll, pitch, yaw):
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    return torch.stack(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        dim=-1,
    )


def quat_mul(a, b):
    scalar = a[:, 3:] * b[:, 3:] - (a[:, :3] * b[:, :3]).sum(dim=-1, keepdim=True)
    vector = a[:, 3:] * b[:, :3] + b[:, 3:] * a[:, :3] + torch.cross(a[:, :3], b[:, :3], dim=-1)
    return torch.cat((vector, scalar), dim=-1)


def yaw_quat(q):
    x, y, z, w = q.unbind(dim=-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
    return quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)


def quat_apply_inverse(q, vector):
    inverse = torch.cat((-q[:, :3], q[:, 3:]), dim=-1)
    vector_q = torch.cat((vector, torch.zeros_like(vector[:, :1])), dim=-1)
    return quat_mul(quat_mul(inverse, vector_q), q)[:, :3]


class BaseEnv:
    def _reset_idx(self, env_ids):
        self.episode_length_buf[env_ids] = 0


def load_task_modules(monkeypatch):
    """Temporarily supply external boundaries, importing the actual task source files."""
    monkeypatch.syspath_prepend(str(TASK_DIR.parents[4]))

    def module(name, **attributes):
        result = types.ModuleType(name)
        result.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, result)
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, result)
        return result

    module("warp", to_torch=lambda x: x)
    module("isaaclab")
    module(
        "isaaclab.sim",
        RigidBodyMaterialCfg=Config,
        MdlFileCfg=Config,
        DomeLightCfg=Config,
        SimulationCfg=Config,
        PreviewSurfaceCfg=Config,
        UsdFileCfg=Config,
        MultiUsdFileCfg=Config,
    )
    module("isaaclab.assets", ArticulationCfg=Config, AssetBaseCfg=Config)
    module("isaaclab.envs", DirectRLEnv=BaseEnv, DirectRLEnvCfg=Config)
    module("isaaclab.physics", PhysxAutoCfg=Config)
    module("isaaclab.scene", InteractiveSceneCfg=type("SceneCfg", (Config,), {"clone_cfg": Config()}))
    ray_cfg = type("RayCasterCfg", (Config,), {"OffsetCfg": Config})
    module("isaaclab.sensors", ContactSensorCfg=Config, RayCasterCfg=ray_cfg)
    module("isaaclab.sensors.patterns", GridPatternCfg=Config)
    module("isaaclab.terrains", TerrainImporterCfg=Config)
    module("isaaclab.terrains.config")
    module(
        "isaaclab.terrains.config.rough",
        ROUGH_TERRAINS_CFG=Config(
            num_rows=10,
            num_cols=20,
            size=(8.0, 8.0),
            curriculum=False,
            sub_terrains={
                name: Config(proportion=proportion)
                for name, proportion in (
                    ("pyramid_stairs", 0.2),
                    ("pyramid_stairs_inv", 0.2),
                    ("boxes", 0.2),
                    ("random_rough", 0.2),
                    ("hf_pyramid_slope", 0.1),
                    ("hf_pyramid_slope_inv", 0.1),
                )
            },
        ),
    )
    module(
        "isaaclab.utils",
        configclass=configclass,
        replace=replace,
        index_fill_=lambda tensor, ids, value: tensor.__setitem__(ids, value),
    )
    module(
        "isaaclab.utils.math",
        quat_apply_inverse=quat_apply_inverse,
        yaw_quat=yaw_quat,
        quat_from_euler_xyz=quat_from_euler_xyz,
        quat_mul=quat_mul,
        wrap_to_pi=wrap_to_pi,
        sample_uniform=lambda low, high, shape, device: low + torch.rand(shape, device=device) * (high - low),
    )
    module("isaaclab.utils.assets", ISAAC_NUCLEUS_DIR="test-assets", ISAACLAB_NUCLEUS_DIR="test-assets")
    module("isaaclab.visualizers", VisualizerCfg=Config)
    module("isaaclab_newton")
    module(
        "isaaclab_newton.physics",
        **{
            name: Config
            for name in (
                "KaminoPADMMSolverCfg",
                "MJWarpSolverCfg",
                "NewtonCfg",
                "NewtonCollisionPipelineCfg",
                "NewtonShapeCfg",
            )
        },
    )
    module("isaaclab_ov")
    module("isaaclab_ov.physics", OvPhysxCfg=Config)
    module("isaaclab_physx")
    module("isaaclab_physx.physics", PhysxCfg=Config)
    module("isaaclab_tasks")
    module("isaaclab_tasks.utils", PresetCfg=Config)
    module("isaaclab_assets", G1_MINIMAL_CFG=Config())
    module("_stage1_unit_task", __path__=[str(TASK_DIR)])
    loaded = {}
    for name in ("race_domain", "rewards", "env_cfg", "race", "env"):
        full_name = f"_stage1_unit_task.{name}"
        # Register removal before import, so fixture teardown cannot leak fake imports to other tests.
        monkeypatch.setitem(sys.modules, full_name, None)
        del sys.modules[full_name]
        loaded[name] = importlib.import_module(full_name)
    # Small scenes keep CPU contract tests from exporting thousands of lane variants.
    loaded["env_cfg"].HumanoidGaitRlEnvCfg.scene.num_envs = 4
    return SimpleNamespace(**loaded)


def proxy(value):
    return SimpleNamespace(torch=value)


class ContactSensor:
    def __init__(self, num_envs):
        times = torch.zeros(num_envs, 3)
        times[:, :2] = 0.2
        forces = torch.zeros(num_envs, 3, 3, 3)
        forces[:, :, :2, 2] = 10.0
        self.data = SimpleNamespace(
            current_contact_time=proxy(times),
            current_air_time=proxy(torch.zeros(num_envs, 3)),
            last_air_time=proxy(torch.full((num_envs, 3), 0.25)),
            net_normal_forces_w_history=proxy(forces),
        )

    def compute_first_contact(self, dt):
        times = self.data.current_contact_time.torch
        return proxy((times > 0) & (times < dt + 1e-8))


class Robot:
    def __init__(self, num_envs):
        pose = torch.zeros(num_envs, 7)
        pose[:, 2] = 0.74
        pose[:, 6] = 1.0
        velocity = torch.zeros(num_envs, 3)
        velocity[:, 0] = 0.5
        gravity = torch.tensor([0.0, 0.0, -1.0]).repeat(num_envs, 1)
        tensors = {
            "default_root_pose": pose,
            "default_root_vel": torch.zeros(num_envs, 6),
            "root_pos_w": pose[:, :3].clone(),
            "root_quat_w": pose[:, 3:].clone(),
            "heading_w": torch.zeros(num_envs),
            "root_lin_vel_w": velocity,
            "root_com_lin_vel_b": velocity.clone(),
            "root_ang_vel_w": torch.zeros(num_envs, 3),
            "root_ang_vel_b": torch.zeros(num_envs, 3),
            "root_com_ang_vel_b": torch.zeros(num_envs, 3),
            "projected_gravity_b": gravity,
            "body_lin_vel_w": torch.zeros(num_envs, 3, 3),
            "soft_joint_pos_limits": torch.tensor([-1.0, 1.0]).repeat(num_envs, 37, 1),
        }
        tensors.update(
            {
                name: torch.zeros(num_envs, 37)
                for name in ("joint_pos", "default_joint_pos", "joint_vel", "default_joint_vel", "joint_acc")
            }
        )
        self.data = SimpleNamespace(**{name: proxy(value) for name, value in tensors.items()})
        self.data.root_lin_vel_b = self.data.root_com_lin_vel_b
        self.actuators = SimpleNamespace(applied_effort=proxy(torch.zeros(num_envs, 37)))
        self._ALL_INDICES = torch.arange(num_envs)

    def reset(self, env_ids):
        pass

    def write_root_pose_to_sim_index(self, root_pose, env_ids):
        self.data.root_pos_w.torch[env_ids] = root_pose[:, :3]
        self.data.root_quat_w.torch[env_ids] = root_pose[:, 3:]
        q = yaw_quat(root_pose[:, 3:])
        self.data.heading_w.torch[env_ids] = wrap_to_pi(2 * torch.atan2(q[:, 2], q[:, 3]))

    def write_root_velocity_to_sim_index(self, root_velocity, env_ids):
        self.data.root_lin_vel_w.torch[env_ids] = root_velocity[:, :3]
        self.data.root_ang_vel_w.torch[env_ids] = root_velocity[:, 3:]
        q = self.data.root_quat_w.torch[env_ids]
        self.data.root_lin_vel_b.torch[env_ids] = quat_apply_inverse(q, root_velocity[:, :3])

    def write_joint_position_to_sim_index(self, position, env_ids):
        self.data.joint_pos.torch[env_ids] = position

    def write_joint_velocity_to_sim_index(self, velocity, env_ids):
        self.data.joint_vel.torch[env_ids] = velocity


class Terrain:
    """Synthetic origins for testing curriculum/reset ordering; no generated mesh or physics."""

    def __init__(self, cfg, num_envs):
        self.cfg = cfg
        self.env_origins = torch.zeros(num_envs, 3)
        self.terrain_levels = torch.zeros(num_envs, dtype=torch.long)
        generator = cfg.terrain_generator
        if generator is not None:
            self.terrain_types = torch.arange(num_envs) * generator.num_cols // num_envs
            self.terrain_origins = torch.zeros(generator.num_rows, generator.num_cols, 3)
            self.terrain_origins[..., 0] = torch.arange(generator.num_rows).unsqueeze(1) * 8.0
            self.terrain_origins[..., 1] = torch.arange(generator.num_cols) * 8.0
            self.terrain_origins[..., 2] = torch.arange(generator.num_rows).unsqueeze(1) * 0.25
        self.updates = []

    def update_env_origins(self, ids, move_up, move_down):
        self.updates.append((ids.clone(), move_up.clone(), move_down.clone()))
        self.terrain_levels[ids] = (self.terrain_levels[ids] + move_up.long() - move_down.long()).clamp(
            0, self.cfg.terrain_generator.num_rows - 1
        )
        self.env_origins[ids] = self.terrain_origins[self.terrain_levels[ids], self.terrain_types[ids]]


def make_env(modules, num_envs=4, stage1=True, terrain_mode=None, randomize=False):
    env = object.__new__(modules.env.HumanoidGaitRlEnv)
    env.cfg = modules.env_cfg.HumanoidGaitRlEnvCfg()
    env.cfg.stage1 = stage1
    env.cfg.scene.num_envs = num_envs
    env.cfg.race_domain_randomization = randomize
    # Existing gait unit tests explicitly exercise plane behavior; rough tests request the new default.
    env.cfg.terrain_mode = terrain_mode or ("plane" if stage1 else "rough")
    env.cfg.configure_stage1()
    env.num_envs, env.device, env.step_dt = num_envs, "cpu", 0.02
    env.max_episode_length_s = env.cfg.episode_length_s
    env.max_episode_length = math.ceil(env.max_episode_length_s / env.step_dt)
    env._robot = Robot(num_envs)
    env._contact_sensor = ContactSensor(num_envs)
    env._height_scanner = SimpleNamespace(
        data=SimpleNamespace(pos_w=proxy(torch.zeros(num_envs, 3)), ray_hits_w=proxy(torch.zeros(num_envs, 187, 3)))
    )
    env._terrain = Terrain(env.cfg.scene.terrain, num_envs)
    env._commands = torch.tensor([0.5, 0.0, 0.0]).repeat(num_envs, 1)
    env._target_heading = torch.zeros(num_envs)
    env._command_time_left = torch.zeros(num_envs)
    env._actions = torch.zeros(num_envs, 37)
    env._previous_actions = torch.zeros_like(env._actions)
    env._episode_sums = {}
    env._error_vel_xy_sum = torch.zeros(num_envs)
    env._error_vel_yaw_sum = torch.zeros(num_envs)
    env.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
    env.reset_terminated = torch.zeros(num_envs, dtype=torch.bool)
    env.reset_time_outs = torch.zeros_like(env.reset_terminated)
    env.extras = {}
    env._feet_sensor_ids, env._feet_body_ids, env._torso_sensor_ids = [0, 1], [0, 1], [2]
    env._leg_torque_joint_ids, env._leg_acc_joint_ids = list(range(12)), list(range(10))
    env._ankle_joint_ids, env._hip_joint_ids = [10, 11], [0, 1]
    env._arm_joint_ids, env._finger_joint_ids, env._torso_joint_ids = list(range(13, 23)), list(range(23, 37)), [12]
    env._init_reset_line_buffers()
    env._reset_line_state(torch.arange(num_envs), env._robot.data.default_root_pose.torch.clone())
    if terrain_mode == "race":
        env._fence_contact_sensor = SimpleNamespace(
            data=SimpleNamespace(normal_force_matrix_w_history=proxy(torch.zeros(num_envs, 3, 3, 2, 3)))
        )
        env._torso_ground_sensor = SimpleNamespace(
            data=SimpleNamespace(normal_force_matrix_w_history=proxy(torch.zeros(num_envs, 3, 1, 1, 3)))
        )
        env._init_race_buffers()
        env._reset_race_state(torch.arange(num_envs))
    elif stage1:
        env._init_stage1_buffers()
        env._reset_stage1_state(torch.arange(num_envs), env._robot.data.default_root_pose.torch.clone())
    return env
