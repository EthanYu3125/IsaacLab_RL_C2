"""Geometry, USD composition and reproducible asset generation without Isaac Lab."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


@pytest.fixture
def track_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    from Miislab_humanoid_gait_RL.assets import race_track

    return race_track


@pytest.mark.parametrize("length,width", [(30, 3), (7.25, 2.13), (50, 5), (0.1, 0.1)])
def test_dimensions_and_noncolliding_markings(track_module, length, width):
    spec = track_module.TrackSpec(length, width)
    parts = spec.parts()
    floor, start = parts[:2]
    assert floor.center[0] == length / 2
    assert floor.size == (length + 4, width, 0)
    assert start.center[:2] == (0, 0) and start.size == (0.2, width, 0)
    finish = [p for p in parts if p.name.startswith("Finish_")]
    assert min(p.center[0] - p.size[0] / 2 for p in finish) == length
    assert max(p.center[0] + p.size[0] / 2 for p in finish) == length + 0.5
    assert sum(p.size[0] * p.size[1] for p in finish) == pytest.approx(0.5 * width)
    for p in parts:
        if p.material == "fence":
            assert p.collision and p.size[2] > 0
            assert abs(p.center[1]) - p.size[1] / 2 == pytest.approx(width / 2)
        else:
            assert not p.collision and p.size[2] == 0
    posts = [p.center[0] for p in parts if p.name.startswith("Left_Post")]
    assert max(b - a for a, b in zip(posts, posts[1:], strict=False)) <= 2.0 + 1e-10
    for axis in (0, 1):
        low = min(p.center[axis] - p.size[axis] / 2 for p in parts)
        high = max(p.center[axis] + p.size[axis] / 2 for p in parts)
        assert spec.env_spacing >= high - low + 2 - 1e-10


def test_cached_assets_are_atomic_portable_and_content_addressed(track_module, tmp_path, monkeypatch):
    spec = track_module.TrackSpec()
    cache = tmp_path / "assets"
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: track_module.ensure_track_asset(spec, cache), range(12)))
    assert len(set(paths)) == 1
    path = paths[0]
    stamp = path.stat().st_mtime_ns
    monkeypatch.chdir(tmp_path)
    assert track_module.ensure_track_asset(spec, cache) == path
    assert path.stat().st_mtime_ns == stamp
    assert path.read_text(encoding="utf-8") == track_module.track_usda(spec)
    assert not list(cache.glob("*.tmp"))
    assert track_module.ensure_track_asset(track_module.TrackSpec(width=4), cache) != path


@pytest.mark.parametrize("length,width", [(30, 3), (10, 2.13)])
def test_usd_parses_materials_colliders_and_two_lane_references(track_module, tmp_path, length, width):
    pytest.importorskip("pxr.Usd", reason="Install usd-core for offline USD validation")
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    spec = track_module.TrackSpec(length, width)
    path = track_module.ensure_track_asset(spec, tmp_path)
    stage = Usd.Stage.Open(str(path))
    assert stage and stage.GetDefaultPrim().GetName() == "RaceTrack"
    assert UsdGeom.GetStageUpAxis(stage) == "Z" and UsdGeom.GetStageMetersPerUnit(stage) == 1
    for part in spec.parts():
        prim = stage.GetPrimAtPath(f"/RaceTrack/{part.name}")
        assert prim.IsValid()
        assert prim.HasAPI(UsdPhysics.CollisionAPI) == part.collision
        assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)  # static, never freely falling fences
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        assert material.GetPrim().GetName() == part.material
    scene = Usd.Stage.CreateInMemory()
    for index in range(2):
        root = UsdGeom.Xform.Define(scene, f"/World/envs/env_{index}")
        root.AddTranslateOp().Set((0, index * spec.env_spacing, 0))
        lane = scene.DefinePrim(f"/World/envs/env_{index}/RaceTrack", "Xform")
        lane.GetReferences().AddReference(str(path))
    cache = UsdGeom.BBoxCache(0, ["default"])
    a, b = [cache.ComputeWorldBound(scene.GetPrimAtPath(f"/World/envs/env_{i}")).ComputeAlignedBox() for i in range(2)]
    assert a.GetMax()[1] < b.GetMin()[1]
    assert scene.GetPrimAtPath("/World/envs/env_1/RaceTrack/Start").IsValid()


def test_randomized_lanes_have_identical_topology_and_actual_dimensions(track_module, tmp_path):
    from pxr import Usd, UsdGeom

    specs = [track_module.TrackSpec(length, width, 32, 12) for length, width in [(40, 2), (60, 3), (52.3, 2.27)]]
    topology = None
    scene = Usd.Stage.CreateInMemory()
    for i, spec in enumerate(specs):
        parts = spec.parts()
        signature = [(p.name, p.collision, p.material) for p in parts]
        if topology is None:
            topology = signature
        assert signature == topology  # same shapes, names and order for homogeneous Newton worlds
        posts = [p for p in parts if p.name.startswith("Left_Post")]
        assert len(posts) == 33
        assert posts[0].center[0] == -2 and posts[-1].center[0] == spec.length + 2
        assert parts[0].size[:2] == (spec.length + 4, spec.width)
        lane = UsdGeom.Xform.Define(scene, f"/World/lane{i}")
        lane.AddTranslateOp().Set((i * 66.12, 0, 0))
        lane.GetPrim().GetReferences().AddReference(str(track_module.ensure_track_asset(spec, tmp_path)))
    cache = UsdGeom.BBoxCache(0, ["default"])
    boxes = [cache.ComputeWorldBound(scene.GetPrimAtPath(f"/World/lane{i}")).ComputeAlignedBox() for i in range(3)]
    assert all(a.GetMax()[0] < b.GetMin()[0] for a, b in zip(boxes, boxes[1:], strict=False))
