"""Deterministic race-lane geometry and portable USDA assets (no simulator dependency)."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

RACE_TRACK_LENGTH = 50.0
RACE_TRACK_WIDTH = 2.5
BUFFER_LENGTH = 2.0
FENCE_THICKNESS = 0.12
COLORS = {"floor": "DCAAA0", "fence": "E9B51D", "white": "FFFFFF", "black": "151515"}


@dataclass(frozen=True)
class Part:
    """One box, or a zero-thickness horizontal visual rectangle."""

    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    material: str
    collision: bool = False


@dataclass(frozen=True)
class TrackSpec:
    """Length is start-to-finish distance; width is the clear distance between fences."""

    length: float = RACE_TRACK_LENGTH
    width: float = RACE_TRACK_WIDTH
    post_intervals: int | None = None
    finish_columns: int | None = None

    def __post_init__(self):
        for name, value in (("length", self.length), ("width", self.width)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Race track {name} must be finite and positive")
        for name, value in (("post_intervals", self.post_intervals), ("finish_columns", self.finish_columns)):
            if value is not None and (not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer")

    @property
    def env_spacing(self) -> float:
        # End posts extend half their thickness past each end of the floor.
        return max(self.length + 2 * BUFFER_LENGTH + FENCE_THICKNESS, self.width + 2 * FENCE_THICKNESS) + 2.0

    def parts(self) -> list[Part]:
        """Return the single source of truth for both USD and the offline preview."""
        extent = self.length + 2 * BUFFER_LENGTH
        result = [
            Part("Floor", (self.length / 2, 0, 0.001), (extent, self.width, 0), "floor"),
            Part("Start", (0, 0, 0.002), (0.2, self.width, 0), "white"),
        ]
        # Two rows of 25 cm squares; trim the last column for non-multiple widths.
        columns = self.finish_columns or math.ceil(self.width / 0.25)
        cell_width = self.width / columns if self.finish_columns else 0.25
        for row in range(2):
            for col in range(columns):
                size_y = min(cell_width, self.width - col * cell_width)
                result.append(
                    Part(
                        f"Finish_{row}_{col}",
                        (self.length + row * 0.25 + 0.125, -self.width / 2 + col * cell_width + size_y / 2, 0.002),
                        (0.25, size_y, 0),
                        "white" if (row + col) % 2 else "black",
                    )
                )
        intervals = self.post_intervals or math.ceil(extent / 2)
        for side, sign in (("Left", 1), ("Right", -1)):
            y = sign * (self.width / 2 + FENCE_THICKNESS / 2)
            for i in range(intervals + 1):
                result.append(
                    Part(
                        f"{side}_Post_{i}",
                        (-BUFFER_LENGTH + extent * i / intervals, y, 0.5),
                        (FENCE_THICKNESS, FENCE_THICKNESS, 1.0),
                        "fence",
                        True,
                    )
                )
            for i, z in enumerate((0.35, 0.75)):
                result.append(
                    Part(f"{side}_Rail_{i}", (self.length / 2, y, z), (extent, FENCE_THICKNESS, 0.12), "fence", True)
                )
        return result


def _linear_color(hex_color: str) -> tuple[float, ...]:
    values = [int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in values)


def track_usda(spec: TrackSpec) -> str:
    """Create a self-contained static lane; the common terrain supplies floor collision."""
    lines = [
        '#usda 1.0\n(\n    defaultPrim = "RaceTrack"\n    metersPerUnit = 1\n    upAxis = "Z"\n)',
        'def Xform "RaceTrack"\n{',
        '    def Scope "Looks"\n    {',
    ]
    for name, color in COLORS.items():
        lines += [
            f'        def Material "{name}" {{',
            f"            token outputs:surface.connect = </RaceTrack/Looks/{name}/Shader.outputs:surface>",
            '            def Shader "Shader" {',
            '                uniform token info:id = "UsdPreviewSurface"',
            f"                color3f inputs:diffuseColor = {_linear_color(color)}",
            "                float inputs:roughness = 0.85",
            "                token outputs:surface",
            "            }\n        }",
        ]
    lines += [
        "    }",
        '    def Material "FencePhysics" (prepend apiSchemas = ["PhysicsMaterialAPI"]) {',
        "        float physics:staticFriction = 1",
        "        float physics:dynamicFriction = 1",
        "        float physics:restitution = 0",
        "    }",
    ]
    for part in spec.parts():
        schemas = '["MaterialBindingAPI", "PhysicsCollisionAPI"]' if part.collision else '["MaterialBindingAPI"]'
        kind = "Cube" if part.collision else "Mesh"
        lines += [f'    def {kind} "{part.name}" (prepend apiSchemas = {schemas}) {{']
        if part.collision:
            lines += [
                "        double size = 1",
                f"        double3 xformOp:translate = {part.center}",
                f"        double3 xformOp:scale = {part.size}",
                '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
                "        bool physics:collisionEnabled = true",
                "        rel material:binding:physics = </RaceTrack/FencePhysics>",
            ]
        else:
            x, y, z = part.center
            hx, hy = part.size[0] / 2, part.size[1] / 2
            points = [(x - hx, y - hy, z), (x + hx, y - hy, z), (x + hx, y + hy, z), (x - hx, y + hy, z)]
            lines += [
                f"        point3f[] points = {points}",
                "        int[] faceVertexCounts = [4]",
                "        int[] faceVertexIndices = [0, 1, 2, 3]",
                '        uniform token subdivisionScheme = "none"',
                "        bool doubleSided = true",
            ]
        lines += [
            f"        color3f[] primvars:displayColor = [{_linear_color(COLORS[part.material])}]",
            f"        rel material:binding = </RaceTrack/Looks/{part.material}>",
            "    }",
        ]
    return "\n".join([*lines, "}", ""])


def ensure_track_asset(spec: TrackSpec, cache_dir: Path | None = None) -> Path:
    """Atomically cache by content, independent of cwd and safe for concurrent workers."""
    content = track_usda(spec)
    root = cache_dir if cache_dir is not None else Path(tempfile.gettempdir()) / "miislab_race_tracks"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"track_{hashlib.sha256(content.encode()).hexdigest()[:20]}.usda"
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return path.resolve()
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=root, suffix=".tmp", delete=False
    ) as f:
        temporary = Path(f.name)
        f.write(content)
    try:
        try:
            os.replace(temporary, path)
        except PermissionError:
            # Windows can deny replacement while another worker reads the completed asset.
            # Only accept the race if the winner published precisely the same content.
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise
    finally:
        temporary.unlink(missing_ok=True)
    return path.resolve()
