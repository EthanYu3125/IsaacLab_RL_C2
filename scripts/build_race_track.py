"""Export a deterministic race asset and an optional geometry preview, without Isaac Lab."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from Miislab_humanoid_gait_RL.assets.race_track import COLORS, TrackSpec, track_usda


def render_preview(spec: TrackSpec, output: Path) -> None:
    """Render the same primitive specification as the USD, not a simulator screenshot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(15, 8), facecolor="#eff3ef")
    ax = fig.add_axes((0.02, 0.20, 0.96, 0.72), projection="3d", computed_zorder=False)
    ax.set_facecolor("#eff3ef")
    ax.set_proj_type("ortho")
    for part in spec.parts():
        x, y, z = part.center
        hx, hy, hz = (v / 2 for v in part.size)
        vertices = [(x + sx * hx, y + sy * hy, z + sz * hz) for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)]
        faces = [[vertices[i] for i in indices] for indices in ((0, 1, 3, 2), (4, 5, 7, 6))]
        if part.collision:
            faces += [
                [vertices[i] for i in indices] for indices in ((0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))
            ]
        ax.add_collection3d(
            Poly3DCollection(
                faces,
                facecolors="#" + COLORS[part.material],
                edgecolors="#946a16" if part.collision else "none",
                linewidths=0.35,
            )
        )
    ax.scatter([0], [0], [0.74], color="#2366a8", s=35, zorder=10)
    ax.set(xlim=(-2.2, spec.length + 2.2), ylim=(-spec.width / 2 - 1, spec.width / 2 + 1), zlim=(0, 1.6))
    ax.set_box_aspect((spec.length + 4.4, spec.width + 2, 1.6))
    ax.view_init(elev=38, azim=-115)
    ax.set_axis_off()
    top = fig.add_axes((0.08, 0.06, 0.84, 0.20))
    for part in spec.parts():
        x, y, _ = part.center
        sx, sy, _ = part.size
        top.add_patch(Rectangle((x - sx / 2, y - sy / 2), sx, sy, color="#" + COLORS[part.material]))
    top.scatter([0], [0], color="#2366a8", s=35)
    top.annotate("Spawn / start", (0, 0), (1, -spec.width / 2 - 0.7), arrowprops={"arrowstyle": "->"})
    top.annotate("Finish", (spec.length, 0), (spec.length - 3, -spec.width / 2 - 0.7), arrowprops={"arrowstyle": "->"})
    top.set(xlim=(-3, spec.length + 3), ylim=(-spec.width / 2 - 1, spec.width / 2 + 0.5), aspect="equal")
    top.axis("off")
    fig.suptitle(f"G1 race lane | {spec.length:g} m x {spec.width:g} m clear width", fontsize=20, y=0.96)
    fig.text(
        0.5,
        0.90,
        "Geometry preview from the USD specification - not a physics simulation",
        ha="center",
        color="#526052",
    )
    fig.savefig(output, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=float, default=TrackSpec().length)
    parser.add_argument("--width", type=float, default=TrackSpec().width)
    parser.add_argument("--output", type=Path, default=Path("docs/race_track.usda"))
    parser.add_argument("--preview", type=Path, help="Optional PNG path; requires matplotlib")
    args = parser.parse_args()
    spec = TrackSpec(args.length, args.width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(track_usda(spec), encoding="utf-8", newline="\n")
    print(args.output.resolve())
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        render_preview(spec, args.preview)
        print(args.preview.resolve())


if __name__ == "__main__":
    main()
