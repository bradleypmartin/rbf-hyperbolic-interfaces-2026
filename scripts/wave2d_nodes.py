"""Node set for the 2-D problem: fixed straddling rows + repulsion-relaxed field.

Left: the whole periodic unit square with both interfaces. Right: a zoom on
the upper interface showing the three hex-staggered rows on each side and
the relaxed nodes beyond them (dissertation Fig. 3-3, all-RBF variant).

    uv run python scripts/wave2d_nodes.py                      # n = 2500, curved
    uv run python scripts/wave2d_nodes.py --n 10000 --amplitude 0
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pdes_demo.plotting import (
    AWARE,
    INK_MUTED,
    INK_SECONDARY,
    NAIVE,
    SURFACE,
    use_demo_style,
)
from pdes_demo.wave2d import (
    LayeredMedium2D,
    SineInterface,
    make_node_set,
    nearest_spacing,
    periodic_knn,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=2500, help="nodes (perfect square)")
    parser.add_argument("--amplitude", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=100, help="repulsion steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/wave2d_nodes.png"))
    args = parser.parse_args()

    medium = LayeredMedium2D(
        lower=SineInterface(0.25, args.amplitude),
        upper=SineInterface(0.5, args.amplitude),
    )
    t0 = time.perf_counter()
    nodes = make_node_set(medium, args.n, repulsion_steps=args.steps, seed=args.seed)
    elapsed = time.perf_counter() - t0
    d = nearest_spacing(nodes) / nodes.h
    print(
        f"{nodes.n} nodes in {elapsed:.1f}s; nearest-neighbour spacing / h: "
        f"min {d.min():.3f}, mean {d.mean():.3f}, max {d.max():.3f}"
    )

    use_demo_style()
    fig, (full, zoom) = plt.subplots(
        1, 2, figsize=(12, 5.2), width_ratios=[1, 1.15], constrained_layout=True
    )
    xs = np.linspace(0, 1, 400)
    # The zoom panel gets markers three times the area of the full panel's:
    # the figure lands at about a third of its size on the slide.
    for ax, scale in ((full, 1), (zoom, 3)):
        ax.grid(False)
        ax.set_aspect("equal")
        for ifc in medium.interfaces:
            ax.plot(xs, ifc.height(xs), "--", color=INK_SECONDARY, lw=1.2, zorder=3)
        ax.scatter(
            nodes.x[~nodes.fixed],
            nodes.y[~nodes.fixed],
            s=4 * scale,
            color=INK_MUTED,
            lw=0,
        )
        ax.scatter(
            nodes.x[nodes.fixed], nodes.y[nodes.fixed], s=6 * scale, color=AWARE, lw=0
        )
    full.set_xlim(0, 1)
    full.set_ylim(0, 1)
    full.set_title(f"{nodes.n} nodes on the periodic unit square")
    full.set_xlabel("x")
    full.set_ylabel("y")

    w = 12 * nodes.h
    zoom.set_xlim(0.5 - w, 0.5 + w)
    zoom.set_ylim(medium.upper.y0 - w * 0.8, medium.upper.y0 + w * 0.8)
    zoom.set_title("Zoom on the upper interface")
    zoom.set_xlabel("x")
    zoom.scatter([], [], s=60, color=AWARE, label="fixed rows straddling the interface")
    zoom.scatter([], [], s=60, color=INK_MUTED, label="repulsion-relaxed nodes")
    zoom.plot([], [], "--", color=INK_SECONDARY, label="material interface")
    # A near-opaque box so the legend reads over the node cloud.
    zoom.legend(
        loc="upper right",
        fontsize=14,
        frameon=True,
        framealpha=0.8,
        facecolor=SURFACE,
        edgecolor=INK_MUTED,
    )
    # Mark one node's 30-nearest stencil to show it draws on both sides.
    i = int(np.argmin(np.abs(nodes.x - 0.5) + np.abs(nodes.y - medium.upper.y0)))
    idx, _ = periodic_knn(nodes.xy, 30, query=nodes.xy[[i]])
    zoom.scatter(
        nodes.x[idx[0]],
        nodes.y[idx[0]],
        s=70,
        facecolors="none",
        edgecolors=NAIVE,
        lw=1.2,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
