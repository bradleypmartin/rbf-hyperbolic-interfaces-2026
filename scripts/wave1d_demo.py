"""Two-panel animation: naive FD vs interface-aware FD on the same grid.

A right-going Gaussian stress pulse hits a layer with a different wave speed
and density. Left: standard 4th-order finite differences straight across
the interfaces. Right: the dissertation's interface-aware stencils. A strip
below each panel shows the error against the exact solution, which is
computed but not drawn: the error strips carry the comparison.

    uv run python scripts/wave1d_demo.py                  # defaults, MP4 + PNG
    uv run python scripts/wave1d_demo.py --layer-width 0.01 --out outputs/thin.mp4
    uv run python scripts/wave1d_demo.py --n 100 --sharpness 150 \
        --out outputs/wave1d_naive_vs_aware_coarse.mp4    # ringing visible to the eye
    uv run python scripts/wave1d_demo.py --n 100 --sharpness 150 \
        --out outputs/wave1d_naive_vs_aware_coarse.mp4 --png-only   # snapshot only
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

from rbf_hyperbolic_interfaces.plotting import (
    COLORS,
    INK_MUTED,
    INK_SECONDARY,
    LABELS,
    LAYER_FILL,
    use_demo_style,
)
from rbf_hyperbolic_interfaces.wave1d import (
    LayeredMedium,
    Material,
    exact_solution,
    periodic_grid,
    run,
)

MODES = ("naive", "aware")
TITLES = {
    "naive": "Standard finite differences",
    "aware": "Interface-aware finite differences",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=400, help="grid nodes (default 400)")
    parser.add_argument("--c2", type=float, default=2.0, help="wave speed in the layer")
    parser.add_argument("--rho2", type=float, default=1.0, help="density in the layer")
    parser.add_argument("--layer-width", type=float, default=0.5)
    parser.add_argument(
        "--sharpness",
        type=float,
        default=600.0,
        help="pulse exp(-s (x - x0)^2); 600 is the dissertation pulse, 150 is wide "
        "enough to resolve on 100 nodes",
    )
    parser.add_argument("--t-end", type=float, default=1.25)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/wave1d_naive_vs_aware.mp4")
    )
    parser.add_argument(
        "--snapshots",
        type=float,
        nargs="*",
        default=[0.4, 0.8, 1.0, 1.2],
        help="times for a static PNG grid saved next to the video",
    )
    parser.add_argument(
        "--png-only",
        action="store_true",
        help="write the snapshot PNG and skip the video (ffmpeg output differs "
        "between runs, so this avoids touching a committed clip)",
    )
    return parser.parse_args()


def shade_layer(ax: plt.Axes, medium: LayeredMedium) -> None:
    ax.axvspan(medium.layer_start, medium.layer_end, color=LAYER_FILL, zorder=0)
    for xi in (medium.layer_start, medium.layer_end):
        ax.axvline(xi, color=INK_MUTED, lw=1, zorder=1)


def main() -> None:
    args = parse_args()
    medium = LayeredMedium(
        layer=Material(c=args.c2, rho=args.rho2), layer_width=args.layer_width
    )
    grid = periodic_grid(args.n)
    runs = {
        mode: run(
            grid,
            medium,
            mode=mode,
            t_end=args.t_end,
            n_snapshots=args.frames,
            pulse_sharpness=args.sharpness,
        )
        for mode in MODES
    }
    times = runs["aware"].t
    exact = np.array(
        [exact_solution(grid.x, t, medium, sharpness=args.sharpness)[1] for t in times]
    )
    errors = {mode: np.abs(runs[mode].f - exact) for mode in MODES}
    ylim = 1.15 * max(np.abs(exact).max(), 1.0)
    err_lim = 1.1 * max(errors["naive"].max(), 1e-3)

    use_demo_style()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    caption = (
        f"Pulse hitting a layer with {args.c2:g}x wave speed and {args.rho2:g}x density"
        f"  |  {args.n} grid nodes, 4th-order stencils, RK4"
    )

    # --- static snapshot grid for the slides ------------------------------
    if args.snapshots:
        n_rows = len(args.snapshots)
        fig, axes = plt.subplots(
            n_rows,
            2,
            figsize=(11, 2.4 * n_rows + 0.6),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        for r, t_snap in enumerate(args.snapshots):
            k = int(np.argmin(np.abs(times - t_snap)))
            for c, mode in enumerate(MODES):
                ax = axes[r, c]
                shade_layer(ax, medium)
                ax.plot(grid.x, runs[mode].f[k], color=COLORS[mode])
                ax.set_xlim(-1, 1)
                ax.set_ylim(-ylim, ylim)
                if r == 0:
                    ax.set_title(TITLES[mode])
                if c == 0:
                    # Large because the figure is scaled to about a quarter
                    # size on the slide; this lands near 7pt there.
                    ax.set_ylabel(f"t = {times[k]:.1f}", fontsize=28)
        for c, mode in enumerate(MODES):
            axes[0, c].plot([], [], color=COLORS[mode], label=LABELS[mode])
            axes[0, c].legend(loc="upper left", fontsize=10)
            axes[-1, c].set_xlabel("position x")
        fig.suptitle(caption, fontsize=11)
        png = args.out.with_suffix(".png")
        fig.savefig(png, dpi=160)
        plt.close(fig)
        print(f"wrote {png}")
    if args.png_only:
        return

    # --- animation ---------------------------------------------------------
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 6.2),
        sharex=True,
        height_ratios=[3, 1],
        constrained_layout=True,
    )
    lines: dict[str, tuple] = {}
    for c, mode in enumerate(MODES):
        top, bottom = axes[0, c], axes[1, c]
        shade_layer(top, medium)
        shade_layer(bottom, medium)
        (line,) = top.plot(grid.x, runs[mode].f[0], color=COLORS[mode])
        (err,) = bottom.plot(grid.x, errors[mode][0], color=COLORS[mode], lw=1.5)
        top.set_title(TITLES[mode])
        top.set_xlim(-1, 1)
        top.set_ylim(-ylim, ylim)
        bottom.set_ylim(0, err_lim)
        bottom.set_xlabel("position x")
        lines[mode] = (line, err)
    axes[0, 0].set_ylabel("stress f")
    axes[1, 0].set_ylabel("error vs exact")
    axes[0, 1].tick_params(labelleft=False)
    axes[1, 1].tick_params(labelleft=False)
    axes[0, 1].sharey(axes[0, 0])
    axes[1, 1].sharey(axes[1, 0])
    clock = axes[0, 1].text(
        0.98, 0.92, "", transform=axes[0, 1].transAxes, ha="right", color=INK_SECONDARY
    )
    fig.suptitle(caption, fontsize=11)

    def update(k: int) -> list:
        artists = []
        for mode in MODES:
            line, err = lines[mode]
            line.set_ydata(runs[mode].f[k])
            err.set_ydata(errors[mode][k])
            artists += [line, err]
        clock.set_text(f"t = {times[k]:.2f}")
        return artists + [clock]

    anim = FuncAnimation(fig, update, frames=len(times), blit=False)
    anim.save(args.out, writer=FFMpegWriter(fps=args.fps, bitrate=2400), dpi=120)
    print(f"wrote {args.out}  ({len(times)} frames at {args.fps} fps)")


if __name__ == "__main__":
    main()
