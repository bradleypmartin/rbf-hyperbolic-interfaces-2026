"""Two-panel animation: naive RBF-FD vs interface-aware RBF-FD on the same nodes.

A plane pressure pulse travels down through a band of stiffer, denser
material (dissertation §3.4.1). Top row: |v|, the vertical particle
velocity, from the two solvers. Bottom row: their errors on one colour
scale, against the exact solution (flat interfaces) or a fine
interface-aware run (curved interfaces). The snapshot grid for the slides
shows the reference wave once, then the two error maps: at this
resolution the two solvers' waves are indistinguishable by eye, so the
second wave column only added clutter. Everything is computed on
scattered nodes and resampled to a pixel grid for display with one-sided
stencils (``wave2d/resample.py``), so the interpolation never crosses an
interface.

    uv run python scripts/wave2d_demo.py                    # flat, 10000 nodes
    uv run python scripts/wave2d_demo.py --amplitude 0.02   # curved (§3.4.2)
    uv run python scripts/wave2d_demo.py --n 2500 --frames 60 --out outputs/quick.mp4
    uv run python scripts/wave2d_demo.py --amplitude 0.02 --png-only   # snapshot only
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

from pdes_demo.plotting import (
    ERROR_CMAP,
    FIELD_CMAP,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    use_demo_style,
)
from pdes_demo.wave2d import (
    LayeredMedium2D,
    NodeSet,
    SineInterface,
    exact_plane_wave,
    make_node_set,
    pixel_grid,
    resample_matrix,
    run,
)

MODES = ("naive", "aware")
TITLES = {
    "naive": "Standard RBF-FD (naive)",
    "aware": "Interface-aware RBF-FD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--n", type=int, default=10000, help="nodes, a perfect square (default 10000)"
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.0,
        help="interface curvature: 0 = flat, exact reference (default); "
        "0.02 = dissertation §3.4.2, reference is a fine interface-aware run",
    )
    parser.add_argument(
        "--ref-n",
        type=int,
        default=None,
        help="nodes of the reference run for curved interfaces (default 4 x n)",
    )
    parser.add_argument("--t-end", type=float, default=0.5)
    parser.add_argument("--frames", type=int, default=250)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--pixels", type=int, default=250, help="display grid resolution per side"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="MP4 path (default outputs/wave2d_naive_vs_aware.mp4, "
        "with a _curved suffix when --amplitude is nonzero)",
    )
    parser.add_argument(
        "--snapshots",
        type=float,
        nargs="*",
        default=[0.15, 0.3, 0.45],
        help="times for a static PNG grid saved next to the video",
    )
    parser.add_argument(
        "--png-only",
        action="store_true",
        help="write the snapshot PNG and skip the video (ffmpeg output differs "
        "between runs, so this avoids touching a committed clip)",
    )
    return parser.parse_args()


def reference_v(
    nodes: NodeSet, medium: LayeredMedium2D, times: np.ndarray, args: argparse.Namespace
) -> tuple[np.ndarray, str]:
    """v at the nodes for every frame, and a label saying where it came from."""
    if medium.is_flat:
        v = np.array([exact_plane_wave(nodes, t, medium)[1] for t in times])
        return v, "the exact solution"
    ref_n = args.ref_n or 4 * args.n
    t0 = time.perf_counter()
    fine = make_node_set(medium, ref_n, seed=args.seed)
    ref = run(
        fine,
        medium,
        mode="aware",
        t_end=args.t_end,
        n_snapshots=args.frames,
        align_snapshots=True,
    )
    if not np.allclose(ref.t, times):
        raise RuntimeError("reference snapshots are not aligned with the runs")
    onto = resample_matrix(fine, nodes.xy, medium)
    v = np.array([onto @ frame for frame in ref.field("v")])
    print(f"reference: {ref_n} nodes, aware, {time.perf_counter() - t0:.1f}s")
    return v, f"a {ref_n}-node interface-aware run"


def style_map(ax: plt.Axes, medium: LayeredMedium2D) -> None:
    ax.grid(False)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(INK_MUTED)
    xs = np.linspace(0, 1, 400)
    for ifc in medium.interfaces:
        ax.plot(xs, ifc.height(xs), "--", color=INK_SECONDARY, lw=1.0)


def main() -> None:
    args = parse_args()
    medium = LayeredMedium2D(
        lower=SineInterface(0.25, args.amplitude),
        upper=SineInterface(0.5, args.amplitude),
    )
    if args.out is None:
        suffix = "_curved" if args.amplitude else ""
        args.out = Path(f"outputs/wave2d_naive_vs_aware{suffix}.mp4")

    t0 = time.perf_counter()
    nodes = make_node_set(medium, args.n, seed=args.seed)
    runs = {
        mode: run(
            nodes,
            medium,
            mode=mode,
            t_end=args.t_end,
            n_snapshots=args.frames,
            align_snapshots=True,
        )
        for mode in MODES
    }
    times = runs["aware"].t
    if not np.allclose(runs["naive"].t, times):
        raise RuntimeError("the two runs must share their snapshot times")
    print(
        f"{nodes.n} nodes, {len(times) - 1} frames to t = {args.t_end:g}, "
        f"{round(args.t_end / runs['aware'].dt)} RK4 steps: "
        f"{time.perf_counter() - t0:.1f}s"
    )
    v_ref, ref_label = reference_v(nodes, medium, times, args)
    if medium.is_flat:
        error_title, ref_short = "error vs exact solution", "exact solution"
    else:
        error_title, ref_short = "error vs reference", "reference run"
    v = {mode: runs[mode].field("v") for mode in MODES}
    diff = {mode: v[mode] - v_ref for mode in MODES}
    rel = {
        mode: np.linalg.norm(diff[mode], axis=1) / np.linalg.norm(v_ref, axis=1)
        for mode in MODES
    }
    for k in np.linspace(0, len(times) - 1, 6).astype(int):
        print(
            f"  t = {times[k]:.2f}  rel. error in v: naive {rel['naive'][k]:.2e}, "
            f"aware {rel['aware'][k]:.2e}"
        )

    # One colour scale for both error maps, saturating the naive method's
    # worst spots so that the aware panel keeps some contrast.
    err_lim = 0.75 * np.abs(diff["naive"]).max()
    grid = pixel_grid(args.pixels)
    to_grid = resample_matrix(nodes, grid, medium)

    def image(values: np.ndarray) -> np.ndarray:
        return np.abs(to_grid @ values).reshape(args.pixels, args.pixels)

    imshow_kw = dict(origin="lower", extent=(0, 1, 0, 1), interpolation="bilinear")
    field_kw = dict(cmap=FIELD_CMAP, vmin=0, vmax=1, **imshow_kw)
    error_kw = dict(cmap=ERROR_CMAP, vmin=0, vmax=err_lim, **imshow_kw)
    text_kw = dict(fontsize=10, color=INK, va="top", ha="left")

    use_demo_style()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shape = "curved" if args.amplitude else "flat"
    caption = (
        "Pressure pulse hitting a band with 4x stiffness and 2x density, "
        f"{shape} interfaces  |  {nodes.n} scattered nodes, RBF-FD, RK4"
    )

    # --- static snapshot grid for the slides ------------------------------
    if args.snapshots:
        n_rows = len(args.snapshots)
        fig, axes = plt.subplots(
            n_rows, 3, figsize=(10, 3.2 * n_rows + 1.2), constrained_layout=True
        )
        axes = np.atleast_2d(axes)
        for r, t_snap in enumerate(args.snapshots):
            k = int(np.argmin(np.abs(times - t_snap)))
            ax_v = axes[r, 0]
            im_v = ax_v.imshow(image(v_ref[k]), **field_kw)
            style_map(ax_v, medium)
            if r == 0:
                ax_v.set_title(f"The wave ({ref_short})\n|v|", fontsize=11)
            for c, mode in enumerate(MODES):
                ax_e = axes[r, 1 + c]
                im_e = ax_e.imshow(image(diff[mode][k]), **error_kw)
                style_map(ax_e, medium)
                ax_e.text(
                    0.03,
                    0.97,
                    f"rel. error {rel[mode][k]:.1%}",
                    transform=ax_e.transAxes,
                    **text_kw,
                )
                if r == 0:
                    ax_e.set_title(f"{TITLES[mode]}\n{error_title}", fontsize=11)
            axes[r, 0].set_ylabel(f"t = {times[k]:.2f}\ny")
        for ax in axes[-1]:
            ax.set_xlabel("x")
        fig.colorbar(
            im_v,
            ax=axes[:, 0].tolist(),
            location="bottom",
            shrink=0.8,
            pad=0.02,
            label="|v|, vertical particle velocity",
        )
        fig.colorbar(
            im_e,
            ax=axes[:, 1:].ravel().tolist(),
            location="bottom",
            shrink=0.5,
            pad=0.02,
            label=f"|error in v| vs {ref_label}, one colour scale",
        )
        fig.suptitle(caption.replace("  |  ", "\n"), fontsize=11)
        png = args.out.with_suffix(".png")
        fig.savefig(png, dpi=160)
        plt.close(fig)
        print(f"wrote {png}")
    if args.png_only:
        return

    # --- animation ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10.4), constrained_layout=True)
    images: dict[str, tuple] = {}
    labels: dict[str, plt.Text] = {}
    for c, mode in enumerate(MODES):
        top, bottom = axes[0, c], axes[1, c]
        im_v = top.imshow(image(v[mode][0]), **field_kw)
        im_e = bottom.imshow(image(diff[mode][0]), **error_kw)
        for ax in (top, bottom):
            style_map(ax, medium)
        top.set_title(f"{TITLES[mode]}: |v|")
        bottom.set_title(error_title)
        bottom.set_xlabel("x")
        labels[mode] = bottom.text(
            0.03, 0.97, "", transform=bottom.transAxes, **text_kw
        )
        images[mode] = (im_v, im_e)
    axes[0, 0].set_ylabel("y")
    axes[1, 0].set_ylabel("y")
    axes[0, 0].text(
        0.02,
        0.375,
        "stiffer, denser band",
        fontsize=9,
        color=INK_SECONDARY,
        va="center",
    )
    fig.colorbar(
        images["aware"][0], ax=axes[0].tolist(), shrink=0.8, pad=0.02, label="|v|"
    )
    fig.colorbar(
        images["aware"][1],
        ax=axes[1].tolist(),
        shrink=0.8,
        pad=0.02,
        label=f"|error in v| vs {ref_label}",
    )
    clock = axes[0, 1].text(
        0.97, 0.97, "", transform=axes[0, 1].transAxes, ha="right", va="top", color=INK
    )
    fig.suptitle(caption, fontsize=11)

    def update(k: int) -> list:
        artists = []
        for mode in MODES:
            im_v, im_e = images[mode]
            im_v.set_data(image(v[mode][k]))
            im_e.set_data(image(diff[mode][k]))
            labels[mode].set_text(f"rel. error {rel[mode][k]:.1%}")
            artists += [im_v, im_e, labels[mode]]
        clock.set_text(f"t = {times[k]:.2f}")
        return artists + [clock]

    t0 = time.perf_counter()
    anim = FuncAnimation(fig, update, frames=len(times), blit=False)
    anim.save(args.out, writer=FFMpegWriter(fps=args.fps, bitrate=3000), dpi=120)
    print(
        f"wrote {args.out}  ({len(times)} frames at {args.fps} fps, "
        f"{time.perf_counter() - t0:.0f}s to render)"
    )


if __name__ == "__main__":
    main()
