"""Stiff smooth edges: standard FD4 vs ODE-continued seed stencils (issue #27).

The layer of ``wave1d_convergence.py`` with its two edges smoothed into tanh
transitions of width delta. The panels sweep delta from the dissertation's
jump (delta = 0, ray-sum reference) through two edges the coarse grids
cannot resolve to one every grid resolves. References for delta > 0 are
pseudo-spectral and cached under ``outputs/``. The pulse is wider than the
dissertation's (sharpness 60, centre -0.6) so that it is resolved on 100
nodes and the edge is the only thing under test.

Also writes a snapshot at t = 1 on 100 nodes through an edge of width h/8,
and the seeds themselves for one stencil across an edge. The errors and
rates go to ``outputs/wave1d_stiff.json`` (and ``--data-dir``, #54); the
convergence and seed figures are drawn by ``pdes_demo.stiff_figures`` from
those records, so ``scripts/paper_figures.py`` can redraw them in print
style without this run. ``--style print --format pdf`` draws them that way
here.

    uv run python scripts/wave1d_stiff.py
    uv run python scripts/wave1d_stiff.py --widths 0 0.001 --ns 100 200 400 800 1600
    uv run python scripts/wave1d_stiff.py --style print --format pdf \\
        --data-dir paper/data
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pdes_demo.plotting import (
    COLORS,
    INK,
    INK_MUTED,
    LAYER_FILL,
    PRINT_LABELS,
    TEXTWIDTH_IN,
    use_demo_style,
    use_print_style,
)
from pdes_demo.results_cache import ResultsCache
from pdes_demo.stiff_figures import DEMO_LABELS_1D, convergence_1d, seeds_1d
from pdes_demo.wave1d import LayeredMedium, exact_solution, periodic_grid, run
from pdes_demo.wave1d.spectral import interpolate, reference_size, run_spectral

REF_DT = 5e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--widths", type=float, nargs="+", default=[0.0, 0.0025, 0.01, 0.04]
    )
    parser.add_argument("--ns", type=int, nargs="+", default=[100, 200, 400, 800, 1600])
    parser.add_argument("--t-end", type=float, default=1.0)
    parser.add_argument("--sharpness", type=float, default=60.0)
    parser.add_argument("--center", type=float, default=-0.6)
    parser.add_argument("--snapshot-n", type=int, default=100)
    parser.add_argument("--snapshot-width", type=float, default=0.0025)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="also write the results JSON here (paper/data for the committed copy)",
    )
    parser.add_argument(
        "--style",
        choices=["demo", "print"],
        default="demo",
        help="deck style (default) or the manuscript's print style",
    )
    parser.add_argument("--format", choices=["png", "pdf"], default="png")
    return parser.parse_args()


def reference(
    medium: LayeredMedium, t_end: float, sharpness: float, center: float, out: Path
) -> tuple:
    """Pseudo-spectral reference at ``t_end``, cached as an npz in ``out``."""
    n_ref = reference_size(medium.edge_width)
    key = (
        f"wave1d_stiff_ref_w{medium.edge_width:g}_s{sharpness:g}_c{center:g}"
        f"_t{t_end:g}_n{n_ref}_dt{REF_DT:g}.npz"
    )
    path = out / key
    if path.exists():
        data = np.load(path)
        return periodic_grid(n_ref), data["u"], data["f"]
    t0 = time.perf_counter()
    grid, snaps = run_spectral(
        medium,
        n_ref,
        t_end=t_end,
        dt=REF_DT,
        n_snapshots=1,
        pulse_center=center,
        pulse_sharpness=sharpness,
    )
    np.savez(path, u=snaps.u[-1], f=snaps.f[-1])
    print(f"  reference n = {n_ref} in {time.perf_counter() - t0:.1f}s -> {path}")
    return grid, snaps.u[-1], snaps.f[-1]


def reference_at(
    x: np.ndarray, medium: LayeredMedium, args: argparse.Namespace
) -> np.ndarray:
    """Reference stress at ``t_end`` on the points ``x``."""
    if not medium.is_smooth:
        _, f = exact_solution(x, args.t_end, medium, args.center, args.sharpness)
        return f
    grid, _, f = reference(
        medium, args.t_end, args.sharpness, args.center, args.out_dir
    )
    return interpolate(f, grid, x)


def rel_error(n: int, medium: LayeredMedium, mode: str, args: argparse.Namespace):
    grid = periodic_grid(n)
    snaps = run(
        grid,
        medium,
        mode=mode,
        t_end=args.t_end,
        n_snapshots=1,
        pulse_center=args.center,
        pulse_sharpness=args.sharpness,
    )
    f_ref = reference_at(grid.x, medium, args)
    return float(np.linalg.norm(snaps.f[-1] - f_ref) / np.linalg.norm(f_ref))


def rates(ns: np.ndarray, errors: list[float]) -> np.ndarray:
    e = np.array(errors)
    return np.log(e[:-1] / e[1:]) / np.log(ns[1:] / ns[:-1])


def convergence_figure(args: argparse.Namespace, cache: ResultsCache) -> None:
    """The sweep: errors and rates into ``cache``, printed, then drawn."""
    ns = np.array(args.ns, dtype=float)
    for width in args.widths:
        medium = LayeredMedium(edge_width=width)
        title = (
            "jump edges (delta = 0)" if width == 0 else f"edge width delta = {width:g}"
        )
        print(f"\n{title}")
        t0 = time.perf_counter()
        errors = {
            mode: [rel_error(n, medium, mode, args) for n in args.ns]
            for mode in ("naive", "aware")
        }
        print(f"  ({time.perf_counter() - t0:.1f}s)")
        print("      N   h/delta      naive   rate      aware   rate")
        r = {mode: rates(ns, errors[mode]) for mode in errors}
        for mode in errors:
            cache.add_errors(
                args.ns,
                errors[mode],
                list(r[mode]),
                h=[2 / n for n in args.ns],
                delta=width,
                mode=mode,
                field="f",
            )
        for i, n in enumerate(args.ns):
            h_over = f"{2 / n / width:8.2f}" if width else "     inf"
            rn = f"{r['naive'][i - 1]:5.1f}" if i else "     "
            ra = f"{r['aware'][i - 1]:5.1f}" if i else "     "
            print(
                f"  {n:5d}  {h_over}  {errors['naive'][i]:9.2e} {rn}  "
                f"{errors['aware'][i]:9.2e} {ra}"
            )
    out = convergence_1d(
        cache,
        args.out_dir / f"wave1d_stiff_convergence.{args.format}",
        print_mode=args.style == "print",
    )
    print(f"\nwrote {out}")


def snapshot_figure(args: argparse.Namespace, cache: ResultsCache) -> None:
    print_mode = args.style == "print"
    labels = PRINT_LABELS if print_mode else DEMO_LABELS_1D
    medium = LayeredMedium(edge_width=args.snapshot_width)
    grid = periodic_grid(args.snapshot_n)
    x_fine = np.linspace(-1, 1, 2001)
    f_ref_fine = reference_at(x_fine, medium, args)
    f_ref = reference_at(grid.x, medium, args)
    sols = {
        mode: run(
            grid,
            medium,
            mode=mode,
            t_end=args.t_end,
            n_snapshots=1,
            pulse_center=args.center,
            pulse_sharpness=args.sharpness,
        ).f[-1]
        for mode in ("naive", "aware")
    }

    fig, (ax, ax_err) = plt.subplots(
        2,
        1,
        figsize=(TEXTWIDTH_IN, 0.6 * TEXTWIDTH_IN) if print_mode else (9, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.3]},
        constrained_layout=True,
    )
    lw = 1.1 if print_mode else 1.6
    for a in (ax, ax_err):
        a.axvspan(medium.layer_start, medium.layer_end, color=LAYER_FILL, zorder=0)
    ax.plot(
        x_fine,
        f_ref_fine,
        color=INK,
        lw=0.9 if print_mode else 1.4,
        label="reference (spectral)",
    )
    for mode in ("naive", "aware"):
        ax.plot(grid.x, sols[mode], "-", color=COLORS[mode], lw=lw, label=labels[mode])
        ax_err.plot(grid.x, sols[mode] - f_ref, "-", color=COLORS[mode], lw=lw)
    ax.set_ylabel("stress $f$" if print_mode else "stress f")
    ax.legend(loc="upper left", fontsize=6.5 if print_mode else 9)
    h = grid.h
    if not print_mode:
        ax.set_title(
            f"t = {args.t_end:g}, {grid.n} nodes, edge width delta = "
            f"{medium.edge_width:g} = h/{h / medium.edge_width:.0f}: the layer's "
            "edges fall between nodes",
            fontsize=11,
        )
    ax_err.set_ylabel("error")
    ax_err.set_xlabel("$x$" if print_mode else "x")
    ax_err.axhline(0, color=INK_MUTED, lw=0.8)
    out = args.out_dir / f"wave1d_stiff_snapshot.{args.format}"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    errs = {m: np.abs(sols[m] - f_ref).max() for m in sols}
    for mode, err in errs.items():
        cache.add(
            "snapshot",
            n=grid.n,
            delta=medium.edge_width,
            t=args.t_end,
            mode=mode,
            field="max_abs_f",
            error=err,
        )
    print(
        f"wrote {out}  (max error naive {errs['naive']:.2e}, seeds {errs['aware']:.2e})"
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.style == "print":
        use_print_style()
    else:
        use_demo_style()
    cache = ResultsCache.new("scripts/wave1d_stiff.py", args)
    convergence_figure(args, cache)
    snapshot_figure(args, cache)
    out = seeds_1d(
        args.out_dir / f"wave1d_stiff_seeds.{args.format}",
        print_mode=args.style == "print",
    )
    print(f"wrote {out}")
    paths = [args.out_dir / "wave1d_stiff.json"]
    if args.data_dir is not None:
        paths.append(args.data_dir / "wave1d_stiff.json")
    cache.write(*paths)
    print("results ->", ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
