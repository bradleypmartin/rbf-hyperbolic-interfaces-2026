"""Stiff smooth edges: standard FD4 vs ODE-continued seed stencils (issue #27).

The layer of ``wave1d_convergence.py`` with its two edges smoothed into tanh
transitions of width delta. The panels sweep delta from the dissertation's
jump (delta = 0, ray-sum reference) through two edges the coarse grids
cannot resolve to one every grid resolves. References for delta > 0 are
pseudo-spectral and cached under ``outputs/``. The pulse is wider than the
dissertation's (sharpness 60, centre -0.6) so that it is resolved on 100
nodes and the edge is the only thing under test.

Also writes a snapshot at t = 1 on 100 nodes through an edge of width h/8,
and the seeds themselves for one stencil across an edge.

    uv run python scripts/wave1d_stiff.py
    uv run python scripts/wave1d_stiff.py --widths 0 0.001 --ns 100 200 400 800 1600
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
    INK_SECONDARY,
    LABELS,
    LAYER_FILL,
    use_demo_style,
)
from pdes_demo.wave1d import LayeredMedium, exact_solution, periodic_grid, run
from pdes_demo.wave1d.spectral import interpolate, reference_size, run_spectral
from pdes_demo.wave1d.stiff import seed_basis

SEED_LABELS = {"aware": "seed stencils (ODE-continued)", "naive": LABELS["naive"]}
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


def convergence_figure(args: argparse.Namespace) -> None:
    ns = np.array(args.ns, dtype=float)
    fig, axes = plt.subplots(
        1,
        len(args.widths),
        figsize=(3.6 * len(args.widths), 4.4),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, width in zip(axes, args.widths, strict=True):
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
        for i, n in enumerate(args.ns):
            h_over = f"{2 / n / width:8.2f}" if width else "     inf"
            rn = f"{r['naive'][i - 1]:5.1f}" if i else "     "
            ra = f"{r['aware'][i - 1]:5.1f}" if i else "     "
            print(
                f"  {n:5d}  {h_over}  {errors['naive'][i]:9.2e} {rn}  "
                f"{errors['aware'][i]:9.2e} {ra}"
            )

        for mode in ("naive", "aware"):
            ax.loglog(
                ns,
                errors[mode],
                "o-",
                color=COLORS[mode],
                label=SEED_LABELS[mode],
                ms=5,
            )
        guides = [("aware", 4, "4th order")]
        if width == 0 or 2 / ns[-2] > width:
            guides.append(("naive", 1, "1st order"))
        for mode, order, label in guides:
            anchor = errors[mode][1]
            ref = anchor * (ns[1] / ns) ** order
            ax.loglog(ns, ref, "--", color=INK_MUTED, lw=1.1)
            ax.annotate(
                label,
                (ns[-1], ref[-1]),
                xytext=(5, 0),
                textcoords="offset points",
                color=INK_SECONDARY,
                fontsize=9,
                va="center",
            )
        if width and ns[0] <= 2 / width <= ns[-1]:
            ax.axvline(2 / width, color=INK_MUTED, lw=1.0, ls=":")
            ax.annotate(
                "h = delta",
                (2 / width, 1.5e-1),
                xytext=(4, 0),
                textcoords="offset points",
                color=INK_SECONDARY,
                fontsize=9,
                rotation=90,
                va="top",
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("number of grid nodes")
        ax.set_xticks(ns, [str(int(n)) for n in ns])
        ax.set_xlim(ns[0] / 1.3, ns[-1] * 2.4)
    axes[0].set_ylabel(f"relative error in stress at t = {args.t_end:g}")
    axes[0].set_ylim(1e-8, 5e-1)
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle(
        "Same equispaced grid, same time step: only the stencils that see the "
        "edge differ",
        fontsize=12,
    )
    out = args.out_dir / "wave1d_stiff_convergence.png"
    fig.savefig(out, dpi=160)
    print(f"\nwrote {out}")


def snapshot_figure(args: argparse.Namespace) -> None:
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
        figsize=(9, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.3]},
        constrained_layout=True,
    )
    for a in (ax, ax_err):
        a.axvspan(medium.layer_start, medium.layer_end, color=LAYER_FILL, zorder=0)
    ax.plot(x_fine, f_ref_fine, color=INK, lw=1.4, label="reference (spectral)")
    for mode in ("naive", "aware"):
        ax.plot(
            grid.x, sols[mode], "-", color=COLORS[mode], lw=1.6, label=SEED_LABELS[mode]
        )
        ax_err.plot(grid.x, sols[mode] - f_ref, "-", color=COLORS[mode], lw=1.6)
    ax.set_ylabel("stress f")
    ax.legend(loc="upper left", fontsize=9)
    h = grid.h
    ax.set_title(
        f"t = {args.t_end:g}, {grid.n} nodes, edge width delta = {medium.edge_width:g}"
        f" = h/{h / medium.edge_width:.0f}: the layer's edges fall between nodes",
        fontsize=11,
    )
    ax_err.set_ylabel("error")
    ax_err.set_xlabel("x")
    ax_err.axhline(0, color=INK_MUTED, lw=0.8)
    out = args.out_dir / "wave1d_stiff_snapshot.png"
    fig.savefig(out, dpi=160)
    errs = {m: np.abs(sols[m] - f_ref).max() for m in sols}
    print(
        f"wrote {out}  (max error naive {errs['naive']:.2e}, seeds {errs['aware']:.2e})"
    )


def seeds_figure(args: argparse.Namespace) -> None:
    """phi_1..phi_4 for the u field on one stencil whose centre is h/2 left of
    the edge at x = 0, for a jump, an edge of width h/4, and one of width 2h."""
    h = 0.01
    xe = -0.5 * h
    x_plot = np.linspace(-3 * h, 3 * h, 301)
    nodes = xe + h * np.arange(-2, 3)
    cases = [("jump (delta -> 0)", 1e-6), ("delta = h/4", h / 4), ("delta = 2h", 2 * h)]
    styles = [(INK, "-"), (COLORS["aware"], "-"), (COLORS["naive"], "-")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), constrained_layout=True)
    xi = (x_plot - xe) / h
    for k, ax in enumerate(axes, start=1):
        for node in nodes:
            ax.axvline((node - xe) / h, color=INK_MUTED, lw=0.6, ls=":")
        ax.axvline(-xe / h, color=INK_SECONDARY, lw=1.0)
        ax.plot(
            xi, xi**k, color=INK_MUTED, lw=1.2, ls="--", label=f"monomial $\\xi^{k}$"
        )
        curves = [xi**k]
        for (label, width), (color, ls) in zip(cases, styles, strict=True):
            medium = LayeredMedium(edge_width=width)
            a_u, _, h_s = seed_basis(x_plot, xe, medium, 5)
            phi = a_u[k] * (h_s / h) ** k
            ax.plot(xi, phi, color=color, ls=ls, lw=1.8, label=label)
            curves.append(phi)
        ax.set_title(f"seed $\\phi_{k}$ (u field)", fontsize=11)
        ax.set_xlabel(r"$\xi = (x - x_e) / h$")
        ax.set_xlim(-3, 3)
        # Frame the stencil's reach, |xi| <= 2.5, whatever the seeds do beyond it.
        inside = np.abs(xi) <= 2.5
        ys = np.concatenate([c[inside] for c in curves])
        span = ys.max() - ys.min()
        ax.set_ylim(ys.min() - 0.05 * span, ys.max() + 0.05 * span)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Seeds look like monomials at the evaluation point and are continued "
        "through the edge by the PDE (c: 1 -> 2, rho: 1 -> 1); edge at "
        r"$\xi = 0.5$",
        fontsize=11,
    )
    out = args.out_dir / "wave1d_stiff_seeds.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    use_demo_style()
    convergence_figure(args)
    snapshot_figure(args)
    seeds_figure(args)


if __name__ == "__main__":
    main()
