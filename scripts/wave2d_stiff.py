"""Stiff smooth edges in 2-D: naive RBF-FD through a tanh edge, vs resolution (#37).

The flat two-interface problem of ``wave2d_convergence.py`` with both
interfaces smoothed into tanh transitions of width delta. Plain RBF-FD
stencils everywhere, material coefficients sampled at the stencil centres,
plane P-wave at normal incidence; errors in v, f and h against the ray sum
(delta = 0) or the 1-D pseudo-spectral reference mapped through
``spectral_plane_wave`` (delta > 0), whose 1-D snapshots are cached under
``outputs/``. "Floor" is the same pulse on the same nodes in a uniform
medium, the pure resolution error. Fixed delta per panel, as in
``wave1d_stiff.py``: the knee, if any, shows as n crosses h = delta.

    uv run python scripts/wave2d_stiff.py
    uv run python scripts/wave2d_stiff.py --widths 0 0.01 --ns 2500 4900
    uv run python scripts/wave2d_stiff.py --sharpness 15 --center 0.875
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from pdes_demo.plotting import COLORS, INK_MUTED, INK_SECONDARY, use_demo_style
from pdes_demo.wave1d import periodic_grid
from pdes_demo.wave2d import (
    ElasticMaterial,
    LayeredMedium2D,
    NodeSet,
    exact_plane_wave,
    make_node_set,
    run,
)
from pdes_demo.wave2d.exact import plane_wave_from_1d, spectral_plane_wave_1d

MODE_LABELS = {"naive": "standard RBF-FD (naive)", "aware": "seed stencils"}
FIELD_ROWS = {"v": 1, "f": 2, "h": 4}
REF_DT = 5e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--widths", type=float, nargs="+", default=[0.0, 0.0025, 0.01, 0.04]
    )
    parser.add_argument("--ns", type=int, nargs="+", default=[2500, 4900, 10000, 19600])
    parser.add_argument("--modes", nargs="+", default=["naive"], choices=["naive"])
    parser.add_argument("--t-end", type=float, default=1.0)
    parser.add_argument("--sharpness", type=float, default=23.0)
    parser.add_argument("--center", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def reference_1d(medium: LayeredMedium2D, args: argparse.Namespace) -> tuple:
    """Cached 1-D spectral snapshot ``(grid, u, f)`` at ``t_end``, smooth media."""
    key = (
        f"wave2d_stiff_ref_w{medium.edge_width:g}_s{args.sharpness:g}"
        f"_c{args.center:g}_t{args.t_end:g}_dt{REF_DT:g}.npz"
    )
    path = args.out_dir / key
    if path.exists():
        data = np.load(path)
        return periodic_grid(int(data["n"])), data["u"], data["f"]
    t0 = time.perf_counter()
    grid, u, f = spectral_plane_wave_1d(
        medium, args.t_end, args.center, args.sharpness, dt=REF_DT
    )
    np.savez(path, n=grid.n, u=u, f=f)
    print(f"  reference n = {grid.n} in {time.perf_counter() - t0:.1f}s -> {path}")
    return grid, u, f


def reference_at(
    nodes: NodeSet, medium: LayeredMedium2D, args: argparse.Namespace
) -> np.ndarray:
    """Reference state ``(5, n)`` at ``t_end`` on the nodes."""
    if not medium.is_smooth:
        return exact_plane_wave(nodes, args.t_end, medium, args.center, args.sharpness)
    grid, u, f = reference_1d(medium, args)
    return plane_wave_from_1d(nodes, medium, grid, u, f, args.center, args.sharpness)


def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def rates(ns: np.ndarray, errors: list[float]) -> np.ndarray:
    # Orders in h = 1 / sqrt(n).
    e = np.array(errors)
    return 2 * np.log(e[:-1] / e[1:]) / np.log(ns[1:] / ns[:-1])


def errors_for(
    medium: LayeredMedium2D, nodes: NodeSet, mode: str, args: argparse.Namespace
) -> dict[str, float]:
    snaps = run(
        nodes,
        medium,
        mode=mode,
        t_end=args.t_end,
        n_snapshots=1,
        pulse_center=args.center,
        pulse_sharpness=args.sharpness,
    )
    state = snaps.state[-1]
    ref = reference_at(nodes, medium, args)
    out = {name: rel_error(state[k], ref[k]) for name, k in FIELD_ROWS.items()}
    out["u_max"] = float(np.abs(state[0]).max())
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    use_demo_style()
    ns = np.array(args.ns, dtype=float)
    uniform = LayeredMedium2D(layer=ElasticMaterial(lam=1.0, mu=1.0, rho=1.0))

    t0 = time.perf_counter()
    node_sets = {
        n: make_node_set(LayeredMedium2D(), n, seed=args.seed) for n in args.ns
    }
    print(f"node sets: {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    floor = [errors_for(uniform, node_sets[n], "naive", args) for n in args.ns]
    print(f"resolution floor (uniform medium): {time.perf_counter() - t0:.1f}s")

    fig, panels = plt.subplots(
        2,
        len(args.widths),
        figsize=(3.6 * len(args.widths), 7.6),
        sharey="row",
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes, axes_u = panels
    for ax, ax_u, width in zip(axes, axes_u, args.widths, strict=True):
        medium = LayeredMedium2D(edge_width=width)
        title = (
            "jump edges (delta = 0)" if width == 0 else f"edge width delta = {width:g}"
        )
        print(f"\n{title}")
        t0 = time.perf_counter()
        results = {
            mode: [errors_for(medium, node_sets[n], mode, args) for n in args.ns]
            for mode in args.modes
        }
        print(f"  ({time.perf_counter() - t0:.1f}s)")
        header = "      N   h/delta " + "".join(
            f"{mode + ' ' + f:>12s} {'rate':>5s}" for mode in args.modes for f in "vfh"
        )
        print(header + f"{'max|u|':>10s}{'floor v':>10s}{'floor |u|':>10s}")
        r = {
            (mode, f): rates(ns, [e[f] for e in results[mode]])
            for mode in args.modes
            for f in "vfh"
        }
        for i, n in enumerate(args.ns):
            h_over = f"{1 / np.sqrt(n) / width:8.2f}" if width else "     inf"
            cells = ""
            for mode in args.modes:
                for f in "vfh":
                    rate = f"{r[(mode, f)][i - 1]:5.1f}" if i else "     "
                    cells += f"{results[mode][i][f]:12.2e} {rate}"
            print(
                f"  {n:5d}  {h_over} {cells}{results['naive'][i]['u_max']:10.1e}"
                f"{floor[i]['v']:10.2e}{floor[i]['u_max']:10.1e}"
            )

        for mode in args.modes:
            ax.loglog(
                ns,
                [e["v"] for e in results[mode]],
                "o-",
                color=COLORS[mode],
                label=MODE_LABELS[mode],
                ms=5,
            )
            ax_u.loglog(
                ns, [e["u_max"] for e in results[mode]], "o-", color=COLORS[mode], ms=5
            )
        ax.loglog(
            ns,
            [e["v"] for e in floor],
            "s--",
            color=INK_SECONDARY,
            label="no interface (resolution floor)",
            ms=5,
            lw=1.4,
        )
        ax_u.loglog(
            ns, [e["u_max"] for e in floor], "s--", color=INK_SECONDARY, ms=5, lw=1.4
        )
        for mode, order, label in (("naive", 2, "2nd order"),):
            anchor = results[mode][0]["v"]
            guide = anchor * (ns[0] / ns) ** (order / 2)
            ax.loglog(ns, guide, ":", color=INK_MUTED, lw=1.1)
            ax.annotate(
                label,
                (ns[-1], guide[-1]),
                xytext=(5, 0),
                textcoords="offset points",
                color=INK_SECONDARY,
                fontsize=9,
                va="center",
            )
        if width and ns[0] <= width**-2 <= ns[-1]:
            ax_u.axvline(width**-2, color=INK_MUTED, lw=1.0, ls=":")
            ax.axvline(width**-2, color=INK_MUTED, lw=1.0, ls=":")
            ax.annotate(
                "h = delta",
                (width**-2, 0.97),
                xycoords=("data", "axes fraction"),
                xytext=(4, 0),
                textcoords="offset points",
                color=INK_SECONDARY,
                fontsize=9,
                rotation=90,
                va="top",
            )
        ax.set_title(title, fontsize=11)
        ax_u.set_xlabel("number of nodes")
        ax_u.set_xticks(ns, [str(int(n)) for n in ns])
        ax_u.xaxis.set_minor_formatter(NullFormatter())
        ax_u.set_xlim(ns[0] / 1.3, ns[-1] * 2.4)
    axes[0].set_ylabel(f"relative error in v at t = {args.t_end:g}")
    axes[0].legend(loc="lower left", fontsize=9)
    axes_u[0].set_ylabel("max |u| (exact: 0)")
    fig.suptitle(
        "Same nodes, same time step: the edge is only as sharp as delta",
        fontsize=12,
    )
    tag = "" if args.sharpness == 23.0 else f"_s{args.sharpness:g}"
    out = args.out_dir / f"wave2d_stiff_naive{tag}.png"
    fig.savefig(out, dpi=160)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
