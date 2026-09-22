"""Error vs resolution for the 1-D layer problem (dissertation Fig. 2-8).

Left: the dissertation's thick layer. Right: a layer thinner than the FD
stencil, where interface stencils straddle both edges at once. Errors are
relative l2 errors in stress f at t = 1 against the exact ray-sum solution.

    uv run python scripts/wave1d_convergence.py
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pdes_demo.plotting import COLORS, INK_MUTED, INK_SECONDARY, LABELS, use_demo_style
from pdes_demo.wave1d import LayeredMedium, exact_solution, periodic_grid, run


def rel_error(n: int, medium: LayeredMedium, mode: str, t_end: float) -> float:
    grid = periodic_grid(n)
    snaps = run(grid, medium, mode=mode, t_end=t_end, n_snapshots=1)
    _, f_exact = exact_solution(grid.x, snaps.t[-1], medium)
    return float(np.linalg.norm(snaps.f[-1] - f_exact) / np.linalg.norm(f_exact))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ns", type=int, nargs="+", default=[100, 200, 400, 800, 1600])
    parser.add_argument("--t-end", type=float, default=1.0)
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/wave1d_convergence.png")
    )
    args = parser.parse_args()

    cases = {
        "layer width 0.5 (dissertation §2.2)": LayeredMedium(layer_width=0.5),
        "layer width 0.01 (thinner than the stencil)": LayeredMedium(layer_width=0.01),
    }
    ns = np.array(args.ns)

    use_demo_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.6), sharey=True, constrained_layout=True
    )
    for ax, (title, medium) in zip(axes, cases.items(), strict=True):
        t0 = time.perf_counter()
        errors = {
            mode: [rel_error(n, medium, mode, args.t_end) for n in ns]
            for mode in ("naive", "aware")
        }
        print(f"{title}  ({time.perf_counter() - t0:.1f}s)")
        print("      N     naive       aware")
        for n, e_n, e_a in zip(ns, errors["naive"], errors["aware"], strict=True):
            print(f"  {n:5d}  {e_n:9.2e}  {e_a:9.2e}")

        for mode in ("naive", "aware"):
            ax.loglog(
                ns, errors[mode], "o-", color=COLORS[mode], label=LABELS[mode], ms=6
            )
        # Reference slopes anchored to the second-coarsest point of each series.
        for mode, order, label in (
            ("naive", 1, "1st order"),
            ("aware", 4, "4th order"),
        ):
            anchor = errors[mode][1]
            ref = anchor * (ns[1] / ns) ** order
            ax.loglog(ns, ref, "--", color=INK_MUTED, lw=1.2)
            ax.annotate(
                label,
                (ns[-1], ref[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                color=INK_SECONDARY,
                fontsize=10,
                va="center",
            )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("number of grid nodes")
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.set_xlim(ns[0] / 1.3, ns[-1] * 2.2)
    axes[0].set_ylabel("relative error in stress at t = 1")
    axes[0].legend(loc="lower left")
    fig.suptitle(
        "Same grid, same time step: only the interface stencils differ", fontsize=13
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
