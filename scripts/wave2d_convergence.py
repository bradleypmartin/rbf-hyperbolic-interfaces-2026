"""Error vs resolution for the 2-D layer problem (dissertation Fig. 3-5 and 3-8).

Left: flat interfaces (§3.4.1), relative l2 errors in v at t = 0.3 against
the exact solution. "Floor" is the same pulse in a uniform medium, i.e.
resolution error with no interface at all. Right: the curved interfaces of
§3.4.2 (amplitude 0.02), errors against a fine interface-aware run resampled
onto each node set with one-sided stencils; only node sets at most a quarter
the size of the reference are shown there.

    uv run python scripts/wave2d_convergence.py
    uv run python scripts/wave2d_convergence.py --ns 2500 4900 10000 19600 --ref-n 90000
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from pdes_demo.plotting import COLORS, INK_MUTED, INK_SECONDARY, use_demo_style
from pdes_demo.wave2d import (
    ElasticMaterial,
    LayeredMedium2D,
    NodeSet,
    SineInterface,
    exact_plane_wave,
    make_node_set,
    resample_matrix,
    run,
)

LABELS = {
    "naive": "standard RBF-FD (naive)",
    "aware": "interface-aware RBF-FD",
    "floor": "no interface (resolution floor)",
}
UNIFORM = LayeredMedium2D(layer=ElasticMaterial(lam=1.0, mu=1.0, rho=1.0))


def rel_error(v: np.ndarray, v_ref: np.ndarray) -> float:
    return float(np.linalg.norm(v - v_ref) / np.linalg.norm(v_ref))


def final_v(nodes: NodeSet, medium: LayeredMedium2D, mode: str, t_end: float):
    snaps = run(nodes, medium, mode=mode, t_end=t_end, n_snapshots=1)
    return snaps.field("v")[-1]


def flat_errors(ns: list[int], t_end: float, seed: int) -> dict[str, list[float]]:
    medium = LayeredMedium2D()
    errors: dict[str, list[float]] = {"naive": [], "aware": [], "floor": []}
    for n in ns:
        nodes = make_node_set(medium, n, seed=seed)
        exact = exact_plane_wave(nodes, t_end, medium)[1]
        for mode in ("naive", "aware"):
            errors[mode].append(rel_error(final_v(nodes, medium, mode, t_end), exact))
        # Same node set: the uniform medium has the same interface geometry.
        exact_u = exact_plane_wave(nodes, t_end, UNIFORM)[1]
        v_u = final_v(nodes, UNIFORM, "naive", t_end)
        errors["floor"].append(rel_error(v_u, exact_u))
    return errors


def curved_errors(
    ns: list[int], ref_n: int, amplitude: float, t_end: float, seed: int
) -> dict[str, list[float]]:
    medium = LayeredMedium2D(
        lower=SineInterface(0.25, amplitude), upper=SineInterface(0.5, amplitude)
    )
    t0 = time.perf_counter()
    fine = make_node_set(medium, ref_n, seed=seed)
    v_fine = final_v(fine, medium, "aware", t_end)
    print(f"curved reference: {ref_n} nodes, aware ({time.perf_counter() - t0:.1f}s)")
    errors: dict[str, list[float]] = {"naive": [], "aware": []}
    for n in ns:
        nodes = make_node_set(medium, n, seed=seed)
        v_ref = resample_matrix(fine, nodes.xy, medium) @ v_fine
        for mode in ("naive", "aware"):
            errors[mode].append(rel_error(final_v(nodes, medium, mode, t_end), v_ref))
    return errors


def print_table(title: str, ns: list[int], errors: dict[str, list[float]]) -> None:
    print(title)
    print("      N  " + "".join(f"{k:>10s}" for k in errors))
    for i, n in enumerate(ns):
        print(f"  {n:6d}  " + "".join(f"{errors[k][i]:10.2e}" for k in errors))


def plot_series(
    ax: plt.Axes,
    ns: list[int],
    errors: dict[str, list[float]],
    title: str,
    x_range: tuple[int, int],
) -> None:
    ns_arr = np.array(ns)
    for key, values in errors.items():
        style = "s--" if key == "floor" else "o-"
        color = INK_SECONDARY if key == "floor" else COLORS[key]
        ax.loglog(ns_arr, values, style, color=color, label=LABELS[key], ms=6, lw=1.6)
    # Reference slopes in h = 1/sqrt(N), anchored at the second-coarsest point.
    for key, order, label in (("naive", 2, "2nd order"), ("aware", 4, "4th order")):
        anchor = errors[key][1]
        ref = anchor * (ns_arr[1] / ns_arr) ** (order / 2)
        ax.loglog(ns_arr, ref, ":", color=INK_MUTED, lw=1.2)
        ax.annotate(
            label,
            (ns_arr[-1], ref[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=INK_SECONDARY,
            fontsize=10,
            va="center",
        )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("number of nodes")
    ax.set_xticks(ns_arr, [str(n) for n in ns])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(x_range[0] / 1.3, x_range[1] * 2.2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ns", type=int, nargs="+", default=[2500, 4900, 10000, 19600])
    parser.add_argument("--ref-n", type=int, default=40000)
    parser.add_argument("--amplitude", type=float, default=0.02)
    parser.add_argument("--t-end", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/wave2d_convergence.png")
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    flat = flat_errors(args.ns, args.t_end, args.seed)
    elapsed = time.perf_counter() - t0
    print_table(f"flat interfaces, exact reference ({elapsed:.0f}s)", args.ns, flat)
    curved_ns = [n for n in args.ns if 4 * n <= args.ref_n]
    t0 = time.perf_counter()
    curved = curved_errors(curved_ns, args.ref_n, args.amplitude, args.t_end, args.seed)
    elapsed = time.perf_counter() - t0
    print_table(
        f"curved interfaces, {args.ref_n}-node reference ({elapsed:.0f}s)",
        curved_ns,
        curved,
    )

    use_demo_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.8), sharey=True, constrained_layout=True
    )
    x_range = (min(args.ns), max(args.ns))
    plot_series(
        axes[0], args.ns, flat, "Flat interfaces, error vs exact solution", x_range
    )
    plot_series(
        axes[1],
        curved_ns,
        curved,
        f"Curved interfaces, error vs {args.ref_n}-node reference",
        x_range,
    )
    axes[0].set_ylabel(f"relative error in v at t = {args.t_end:g}")
    axes[0].legend(loc="lower left", fontsize=10)
    fig.suptitle(
        "Same nodes, same time step: only the stencils near the interfaces differ",
        fontsize=13,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
