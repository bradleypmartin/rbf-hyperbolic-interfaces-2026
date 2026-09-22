"""Eigenvalues of the RBF-FD elastic operator with and without hyperviscosity.

Reproduces dissertation Fig. 3-2: plain RBF-FD stencils on scattered nodes
put eigenvalues on both sides of the imaginary axis, so any explicit
integrator eventually blows up; a small Delta^3 hyperviscosity term pulls
them into the left half-plane, inside the RK4 stability region for the
chosen time step. Dense eigenvalue solves, so keep n small (900 nodes is
4500 eigenvalues and takes a few seconds; 2500 nodes takes minutes).

With ``--edge-width delta`` the interfaces are smooth tanh edges (Part 3,
demo#37): flat, coefficients sampled at the stencil centres. ``--mode aware``
rebuilds the interface rows: the piecewise-polynomial stencils for a jump,
the seed stencils for a smooth edge (demo#39). ``scripts/wave2d_stiff_eigenvalues.py``
runs the comparison across widths.

    uv run python scripts/wave2d_eigenvalues.py
    uv run python scripts/wave2d_eigenvalues.py --n 2500 --gamma-scale 2
    uv run python scripts/wave2d_eigenvalues.py --edge-width 0.004 --mode aware
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rbf_hyperbolic_interfaces.plotting import (
    AWARE,
    INK_MUTED,
    INK_SECONDARY,
    NAIVE,
    use_demo_style,
)
from rbf_hyperbolic_interfaces.wave2d import (
    LayeredMedium2D,
    SineInterface,
    make_node_set,
)
from rbf_hyperbolic_interfaces.wave2d.operators import (
    build_operators,
    hyperviscosity_gamma,
)


def rk4_boundary(n_pts: int = 800) -> np.ndarray:
    """Boundary of the RK4 stability region: |1 + z + z^2/2 + z^3/6 + z^4/24| = 1."""
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    pts = []
    for th in theta:
        # Solve R(z) = exp(i theta) for the root continuing from the origin.
        roots = np.roots([1 / 24, 1 / 6, 1 / 2, 1, 1 - np.exp(1j * th)])
        pts.append(roots[np.argmin(np.abs(roots))])
    pts = np.array(pts)
    return pts[np.argsort(np.angle(pts))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=900, help="nodes (perfect square)")
    parser.add_argument(
        "--amplitude",
        type=float,
        default=None,
        help="interface amplitude (default 0.02, or 0 with --edge-width)",
    )
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument("--gamma-scale", type=float, default=1.0)
    parser.add_argument(
        "--edge-width", type=float, default=0.0, help="tanh edge width (flat only)"
    )
    parser.add_argument("--mode", default="naive", choices=["naive", "aware"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.amplitude is None:
        args.amplitude = 0.0 if args.edge_width else 0.02
    if args.out is None:
        suffix = f"_w{args.edge_width:g}" if args.edge_width else ""
        suffix += "_aware" if args.mode == "aware" else ""
        args.out = Path(f"outputs/wave2d_eigenvalues{suffix}.png")

    medium = LayeredMedium2D(
        lower=SineInterface(0.25, args.amplitude),
        upper=SineInterface(0.5, args.amplitude),
        edge_width=args.edge_width,
    )
    nodes = make_node_set(medium, args.n)
    t0 = time.perf_counter()
    ops = build_operators(nodes, medium, mode=args.mode)
    if args.mode == "aware":
        rebuilt = ops.interface_nodes.size
        print(f"{rebuilt} rebuilt rows in {time.perf_counter() - t0:.1f}s")
    gamma = args.gamma_scale * hyperviscosity_gamma(nodes.h, ops.hyper_power)
    dt = args.cfl * nodes.h / medium.c_max

    t0 = time.perf_counter()
    dense = ops.elastic.toarray()
    ev_plain = np.linalg.eigvals(dense)
    ev_hyper = np.linalg.eigvals(dense + gamma * ops.hyper_block.toarray())
    print(f"{5 * nodes.n} eigenvalues x2 in {time.perf_counter() - t0:.1f}s")

    def amp(ev: np.ndarray) -> float:
        z = ev * dt
        return float(np.abs(1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24).max())

    for label, ev in (("no hyperviscosity", ev_plain), ("with", ev_hyper)):
        print(
            f"{label:18s} max Re = {ev.real.max():+.3e}  min Re = {ev.real.min():+.3e}"
            f"  max |Im| = {np.abs(ev.imag).max():.1f}  RK4 max |R| = {amp(ev):.4f}"
        )

    use_demo_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), constrained_layout=True)
    boundary = rk4_boundary()
    for ax, ev, title, color in (
        (axes[0], ev_plain, "RBF-FD operator, no hyperviscosity", NAIVE),
        (axes[1], ev_hyper, "with $\\Delta^3$ hyperviscosity", AWARE),
    ):
        z = ev * dt
        ax.fill(boundary.real, boundary.imag, color="#eef3fa", zorder=0)
        ax.plot(boundary.real, boundary.imag, color=INK_MUTED, lw=1, zorder=1)
        ax.axvline(0, color=INK_SECONDARY, lw=0.8)
        ax.scatter(z.real, z.imag, s=5, color=color, lw=0, zorder=2)
        ax.set_title(title)
        ax.set_xlabel(r"Re($\lambda \, \Delta t$)")
        ax.set_xlim(-3.2, 1.2)
        ax.set_ylim(-3.2, 3.2)
        ax.set_aspect("equal")
    axes[0].set_ylabel(r"Im($\lambda \, \Delta t$)")
    axes[0].text(
        0.03,
        0.97,
        "shaded: RK4 stability region",
        transform=axes[0].transAxes,
        va="top",
        fontsize=10,
        color=INK_SECONDARY,
    )
    edge = f", edge width {medium.edge_width:g}" if medium.is_smooth else ""
    if args.mode == "aware":
        edge += ", seed stencils" if medium.is_smooth else ", interface-aware"
    fig.suptitle(
        f"{5 * nodes.n} eigenvalues of the elastic operator on {nodes.n} nodes{edge}"
        f"  |  $\\gamma$ = {gamma:.2e}, $\\Delta t$ = {dt:.4f}",
        fontsize=12,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
