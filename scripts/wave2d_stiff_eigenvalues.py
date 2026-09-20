"""Spectra of the seed-aware elastic operator across edge widths (#39).

Does RBF-FD stability (Delta^3 hyperviscosity at the MATLAB gamma, straddling
rows) survive seed-augmented stencils? For each edge width the operator is
built naive and with the seed stencils on every row that sees the edge, the
dense spectrum is computed with hyperviscosity, and the largest real part and
the RK4 amplification at CFL 0.5 are reported next to the jump-aware operator.
``--variants`` adds the configurations the study tried and rejected
(``docs/stiff-features.md`` §5.3). ``--run`` integrates the wide plane pulse
to ``t_end`` for every case and reports the energy ratio and the spurious u.
``--amplitude 0.02`` bends the interfaces (#42): same spectra on the curved
node set, with the seeds along the true normals; ``--run`` then reports the
energy ratio and the spurious u only (the curved reference lives in
``wave2d_stiff.py --amplitude``).

    uv run python scripts/wave2d_stiff_eigenvalues.py                 # 900 nodes
    uv run python scripts/wave2d_stiff_eigenvalues.py --n 2500 --run  # minutes
    uv run python scripts/wave2d_stiff_eigenvalues.py --variants seeds naive19 seeds30
    uv run python scripts/wave2d_stiff_eigenvalues.py --n 2500 --amplitude 0.02 --run
"""

import argparse
import os
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from pdes_demo.plotting import (
    AWARE,
    INK_MUTED,
    INK_SECONDARY,
    NAIVE,
    TEXTWIDTH_IN,
    use_demo_style,
    use_print_style,
)
from pdes_demo.results_cache import ResultsCache
from pdes_demo.wave1d import periodic_grid
from pdes_demo.wave2d import (
    LayeredMedium2D,
    NodeSet,
    Operators,
    SineInterface,
    build_operators,
    energy,
    exact_plane_wave,
    hyperviscosity_gamma,
    make_node_set,
    run,
)
from pdes_demo.wave2d.exact import plane_wave_from_1d, spectral_plane_wave_1d

VARIANTS = {
    "naive": "plain 30-node degree-4 stencils everywhere",
    "seeds": "19-node seed rows, 30-node seed-annihilating Delta^3 rows (default)",
    "hyper19": "19-node seed rows with their own 19-node Delta^3 rows",
    "naive-hyper": "19-node seed rows keeping the naive 30-node Delta^3 rows",
    "seeds30": "30-node degree-4 seed stencils for both",
    "naive19": "plain 19-node degree-3 stencils everywhere (control, no seeds)",
}
COLORS = {name: (NAIVE if name.startswith("naive") else AWARE) for name in VARIANTS}
REF_DT = 5e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=900)
    parser.add_argument(
        "--widths",
        type=float,
        nargs="+",
        default=[1 / 8, 1 / 2, 2.0],
        help="edge widths in units of h (2h is capped at the band's limit)",
    )
    parser.add_argument(
        "--variants", nargs="+", default=["seeds"], choices=list(VARIANTS)[1:]
    )
    parser.add_argument(
        "--floor",
        action="store_true",
        help="with --run, also the naive scheme in a uniform medium",
    )
    parser.add_argument("--gamma-scale", type=float, default=1.0)
    parser.add_argument(
        "--seed-hyper-scale",
        type=float,
        default=1.0,
        help="extra factor on the hyperviscosity rows of the seed stencils",
    )
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument("--run", action="store_true", help="RK4 to --t-end")
    parser.add_argument("--t-end", type=float, default=1.0)
    parser.add_argument("--sharpness", type=float, default=15.0)
    parser.add_argument("--center", type=float, default=0.875)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--amplitude", type=float, default=0.0, help="sine amplitude of the interfaces"
    )
    parser.add_argument("--out", type=Path, default=None)
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
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
        help="processes for the seed marches (default: all cores but two)",
    )
    return parser.parse_args()


def medium_for(width: float, args: argparse.Namespace) -> LayeredMedium2D:
    if args.amplitude == 0.0:
        return LayeredMedium2D(edge_width=width)
    return LayeredMedium2D(
        lower=SineInterface(0.25, args.amplitude),
        upper=SineInterface(0.5, args.amplitude),
        edge_width=width,
    )


def build(
    variant: str,
    nodes: NodeSet,
    medium: LayeredMedium2D,
    seed_hyper_scale: float,
    workers: int | None = None,
) -> Operators:
    if variant == "naive":
        return build_operators(nodes, medium)
    if variant == "naive19":
        return build_operators(nodes, medium, stencil_size=19, poly_degree=3)
    if variant == "seeds30":
        ops = build_operators(
            nodes,
            medium,
            mode="aware",
            interface_stencil=30,
            interface_degree=4,
            workers=workers,
        )
    elif variant == "hyper19":
        ops = build_operators(
            nodes, medium, mode="aware", seed_hyper_stencil=19, workers=workers
        )
    else:
        ops = build_operators(nodes, medium, mode="aware", workers=workers)
    if variant == "naive-hyper":
        ops = replace(ops, hyper_block=build_operators(nodes, medium).hyper_block)
    if seed_hyper_scale != 1.0 and ops.interface_nodes.size:
        d = np.zeros(5 * nodes.n)
        for f in range(5):
            d[f * nodes.n + ops.interface_nodes] = seed_hyper_scale - 1.0
        extra = sp.diags_array(d) @ ops.hyper_block
        ops = replace(ops, hyper_block=sp.csr_array(ops.hyper_block + extra))
    return ops


def reference(
    nodes: NodeSet, medium: LayeredMedium2D, args: argparse.Namespace
) -> np.ndarray:
    """Reference state at ``t_end`` on the nodes: the ray sum for a jump, the
    1-D spectral snapshot (cached under outputs/ as ``wave2d_stiff.py`` does)
    mapped to the nodes for a smooth edge."""
    if not medium.is_smooth:
        return exact_plane_wave(nodes, args.t_end, medium, args.center, args.sharpness)
    path = Path("outputs") / (
        f"wave2d_stiff_ref_w{medium.edge_width:g}_s{args.sharpness:g}"
        f"_c{args.center:g}_t{args.t_end:g}_dt{REF_DT:g}.npz"
    )
    if path.exists():
        data = np.load(path)
        grid, u, f = periodic_grid(int(data["n"])), data["u"], data["f"]
    else:
        grid, u, f = spectral_plane_wave_1d(
            medium, args.t_end, args.center, args.sharpness, dt=REF_DT
        )
        np.savez(path, n=grid.n, u=u, f=f)
    return plane_wave_from_1d(nodes, medium, grid, u, f, args.center, args.sharpness)


def rk4_amplification(ev: np.ndarray, dt: float) -> float:
    z = ev * dt
    return float(np.abs(1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24).max())


def rk4_boundary(n_pts: int = 800) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    pts = []
    for th in theta:
        roots = np.roots([1 / 24, 1 / 6, 1 / 2, 1, 1 - np.exp(1j * th)])
        pts.append(roots[np.argmin(np.abs(roots))])
    pts = np.array(pts)
    return pts[np.argsort(np.angle(pts))]


def main() -> None:
    args = parse_args()
    if args.style == "print":
        use_print_style()
    else:
        use_demo_style()
    jump = medium_for(0.0, args)
    nodes = make_node_set(jump, args.n, seed=args.seed)
    h = nodes.h
    dt = args.cfl * h / jump.c_max
    gamma = args.gamma_scale * hyperviscosity_gamma(h)
    widths = []
    limit = (0.25 - 2 * abs(args.amplitude)) / 4
    for w in args.widths:
        width = w * h
        if width > limit:
            width = limit
            print(f"edge width {w:g} h = {w * h:.4f} capped at {limit:g} (band limit)")
        widths.append(width)
    variants = ["naive"] + args.variants

    def spectrum(ops: Operators) -> tuple[np.ndarray, np.ndarray]:
        dense = ops.elastic.toarray()
        plain = np.linalg.eigvals(dense)
        hyper = np.linalg.eigvals(dense + gamma * ops.hyper_block.toarray())
        return plain, hyper

    print(f"{args.n} nodes, h = {h:.4f}, gamma = {gamma:.2e}, dt = {dt:.4f}")
    header = (
        f"{'case':44s} {'rows':>5s} {'build':>7s} {'max Re':>10s} {'+hyper':>10s}"
        f" {'min Re':>10s} {'RK4 |R|':>8s}"
    )
    if args.run:
        header += f" {'E(t)/E(0)':>10s} {'max|u|':>9s}"
        if args.amplitude == 0.0:
            header += f" {'err v':>9s}"
    print(header)
    refs: dict[LayeredMedium2D, np.ndarray] = {}  # frozen, so hashable
    cache = ResultsCache.new("scripts/wave2d_stiff_eigenvalues.py", args)

    def report(
        label: str,
        ops: Operators,
        medium: LayeredMedium2D,
        built: float,
        *,
        width_label: str,
        variant: str,
    ):
        plain, hyper = spectrum(ops)
        line = (
            f"{label:44s} {ops.interface_nodes.size:5d} {built:6.1f}s"
            f" {plain.real.max():+10.2e} {hyper.real.max():+10.2e}"
            f" {hyper.real.min():+10.2e} {rk4_amplification(hyper, dt):8.5f}"
        )
        measured: dict[str, float] = {}
        if args.run:
            snaps = run(
                nodes,
                medium,
                t_end=args.t_end,
                n_snapshots=1,
                gamma_scale=args.gamma_scale,
                pulse_center=args.center,
                pulse_sharpness=args.sharpness,
                operators=ops,
            )
            e0 = energy(snaps.state[0], nodes, medium)
            e1 = energy(snaps.state[-1], nodes, medium)
            measured["energy_ratio"] = e1 / e0
            measured["max_u"] = float(np.abs(snaps.state[-1][0]).max())
            line += f" {e1 / e0:10.4f} {np.abs(snaps.state[-1][0]).max():9.1e}"
            if args.amplitude == 0.0:
                if medium not in refs:
                    refs[medium] = reference(nodes, medium, args)
                v, v_ref = snaps.state[-1][1], refs[medium][1]
                err = np.linalg.norm(v - v_ref) / np.linalg.norm(v_ref)
                measured["err_v"] = float(err)
                line += f" {err:9.2e}"
        print(line, flush=True)
        cache.add(
            "spectrum",
            n=nodes.n,
            h=h,
            delta=medium.edge_width,
            width_label=width_label,
            variant=variant,
            rows=int(ops.interface_nodes.size),
            dt=dt,
            gamma=gamma,
            max_re=float(plain.real.max()),
            max_re_hyper=float(hyper.real.max()),
            min_re_hyper=float(hyper.real.min()),
            rk4_max=rk4_amplification(hyper, dt),
            **measured,
        )
        return hyper

    t0 = time.perf_counter()
    report(
        "jump, naive",
        build_operators(nodes, jump),
        jump,
        time.perf_counter() - t0,
        width_label="jump",
        variant="naive",
    )
    t0 = time.perf_counter()
    report(
        "jump, interface-aware (19/3)",
        build_operators(nodes, jump, mode="aware"),
        jump,
        time.perf_counter() - t0,
        width_label="jump",
        variant="aware",
    )

    if args.run and args.floor:
        uniform = LayeredMedium2D(layer=jump.background)
        report(
            "uniform medium, naive (floor)",
            build_operators(nodes, uniform),
            uniform,
            0.0,
            width_label="uniform",
            variant="floor",
        )

    spectra: dict[tuple[str, float], np.ndarray] = {}
    width_labels: dict[float, str] = {}
    for width in widths:
        medium = medium_for(width, args)
        ratio = h / width
        if abs(ratio - 1) < 1e-9:
            width_label = "h"
        elif ratio > 1:
            width_label = f"h/{ratio:.3g}"
        else:
            width_label = f"{1 / ratio:.2g}h"
        width_labels[width] = width_label
        for variant in variants:
            t0 = time.perf_counter()
            ops = build(variant, nodes, medium, args.seed_hyper_scale, args.workers)
            built = time.perf_counter() - t0
            label = f"delta = {width:.4f} = h/{h / width:.3g}, {variant}"
            spectra[(variant, width)] = report(
                label, ops, medium, built, width_label=width_label, variant=variant
            )

    boundary = rk4_boundary()
    print_mode = args.style == "print"
    if print_mode:
        panel = TEXTWIDTH_IN / len(widths)
        figsize = (TEXTWIDTH_IN, panel * 1.45 * len(variants) + 0.2)
    else:
        figsize = (3.9 * len(widths), 3.9 * len(variants))
    fig, axes = plt.subplots(
        len(variants),
        len(widths),
        figsize=figsize,
        constrained_layout=True,
        squeeze=False,
    )
    for i, variant in enumerate(variants):
        for j, width in enumerate(widths):
            ax = axes[i, j]
            z = spectra[(variant, width)] * dt
            ax.fill(boundary.real, boundary.imag, color="#eef3fa", zorder=0)
            ax.plot(boundary.real, boundary.imag, color=INK_MUTED, lw=0.8, zorder=1)
            ax.axvline(0, color=INK_SECONDARY, lw=0.6)
            ax.scatter(
                z.real, z.imag, s=1 if print_mode else 4, color=COLORS[variant], lw=0
            )
            ax.text(
                0.03,
                0.97,
                f"max Re = {z.real.max():+.1e}\nRK4 max |R| = "
                f"{rk4_amplification(spectra[(variant, width)], dt):.4f}",
                transform=ax.transAxes,
                va="top",
                fontsize=5.5 if print_mode else 9,
                color=INK_SECONDARY,
            )
            ax.set_xlim(-3.2, 1.2)
            ax.set_ylim(-3.2, 3.2)
            ax.set_aspect("equal")
            if print_mode:
                ax.grid(False)
                ax.set_xticks([-3, -2, -1, 0, 1])
            if i == 0:
                ax.set_title(
                    rf"$\delta = {width_labels[width]}$"
                    if print_mode
                    else f"edge width $\\delta$ = {width_labels[width]}",
                    fontsize=None if print_mode else 11,
                )
            if i == len(variants) - 1:
                ax.set_xlabel(r"Re($\lambda \, \Delta t$)")
            if j == 0:
                ax.set_ylabel(f"{variant}\n" + r"Im($\lambda \, \Delta t$)")
    if not print_mode:
        geometry = f", curved (amplitude {args.amplitude:g})" if args.amplitude else ""
        fig.suptitle(
            f"Spectra with $\\Delta^3$ hyperviscosity on {nodes.n} nodes{geometry}, "
            f"$\\gamma$ = {gamma:.2e}, $\\Delta t$ = {dt:.4f} (shaded: RK4 region)",
            fontsize=12,
        )
    tag = f"_a{args.amplitude:g}" if args.amplitude else ""
    out = args.out or Path(
        f"outputs/wave2d_stiff_eigenvalues_n{args.n}{tag}.{args.format}"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")
    vtag = "" if args.variants == ["seeds"] else "_variants"
    name = f"wave2d_stiff_eigenvalues_n{args.n}{tag}{vtag}.json"
    paths = [Path("outputs") / name]
    if args.data_dir is not None:
        paths.append(args.data_dir / name)
    cache.write(*paths)
    print("results ->", ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
