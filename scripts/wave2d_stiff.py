"""Stiff smooth edges in 2-D: naive RBF-FD vs seed stencils through a tanh edge (#40).

The flat two-interface problem of ``wave2d_convergence.py`` with both
interfaces smoothed into tanh transitions of width delta. Naive: plain RBF-FD
stencils everywhere, material coefficients sampled at the stencil centres.
Seeds: ``build_operators(mode="aware")``, the ODE-continued seed stencils of
``wave2d/seeds.py`` on every row that sees the edge (at delta = 0 that is
Part 2's interface-aware operator). Plane P-wave at normal incidence; errors
in v and h at t = 1 against the ray sum (delta = 0) or the 1-D pseudo-spectral
reference mapped through ``spectral_plane_wave`` (delta > 0), whose 1-D
snapshots are cached under ``outputs/``, and the largest spurious u (exactly
0 in the true solution). "Floor" is the same pulse on the same nodes in a
uniform medium, the pure resolution error. Fixed delta per panel, as in
``wave1d_stiff.py``: the knee, if any, shows as n crosses h = delta.

A seed row costs two ODE marches, about 50 ms, and a resolved edge seeds
every row (17 min serially at 19,600 nodes), so the marches run on
``--workers`` processes and the seed operators are cached under ``outputs/``
keyed on (n, seed, delta); delete the ``wave2d_stiff_ops_*`` files after any
change to the seed construction.

Also writes a still at the clip resolution (``--snapshot-n``) through an edge
that sits between the straddling rows (``--snapshot-width``): the reference
wave, then each method's error map, at ``--snapshot-times``.

    uv run python scripts/wave2d_stiff.py                      # both modes, 4 widths
    uv run python scripts/wave2d_stiff.py --modes naive        # the #37 baseline
    uv run python scripts/wave2d_stiff.py --widths 0 0.0025 --ns 2500 4900
    uv run python scripts/wave2d_stiff.py --snapshot-only
"""

import argparse
import os
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from matplotlib.ticker import NullFormatter

from pdes_demo.plotting import (
    COLORS,
    ERROR_CMAP,
    FIELD_CMAP,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    use_demo_style,
)
from pdes_demo.wave1d import periodic_grid
from pdes_demo.wave2d import (
    ElasticMaterial,
    LayeredMedium2D,
    NodeSet,
    Operators,
    build_operators,
    exact_plane_wave,
    make_node_set,
    pixel_grid,
    resample_matrix,
    run,
)
from pdes_demo.wave2d.exact import plane_wave_from_1d, spectral_plane_wave_1d

MODE_LABELS = {
    "naive": "standard RBF-FD (naive)",
    "aware": "seed stencils (interface-aware at delta = 0)",
}
MODE_SHORT = {"naive": "naive", "aware": "seeds"}
TITLES = {"naive": "Standard RBF-FD (naive)", "aware": "Seed stencils"}
UNIFORM = LayeredMedium2D(layer=ElasticMaterial(lam=1.0, mu=1.0, rho=1.0))
REF_DT = 5e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--widths", type=float, nargs="+", default=[0.0, 0.0025, 0.01, 0.04]
    )
    parser.add_argument("--ns", type=int, nargs="+", default=[2500, 4900, 10000, 19600])
    parser.add_argument(
        "--modes", nargs="+", default=["naive", "aware"], choices=["naive", "aware"]
    )
    parser.add_argument("--t-end", type=float, default=1.0)
    parser.add_argument("--sharpness", type=float, default=15.0)
    parser.add_argument("--center", type=float, default=0.875)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
        help="processes for the seed marches (default: all cores but two)",
    )
    parser.add_argument("--snapshot-n", type=int, default=10000)
    parser.add_argument("--snapshot-width", type=float, default=0.0025)
    parser.add_argument("--snapshot-times", type=float, nargs="+", default=[0.25, 1.0])
    parser.add_argument("--pixels", type=int, default=250)
    parser.add_argument("--no-snapshot", action="store_true")
    parser.add_argument(
        "--snapshot-only", action="store_true", help="skip the sweep, write the still"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


# --- references ---------------------------------------------------------------


def reference_1d(medium: LayeredMedium2D, args: argparse.Namespace, t: float):
    """Cached 1-D spectral snapshot ``(grid, u, f)`` at ``t``, smooth media."""
    key = (
        f"wave2d_stiff_ref_w{medium.edge_width:g}_s{args.sharpness:g}"
        f"_c{args.center:g}_t{t:g}_dt{REF_DT:g}.npz"
    )
    path = args.out_dir / key
    if path.exists():
        data = np.load(path)
        return periodic_grid(int(data["n"])), data["u"], data["f"]
    t0 = time.perf_counter()
    grid, u, f = spectral_plane_wave_1d(
        medium, t, args.center, args.sharpness, dt=REF_DT
    )
    np.savez(path, n=grid.n, u=u, f=f)
    print(f"  reference n = {grid.n} in {time.perf_counter() - t0:.1f}s -> {path}")
    return grid, u, f


def reference_at(
    nodes: NodeSet, medium: LayeredMedium2D, args: argparse.Namespace, t: float
) -> np.ndarray:
    """Reference state ``(5, n)`` at ``t`` on the nodes."""
    if not medium.is_smooth:
        return exact_plane_wave(nodes, t, medium, args.center, args.sharpness)
    grid, u, f = reference_1d(medium, args, t)
    return plane_wave_from_1d(nodes, medium, grid, u, f, args.center, args.sharpness)


# --- operators ----------------------------------------------------------------


def _csr(data: np.lib.npyio.NpzFile, tag: str, n: int) -> sp.csr_array:
    parts = (data[f"{tag}_data"], data[f"{tag}_indices"], data[f"{tag}_indptr"])
    return sp.csr_array(parts, shape=(5 * n, 5 * n))


def operators_for(
    nodes: NodeSet, medium: LayeredMedium2D, mode: str, args: argparse.Namespace
) -> Operators:
    """Operators for the run; seed operators come from the cache when present.

    Only the seed rows are stored (the elastic and hyperviscosity blocks and
    the row list); everything else in :class:`Operators` is the naive build,
    which is what ``build_operators(mode="aware")`` starts from.
    """
    if mode == "naive" or not medium.is_smooth:
        return build_operators(nodes, medium, mode=mode)
    path = args.out_dir / (
        f"wave2d_stiff_ops_n{nodes.n}_seed{args.seed}_w{medium.edge_width:g}.npz"
    )
    naive = build_operators(nodes, medium)
    if path.exists():
        data = np.load(path)
        return replace(
            naive,
            elastic=_csr(data, "elastic", nodes.n),
            hyper_block=_csr(data, "hyper", nodes.n),
            interface_nodes=data["rows"],
        )
    t0 = time.perf_counter()
    ops = build_operators(nodes, medium, mode="aware", workers=args.workers)
    np.savez(
        path,
        elastic_data=ops.elastic.data,
        elastic_indices=ops.elastic.indices,
        elastic_indptr=ops.elastic.indptr,
        hyper_data=ops.hyper_block.data,
        hyper_indices=ops.hyper_block.indices,
        hyper_indptr=ops.hyper_block.indptr,
        rows=ops.interface_nodes,
    )
    print(
        f"  seed operator, {ops.interface_nodes.size} of {nodes.n} rows, "
        f"{time.perf_counter() - t0:.0f}s on {args.workers} workers -> {path}"
    )
    return ops


# --- the sweep -----------------------------------------------------------------


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
        t_end=args.t_end,
        n_snapshots=1,
        pulse_center=args.center,
        pulse_sharpness=args.sharpness,
        operators=operators_for(nodes, medium, mode, args),
    )
    state = snaps.state[-1]
    ref = reference_at(nodes, medium, args, args.t_end)
    return {
        "v": rel_error(state[1], ref[1]),
        "h": rel_error(state[4], ref[4]),
        "u": float(np.abs(state[0]).max()),
    }


def print_table(
    width: float,
    ns: np.ndarray,
    results: dict[str, list[dict[str, float]]],
    floor: list[dict[str, float]],
) -> None:
    modes = list(results)
    head = "      N  h/delta"
    for mode in modes:
        m = MODE_SHORT[mode]
        head += f" | {m + ' v':>9s} {'rate':>4s} {m + ' h':>9s} {'rate':>4s}"
        head += f" {'max|u|':>7s}"
    print(head + f" | {'floor v':>8s} {'floor |u|':>9s}")
    r = {
        (mode, f): rates(ns, [e[f] for e in results[mode]])
        for mode in modes
        for f in "vh"
    }
    for i, n in enumerate(ns):
        h_over = f"{1 / np.sqrt(n) / width:7.2f}" if width else "    inf"
        line = f"  {int(n):5d}  {h_over}"
        for mode in modes:
            cells = ""
            for f in "vh":
                rate = f"{r[(mode, f)][i - 1]:4.1f}" if i else "    "
                cells += f" {results[mode][i][f]:9.2e} {rate}"
            line += f" |{cells} {results[mode][i]['u']:7.1e}"
        print(line + f" | {floor[i]['v']:8.2e} {floor[i]['u']:9.1e}")


def sweep(args: argparse.Namespace, node_sets: dict[int, NodeSet]) -> None:
    ns = np.array(args.ns, dtype=float)
    t0 = time.perf_counter()
    floor = [errors_for(UNIFORM, node_sets[n], "naive", args) for n in args.ns]
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
        print_table(width, ns, results, floor)

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
                ns, [e["u"] for e in results[mode]], "o-", color=COLORS[mode], ms=5
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
            ns, [e["u"] for e in floor], "s--", color=INK_SECONDARY, ms=5, lw=1.4
        )
        guides = [("naive", 2, "2nd order")]
        if "aware" in args.modes:
            guides.append(("aware", 4, "4th order"))
        for mode, order, label in guides:
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
            for a in (ax, ax_u):
                a.axvline(width**-2, color=INK_MUTED, lw=1.0, ls=":")
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
    tag = "_naive" if args.modes == ["naive"] else ""
    tag += "" if args.sharpness == 15.0 else f"_s{args.sharpness:g}"
    out = args.out_dir / f"wave2d_stiff{tag}.png"
    fig.savefig(out, dpi=160)
    print(f"\nwrote {out}")


# --- the still -----------------------------------------------------------------


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
    for ifc in medium.interfaces:
        ax.axhline(ifc.y0, ls="--", color=INK_SECONDARY, lw=1.0)


def snapshot(args: argparse.Namespace, node_sets: dict[int, NodeSet]) -> None:
    n, width = args.snapshot_n, args.snapshot_width
    nodes = node_sets.get(n) or make_node_set(LayeredMedium2D(), n, seed=args.seed)
    medium = LayeredMedium2D(edge_width=width)
    t0 = time.perf_counter()
    n_frames = 40
    runs = {
        mode: run(
            nodes,
            medium,
            t_end=args.t_end,
            n_snapshots=n_frames,
            align_snapshots=True,
            pulse_center=args.center,
            pulse_sharpness=args.sharpness,
            operators=operators_for(nodes, medium, mode, args),
        )
        for mode in args.modes
    }
    times = runs[args.modes[0]].t
    frames = [int(np.argmin(np.abs(times - t))) for t in args.snapshot_times]
    refs = {k: reference_at(nodes, medium, args, float(times[k])) for k in frames}
    print(
        f"\nstill: {nodes.n} nodes, delta = {width:g} = h/{nodes.h / width:.3g}, "
        f"{time.perf_counter() - t0:.1f}s"
    )
    diff = {
        mode: {k: runs[mode].state[k][1] - refs[k][1] for k in frames}
        for mode in args.modes
    }
    grid = pixel_grid(args.pixels)
    to_grid = resample_matrix(nodes, grid, medium)

    def image(values: np.ndarray) -> np.ndarray:
        return np.abs(to_grid @ values).reshape(args.pixels, args.pixels)

    err_lim = 0.75 * max(np.abs(d).max() for d in diff["naive"].values())
    imshow_kw = dict(origin="lower", extent=(0, 1, 0, 1), interpolation="bilinear")
    field_kw = dict(cmap=FIELD_CMAP, vmin=0, vmax=1, **imshow_kw)
    error_kw = dict(cmap=ERROR_CMAP, vmin=0, vmax=err_lim, **imshow_kw)
    text_kw = dict(fontsize=10, color=INK, va="top", ha="left")

    n_rows, n_cols = len(frames), 1 + len(args.modes)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.4 * n_cols, 3.2 * n_rows + 1.2),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for r, k in enumerate(frames):
        ax_v = axes[r, 0]
        im_v = ax_v.imshow(image(refs[k][1]), **field_kw)
        style_map(ax_v, medium)
        if r == 0:
            ax_v.set_title("The wave (spectral reference)\n|v|", fontsize=11)
        for c, mode in enumerate(args.modes):
            ax_e = axes[r, 1 + c]
            im_e = ax_e.imshow(image(diff[mode][k]), **error_kw)
            style_map(ax_e, medium)
            rel = rel_error(runs[mode].state[k][1], refs[k][1])
            u_max = np.abs(runs[mode].state[k][0]).max()
            ax_e.text(
                0.03,
                0.97,
                f"rel. error in v {rel:.1%}\nmax |u| {u_max:.1e}",
                transform=ax_e.transAxes,
                **text_kw,
            )
            print(f"  t = {times[k]:.2f}  {mode:5s}  v {rel:.2e}  max|u| {u_max:.1e}")
            if r == 0:
                ax_e.set_title(f"{TITLES[mode]}\nerror in v vs reference", fontsize=11)
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
        label="|error in v| vs the spectral reference, one colour scale",
    )
    fig.suptitle(
        "Pressure pulse through a band with 4x stiffness and 2x density, "
        f"edges of width delta = {width:g} = h/{nodes.h / width:.3g}\n"
        f"{nodes.n} scattered nodes, RBF-FD, RK4",
        fontsize=11,
    )
    out = args.out_dir / "wave2d_stiff_snapshot.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    use_demo_style()
    t_all = time.perf_counter()
    t0 = time.perf_counter()
    wanted = set() if args.snapshot_only else set(args.ns)
    if not args.no_snapshot:
        wanted.add(args.snapshot_n)
    node_sets = {
        n: make_node_set(LayeredMedium2D(), n, seed=args.seed) for n in sorted(wanted)
    }
    print(f"node sets: {time.perf_counter() - t0:.1f}s")
    if not args.snapshot_only:
        sweep(args, node_sets)
    if not args.no_snapshot:
        snapshot(args, node_sets)
    print(f"total {time.perf_counter() - t_all:.0f}s")


if __name__ == "__main__":
    main()
