"""Stiff smooth edges in 2-D: naive RBF-FD vs seed stencils through a tanh edge.

The flat two-interface problem of ``wave2d_convergence.py`` with both
interfaces smoothed into tanh transitions of width delta. Naive: plain RBF-FD
stencils everywhere, material coefficients sampled at the stencil centres.
Seeds: ``build_operators(mode="aware")``, the ODE-continued seed stencils of
``wave2d/seeds.py`` on every row that sees the edge (at delta = 0 that is
Part 2's interface-aware operator). Plane P-wave at normal incidence by
default; ``--direction m_x m_y`` sends the plane-wave train of
``oblique_p_wave`` at the angle atan(m_x / m_y) to the edge normal (#41),
where the seeds' x'-dependent columns act for the first time and P-to-S
conversion appears. Errors in v, h and u at t = 1 against the ray sum
(delta = 0, normal incidence), the 1-D pseudo-spectral reference mapped
through ``spectral_plane_wave`` (delta > 0, normal incidence) or the
Fourier-in-x reference of ``wave2d/spectral.py`` (delta > 0, oblique); the
reference snapshots are cached under ``outputs/``. At normal incidence the
u column is the largest spurious u (exactly 0 in the true solution); at
oblique incidence it is the relative error in u. "Floor" is the same pulse
on the same nodes in a uniform medium, the pure resolution error;
``--seed-floor`` adds the seed operator's own floor, the seed rows built
through a contrast of 1e-6 (seeds equal to monomials) against the exact
uniform-medium solution. ``--modes ablate`` runs the seeds with the
x'-dependent columns replaced by monomials (``seed_tangential=False``), the
ablation of #41. Fixed delta per panel, as in ``wave1d_stiff.py``: the
knee, if any, shows as n crosses h = delta.

A seed row costs two ODE marches, about 50 ms, and a resolved edge seeds
every row (17 min serially at 19,600 nodes), so the marches run on
``--workers`` processes and the seed operators are cached under ``outputs/``
keyed on (n, seed, delta, band material, ablation); they are pulse
independent, so the oblique runs reuse the normal-incidence ones. Delete
the ``wave2d_stiff_ops_*`` files after any change to the seed construction.

``--amplitude 0.02`` bends both interfaces into the sine curves of Part 2's
curved case (#42): the seeds of every row are the straight-feature seeds
along the true normal through the stencil's foot point, route (a) of the
issue. References: for delta > 0 the product-grid Fourier solver of
``wave2d/spectral.py: run_fourier_2d`` (``--ref-nx``, ``--ref-ny``, cached
under ``outputs/`` per snapshot time; ``--ref-cfl`` halves the step for a
self-convergence check), for delta = 0 the jump-aware operator on a
``--ref-n``-node set of its own, resampled one-sided onto the sweep's nodes
as ``wave2d_convergence.py`` does. ``--truncation`` adds the probe of #41:
the elastic operator applied to the reference state at t_end on the nodes
against the exact rate from the reference's derivatives, per row group.
Normal incidence only.

Also writes a still at the clip resolution (``--snapshot-n``) through an edge
that sits between the straddling rows (``--snapshot-width``): the reference
wave (and, at oblique incidence or on a curved edge, its curl, which maps
the S waves), then each method's error map, at ``--snapshot-times``.

    uv run python scripts/wave2d_stiff.py                      # both modes, 4 widths
    uv run python scripts/wave2d_stiff.py --modes naive        # the #37 baseline
    uv run python scripts/wave2d_stiff.py --widths 0 0.0025 --ns 2500 4900
    uv run python scripts/wave2d_stiff.py --snapshot-only
    uv run python scripts/wave2d_stiff.py --direction 1 2 --widths 0.0025 0.01 \\
        --modes naive aware ablate --seed-floor           # the #41 oblique sweep
    uv run python scripts/wave2d_stiff.py --amplitude 0.02 --widths 0 0.005 0.01 \\
        --seed-floor --truncation --snapshot-width 0.005  # the #42 curved sweep
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
    ERROR_CMAP,
    FIELD_CMAP,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    TEXTWIDTH_IN,
    use_demo_style,
    use_print_style,
)
from pdes_demo.results_cache import ResultsCache
from pdes_demo.stiff_figures import convergence_2d
from pdes_demo.wave1d import periodic_grid
from pdes_demo.wave1d.spectral import reference_size
from pdes_demo.wave2d import (
    ElasticMaterial,
    GridState,
    LayeredMedium2D,
    ModeState,
    NodeSet,
    Operators,
    SineInterface,
    build_operators,
    exact_plane_wave,
    make_node_set,
    oblique_p_wave,
    pixel_grid,
    resample_matrix,
    run,
    run_fourier,
    run_fourier_2d,
)
from pdes_demo.wave2d.exact import plane_wave_from_1d, spectral_plane_wave_1d

MODE_SHORT = {"naive": "naive", "aware": "seeds", "ablate": "ablat"}
TITLES = {"naive": "Standard RBF-FD (naive)", "aware": "Seed stencils"}
PRINT_TITLES = {"naive": "naive", "aware": "seeds"}
BUILD_MODE = {"naive": "naive", "aware": "aware", "ablate": "aware"}
UNIFORM = LayeredMedium2D(layer=ElasticMaterial(lam=1.0, mu=1.0, rho=1.0))
# A band the exact uniform solution cannot tell from the background, on
# which the seed rows are still built (float inequality): the seed floor.
SEED_FLOOR_LAYER = ElasticMaterial(lam=1 + 1e-6, mu=1 + 1e-6, rho=1 + 1e-6)
REF_DT = 5e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--widths", type=float, nargs="+", default=[0.0, 0.0025, 0.01, 0.04]
    )
    parser.add_argument("--ns", type=int, nargs="+", default=[2500, 4900, 10000, 19600])
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["naive", "aware"],
        choices=["naive", "aware", "ablate"],
    )
    parser.add_argument(
        "--direction",
        type=int,
        nargs=2,
        default=[0, 1],
        metavar=("M_X", "M_Y"),
        help="lattice direction of the plane-wave train (default: normal incidence)",
    )
    parser.add_argument(
        "--seed-floor",
        action="store_true",
        help="also measure the seed operator's own floor (1e-6 contrast)",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.0,
        help="sine amplitude of both interfaces (0.02 is Part 2's curved case)",
    )
    parser.add_argument(
        "--ref-nx", type=int, default=None, help="product-grid reference n_x"
    )
    parser.add_argument(
        "--ref-ny", type=int, default=None, help="product-grid reference n_y"
    )
    parser.add_argument("--ref-cfl", type=float, default=0.5)
    parser.add_argument(
        "--ref-n",
        type=int,
        default=122500,
        help="nodes of the jump-aware reference run for a curved jump (delta = 0)",
    )
    parser.add_argument(
        "--truncation",
        action="store_true",
        help="truncation error of the operators on the reference state, per row group",
    )
    parser.add_argument(
        "--seed-rtol",
        type=float,
        default=0.0,
        help="seed only the rows whose stencil sees a relative material spread "
        "above this (0: any spread, tails to 19 delta; 1e-3 trims to 3.8 delta)",
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
    args = parser.parse_args()
    args.direction = tuple(args.direction)
    args.oblique = args.direction != (0, 1)
    args.curved = args.amplitude != 0.0
    # u is exactly 0 only for the flat pulse at normal incidence; otherwise
    # the u column is the relative error in u.
    args.u_error = args.oblique or args.curved
    if args.oblique and 0.0 in args.widths:
        parser.error("oblique incidence has no jump reference; use widths > 0")
    if args.oblique and args.curved:
        parser.error("the curved case runs at normal incidence")
    return args


def medium_for(
    width: float, args: argparse.Namespace, layer: ElasticMaterial | None = None
) -> LayeredMedium2D:
    """The band with edges of ``width`` on the sweep's geometry."""
    kw = {} if layer is None else {"layer": layer}
    if args.curved:
        kw["lower"] = SineInterface(0.25, args.amplitude)
        kw["upper"] = SineInterface(0.5, args.amplitude)
    return LayeredMedium2D(edge_width=width, **kw)


def geometry_tag(args: argparse.Namespace) -> str:
    return f"_a{args.amplitude:g}" if args.curved else ""


def direction_tag(args: argparse.Namespace) -> str:
    return f"_d{args.direction[0]}{args.direction[1]}" if args.oblique else ""


def angle_deg(args: argparse.Namespace) -> float:
    return float(np.degrees(np.arctan2(*args.direction)))


def run_tag(args: argparse.Namespace) -> str:
    """Suffix that names this configuration's figure and results files."""
    tag = "_naive" if args.modes == ["naive"] else ""
    tag += "" if args.sharpness == 15.0 else f"_s{args.sharpness:g}"
    tag += direction_tag(args) + geometry_tag(args)
    tag += f"_r{args.seed_rtol:g}" if args.seed_rtol else ""
    return tag


def u_field(args: argparse.Namespace) -> str:
    """Cache field name of the u column: a relative error, or the spurious max."""
    return "u" if args.u_error else "max_u"


# --- references ---------------------------------------------------------------


def reference_1d(medium: LayeredMedium2D, args: argparse.Namespace, t: float):
    """Cached 1-D spectral snapshot ``(grid, u, f)`` at ``t``, smooth media,
    normal incidence."""
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


def _modes_path(medium: LayeredMedium2D, args: argparse.Namespace, t: float) -> Path:
    return args.out_dir / (
        f"wave2d_stiff_ref2d_w{medium.edge_width:g}{direction_tag(args)}"
        f"_s{args.sharpness:g}_c{args.center:g}_t{t:g}_dt{REF_DT:g}.npz"
    )


def reference_modes(
    medium: LayeredMedium2D, args: argparse.Namespace, t: float
) -> ModeState:
    """Cached Fourier-in-x snapshot at ``t``, smooth media, any direction.

    One run serves every time the driver will ask for (the sweep's ``t_end``
    and the still's times), so the missing ones are computed together.
    """
    path = _modes_path(medium, args, t)
    if path.exists():
        data = np.load(path)
        return ModeState(
            modes=data["modes"], y=data["y"], t=float(data["t"]), coef=data["coef"]
        )
    wanted = {args.t_end}
    if not args.no_snapshot and medium.edge_width == args.snapshot_width:
        wanted.update(args.snapshot_times)
    missing = sorted(
        s for s in wanted | {t} if not _modes_path(medium, args, s).exists()
    )
    t0 = time.perf_counter()
    states = run_fourier(
        medium,
        lambda xy: oblique_p_wave(
            xy, medium, args.direction, args.center, args.sharpness
        ),
        max(missing),
        snapshot_times=missing,
        dt=REF_DT,
    )
    for state in states:
        np.savez(
            _modes_path(medium, args, state.t),
            modes=state.modes,
            y=state.y,
            t=state.t,
            coef=state.coef,
        )
    print(
        f"  Fourier reference, {states[0].modes.size} modes, n_y = {states[0].n_y}, "
        f"t = {missing} in {time.perf_counter() - t0:.0f}s -> {path.parent}"
    )
    return next(s for s in states if s.t == t)


def _grid_path(medium: LayeredMedium2D, args: argparse.Namespace, t: float) -> Path:
    n_x, n_y = grid_size(medium, args)
    cfl = "" if args.ref_cfl == 0.5 else f"_cfl{args.ref_cfl:g}"
    return args.out_dir / (
        f"wave2d_stiff_refgrid_w{medium.edge_width:g}{geometry_tag(args)}"
        f"_s{args.sharpness:g}_c{args.center:g}_t{t:g}_{n_x}x{n_y}{cfl}.npz"
    )


def grid_size(medium: LayeredMedium2D, args: argparse.Namespace) -> tuple[int, int]:
    n_y = args.ref_ny or reference_size(2 * medium.edge_width)
    return args.ref_nx or max(64, n_y // 2), n_y


def reference_grid(
    medium: LayeredMedium2D, args: argparse.Namespace, t: float
) -> GridState:
    """Cached product-grid snapshot at ``t`` (curved smooth media); the
    sweep's ``t_end`` and the still's times come from one run."""
    path = _grid_path(medium, args, t)
    if path.exists():
        data = np.load(path)
        return GridState(
            x=data["x"], y=data["y"], t=float(data["t"]), fields=data["fields"]
        )
    wanted = {args.t_end}
    if not args.no_snapshot and medium.edge_width == args.snapshot_width:
        wanted.update(args.snapshot_times)
    missing = sorted(
        s for s in wanted | {t} if not _grid_path(medium, args, s).exists()
    )
    n_x, n_y = grid_size(medium, args)
    t0 = time.perf_counter()
    states = run_fourier_2d(
        medium,
        lambda xy: oblique_p_wave(
            xy, medium, args.direction, args.center, args.sharpness
        ),
        max(missing),
        snapshot_times=missing,
        n_x=n_x,
        n_y=n_y,
        cfl=args.ref_cfl,
    )
    for state in states:
        np.savez(
            _grid_path(medium, args, state.t),
            x=state.x,
            y=state.y,
            t=state.t,
            fields=state.fields,
        )
    print(
        f"  product-grid reference {n_x} x {n_y}, t = {missing} in "
        f"{time.perf_counter() - t0:.0f}s -> {path.parent}"
    )
    return next(s for s in states if s.t == t)


def reference_curved_jump(
    medium: LayeredMedium2D, args: argparse.Namespace, t: float
) -> tuple[NodeSet, np.ndarray]:
    """Cached jump-aware run on ``--ref-n`` nodes at ``t`` (curved jump)."""
    path = args.out_dir / (
        f"wave2d_stiff_refjump{geometry_tag(args)}_n{args.ref_n}_seed{args.seed}"
        f"_s{args.sharpness:g}_c{args.center:g}_t{t:g}.npz"
    )
    if path.exists():
        data = np.load(path)
        fine = NodeSet(xy=data["xy"], h=float(data["h"]), fixed=data["fixed"])
        return fine, data["state"]
    t0 = time.perf_counter()
    fine = make_node_set(medium, args.ref_n, seed=args.seed)
    snaps = run(
        fine,
        medium,
        mode="aware",
        t_end=t,
        n_snapshots=1,
        pulse_center=args.center,
        pulse_sharpness=args.sharpness,
    )
    state = snaps.state[-1]
    np.savez(path, xy=fine.xy, h=fine.h, fixed=fine.fixed, state=state)
    print(
        f"  jump-aware reference on {fine.n} nodes, t = {t:g} in "
        f"{time.perf_counter() - t0:.0f}s -> {path}"
    )
    return fine, state


_REFERENCE_CACHE: dict[tuple[int, float, float], np.ndarray] = {}


def reference_at(
    nodes: NodeSet, medium: LayeredMedium2D, args: argparse.Namespace, t: float
) -> np.ndarray:
    """Reference state ``(5, n)`` at ``t`` on the nodes."""
    if not args.oblique and not args.curved:
        if not medium.is_smooth:
            return exact_plane_wave(nodes, t, medium, args.center, args.sharpness)
        grid, u, f = reference_1d(medium, args, t)
        return plane_wave_from_1d(
            nodes, medium, grid, u, f, args.center, args.sharpness
        )
    key = (nodes.n, medium.edge_width, t)
    if key not in _REFERENCE_CACHE:
        if args.oblique:
            _REFERENCE_CACHE[key] = reference_modes(medium, args, t).evaluate(nodes.xy)
        elif medium.is_smooth:
            _REFERENCE_CACHE[key] = reference_grid(medium, args, t).evaluate(nodes.xy)
        else:
            fine, state = reference_curved_jump(medium, args, t)
            to_nodes = resample_matrix(fine, nodes.xy, medium)
            _REFERENCE_CACHE[key] = np.stack([to_nodes @ s for s in state])
    return _REFERENCE_CACHE[key]


def exact_uniform(nodes: NodeSet, args: argparse.Namespace, t: float) -> np.ndarray:
    """The pulse translated through the uniform background: the floors' reference."""
    if not args.oblique:
        return exact_plane_wave(nodes, t, UNIFORM, args.center, args.sharpness)
    return oblique_p_wave(
        nodes, UNIFORM, args.direction, args.center, args.sharpness, t=t
    )


def initial_state(
    nodes: NodeSet, medium: LayeredMedium2D, args: argparse.Namespace
) -> np.ndarray | None:
    """``None`` at normal incidence (``run`` sets the plane pulse itself, so
    the #40 numbers are reproduced bit for bit), the train otherwise."""
    if not args.oblique:
        return None
    return oblique_p_wave(nodes, medium, args.direction, args.center, args.sharpness)


# --- operators ----------------------------------------------------------------


def _csr(data: np.lib.npyio.NpzFile, tag: str, n: int) -> sp.csr_array:
    parts = (data[f"{tag}_data"], data[f"{tag}_indices"], data[f"{tag}_indptr"])
    return sp.csr_array(parts, shape=(5 * n, 5 * n))


def _ops_path(
    nodes: NodeSet, medium: LayeredMedium2D, mode: str, args: argparse.Namespace
) -> Path:
    layer = medium.layer
    tag = (
        ""
        if layer == LayeredMedium2D().layer
        else f"_L{layer.lam:g}-{layer.mu:g}-{layer.rho:g}"
    )
    tag += "_abl" if mode == "ablate" else ""
    tag += geometry_tag(args)
    tag += f"_r{args.seed_rtol:g}" if args.seed_rtol else ""
    return args.out_dir / (
        f"wave2d_stiff_ops_n{nodes.n}_seed{args.seed}_w{medium.edge_width:g}{tag}.npz"
    )


def operators_for(
    nodes: NodeSet, medium: LayeredMedium2D, mode: str, args: argparse.Namespace
) -> Operators:
    """Operators for the run; seed operators come from the cache when present.

    Only the seed rows are stored (the elastic and hyperviscosity blocks and
    the row list); everything else in :class:`Operators` is the naive build,
    which is what ``build_operators(mode="aware")`` starts from.
    """
    if mode == "naive" or not medium.is_smooth:
        return build_operators(nodes, medium, mode=BUILD_MODE[mode])
    path = _ops_path(nodes, medium, mode, args)
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
    ops = build_operators(
        nodes,
        medium,
        mode="aware",
        seed_tangential=mode != "ablate",
        seed_rtol=args.seed_rtol,
        workers=args.workers,
    )
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
        f"  seed operator ({mode}), {ops.interface_nodes.size} of {nodes.n} rows, "
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
    medium: LayeredMedium2D,
    nodes: NodeSet,
    mode: str,
    args: argparse.Namespace,
    reference: np.ndarray | None = None,
) -> dict[str, float]:
    """Relative l2 errors in v, h and u at ``t_end`` against ``reference``
    (the medium's own by default); at normal incidence the u entry is the
    largest spurious |u| instead."""
    snaps = run(
        nodes,
        medium,
        t_end=args.t_end,
        n_snapshots=1,
        pulse_center=args.center,
        pulse_sharpness=args.sharpness,
        operators=operators_for(nodes, medium, mode, args),
        initial=initial_state(nodes, medium, args),
    )
    state = snaps.state[-1]
    ref = reference
    if ref is None:
        ref = reference_at(nodes, medium, args, args.t_end)
    # A reference without u (the floors' uniform-medium pulse) gets the
    # largest spurious |u| instead of a relative error.
    if args.u_error and np.linalg.norm(ref[0]) > 0:
        u = rel_error(state[0], ref[0])
    else:
        u = float(np.abs(state[0]).max())
    return {"v": rel_error(state[1], ref[1]), "h": rel_error(state[4], ref[4]), "u": u}


def u_label(args: argparse.Namespace) -> str:
    return "u err" if args.u_error else "max|u|"


def truncation_errors(
    medium: LayeredMedium2D, nodes: NodeSet, args: argparse.Namespace
) -> dict[str, dict[str, float]]:
    """Relative l2 truncation error of each operator on the reference state
    at ``t_end``, per row group (the rebuilt rows and the rest): the exact
    rate is eq. 32 with the reference's own derivatives at the nodes and
    the material there. Curved smooth media only (the grid reference has
    derivatives)."""
    grid = reference_grid(medium, args, args.t_end)
    state = grid.evaluate(nodes.xy)
    d_x = grid.evaluate(nodes.xy, dx=1)
    d_y = grid.evaluate(nodes.xy, dy=1)
    lam, mu, rho = medium.material_at(nodes.x, nodes.y)
    exact = np.stack(
        [
            (d_x[2] + d_y[3]) / rho,
            (d_x[3] + d_y[4]) / rho,
            (lam + 2 * mu) * d_x[0] + lam * d_y[1],
            mu * (d_y[0] + d_x[1]),
            lam * d_x[0] + (lam + 2 * mu) * d_y[1],
        ]
    )
    out = {}
    for mode in args.modes:
        ops = operators_for(nodes, medium, mode, args)
        rate = (ops.elastic @ state.ravel()).reshape(5, -1)
        rebuilt = np.zeros(nodes.n, dtype=bool)
        rebuilt[ops.interface_nodes] = True
        groups = {"edge": rebuilt, "bulk": ~rebuilt}
        if mode == "naive":
            aware = operators_for(nodes, medium, "aware", args)
            rebuilt = np.zeros(nodes.n, dtype=bool)
            rebuilt[aware.interface_nodes] = True
            groups = {"edge": rebuilt, "bulk": ~rebuilt}
        out[mode] = {
            name: rel_error(rate[:, sel], exact[:, sel]) for name, sel in groups.items()
        }
    return out


def print_table(
    width: float,
    ns: np.ndarray,
    results: dict[str, list[dict[str, float]]],
    floors: dict[str, list[dict[str, float]]],
    args: argparse.Namespace,
) -> None:
    modes = list(results)
    rated = "vhu" if args.u_error else "vh"
    head = "      N  h/delta"
    for mode in modes:
        m = MODE_SHORT[mode]
        for f in "vh":
            head += f" | {m + ' ' + f:>9s} {'rate':>4s}"
        head += f" {u_label(args):>7s}" + (" rate" if args.u_error else "")
    for name in floors:
        head += f" | {name + ' v':>8s} {name + ' u':>9s}"
    print(head)
    r = {
        (mode, f): rates(ns, [e[f] for e in results[mode]])
        for mode in modes
        for f in rated
    }
    for i, n in enumerate(ns):
        h_over = f"{1 / np.sqrt(n) / width:7.2f}" if width else "    inf"
        line = f"  {int(n):5d}  {h_over}"
        for mode in modes:
            cells = ""
            for f in "vh":
                rate = f"{r[(mode, f)][i - 1]:4.1f}" if i else "    "
                cells += f" | {results[mode][i][f]:9.2e} {rate}"
            cells += f" {results[mode][i]['u']:7.1e}"
            if args.u_error:
                cells += f" {r[(mode, 'u')][i - 1]:4.1f}" if i else "     "
            line += cells
        for name in floors:
            line += f" | {floors[name][i]['v']:8.2e} {floors[name][i]['u']:9.1e}"
        print(line)


def record_errors(
    cache: ResultsCache,
    ns: np.ndarray,
    results: dict[str, list[dict[str, float]]],
    delta: float | None,
    args: argparse.Namespace,
) -> None:
    """``error`` records for every (mode, field) of one edge width; ``delta``
    is ``None`` for the floors, which do not depend on it."""
    h = [1 / np.sqrt(n) for n in ns]
    for mode, errs in results.items():
        for f in "vhu":
            values = [e[f] for e in errs]
            cache.add_errors(
                [int(n) for n in ns],
                values,
                list(rates(ns, values)),
                h=h,
                delta=delta,
                mode=mode,
                field=u_field(args) if f == "u" else f,
            )


def sweep(
    args: argparse.Namespace, node_sets: dict[int, NodeSet], cache: ResultsCache
) -> None:
    ns = np.array(args.ns, dtype=float)
    t0 = time.perf_counter()
    floor = [
        errors_for(
            UNIFORM,
            node_sets[n],
            "naive",
            args,
            exact_uniform(node_sets[n], args, args.t_end),
        )
        for n in args.ns
    ]
    print(f"resolution floor (uniform medium): {time.perf_counter() - t0:.1f}s")
    record_errors(cache, ns, {"floor": floor}, None, args)

    for width in args.widths:
        medium = medium_for(width, args)
        title = (
            "jump edges (delta = 0)" if width == 0 else f"edge width delta = {width:g}"
        )
        print(f"\n{title}")
        t0 = time.perf_counter()
        results = {
            mode: [errors_for(medium, node_sets[n], mode, args) for n in args.ns]
            for mode in args.modes
        }
        floors = {"floor": floor}
        if args.seed_floor and width > 0:
            seed_medium = medium_for(width, args, layer=SEED_FLOOR_LAYER)
            floors["sfloor"] = [
                errors_for(
                    seed_medium,
                    node_sets[n],
                    "aware",
                    args,
                    exact_uniform(node_sets[n], args, args.t_end),
                )
                for n in args.ns
            ]
        print(f"  ({time.perf_counter() - t0:.1f}s)")
        print_table(width, ns, results, floors, args)
        record_errors(cache, ns, results, width, args)
        if "sfloor" in floors:
            record_errors(cache, ns, {"sfloor": floors["sfloor"]}, width, args)
        if args.truncation and args.curved and width > 0:
            print("  truncation error on the reference state, edge rows | bulk rows")
            for n in args.ns:
                errs = truncation_errors(medium, node_sets[n], args)
                cells = "  ".join(
                    f"{MODE_SHORT[m]} {e['edge']:8.2e} | {e['bulk']:8.2e}"
                    for m, e in errs.items()
                )
                print(f"  {int(n):5d}  {cells}")
                for m, e in errs.items():
                    for group, err in e.items():
                        cache.add(
                            "truncation",
                            n=int(n),
                            h=1 / np.sqrt(n),
                            delta=width,
                            mode=m,
                            group=group,
                            error=err,
                        )

    out = convergence_2d(
        cache,
        args.out_dir / f"wave2d_stiff{run_tag(args)}.{args.format}",
        print_mode=args.style == "print",
    )
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
    xs = np.linspace(0, 1, 201)
    for ifc in medium.interfaces:
        ax.plot(xs, ifc.height(xs), ls="--", color=INK_SECONDARY, lw=1.0)


def snapshot(
    args: argparse.Namespace, node_sets: dict[int, NodeSet], cache: ResultsCache
) -> None:
    n, width = args.snapshot_n, args.snapshot_width
    nodes = node_sets.get(n) or make_node_set(medium_for(0.0, args), n, seed=args.seed)
    medium = medium_for(width, args)
    modes = [m for m in args.modes if m != "ablate"]
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
            initial=initial_state(nodes, medium, args),
        )
        for mode in modes
    }
    times = runs[modes[0]].t
    frames = [int(np.argmin(np.abs(times - t))) for t in args.snapshot_times]
    refs = {k: reference_at(nodes, medium, args, float(times[k])) for k in frames}
    print(
        f"\nstill: {nodes.n} nodes, delta = {width:g} = h/{nodes.h / width:.3g}, "
        f"{time.perf_counter() - t0:.1f}s"
    )
    diff = {
        mode: {k: runs[mode].state[k][1] - refs[k][1] for k in frames} for mode in modes
    }
    grid = pixel_grid(args.pixels)
    to_grid = resample_matrix(nodes, grid, medium)

    def image(values: np.ndarray) -> np.ndarray:
        return np.abs(values).reshape(args.pixels, args.pixels)

    # The reference itself is drawn from its own representation: resampled
    # from the nodes at flat normal incidence, evaluated on the pixels (and
    # its curl, the S waves) from the Fourier modes or the product grid.
    wave, curl = {}, {}
    show_curl = args.oblique or args.curved
    for k in frames:
        if args.oblique:
            state = reference_modes(medium, args, float(times[k]))
        elif args.curved:
            state = reference_grid(medium, args, float(times[k]))
        else:
            wave[k] = image(to_grid @ refs[k][1])
            continue
        wave[k] = image(state.evaluate(grid)[1])
        curl[k] = image(state.curl(grid))
    err_lim = 0.75 * max(np.abs(d).max() for d in diff[modes[0]].values())
    imshow_kw = dict(origin="lower", extent=(0, 1, 0, 1), interpolation="bilinear")
    field_kw = dict(cmap=FIELD_CMAP, vmin=0, vmax=1, **imshow_kw)
    error_kw = dict(cmap=ERROR_CMAP, vmin=0, vmax=err_lim, **imshow_kw)
    print_mode = args.style == "print"
    fs = 7 if print_mode else 11
    text_kw = dict(fontsize=6 if print_mode else 10, color=INK, va="top", ha="left")
    titles = PRINT_TITLES if print_mode else TITLES

    n_ref = 2 if show_curl else 1
    n_rows, n_cols = len(frames), n_ref + len(modes)
    if print_mode:
        # One text width across; the colorbars below add about 0.5 in per figure.
        figsize = (TEXTWIDTH_IN, TEXTWIDTH_IN / n_cols * 1.15 * n_rows + 0.55)
    else:
        figsize = (3.4 * n_cols, 3.2 * n_rows + 1.2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
    axes = np.atleast_2d(axes)
    ref_name = (
        "Fourier reference" if (args.oblique or args.curved) else "spectral reference"
    )
    for r, k in enumerate(frames):
        ax_v = axes[r, 0]
        im_v = ax_v.imshow(wave[k], **field_kw)
        style_map(ax_v, medium)
        if r == 0:
            ax_v.set_title(
                "$|v|$, reference" if print_mode else f"The wave ({ref_name})\n|v|",
                fontsize=fs,
            )
        if show_curl:
            ax_c = axes[r, 1]
            curl_kw = dict(
                cmap=FIELD_CMAP, vmin=0, vmax=max(c.max() for c in curl.values())
            )
            im_c = ax_c.imshow(curl[k], **curl_kw, **imshow_kw)
            style_map(ax_c, medium)
            if r == 0:
                ax_c.set_title(
                    r"$|u_y - v_x|$, S waves"
                    if print_mode
                    else "S waves: |u_y - v_x|\n(zero for a P wave)",
                    fontsize=fs,
                )
        for c, mode in enumerate(modes):
            ax_e = axes[r, n_ref + c]
            im_e = ax_e.imshow(image(to_grid @ diff[mode][k]), **error_kw)
            style_map(ax_e, medium)
            rel = rel_error(runs[mode].state[k][1], refs[k][1])
            if args.u_error:
                rel_u = rel_error(runs[mode].state[k][0], refs[k][0])
                u_line, u_print = f"rel. error in u {rel_u:.1%}", f"u {rel_u:.2e}"
                u_short = f"$u$: {rel_u:.1%}"
            else:
                u_max = np.abs(runs[mode].state[k][0]).max()
                u_line, u_print = f"max |u| {u_max:.1e}", f"max|u| {u_max:.1e}"
                u_short = f"max $|u|$: {u_max:.1e}"
            ax_e.text(
                0.03,
                0.97,
                f"$v$: {rel:.1%}\n{u_short}"
                if print_mode
                else f"rel. error in v {rel:.1%}\n{u_line}",
                transform=ax_e.transAxes,
                **text_kw,
            )
            print(f"  t = {times[k]:.2f}  {mode:5s}  v {rel:.2e}  {u_print}")
            u_value = rel_u if args.u_error else u_max
            for field, value in (("v", rel), (u_field(args), u_value)):
                cache.add(
                    "snapshot",
                    n=nodes.n,
                    delta=width,
                    t=float(times[k]),
                    mode=mode,
                    field=field,
                    error=value,
                )
            if r == 0:
                ax_e.set_title(
                    f"error in $v$, {titles[mode]}"
                    if print_mode
                    else f"{titles[mode]}\nerror in v vs reference",
                    fontsize=fs,
                )
        axes[r, 0].set_ylabel(
            f"$t = {times[k]:.2f}$" if print_mode else f"t = {times[k]:.2f}\ny"
        )
    for ax in axes[-1]:
        ax.set_xlabel("$x$" if print_mode else "x")
    cb_kw = dict(location="bottom", pad=0.02)
    fig.colorbar(
        im_v,
        ax=axes[:, 0].tolist(),
        shrink=0.8,
        label="$|v|$" if print_mode else "|v|, vertical particle velocity",
        **cb_kw,
    )
    if show_curl:
        fig.colorbar(
            im_c,
            ax=axes[:, 1].tolist(),
            shrink=0.8,
            label="|curl|" if print_mode else "|curl of the velocity|",
            **cb_kw,
        )
    fig.colorbar(
        im_e,
        ax=axes[:, n_ref:].ravel().tolist(),
        shrink=0.5,
        label=(
            "$|$error in $v|$, one scale"
            if print_mode
            else f"|error in v| vs the {ref_name}, one colour scale"
        ),
        **cb_kw,
    )
    incidence = (
        f"P train at {angle_deg(args):.1f} deg to the normal"
        if args.oblique
        else "Pressure pulse"
    )
    band = (
        f"a band with curved edges (amplitude {args.amplitude:g}), 4x stiffness and "
        "2x density"
        if args.curved
        else "a band with 4x stiffness and 2x density"
    )
    if not print_mode:
        fig.suptitle(
            f"{incidence} through {band}, "
            f"edges of width delta = {width:g} = h/{nodes.h / width:.3g}\n"
            f"{nodes.n} scattered nodes, RBF-FD, RK4",
            fontsize=11,
        )
    out = args.out_dir / (
        f"wave2d_stiff_snapshot{direction_tag(args)}{geometry_tag(args)}.{args.format}"
    )
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.style == "print":
        use_print_style()
    else:
        use_demo_style()
    t_all = time.perf_counter()
    t0 = time.perf_counter()
    wanted = set() if args.snapshot_only else set(args.ns)
    if not args.no_snapshot:
        wanted.add(args.snapshot_n)
    geometry = medium_for(0.0, args)
    node_sets = {n: make_node_set(geometry, n, seed=args.seed) for n in sorted(wanted)}
    print(f"node sets: {time.perf_counter() - t0:.1f}s")
    cache = ResultsCache.new("scripts/wave2d_stiff.py", args)
    if not args.snapshot_only:
        sweep(args, node_sets, cache)
    if not args.no_snapshot:
        snapshot(args, node_sets, cache)
    # A still-only run must not overwrite the sweep's cache of the same tag.
    kind = "_snapshot" if args.snapshot_only else ""
    name = f"wave2d_stiff{run_tag(args)}{kind}.json"
    paths = [args.out_dir / name]
    if args.data_dir is not None:
        paths.append(args.data_dir / name)
    cache.write(*paths)
    print("results ->", ", ".join(str(p) for p in paths))
    print(f"total {time.perf_counter() - t_all:.0f}s")


if __name__ == "__main__":
    main()
