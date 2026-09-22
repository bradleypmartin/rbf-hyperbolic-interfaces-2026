"""Sparse RBF-FD operators for the 2-D elastic wave equation.

``build_operators`` turns a node set into

- ``dx``, ``dy``: ``n x n`` first-derivative matrices,
- ``hyper``: ``n x n`` hyperviscosity matrix ``(-1)^(k+1) Delta^k`` so that
  adding ``gamma * hyper`` to the right-hand side damps (for ``k = 3`` the
  sign is ``+Delta^3``, whose Fourier symbol is ``-|xi|^6``),
- ``elastic``: the ``5n x 5n`` block operator of dissertation eq. 32 acting
  on the stacked state ``(u, v, f, g, h)``, with the material coefficients
  evaluated at each stencil's centre node (as in ``createRBFLoperator1``),
- ``hyper_block``: ``hyper`` repeated on the diagonal for all five fields.

``mode="naive"`` uses the same plain RBF-FD stencils everywhere, including
across the interfaces; that is the wrong-but-standard treatment the demo
contrasts against. ``mode="aware"`` replaces the rows of nodes near an
interface with the coupled piecewise-polynomial stencils of
:mod:`rbf_hyperbolic_interfaces.wave2d.interface` (dissertation §3.3) for a jump, and
with the seed stencils of :mod:`rbf_hyperbolic_interfaces.wave2d.seeds` for a smooth
edge (``medium.is_smooth``, Part 3): every row whose stencil sees varying
material, the 1-D rule of ``wave1d.operators``.
"""

from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp

from .domain import LayeredMedium2D, NodeSet, SineInterface
from .interface import (
    InterfaceWeights,
    closest_point,
    interface_basis,
    interface_weights,
)
from .neighbors import minimal_image, periodic_knn, stencil_offsets
from .rbf import StencilWeights, rbf_fd_weights
from .seeds import NormalProfile, SeedBasis, normal_profile, seed_basis, seed_weights

Mode = Literal["naive", "aware"]


@dataclass(frozen=True)
class Operators:
    dx: sp.csr_array
    dy: sp.csr_array
    hyper: sp.csr_array
    elastic: sp.csr_array
    hyper_block: sp.csr_array
    stencils: np.ndarray  # (n, stencil_size) neighbour indices
    weights: StencilWeights
    hyper_power: int
    interface_nodes: np.ndarray  # rows rebuilt with interface or seed stencils


def _scatter(nodes: NodeSet, idx: np.ndarray, w: np.ndarray) -> sp.csr_array:
    rows = np.repeat(np.arange(nodes.n), idx.shape[1])
    return sp.csr_array((w.ravel(), (rows, idx.ravel())), shape=(nodes.n, nodes.n))


def elastic_block(
    dx: sp.csr_array,
    dy: sp.csr_array,
    lam: np.ndarray,
    mu: np.ndarray,
    rho: np.ndarray,
) -> sp.csr_array:
    """Assemble the ``5n x 5n`` operator of eq. 32 from derivative matrices."""
    d = sp.diags_array
    inv_rho = d(1.0 / rho)
    lam2mu = d(lam + 2 * mu)
    lam_d, mu_d = d(lam), d(mu)
    blocks = [
        [None, None, inv_rho @ dx, inv_rho @ dy, None],
        [None, None, None, inv_rho @ dx, inv_rho @ dy],
        [lam2mu @ dx, lam_d @ dy, None, None, None],
        [mu_d @ dy, mu_d @ dx, None, None, None],
        [lam_d @ dx, lam2mu @ dy, None, None, None],
    ]
    return sp.csr_array(sp.block_array(blocks, format="csr"))


def build_operators(
    nodes: NodeSet,
    medium: LayeredMedium2D,
    *,
    stencil_size: int = 30,
    poly_degree: int = 4,
    shape: float = 0.4,
    hyper_power: int = 3,
    mode: Mode = "naive",
    interface_stencil: int = 19,
    interface_degree: int = 3,
    interface_band: float = 4.0,
    seed_rtol: float = 0.0,
    seed_hyper_stencil: int | None = None,
    seed_tangential: bool = True,
    workers: int | None = None,
) -> Operators:
    """Operators on ``nodes``; ``mode="aware"`` rebuilds the rows of nodes within
    ``interface_band * h`` of an interface with the coupled stencils of
    :mod:`rbf_hyperbolic_interfaces.wave2d.interface` (19 nodes, degree 3, as in the
    MATLAB).

    For a smooth edge (``medium.is_smooth``) the rebuilt rows are instead
    those whose ``interface_stencil`` nearest nodes see different material
    values, ``LayeredMedium2D.varies_over`` with ``seed_rtol`` (exact
    inequality by default, which reaches about ``19 edge_width`` from an
    edge centre), and they get the seed stencils of
    :mod:`rbf_hyperbolic_interfaces.wave2d.seeds`: ``interface_stencil`` nodes for the
    elastic rows, and for the hyperviscosity rows the naive footprint of
    ``seed_hyper_stencil`` (default ``stencil_size``) nodes with the same
    seeds annihilated, since a 19-node stencil carrying 20 coupled
    constraints has too few degrees of freedom left for a ``Delta^3``
    (``docs/stiff-features.md`` §5.3). Two ODE marches per stencil, about
    50 ms; ``workers`` > 1 spreads them over that many processes (the march
    is Python-bound, so it scales with the cores), same weights to rounding.
    ``seed_tangential=False`` keeps the marched seeds for the pure normal
    monomials only (:func:`~.seeds.seed_basis`, the demo#41 ablation).
    """
    if mode not in ("naive", "aware"):
        raise ValueError(f"unknown mode {mode!r}")
    idx, _ = periodic_knn(nodes.xy, stencil_size)
    offsets = stencil_offsets(nodes.xy, idx)
    weights = rbf_fd_weights(
        offsets, shape=shape, poly_degree=poly_degree, hyper_power=hyper_power
    )
    dx = _scatter(nodes, idx, weights.dx)
    dy = _scatter(nodes, idx, weights.dy)
    hyper = _scatter(nodes, idx, (-1) ** (hyper_power + 1) * weights.hyper)
    lam = medium.lam_at(nodes.x, nodes.y)
    mu = medium.mu_at(nodes.x, nodes.y)
    rho = medium.rho_at(nodes.x, nodes.y)
    elastic = elastic_block(dx, dy, lam, mu, rho)
    hyper_block = sp.csr_array(sp.block_diag([hyper] * 5, format="csr"))
    interface_nodes = np.zeros(0, dtype=int)
    if mode == "aware" and medium.is_smooth:
        elastic, hyper_block, interface_nodes = _apply_seed_rows(
            nodes,
            medium,
            elastic,
            hyper_block,
            stencil_size=interface_stencil,
            hyper_stencil=seed_hyper_stencil or stencil_size,
            degree=interface_degree,
            rtol=seed_rtol,
            tangential=seed_tangential,
            shape=shape,
            hyper_power=hyper_power,
            workers=workers,
        )
    elif mode == "aware":
        elastic, hyper_block, interface_nodes = _apply_interface_rows(
            nodes,
            medium,
            elastic,
            hyper_block,
            stencil_size=interface_stencil,
            degree=interface_degree,
            band=interface_band * nodes.h,
            shape=shape,
            hyper_power=hyper_power,
        )
    return Operators(
        dx=dx,
        dy=dy,
        hyper=hyper,
        elastic=elastic,
        hyper_block=hyper_block,
        stencils=idx,
        weights=weights,
        hyper_power=hyper_power,
        interface_nodes=interface_nodes,
    )


def _local_frames(
    nodes: NodeSet, ifc: SineInterface, centre: np.ndarray, idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotated stencil coordinates about the nearest interface point.

    Returns ``(local, above, theta)`` with ``local`` ``(s, k, 2)`` in the
    frame of eq. 37 (x' along the tangent), ``above`` ``(s, k)`` from the
    true curve, and ``theta`` ``(s,)``.
    """
    x0, theta = closest_point(ifc, nodes.x[centre], nodes.y[centre])
    origin = np.stack([x0, ifc.height(x0)], axis=-1)
    disp = minimal_image(nodes.xy[idx] - origin[:, None, :])
    c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]
    local = np.stack(
        [c * disp[..., 0] + s * disp[..., 1], -s * disp[..., 0] + c * disp[..., 1]],
        axis=-1,
    )
    above = ifc.vertical_offset(nodes.x[idx], nodes.y[idx]) > 0
    return local, above, theta


def _apply_interface_rows(
    nodes: NodeSet,
    medium: LayeredMedium2D,
    elastic: sp.csr_array,
    hyper_block: sp.csr_array,
    *,
    stencil_size: int,
    degree: int,
    band: float,
    shape: float,
    hyper_power: int,
) -> tuple[sp.csr_array, sp.csr_array, np.ndarray]:
    """Rows within ``band`` of a jump interface get the piecewise-polynomial
    stencils, standard monomials inside the band as in the MATLAB."""
    off_lower = medium.lower.vertical_offset(nodes.x, nodes.y)
    off_upper = medium.upper.vertical_offset(nodes.x, nodes.y)
    near_lower = np.abs(off_lower) <= band
    near_upper = np.abs(off_upper) <= band
    use_lower = near_lower & (np.abs(off_lower) <= np.abs(off_upper))
    use_upper = near_upper & ~use_lower
    bg, ly = medium.background, medium.layer
    setups = [
        (
            medium.lower,
            np.flatnonzero(use_lower),
            interface_basis(bg, ly, degree, True),
        ),
        (
            medium.upper,
            np.flatnonzero(use_upper),
            interface_basis(ly, bg, degree, False),
        ),
    ]
    groups = []
    for ifc, centre, basis in setups:
        if centre.size == 0:
            continue
        idx, _ = periodic_knn(nodes.xy, stencil_size, query=nodes.xy[centre])
        # A stencil built for one interface must not reach past the other.
        if ifc is medium.lower:
            beyond = medium.upper.vertical_offset(nodes.x[idx], nodes.y[idx]) > 0
        else:
            beyond = medium.lower.vertical_offset(nodes.x[idx], nodes.y[idx]) < 0
        if np.any(beyond):
            raise NotImplementedError(
                "a stencil sees both interfaces (thin layer); not ported yet"
            )
        local, above, theta = _local_frames(nodes, ifc, centre, idx)
        w = interface_weights(
            local, above, theta, basis, shape=shape, hyper_power=hyper_power
        )
        groups.append((centre, idx, w))
    interface_nodes = np.concatenate([setups[0][1], setups[1][1]])
    return _merge_coupled_rows(
        nodes.n, elastic, hyper_block, groups, groups, interface_nodes, hyper_power
    )


def _stencils_seeing_variation(
    nodes: NodeSet, medium: LayeredMedium2D, idx: np.ndarray, rtol: float
) -> np.ndarray:
    """Indices of the stencils ``idx`` ``(n, k)`` over which ``lam``, ``mu`` or
    ``rho`` differ: ``LayeredMedium2D.varies_over`` for every stencil at once."""
    spread = np.zeros(nodes.n)
    for values in medium.material_at(nodes.x, nodes.y):
        at_nodes = values[idx]
        spread = np.maximum(
            spread, np.ptp(at_nodes, axis=1) / np.max(np.abs(at_nodes), axis=1)
        )
    return np.flatnonzero(spread > rtol)


def _apply_seed_rows(
    nodes: NodeSet,
    medium: LayeredMedium2D,
    elastic: sp.csr_array,
    hyper_block: sp.csr_array,
    *,
    stencil_size: int,
    hyper_stencil: int,
    degree: int,
    rtol: float,
    tangential: bool,
    shape: float,
    hyper_power: int,
    workers: int | None = None,
) -> tuple[sp.csr_array, sp.csr_array, np.ndarray]:
    """Rows whose ``stencil_size``-node stencils see a smooth edge get the seed
    stencils: elastic rows on those nodes, hyperviscosity rows on the
    ``hyper_stencil`` nearest nodes (the naive footprint), each from its own
    seed march.

    The nearest interface only sets the frame (its foot point is the origin
    of ``y'``); the material profile along the normal carries both edges, so
    a stencil that sees both is marched through both and needs no
    thin-layer guard.
    """
    idx_all, _ = periodic_knn(nodes.xy, max(stencil_size, hyper_stencil))
    idx_all = idx_all[:, :stencil_size]
    centre_all = _stencils_seeing_variation(nodes, medium, idx_all, rtol)
    pool_or_none = (
        ProcessPoolExecutor(max_workers=workers)
        if workers is not None and workers > 1 and centre_all.size > 1
        else nullcontext(None)
    )
    with pool_or_none as pool:
        return _seed_rows_with(
            pool,
            nodes,
            medium,
            elastic,
            hyper_block,
            centre_all,
            stencil_size=stencil_size,
            hyper_stencil=hyper_stencil,
            degree=degree,
            tangential=tangential,
            shape=shape,
            hyper_power=hyper_power,
        )


def _seed_basis_task(task: tuple[np.ndarray, NormalProfile, int, bool]) -> SeedBasis:
    local, profile, degree, tangential = task
    return seed_basis(local, profile, degree, tangential=tangential)


def _seed_bases(
    pool: Executor | None,
    local: np.ndarray,
    profiles: list[NormalProfile],
    degree: int,
    tangential: bool,
) -> list[SeedBasis]:
    """One :func:`seed_basis` per stencil, serially or on the pool (each task
    carries its 19 or 30 local coordinates and the profile, a few hundred
    bytes, and returns about 20 KB of seed values)."""
    tasks = [(local[s], profiles[s], degree, tangential) for s in range(len(profiles))]
    if pool is None:
        return [_seed_basis_task(t) for t in tasks]
    return list(pool.map(_seed_basis_task, tasks, chunksize=8))


def _seed_rows_with(
    pool: Executor | None,
    nodes: NodeSet,
    medium: LayeredMedium2D,
    elastic: sp.csr_array,
    hyper_block: sp.csr_array,
    centre_all: np.ndarray,
    *,
    stencil_size: int,
    hyper_stencil: int,
    degree: int,
    tangential: bool,
    shape: float,
    hyper_power: int,
) -> tuple[sp.csr_array, sp.csr_array, np.ndarray]:
    offsets = np.stack(
        [
            np.abs(ifc.vertical_offset(nodes.x[centre_all], nodes.y[centre_all]))
            for ifc in medium.interfaces
        ]
    )
    which = np.argmin(offsets, axis=0)
    groups_e, groups_h = [], []
    for j, ifc in enumerate(medium.interfaces):
        centre = centre_all[which == j]
        if centre.size == 0:
            continue
        x0, _ = closest_point(ifc, nodes.x[centre], nodes.y[centre])
        profiles = [normal_profile(medium, ifc, x) for x in x0]
        for k, groups in ((stencil_size, groups_e), (hyper_stencil, groups_h)):
            if k == stencil_size and groups is groups_h:
                groups.append(groups_e[-1])  # same footprint: one march serves both
                continue
            idx, _ = periodic_knn(nodes.xy, k, query=nodes.xy[centre])
            local, _, theta = _local_frames(nodes, ifc, centre, idx)
            bases = _seed_bases(pool, local, profiles, degree, tangential)
            w = seed_weights(local, theta, bases, shape=shape, hyper_power=hyper_power)
            groups.append((centre, idx, w))
    return _merge_coupled_rows(
        nodes.n, elastic, hyper_block, groups_e, groups_h, centre_all, hyper_power
    )


Group = tuple[np.ndarray, np.ndarray, InterfaceWeights]


def _merge_coupled_rows(
    n: int,
    elastic: sp.csr_array,
    hyper_block: sp.csr_array,
    groups_e: list[Group],
    groups_h: list[Group],
    rebuilt: np.ndarray,
    hyper_power: int,
) -> tuple[sp.csr_array, sp.csr_array, np.ndarray]:
    """Replace the rows ``rebuilt`` of the block operators by the coupled
    weights of the groups (``centre``, ``idx`` ``(s, k)``, lab-frame weights):
    ``groups_e`` for the elastic operator, ``groups_h`` for hyperviscosity."""
    rows_e, cols_e, vals_e = [], [], []
    rows_h, cols_h, vals_h = [], [], []
    sign = (-1) ** (hyper_power + 1)
    for centre, idx, w in groups_e:
        k = idx.shape[1]
        # Rates of (f, g, h) from (u, v) data and of (u, v) from (f, g, h).
        for r, row_field in enumerate((2, 3, 4)):
            for c, col_field in enumerate((0, 1)):
                rows_e.append(np.repeat(row_field * n + centre, k))
                cols_e.append((col_field * n + idx).ravel())
                vals_e.append(w.fgh_from_uv[:, r, c, :].ravel())
        for r, row_field in enumerate((0, 1)):
            for c, col_field in enumerate((2, 3, 4)):
                rows_e.append(np.repeat(row_field * n + centre, k))
                cols_e.append((col_field * n + idx).ravel())
                vals_e.append(w.uv_from_fgh[:, r, c, :].ravel())
    for centre, idx, w in groups_h:
        k = idx.shape[1]
        for r, row_field in enumerate((0, 1)):
            for c, col_field in enumerate((0, 1)):
                rows_h.append(np.repeat(row_field * n + centre, k))
                cols_h.append((col_field * n + idx).ravel())
                vals_h.append(sign * w.hyper_uv[:, r, c, :].ravel())
        for r, row_field in enumerate((2, 3, 4)):
            for c, col_field in enumerate((2, 3, 4)):
                rows_h.append(np.repeat(row_field * n + centre, k))
                cols_h.append((col_field * n + idx).ravel())
                vals_h.append(sign * w.hyper_fgh[:, r, c, :].ravel())

    keep = np.ones(5 * n)
    for field in range(5):
        keep[field * n + rebuilt] = 0.0
    mask = sp.diags_array(keep)

    def merge(base: sp.csr_array, rows, cols, vals) -> sp.csr_array:
        if not vals:
            return base
        add = sp.coo_array(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(5 * n, 5 * n),
        )
        out = sp.csr_array(mask @ base + add)
        out.eliminate_zeros()
        return out

    return (
        merge(elastic, rows_e, cols_e, vals_e),
        merge(hyper_block, rows_h, cols_h, vals_h),
        rebuilt,
    )


def hyperviscosity_gamma(
    h: float, hyper_power: int = 3, gamma0: float = 2.4e-11, h0: float = 0.02
) -> float:
    """MATLAB scaling ``gamma = gamma0 (h/h0)^(2k-1)``.

    From ``executeRbfSimulationDriver.m``: ``2.4e-11 * relH^(2*dampLk-1)``
    with ``relH = sqrt(2500/N)``, i.e. ``h0 = 0.02``.

    ``Delta^k`` weights grow like ``h^-2k``, so this keeps ``gamma * hyper``
    proportional to ``1/h`` like the wave operator itself: the eigenvalue
    cloud scales uniformly under refinement and one CFL number keeps RK4
    stable at every resolution.
    """
    return gamma0 * (h / h0) ** (2 * hyper_power - 1)
