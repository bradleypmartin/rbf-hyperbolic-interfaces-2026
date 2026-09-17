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
:mod:`pdes_demo.wave2d.interface` (dissertation §3.3).
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp

from .domain import LayeredMedium2D, NodeSet, SineInterface
from .interface import closest_point, interface_basis, interface_weights
from .neighbors import minimal_image, periodic_knn, stencil_offsets
from .rbf import StencilWeights, rbf_fd_weights

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
    interface_nodes: np.ndarray  # indices whose rows use interface-aware stencils


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
) -> Operators:
    """Operators on ``nodes``; ``mode="aware"`` rebuilds the rows of nodes within
    ``interface_band * h`` of an interface with the coupled stencils of
    :mod:`pdes_demo.wave2d.interface` (19 nodes, degree 3, as in the MATLAB).
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
    if mode == "aware":
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
    n = nodes.n
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
    rows_e, cols_e, vals_e = [], [], []
    rows_h, cols_h, vals_h = [], [], []
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
        s, k = idx.shape
        sign = (-1) ** (hyper_power + 1)
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

    interface_nodes = np.concatenate([setups[0][1], setups[1][1]])
    keep = np.ones(5 * n)
    for field in range(5):
        keep[field * n + interface_nodes] = 0.0
    mask = sp.diags_array(keep)

    def merge(base: sp.csr_array, rows, cols, vals) -> sp.csr_array:
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
        interface_nodes,
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
