"""Spatial differentiation matrices for the 1-D wave equation.

Two flavours with identical sparsity:

- ``mode="naive"``: Fornberg finite-difference weights everywhere, straight
  across the material interfaces. This is "standard FD4" in dissertation
  §2.2 and produces the non-physical ringing the demo is about.
- ``mode="aware"``: stencils that straddle an interface are rebuilt from
  piecewise polynomials that satisfy the PDE's interface conditions
  (dissertation §2.1; ``FD4wave1DAC.m``). A stencil that straddles *both*
  edges of a thin layer chains the construction across both interfaces.
  For a medium with smooth edges (``edge_width > 0``) the same mode uses
  the ODE-continued seeds of :mod:`.stiff` in every stencil whose nodes see
  a varying material (issue #27).

The construction, following the dissertation:

1. Expand u and f on each side of an interface as polynomials of degree
   ``order`` about the interface.
2. Continuity of u and f, and of all their time derivatives, must hold at
   the interface. Time derivatives are traded for space derivatives through
   powers of the PDE operator ``D`` (eq. 2-3), which gives a square
   "continuity matrix" ``C`` per material such that
   ``C_left @ coeffs_left == C_right @ coeffs_right`` (eq. 22).
3. Take the standard monomial basis on the leftmost piece and translate it
   across each interface with ``C_right^-1 @ C_left`` (eq. 23-24), shifting
   the expansion point between interfaces when there is more than one.
4. Stencil weights are the ones that differentiate every basis function
   exactly at the evaluation point (a small dense solve per stencil).

The operators return ``(Du, Df)``: ``Du @ u`` approximates ``u_x`` and
``Df @ f`` approximates ``f_x``. They differ only in interface-straddling
rows, because u and f obey different continuity conditions.
"""

from math import comb
from typing import Literal

import numpy as np
import scipy.sparse as sp

from ..fd_weights import fornberg_weights
from .domain import PERIOD, Grid1D, Interface, LayeredMedium, Material
from .stiff import stiff_weights

Mode = Literal["naive", "aware"]


def build_operators(
    grid: Grid1D,
    medium: LayeredMedium,
    order: int = 4,
    mode: Mode = "aware",
) -> tuple[sp.csr_array, sp.csr_array]:
    if order % 2:
        raise ValueError("order must be even (centred stencils)")
    if mode not in ("naive", "aware"):
        raise ValueError(f"unknown mode {mode!r}")
    half = order // 2
    p = order + 1
    n = grid.n
    if n < p:
        raise ValueError("grid too coarse for the requested order")

    rows = np.repeat(np.arange(n), p)
    cols = np.empty(n * p, dtype=int)
    wu = np.empty(n * p)
    wf = np.empty(n * p)

    for i in range(n):
        idx, xs = _stencil(grid, i, half)
        xe = grid.x[i]
        if mode == "naive":
            w_u = w_f = fornberg_weights(xe, xs, 1)[1]
        elif medium.is_smooth:
            if medium.varies_over(xs):
                w_u, w_f = stiff_weights(xs, xe, medium, order)
            else:
                w_u = w_f = fornberg_weights(xe, xs, 1)[1]
        else:
            crossed = _crossed_interfaces(xs, medium)
            if crossed:
                w_u, w_f = interface_weights(xs, xe, crossed, order)
            else:
                w_u = w_f = fornberg_weights(xe, xs, 1)[1]

        sl = slice(i * p, (i + 1) * p)
        cols[sl] = idx
        wu[sl] = w_u
        wf[sl] = w_f

    du = sp.csr_array((wu, (rows, cols)), shape=(n, n))
    df = sp.csr_array((wf, (rows, cols)), shape=(n, n))
    return du, df


def _stencil(grid: Grid1D, i: int, half: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and unwrapped coordinates of the centred stencil at node ``i``."""
    idx = np.arange(i - half, i + half + 1) % grid.n
    xe = grid.x[i]
    xs = grid.x[idx]
    # Unwrap periodic neighbours so stencil coordinates are contiguous.
    xs = xs + PERIOD * ((xe - xs) > PERIOD / 2) - PERIOD * ((xe - xs) < -PERIOD / 2)
    return idx, xs


def _crossed_interfaces(xs: np.ndarray, medium: LayeredMedium) -> list[Interface]:
    """Interfaces with a stencil node strictly left and a node at-or-right."""
    return [ifc for ifc in medium.interfaces if xs[0] < ifc.x <= xs[-1]]


def aware_rows(grid: Grid1D, medium: LayeredMedium, order: int = 4) -> np.ndarray:
    """Boolean mask of the rows ``mode="aware"`` rebuilds for this medium.

    Jump edges: stencils crossing an interface. Smooth edges: stencils whose
    nodes see different materials, which reaches about ``19 * edge_width``
    from each edge centre before the tanh tails round away.
    """
    half = order // 2
    rows = np.zeros(grid.n, dtype=bool)
    for i in range(grid.n):
        _, xs = _stencil(grid, i, half)
        if medium.is_smooth:
            rows[i] = medium.varies_over(xs)
        else:
            rows[i] = bool(_crossed_interfaces(xs, medium))
    return rows


def stencil_crossings(
    grid: Grid1D, medium: LayeredMedium, order: int = 4
) -> np.ndarray:
    """Number of interfaces (0, 1 or 2) each node's stencil straddles.

    Diagnostic: a count of 2 means the thin-layer "double-cross" path is in
    use for that node.
    """
    half = order // 2
    counts = np.empty(grid.n, dtype=int)
    for i in range(grid.n):
        _, xs = _stencil(grid, i, half)
        counts[i] = len(_crossed_interfaces(xs, medium))
    return counts


# --- interface-aware stencils -------------------------------------------------


def _multiplication_matrix(taylor: np.ndarray) -> np.ndarray:
    """Matrix of "multiply a degree<p polynomial by g" on monomial coefficients.

    Lower-triangular Toeplitz built from the Taylor coefficients of g
    (dissertation eq. 11-12); products are truncated at degree p-1.
    """
    p = taylor.size
    m = np.zeros((p, p))
    for i in range(p):
        m[i, : i + 1] = taylor[i::-1]
    return m


def _derivative_matrix(p: int) -> np.ndarray:
    """d/dx on monomial coefficients (dissertation eq. 9)."""
    d = np.zeros((p, p))
    for k in range(1, p):
        d[k - 1, k] = k
    return d


def _pde_operator(material: Material, p: int, scale: float) -> np.ndarray:
    """Block operator D acting on stacked (u, f) coefficient vectors (eq. 7-8).

    ``scale`` is the coordinate scaling x' = scale * x applied to the stencil
    for conditioning; Taylor coefficients of degree k pick up 1/scale**k.
    """
    inv_rho, rho_c2 = material.taylor(p)
    rescale = scale ** -np.arange(p)
    dx = _derivative_matrix(p)
    zero = np.zeros((p, p))
    upper = _multiplication_matrix(inv_rho * rescale) @ dx
    lower = _multiplication_matrix(rho_c2 * rescale) @ dx
    return np.block([[zero, upper], [lower, zero]])


def continuity_matrices(
    material: Material, p: int, scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Continuity matrices (C_u, C_f) for one side of an interface (eq. 17-22).

    Row k collects the constant term of the k-th time derivative of the data
    field, expressed through spatial derivatives via D**k. Even k relate a
    field to itself, odd k route through the other field.
    """
    d = _pde_operator(material, p, scale)
    dk = np.eye(2 * p)
    cu = np.zeros((p, p))
    cf = np.zeros((p, p))
    for k in range(p):
        if k % 2 == 0:
            cu[k] = dk[0, :p]
            cf[k] = dk[p, p:]
        else:
            cu[k] = dk[p, :p]
            cf[k] = dk[0, p:]
        dk = dk @ d
    return cu, cf


def _shift_matrix(p: int, s: float) -> np.ndarray:
    """Re-expand a polynomial about a point ``s`` to the right of its origin."""
    m = np.zeros((p, p))
    for k in range(p):
        for j in range(k, p):
            m[k, j] = comb(j, k) * s ** (j - k)
    return m


def piecewise_bases(
    interfaces: list[Interface], p: int, scale: float
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """Polynomial bases for u and f on each piece cut out by ``interfaces``.

    Returns ``(bases_u, bases_f, origins)``. ``bases_u[k]`` is a ``p x p``
    matrix whose column i holds the monomial coefficients of basis function
    i on piece k, expanded about ``origins[k]`` (scaled coordinates). Piece 0
    lies left of the first interface, piece k between interfaces k-1 and k.
    """
    xi = [scale * ifc.x for ifc in interfaces]
    materials = [interfaces[0].left] + [ifc.right for ifc in interfaces]
    for a, b in zip(interfaces[:-1], interfaces[1:], strict=True):
        if a.right != b.left:
            raise ValueError("adjacent interfaces disagree on the shared material")

    origins = [xi[0]] + xi
    bases_u = [np.eye(p)]
    bases_f = [np.eye(p)]
    for k in range(1, len(materials)):
        cu_l, cf_l = continuity_matrices(materials[k - 1], p, scale)
        cu_r, cf_r = continuity_matrices(materials[k], p, scale)
        shift = _shift_matrix(p, origins[k] - origins[k - 1])
        bases_u.append(np.linalg.solve(cu_r, cu_l @ shift @ bases_u[k - 1]))
        bases_f.append(np.linalg.solve(cf_r, cf_l @ shift @ bases_f[k - 1]))
    return bases_u, bases_f, origins


def interface_weights(
    xs: np.ndarray, xe: float, interfaces: list[Interface], order: int
) -> tuple[np.ndarray, np.ndarray]:
    """First-derivative weights at ``xe`` for a stencil straddling interfaces.

    Every basis function must be differentiated exactly at ``xe``; with
    ``order + 1`` nodes and ``order + 1`` basis functions this is one square
    solve per field.
    """
    p = order + 1
    interfaces = sorted(interfaces, key=lambda ifc: ifc.x)
    scale = 1.0 / np.max(np.abs(xs - interfaces[0].x))
    bases_u, bases_f, origins = piecewise_bases(interfaces, p, scale)
    xi = np.array([scale * ifc.x for ifc in interfaces])

    def piece(x: float) -> int:
        return int(np.searchsorted(xi, x, side="right"))

    powers = np.arange(p)
    a_u = np.empty((p, p))
    a_f = np.empty((p, p))
    for j, xj in enumerate(scale * xs):
        k = piece(xj)
        vals = (xj - origins[k]) ** powers
        a_u[:, j] = bases_u[k].T @ vals
        a_f[:, j] = bases_f[k].T @ vals

    k = piece(scale * xe)
    dx = scale * xe - origins[k]
    dvals = np.zeros(p)
    dvals[1:] = powers[1:] * dx ** (powers[1:] - 1)
    b_u = bases_u[k].T @ dvals
    b_f = bases_f[k].T @ dvals

    w_u = np.linalg.solve(a_u, b_u) * scale
    w_f = np.linalg.solve(a_f, b_f) * scale
    return w_u, w_f
