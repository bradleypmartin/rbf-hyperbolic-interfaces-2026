"""Seed stencils for stiff smooth edges (issue demo#27, demo#31).

Between a jump (dissertation ch. 2, ``operators.interface_weights``) and a
smoothly varying material lies the "twilight zone": an edge of width
``edge_width`` that is technically smooth but far too steep for the grid.
Standard stencils sample the coefficients at the nodes and see a jump they
handle badly; the interface machinery has no jump to translate across.

Brad's construction (issue demo#27) keeps the equispaced grid and replaces the
monomials ``1, x, x**2, ...`` of a stencil that sees the edge by *seeds*:
functions that look like those monomials at the evaluation point and are
continued through the edge by ODEs that say what the PDE allows. They are
the ``t = 0`` profiles of solutions that are polynomial in time. With
``rho u_t = f_x``, ``f_t = K u_x`` (``K = rho c**2``) each field has its own
second-order operator,

    L_u = (1/rho) d/dx K d/dx,        L_f = K d/dx (1/rho) d/dx,

and the seeds anchored at the evaluation point ``x_e`` satisfy

    phi_0 = 1
    K phi_1' = K(x_e)                                  (constant flux)
    L phi_k = k (k-1) c_e**2 phi_{k-2},  phi_k(x_e) = phi_k'(x_e) = 0.

Even ``k`` is "d^k u / dt^k = C"; odd ``k`` comes out as "d^k f / dt^k = C"
(the field swap of ``operators.continuity_matrices``). Constant coefficients
give ``phi_k = (x - x_e)**k`` exactly; a jump gives the dissertation's
translated Taylor basis. The chain is integrated in first-order form
``(phi, psi = K phi')`` for u and ``(phi, psi = phi' / rho)`` for f, so the
coefficients are only ever evaluated, never differentiated, and in the
stencil coordinate ``xi = (x - x_e) / h_s`` so that ``phi_k ~ xi**k`` and
the weight solve is as well conditioned as Fornberg's. The interpolation
matrix is nonsingular for any distinct nodes: ``span(phi_0..phi_k)`` is the
kernel of an operator in Pólya form, an extended complete Chebyshev system
(``docs/stiff-features.md``).
"""

import numpy as np
from scipy.integrate import solve_ivp

from .domain import PERIOD, LayeredMedium

# Tail of the tanh at which the ODE march gets a fresh start: a segment
# boundary here means the adaptive step never has to discover the edge in
# the middle of a long step across flat material.
_EDGE_REACH = 10.0


def seed_basis(
    xs: np.ndarray,
    xe: float,
    medium: LayeredMedium,
    p: int,
    rtol: float = 1e-13,
    atol: float = 1e-15,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Seed values at the stencil nodes for both fields.

    Returns ``(a_u, a_f, h_s)`` with ``a_u[k, j] = phi_k(x_j)`` for the
    u-field seeds in the stencil coordinate ``(x - x_e) / h_s``, ``a_f``
    likewise for f. In that coordinate ``phi_k'(0) = delta_{k1}`` for both.
    """
    xs = np.asarray(xs, dtype=float)
    h_s = float(np.max(np.abs(xs - xe)))
    xi = (xs - xe) / h_s
    k_e = float(medium.rho_at(xe) * medium.c_at(xe) ** 2)
    rho_e = float(medium.rho_at(xe))
    c2_e = k_e / rho_e
    kk = np.arange(p)
    coef = kk * (kk - 1) * c2_e  # k (k-1) c_e**2, zero for k < 2

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        x = xe + h_s * t
        rho = float(medium.rho_at(x))
        k = rho * float(medium.c_at(x)) ** 2
        phi_u, psi_u, phi_f, psi_f = np.split(y, 4)
        lower_u = np.concatenate(([0.0, 0.0], phi_u[:-2]))
        lower_f = np.concatenate(([0.0, 0.0], phi_f[:-2]))
        return np.concatenate(
            (psi_u / k, rho * coef * lower_u, rho * psi_f, coef * lower_f / k)
        )

    y0 = np.zeros(4 * p)
    y0[0] = 1.0  # phi_0 = 1 for u
    y0[p + 1] = k_e  # psi_1 = K phi_1' = K(x_e) -> phi_1'(0) = 1
    y0[2 * p] = 1.0  # phi_0 = 1 for f
    y0[3 * p + 1] = 1.0 / rho_e  # psi_1 = phi_1' / rho -> phi_1'(0) = 1

    stops = _edge_stops(xs, xe, medium) / h_s
    ys = np.empty((4 * p, xi.size))
    for side in (-1, 1):
        sel = np.flatnonzero(np.sign(xi) == side)
        targets = np.concatenate((xi[sel], stops[np.sign(stops) == side]))
        targets = np.unique(targets)[::side]  # away from 0 first
        y = y0
        t0 = 0.0
        for t1 in targets:
            sol = solve_ivp(rhs, (t0, t1), y, method="DOP853", rtol=rtol, atol=atol)
            if not sol.success:
                raise RuntimeError(f"seed ODE failed at xi = {t1}: {sol.message}")
            y = sol.y[:, -1]
            t0 = t1
            for j in sel[np.isclose(xi[sel], t1, rtol=0, atol=0)]:
                ys[:, j] = y
    ys[:, xi == 0] = y0[:, None]
    a_u = ys[:p]
    a_f = ys[2 * p : 3 * p]
    return a_u, a_f, h_s


def _edge_stops(xs: np.ndarray, xe: float, medium: LayeredMedium) -> np.ndarray:
    """Edge centres and their ``+-_EDGE_REACH * edge_width`` flanks inside the
    stencil's span, as offsets from ``xe`` (periodic images included)."""
    lo, hi = xs.min(), xs.max()
    reach = _EDGE_REACH * medium.edge_width
    stops = []
    for ifc in medium.interfaces:
        for m in (-1, 0, 1):
            centre = ifc.x + m * PERIOD
            for s in (centre - reach, centre, centre + reach):
                if lo < s < hi:
                    stops.append(s - xe)
    return np.array(stops)


def stiff_weights(
    xs: np.ndarray, xe: float, medium: LayeredMedium, order: int
) -> tuple[np.ndarray, np.ndarray]:
    """First-derivative weights at ``xe`` from ODE-continued seeds.

    Same contract as :func:`~.operators.interface_weights`: every seed is
    differentiated exactly at ``xe``, one square solve per field. Only
    ``phi_1`` has a non-zero derivative there, so the right-hand side is the
    unit vector ``e_1``, and the scaling by ``h_s`` undoes the stencil
    coordinate.
    """
    p = order + 1
    a_u, a_f, h_s = seed_basis(xs, xe, medium, p)
    b = np.zeros(p)
    b[1] = 1.0
    w_u = np.linalg.solve(a_u, b) / h_s
    w_f = np.linalg.solve(a_f, b) / h_s
    return w_u, w_f
