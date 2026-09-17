"""Exact solution of the flat-interface plane P-wave problem (dissertation §3.4.1).

With flat interfaces and a plane pulse that depends on y only, the 2-D
elastic system reduces to the 1-D two-way wave equation in y for the pair
(v, h): ``rho v_t = h_y``, ``h_t = (lam + 2 mu) v_y``, while ``u = g = 0``
and ``f = lam / (lam + 2 mu) h`` (``f_t = lam v_y`` in step with ``h_t``).
The ray-sum solver of :mod:`pdes_demo.wave1d.exact` handles exactly that
after a change of variables:

    x = 1 - 2 y      (maps y = 0.75 to x = -0.5 and -y travel to +x travel)
    c' = 2 c_p       (dx = -2 dy)
    rho' = rho / 2   (so that rho' u_t = f_x reproduces rho v_t = h_y)

Impedances are unchanged (``rho' c' = rho c_p``), hence so are the
reflection and transmission coefficients of eq. 49. Field mapping:
``h = Z_p f_1D``, ``v = -Z_p u_1D`` (the mirror flips velocity), with the
factor ``Z_p`` giving unit amplitude in v. ``f = lam / (lam + 2 mu) h`` holds
with the local ratio on each side because ``f_t = lam v_y`` and
``h_t = (lam + 2 mu) v_y`` share ``v_y``, and both f and h start at zero
inside the layer (the 1-D solver enforces negligible tails there).
"""

import numpy as np

from ..wave1d.domain import LayeredMedium, Material
from ..wave1d.exact import exact_solution as exact_1d
from .domain import FIELDS, LayeredMedium2D, NodeSet


def _as_1d_medium(medium: LayeredMedium2D) -> LayeredMedium:
    if not medium.is_flat:
        raise ValueError("exact solution needs flat interfaces")
    bg, ly = medium.background, medium.layer
    y1, y2 = medium.lower.y0, medium.upper.y0
    # The order-reversing map turns y in [y1, y2) into x in (1 - 2 y2, 1 - 2 y1];
    # the 1-D medium closes the other end. Only the two boundary points differ.
    return LayeredMedium(
        background=Material(c=2 * bg.c_p, rho=bg.rho / 2),
        layer=Material(c=2 * ly.c_p, rho=ly.rho / 2),
        layer_start=1 - 2 * y2,
        layer_width=2 * (y2 - y1),
    )


def exact_plane_wave(
    nodes: NodeSet | np.ndarray,
    t: float,
    medium: LayeredMedium2D,
    center: float = 0.75,
    sharpness: float = 23.0,
) -> np.ndarray:
    """State ``(5, n)`` at time ``t`` for the pulse of :func:`plane_p_wave`.

    ``nodes`` may be a :class:`NodeSet` or an ``(n, 2)`` array of points.
    """
    xy = nodes.xy if isinstance(nodes, NodeSet) else np.asarray(nodes, dtype=float)
    medium_1d = _as_1d_medium(medium)
    x = 1 - 2 * xy[:, 1]
    # exp(-s^2 (y - c)^2) = exp(-(s^2 / 4) (x - x_c)^2) in the mapped coordinate.
    try:
        u1, f1 = exact_1d(
            x, t, medium_1d, center=1 - 2 * center, sharpness=sharpness**2 / 4
        )
    except ValueError as err:
        raise ValueError(
            "plane pulse must start in the background with negligible tails at "
            f"both interfaces ({err})"
        ) from err
    z = medium.background.p_impedance
    lam = medium.lam_at(xy[:, 0], xy[:, 1])
    mu = medium.mu_at(xy[:, 0], xy[:, 1])
    state = np.zeros((len(FIELDS), xy.shape[0]))
    state[1] = -z * u1
    state[4] = z * f1
    state[2] = lam / (lam + 2 * mu) * state[4]
    return state
