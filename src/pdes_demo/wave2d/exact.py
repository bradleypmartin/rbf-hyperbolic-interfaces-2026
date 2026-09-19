"""References for the flat-interface plane P-wave problem (dissertation §3.4.1).

With flat interfaces and a plane pulse that depends on y only, the 2-D
elastic system reduces to the 1-D two-way wave equation in y for the pair
(v, h): ``rho v_t = h_y``, ``h_t = (lam + 2 mu) v_y``, while ``u = g = 0``
and ``f_t = lam v_y`` follows h. The 1-D solvers of :mod:`pdes_demo.wave1d`
handle exactly that after a change of variables:

    x = 1 - 2 y      (maps y = 0.75 to x = -0.5 and -y travel to +x travel)
    c' = 2 c_p       (dx = -2 dy)
    rho' = rho / 2   (so that rho' u_t = f_x reproduces rho v_t = h_y)

Impedances are unchanged (``rho' c' = rho c_p``), hence so are the
reflection and transmission coefficients of eq. 49. Field mapping:
``h = Z_p f_1D``, ``v = -Z_p u_1D`` (the mirror flips velocity), with the
background's ``Z_p`` giving unit amplitude in v.

:func:`exact_plane_wave` is the ray sum for jump interfaces, with
``f = lam / (lam + 2 mu) h`` on each side; that holds because ``f_t`` and
``h_t`` share ``v_y`` and both f and h start at zero inside the layer (the
ray sum enforces negligible tails there). :func:`spectral_plane_wave` is
the Fourier pseudo-spectral solver for smooth edges (``edge_width > 0``,
issue #36) on the pointwise image of the 2-D medium, fed the exact image of
the 2-D initial state, with f recovered from ``f_t = lam v_y`` exactly.
"""

import numpy as np

from ..wave1d.domain import Grid1D, LayeredMedium, Material, gaussian, periodic_grid
from ..wave1d.exact import exact_solution as exact_1d
from ..wave1d.spectral import interpolate, reference_size, run_spectral
from .domain import FIELDS, LayeredMedium2D, NodeSet, require_background_start


def _points(nodes: NodeSet | np.ndarray) -> np.ndarray:
    return nodes.xy if isinstance(nodes, NodeSet) else np.asarray(nodes, dtype=float)


def _as_1d_medium(medium: LayeredMedium2D) -> LayeredMedium:
    """The jump medium's image as a :class:`LayeredMedium`, for the ray sum."""
    if not medium.is_flat:
        raise ValueError("exact solution needs flat interfaces")
    if medium.is_smooth:
        raise ValueError(
            "the ray sum needs jump interfaces; this medium has smooth edges "
            f"(edge_width = {medium.edge_width:g}), use spectral_plane_wave"
        )
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


class _MappedMedium1D:
    """The 1-D image of a flat 2-D medium under ``x = 1 - 2 y``, pointwise.

    ``rho' = rho / 2`` and ``K' = 2 (lam + 2 mu)`` at every point, so
    ``c' = 2 c_p`` with the local P speed, and an edge of width delta in y is
    one of width 2 delta in x. Built pointwise rather than as a
    :class:`~pdes_demo.wave1d.domain.LayeredMedium` because that class blends
    c and rho linearly across an edge while the 2-D medium blends lam, mu and
    rho, which makes K linear in the blend weight and c_p not; the two
    conventions agree only in the jump limit. Satisfies
    :class:`~pdes_demo.wave1d.domain.Medium1D`.
    """

    def __init__(self, medium: LayeredMedium2D) -> None:
        if not medium.is_flat:
            raise ValueError("the 1-D image needs flat interfaces")
        self._medium = medium

    def _coefficients(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = np.mod((1.0 - np.asarray(x, dtype=float)) / 2, 1.0)
        zeros = np.zeros_like(y)
        m = self._medium
        return m.lam_at(zeros, y), m.mu_at(zeros, y), m.rho_at(zeros, y)

    @property
    def c_max(self) -> float:
        return 2 * self._medium.c_max

    def rho_at(self, x: np.ndarray) -> np.ndarray:
        return self._coefficients(x)[2] / 2

    def c_at(self, x: np.ndarray) -> np.ndarray:
        lam, mu, rho = self._coefficients(x)
        return 2 * np.sqrt((lam + 2 * mu) / rho)

    def impedance_at(self, x: np.ndarray) -> np.ndarray:
        return self.rho_at(x) * self.c_at(x)


def exact_plane_wave(
    nodes: NodeSet | np.ndarray,
    t: float,
    medium: LayeredMedium2D,
    center: float = 0.75,
    sharpness: float = 23.0,
) -> np.ndarray:
    """State ``(5, n)`` at time ``t`` for the pulse of :func:`plane_p_wave`.

    ``nodes`` may be a :class:`NodeSet` or an ``(n, 2)`` array of points.
    Jump interfaces only; smooth edges go to :func:`spectral_plane_wave`.
    """
    xy = _points(nodes)
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


def spectral_plane_wave_1d(
    medium: LayeredMedium2D,
    t: float,
    center: float = 0.75,
    sharpness: float = 23.0,
    *,
    n_ref: int | None = None,
    dt: float | None = None,
    cfl: float = 0.25,
) -> tuple[Grid1D, np.ndarray, np.ndarray]:
    """The 1-D image of the smooth-edge problem at time ``t``: ``(grid, u, f)``.

    The 1-D solver's own pulse scales velocity by the *local* impedance,
    while :func:`plane_p_wave` uses the background's everywhere, so the 2-D
    initial state is mapped explicitly (``u_1D = -v / Z_p``,
    ``f_1D = h / Z_p``): the reference solves the same initial-value problem
    as the 2-D runs whatever the overlap between the pulse's tail and the
    edge's. Drivers cache this triple and map it onto node sets with
    :func:`plane_wave_from_1d`.

    Resolution: ``reference_size(2 * edge_width)`` 1-D nodes unless ``n_ref``
    is given (2048 at delta = 0.005); ``dt`` and ``cfl`` go to
    :func:`~pdes_demo.wave1d.spectral.run_spectral`. The default CFL gives
    dt = 5e-5 at 2048 nodes, where doubling the grid or halving the step
    changes the result by under 3e-10 at t = 1 for the default pulse; pass
    ``dt`` to pin the step when comparing grid sizes. Cost is the 1-D run:
    about 4 s at 2048 nodes to t = 1, 14 s at 8192 nodes to t = 0.3.
    """
    if not medium.is_smooth:
        raise ValueError("a jump has the exact ray sum; use exact_plane_wave")
    # The mapping below is plane_p_wave's background-material pulse; unlike
    # the ray sum there is no tail check because the reference solves whatever
    # initial state it is given, and that state is mapped exactly.
    require_background_start(medium, center)
    mapped = _MappedMedium1D(medium)
    n = reference_size(2 * medium.edge_width) if n_ref is None else n_ref
    grid = periodic_grid(n)
    z = medium.background.p_impedance
    # plane_p_wave on the 1-D grid: v0 = gaussian, h0 = Z_p v0, so u_1D = -v0 / Z_p
    # and f_1D = h0 / Z_p = v0.
    v0 = gaussian(grid.x, 1 - 2 * center, sharpness**2 / 4)
    _, snaps = run_spectral(
        mapped, n, t_end=t, cfl=cfl, dt=dt, n_snapshots=1, initial=(-v0 / z, v0)
    )
    return grid, snaps.u[-1], snaps.f[-1]


def plane_wave_from_1d(
    nodes: NodeSet | np.ndarray,
    medium: LayeredMedium2D,
    grid: Grid1D,
    u1: np.ndarray,
    f1: np.ndarray,
    center: float = 0.75,
    sharpness: float = 23.0,
) -> np.ndarray:
    """State ``(5, n)`` on the nodes from a 1-D snapshot of
    :func:`spectral_plane_wave_1d` (trigonometric interpolation to ``x = 1 - 2y``).

    f follows from ``f_t = lam v_y`` and ``h_t = (lam + 2 mu) v_y`` sharing
    ``v_y``: ``f = f_0 + lam / (lam + 2 mu) (h - h_0)`` pointwise in y,
    exact for any overlap. The jump case's shortcut ``f = lam / (lam + 2 mu) h``
    is off by ``|h_0| |ratio - ratio_bg|``, frozen in time: nothing for the
    default materials (lam = mu on both sides, so the ratio is 1/3
    everywhere), and for a band with a different ratio it grows with the
    overlap of the tanh tails (about 19 delta long) and the pulse's: 2e-14
    at delta = 0.01, 1e-9 at 0.02 and 7e-6 at 0.04 for the default pulse
    against the edge at y = 0.5 (``tests/test_wave2d_smooth_edges.py``).
    """
    xy = _points(nodes)
    x = 1 - 2 * xy[:, 1]
    z = medium.background.p_impedance
    h = z * interpolate(f1, grid, x)
    h0 = z * gaussian(x, 1 - 2 * center, sharpness**2 / 4)
    lam = medium.lam_at(xy[:, 0], xy[:, 1])
    mu = medium.mu_at(xy[:, 0], xy[:, 1])
    bg = medium.background
    state = np.zeros((len(FIELDS), xy.shape[0]))
    state[1] = -z * interpolate(u1, grid, x)
    state[4] = h
    state[2] = bg.lam / (bg.lam + 2 * bg.mu) * h0 + lam / (lam + 2 * mu) * (h - h0)
    return state


def spectral_plane_wave(
    nodes: NodeSet | np.ndarray,
    t: float,
    medium: LayeredMedium2D,
    center: float = 0.75,
    sharpness: float = 23.0,
    *,
    n_ref: int | None = None,
    dt: float | None = None,
    cfl: float = 0.25,
) -> np.ndarray:
    """State ``(5, n)`` at time ``t`` for the pulse of :func:`plane_p_wave`
    through smooth flat edges: :func:`spectral_plane_wave_1d` mapped onto the
    nodes by :func:`plane_wave_from_1d`. See both for the details.
    """
    grid, u1, f1 = spectral_plane_wave_1d(
        medium, t, center, sharpness, n_ref=n_ref, dt=dt, cfl=cfl
    )
    return plane_wave_from_1d(nodes, medium, grid, u1, f1, center, sharpness)
