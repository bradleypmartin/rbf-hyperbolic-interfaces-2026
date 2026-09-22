"""Fourier pseudo-spectral reference solution for smooth-edged layers (#30).

The ray-sum exact solution only covers jump edges. With ``edge_width > 0``
the coefficients are smooth and periodic, so a Fourier pseudo-spectral
discretisation of

    rho u_t = f_x,        f_t = rho c**2 u_x

converges exponentially once the edge is resolved, and it shares nothing
with the finite-difference stencils under test. Derivatives are taken in
Fourier space, the variable coefficients are applied pointwise, and the
result is stepped with the same RK4 loop as the FD solvers. The Fourier
derivative is skew-symmetric, so the weighted energy
``sum(f**2 / K + rho u**2)`` is conserved exactly by the semi-discretisation.

Resolution: on the period-2 interval ``tanh(x / d)`` has poles at
``x = i pi d / 2``, so its Fourier coefficients decay like
``exp(-pi**2 d m / 2)`` in the mode number ``m`` and about ``33 / (pi**2 d)``
modes reach 1e-14. :func:`reference_size` turns that into a grid size.
Time accuracy is RK4's; the default CFL keeps it well below the spatial
error for the pulses used here (see ``tests/test_wave1d_spectral.py``).
"""

import math

import numpy as np
import scipy.sparse.linalg as spla

from .domain import Grid1D, Medium1D, periodic_grid, right_going_pulse
from .simulate import Snapshots, rk4_wave


def spectral_derivative(v: np.ndarray) -> np.ndarray:
    """d/dx of periodic samples on the interval of length 2 (any node offset)."""
    n = v.size
    vh = np.fft.rfft(v)
    k = np.pi * np.arange(vh.size)
    if n % 2 == 0:
        # The Nyquist mode is a pure cosine on the nodes; its derivative
        # samples to zero there.
        k[-1] = 0.0
    return np.fft.irfft(1j * k * vh, n)


def reference_size(edge_width: float, tol: float = 1e-14, n_min: int = 1024) -> int:
    """Power-of-two grid size that resolves a tanh edge of ``edge_width``.

    Twice the mode count at which the edge's Fourier coefficients fall
    below ``tol``, so the grid carries every mode down to that level.
    """
    if edge_width <= 0:
        raise ValueError("a jump has no spectral resolution; use exact_solution")
    modes = 2 * math.log(1 / tol) / (math.pi**2 * edge_width)
    n = max(n_min, 2 * modes)
    return 1 << math.ceil(math.log2(n))


def run_spectral(
    medium: Medium1D,
    n: int,
    *,
    t_end: float = 1.0,
    cfl: float = 0.25,
    dt: float | None = None,
    n_snapshots: int | None = 1,
    pulse_center: float = -0.5,
    pulse_sharpness: float = 600.0,
    initial: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[Grid1D, Snapshots]:
    """Pseudo-spectral run of the layer problem on ``n`` nodes to ``t_end``.

    ``cfl`` is relative to ``h / c_max`` like :func:`~.simulate.stable_dt`;
    Fourier + RK4 is stable below ``cfl * pi < 2.8``. Spatial error is at
    round-off once the edge is resolved, so the time step sets the accuracy:
    for the default pulse the RK4 error is 2e-6 at ``dt = 6.5e-4`` and falls
    as ``dt**4``, so ``dt = 5e-5`` reaches about 1e-10. Pass ``dt`` for a
    reference of known accuracy; it is capped at the CFL step.

    ``initial`` replaces the right-going pulse by samples ``(u0, f0)`` on the
    grid's nodes; :func:`pdes_demo.wave2d.exact.spectral_plane_wave` uses it
    to hand over the exact image of the 2-D initial state.
    """
    grid = periodic_grid(n)
    d = spla.LinearOperator((n, n), matvec=spectral_derivative, dtype=float)
    dt_target = cfl * grid.h / medium.c_max
    if dt is not None:
        dt_target = min(dt, dt_target)
    n_steps = max(1, math.ceil(t_end / dt_target))
    dt = t_end / n_steps
    store_every = 1 if n_snapshots is None else max(1, n_steps // n_snapshots)
    if initial is None:
        u0, f0 = right_going_pulse(grid.x, medium, pulse_center, pulse_sharpness)
    else:
        u0, f0 = (np.asarray(a, dtype=float) for a in initial)
        if u0.shape != (n,) or f0.shape != (n,):
            raise ValueError(f"initial state must be two arrays of length {n}")
    snaps = rk4_wave(
        u0,
        f0,
        d,
        d,
        medium.rho_at(grid.x),
        medium.c_at(grid.x),
        dt,
        n_steps,
        store_every,
    )
    return grid, snaps


def interpolate(
    values: np.ndarray, grid: Grid1D, x_target: np.ndarray, chunk: int = 256
) -> np.ndarray:
    """Trigonometric interpolation from ``grid`` to arbitrary points.

    Cell-centred grids of different sizes share no nodes (the ratio of
    spacings is even), so the reference is evaluated at the coarse nodes
    through its Fourier series. Chunked so the mode-by-point matrix stays
    small. For even ``n`` the Nyquist mode is taken as a cosine, which
    makes the interpolant real and reproduces the samples.
    """
    n = values.size
    vh = np.fft.fft(values) / n
    m = np.fft.fftfreq(n, d=1.0 / n)
    theta = np.pi * (np.asarray(x_target, dtype=float) - grid.x[0])
    out = np.empty(theta.size)
    for start in range(0, theta.size, chunk):
        th = theta[start : start + chunk]
        basis = np.exp(1j * np.outer(th, m))
        if n % 2 == 0:
            basis[:, n // 2] = np.cos(th * m[n // 2])
        out[start : start + chunk] = (basis @ vh).real
    return out


def energy(u: np.ndarray, f: np.ndarray, grid: Grid1D, medium: Medium1D) -> float:
    """Discrete wave energy ``h/2 * sum(f**2 / K + rho u**2)``."""
    rho = medium.rho_at(grid.x)
    k = rho * medium.c_at(grid.x) ** 2
    return float(0.5 * grid.h * np.sum(f**2 / k + rho * u**2))
