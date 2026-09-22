"""Fourier pseudo-spectral references for smooth media: Fourier in x on a
flat medium (issue #41), and on a product grid for a curved one (#42).

The references of :mod:`.exact` need a state that depends on y only. An
oblique plane wave does not, but on a flat medium the coefficients still
do, so the elastic system of eq. 32 separates in x: with

    s(x, y, t) = sum_k c_k(y, t) exp(2 pi i k x)

each x-mode ``k`` obeys the 1-D system in y with ``d/dx -> 2 pi i k``, the
material multiplies pointwise in y without mixing modes, and the modes
never talk to each other. So the reference is one complex-coefficient
Fourier pseudo-spectral solve in y per mode, all modes marched together as
one array with the RK4 loop of :mod:`rbf_hyperbolic_interfaces.wave1d.spectral`: y
derivatives by FFT along y, x derivatives by the mode number. It shares
nothing with the RBF-FD stencils under test and converges exponentially in
y once the edge is resolved
(:func:`~rbf_hyperbolic_interfaces.wave1d.spectral.reference_size` sets the grid for a
tanh edge). Which modes are carried is read off the initial data: those whose x-Fourier
coefficients on an ``n_x`` grid are above ``tol`` of the largest, 13 for the (1, 2)
train of :func:`~.domain.oblique_p_wave` at sharpness 15 and one for the normal
pulse. The Fourier derivative is skew-adjoint, so the elastic energy of
:func:`ModeState.energy` is conserved by the semi-discretisation and the
time step sets the accuracy, as in 1-D.

Cost: the state is ``(5, n_modes, n_y)`` complex and a step is four FFT
pairs on it; 13 modes at ``n_y = 4096`` (delta = 0.0025) take about 20 ms
per step, 20,000 steps to t = 1 at the default step.

On a curved medium the coefficients depend on x as well and the x-modes
mix, so :func:`run_fourier_2d` keeps the real fields on the full
``n_x x n_y`` product grid, differentiates by FFT along each axis and
multiplies by the material in physical space (``x`` at ``j / n_x`` and
``y`` cell-centred as above, so the two solvers sample the same points
and agree to rounding on a flat medium, `tests/test_wave2d_curved_edges.py`).
:class:`GridState` evaluates the fields and their derivatives anywhere
by trigonometric interpolation in both directions, which costs
``n_x n_y`` per point and field (about 10 s for 19,600 nodes at
512 x 1024). The x-grid only has to resolve the material's variation
along x, which for a tilt of at most 8 degrees is 7x gentler than along
y, and the scattered waves' x-content, so ``n_x`` can be a fraction of
``n_y``; the driver checks that by self-convergence.
"""

import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
from scipy import fft as sfft

from ..wave1d.spectral import reference_size
from .domain import FIELDS, LayeredMedium2D

InitialState = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ModeState:
    """The five fields at time ``t`` as x-Fourier modes on a y grid.

    ``coef`` ``(5, n_modes, n_y)`` complex holds ``c_k(y_i)`` for the
    non-negative mode numbers ``modes`` (real fields: the negative modes are
    the conjugates, so mode 0 counts once and every other mode twice). ``y``
    is the cell-centred grid ``(i + 1/2) / n_y``.
    """

    modes: np.ndarray
    y: np.ndarray
    t: float
    coef: np.ndarray

    @property
    def n_y(self) -> int:
        return self.y.size

    @property
    def weights(self) -> np.ndarray:
        """1 for mode 0, 2 for every other mode: the real field's mode sum."""
        return np.where(self.modes == 0, 1.0, 2.0)

    def evaluate(
        self, xy: np.ndarray, *, dx: int = 0, dy: int = 0, chunk: int = 256
    ) -> np.ndarray:
        """The fields ``(5, n)`` at the points ``xy`` ``(n, 2)`` by trigonometric
        interpolation in both directions, exact for the mode sum on the
        grid; ``dx``, ``dy`` differentiate that many times first. The
        Nyquist mode in y is taken as a cosine (real interpolant through
        the samples) and its derivative as zero, as in the solver.
        """
        xy = np.asarray(xy, dtype=float)
        n_y = self.n_y
        hat = np.fft.fft(self.coef, axis=-1) / n_y  # (5, K, n_y)
        ell = np.fft.fftfreq(n_y, d=1.0 / n_y)
        k_y = 2 * np.pi * ell
        if n_y % 2 == 0:
            k_y[n_y // 2] = 0.0
        hat = hat * (1j * k_y) ** dy
        k_x = 2 * np.pi * self.modes
        w = self.weights * (1j * k_x) ** dx  # (K,)
        out = np.empty((len(FIELDS), xy.shape[0]))
        for start in range(0, xy.shape[0], chunk):
            pts = xy[start : start + chunk]
            th = 2 * np.pi * (pts[:, 1] - self.y[0])
            basis = np.exp(1j * np.outer(th, ell))  # (c, n_y)
            if n_y % 2 == 0:
                basis[:, n_y // 2] = np.cos(th * ell[n_y // 2])
            phase = np.exp(1j * np.outer(pts[:, 0], k_x)) * w  # (c, K)
            for field in range(len(FIELDS)):
                values = basis @ hat[field].T  # (c, K): c_k at the points
                out[field, start : start + chunk] = np.sum(values * phase, axis=1).real
        return out

    def curl(self, xy: np.ndarray) -> np.ndarray:
        """``u_y - v_x`` at the points: zero for a P wave, so it maps the
        S waves alone."""
        return self.evaluate(xy, dy=1)[0] - self.evaluate(xy, dx=1)[1]

    def energy(self, medium: LayeredMedium2D) -> float:
        """Elastic energy of the real field, kinetic plus strain, integrated
        over the square: Parseval in x, the trapezoidal rule (exact for
        trigonometric polynomials) in y. Same density as
        :func:`~.simulate.energy`."""
        lam, mu, rho = medium.material_at(np.zeros_like(self.y), self.y)
        u, v, f, g, h = (np.abs(c) ** 2 for c in self.coef)
        fh = (self.coef[2] * np.conj(self.coef[4])).real
        d = 4 * mu * (lam + mu)
        kinetic = 0.5 * rho * (u + v)
        strain = 0.5 * (((lam + 2 * mu) * (f + h) - 2 * lam * fh) / d + g / mu)
        per_mode = np.sum(kinetic + strain, axis=-1) / self.n_y  # (K,)
        return float(np.sum(self.weights * per_mode))


def _grid(n_x: int, n_y: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``x`` ``(n_x,)`` at ``j / n_x`` (plain DFT phases), ``y`` ``(n_y,)``
    cell-centred, and the ``(n_x n_y, 2)`` points in ``(x, y)`` order."""
    x = np.arange(n_x) / n_x
    y = (np.arange(n_y) + 0.5) / n_y
    gx, gy = np.meshgrid(x, y, indexing="ij")
    return x, y, np.stack([gx.ravel(), gy.ravel()], axis=-1)


def x_modes(
    initial: InitialState, n_x: int, n_y: int, tol: float = 1e-15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The x-Fourier modes of the initial state: ``(modes, y, coef)`` with
    ``coef`` ``(5, n_modes, n_y)``, keeping the non-negative modes whose
    largest coefficient over all fields and y exceeds ``tol`` times the
    overall largest (a mode absent at t = 0 stays absent).
    """
    _, y, pts = _grid(n_x, n_y)
    state = np.asarray(initial(pts), dtype=float).reshape(len(FIELDS), n_x, n_y)
    hat = np.fft.rfft(state, axis=1) / n_x  # (5, n_x // 2 + 1, n_y)
    size = np.abs(hat).max(axis=(0, 2))
    keep = np.flatnonzero(size > tol * size.max())
    return keep, y, np.ascontiguousarray(hat[:, keep, :])


def _step_count(t_end: float, dt_target: float, times: Sequence[float]) -> int:
    """Steps to ``t_end`` at a step no larger than ``dt_target`` such that
    every snapshot time is hit exactly (each ``t / t_end`` a fraction with
    a small denominator; the step count is a multiple of all of them)."""
    n = max(1, math.ceil(t_end / dt_target))
    lcm = 1
    for t in times:
        frac = Fraction(t / t_end).limit_denominator(10_000)
        if not 0 <= frac <= 1:
            raise ValueError(f"snapshot time {t} outside [0, {t_end}]")
        lcm = math.lcm(lcm, frac.denominator)
    return lcm * math.ceil(n / lcm)


def run_fourier(
    medium: LayeredMedium2D,
    initial: InitialState,
    t_end: float,
    *,
    snapshot_times: Sequence[float] | None = None,
    n_y: int | None = None,
    n_x: int = 64,
    dt: float | None = 5e-5,
    cfl: float = 0.5,
    tol: float = 1e-15,
) -> list[ModeState]:
    """Fourier-in-x reference for a flat medium from the initial state
    ``initial(xy) -> (5, n)``: one :class:`ModeState` per snapshot time
    (``t_end`` alone by default), integrated with RK4.

    ``n_y`` defaults to
    :func:`~rbf_hyperbolic_interfaces.wave1d.spectral.reference_size` of the edge (4096
    at delta = 0.0025, 1024 at 0.01); ``n_x`` only has to hold the initial data's
    x-modes. The step is ``cfl * h_y / c_max`` capped at ``dt``: Fourier + RK4 is stable
    to ``cfl * pi < 2.8`` and the RK4 error at ``dt = 5e-5`` is about 1e-10 for the
    pulses used here (``tests/test_wave2d_oblique.py`` halves the step and doubles the
    grid).
    """
    if not medium.is_flat:
        raise ValueError("the Fourier-in-x reference needs flat interfaces")
    if not medium.is_smooth:
        raise ValueError("a jump is not resolvable by Fourier; use the ray sum")
    if n_y is None:
        n_y = reference_size(2 * medium.edge_width)
    modes, y, state = x_modes(initial, n_x, n_y, tol)
    lam, mu, rho = medium.material_at(np.zeros_like(y), y)
    lam2mu = lam + 2 * mu
    inv_rho = 1.0 / rho
    k_x = (2j * np.pi * modes)[:, None]  # (K, 1)
    k_y = 2j * np.pi * np.fft.fftfreq(n_y, d=1.0 / n_y)
    if n_y % 2 == 0:
        k_y[n_y // 2] = 0.0
    dy_fields = [0, 1, 3, 4]  # u, v, g, h: what the y derivatives act on

    def rhs(s: np.ndarray) -> np.ndarray:
        u_y, v_y, g_y, h_y = np.fft.ifft(np.fft.fft(s[dy_fields], axis=-1) * k_y)
        u_x, v_x, f_x, g_x = (k_x * s[i] for i in (0, 1, 2, 3))
        return np.stack(
            [
                inv_rho * (f_x + g_y),
                inv_rho * (g_x + h_y),
                lam2mu * u_x + lam * v_y,
                mu * (u_y + v_x),
                lam * u_x + lam2mu * v_y,
            ]
        )

    times = list(snapshot_times) if snapshot_times is not None else [t_end]
    if t_end <= 0:
        return [ModeState(modes=modes, y=y, t=0.0, coef=state)]
    dt_target = cfl / (n_y * medium.c_max)
    if dt is not None:
        dt_target = min(dt, dt_target)
    n_steps = _step_count(t_end, dt_target, times)
    step = t_end / n_steps
    wanted = {round(t / t_end * n_steps): t for t in times}
    out = []
    if 0 in wanted:
        out.append(ModeState(modes=modes, y=y, t=0.0, coef=state.copy()))
    for i in range(1, n_steps + 1):
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * step * k1)
        k3 = rhs(state + 0.5 * step * k2)
        k4 = rhs(state + step * k3)
        state = state + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if i in wanted:
            out.append(ModeState(modes=modes, y=y, t=wanted[i], coef=state.copy()))
    return out


# --- the product grid (#42) -----------------------------------------------------


def _trig_basis(coords: np.ndarray, n: int, origin: float, order: int) -> np.ndarray:
    """``(len(coords), n)`` complex: ``(i k)^order exp(i k (c - origin))`` for
    the DFT wavenumbers ``k = 2 pi ell`` of an ``n``-grid, the Nyquist mode
    as a cosine with derivative zero (a real interpolant through the
    samples, as the solvers take it)."""
    ell = np.fft.fftfreq(n, d=1.0 / n)
    k = 2 * np.pi * ell
    th = 2 * np.pi * (np.asarray(coords, dtype=float) - origin)
    basis = np.exp(1j * np.outer(th, ell))
    if n % 2 == 0:
        k[n // 2] = 0.0
        basis[:, n // 2] = np.cos(th * ell[n // 2])
    return basis * (1j * k) ** order


@dataclass(frozen=True)
class GridState:
    """The five real fields at time ``t`` on the ``n_x x n_y`` product grid,
    ``x`` at ``j / n_x`` and ``y`` cell-centred ``(i + 1/2) / n_y`` (the
    conventions of :class:`ModeState`); ``fields`` is ``(5, n_x, n_y)``."""

    x: np.ndarray
    y: np.ndarray
    t: float
    fields: np.ndarray

    @property
    def n_x(self) -> int:
        return self.x.size

    @property
    def n_y(self) -> int:
        return self.y.size

    def evaluate(
        self, xy: np.ndarray, *, dx: int = 0, dy: int = 0, chunk: int = 512
    ) -> np.ndarray:
        """The fields ``(5, n)`` at the points ``xy`` ``(n, 2)`` by
        trigonometric interpolation in x and y, exact on the grid, ``dx``
        and ``dy`` derivatives first (Nyquist modes as in
        :meth:`ModeState.evaluate`)."""
        xy = np.asarray(xy, dtype=float)
        n_x, n_y = self.n_x, self.n_y
        hat = np.fft.fft2(self.fields, axes=(1, 2)) / (n_x * n_y)  # (5, n_x, n_y)
        out = np.empty((len(FIELDS), xy.shape[0]))
        for start in range(0, xy.shape[0], chunk):
            pts = xy[start : start + chunk]
            b_x = _trig_basis(pts[:, 0], n_x, self.x[0], dx)  # (c, n_x)
            b_y = _trig_basis(pts[:, 1], n_y, self.y[0], dy)  # (c, n_y)
            for field in range(len(FIELDS)):
                partial = b_y @ hat[field].T  # (c, n_x): summed over k_y
                out[field, start : start + chunk] = np.sum(b_x * partial, axis=1).real
        return out

    def curl(self, xy: np.ndarray) -> np.ndarray:
        """``u_y - v_x`` at the points: zero for a P wave, the S waves alone."""
        return self.evaluate(xy, dy=1)[0] - self.evaluate(xy, dx=1)[1]

    def energy(self, medium: LayeredMedium2D) -> float:
        """Elastic energy over the square, the trapezoidal rule on the grid
        (exact for trigonometric polynomials); the density of
        :func:`~.simulate.energy`."""
        gx, gy = np.meshgrid(self.x, self.y, indexing="ij")
        lam, mu, rho = medium.material_at(gx, gy)
        u, v, f, g, h = self.fields
        d = 4 * mu * (lam + mu)
        kinetic = 0.5 * rho * (u**2 + v**2)
        strain = 0.5 * (
            ((lam + 2 * mu) * (f**2 + h**2) - 2 * lam * f * h) / d + g**2 / mu
        )
        return float(np.mean(kinetic + strain))


def run_fourier_2d(
    medium: LayeredMedium2D,
    initial: InitialState,
    t_end: float,
    *,
    snapshot_times: Sequence[float] | None = None,
    n_x: int | None = None,
    n_y: int | None = None,
    cfl: float = 0.5,
    dt: float | None = None,
    workers: int | None = None,
    snapshot_transform: Callable[[GridState], Any] | None = None,
) -> list[Any]:
    """Product-grid pseudo-spectral reference for any smooth medium from the
    initial state ``initial(xy) -> (5, n)``: one :class:`GridState` per
    snapshot time (``t_end`` alone by default), RK4 in time. With
    ``snapshot_transform`` each snapshot is replaced by its image under it
    as soon as it is taken (a clip wants the fields at its nodes for
    hundreds of frames, not hundreds of 20 MB grids).

    ``n_y`` defaults to
    :func:`~rbf_hyperbolic_interfaces.wave1d.spectral.reference_size` of the edge and
    ``n_x`` to ``n_y / 2``. The step is ``cfl / (c_max sqrt(n_x^2 + n_y^2))``, capped at
    ``dt`` when given: the largest eigenvalue of the semi-discretisation is ``c_p |k|``
    with ``|k| < pi sqrt(n_x^2 + n_y^2)``, and RK4 holds the imaginary axis to
    2.83, so ``cfl < 0.9`` is stable and 0.5 leaves the RK4 error near
    ``1e-8`` for the pulses used here (the driver halves it once). FFTs
    run on ``workers`` threads (all cores by default).
    """
    if not medium.is_smooth:
        raise ValueError("a jump is not resolvable by Fourier; use the ray sum")
    if n_y is None:
        n_y = reference_size(2 * medium.edge_width)
    if n_x is None:
        n_x = max(64, n_y // 2)
    if workers is None:
        workers = os.cpu_count() or 1
    x, y, pts = _grid(n_x, n_y)
    state = np.ascontiguousarray(
        np.asarray(initial(pts), dtype=float).reshape(len(FIELDS), n_x, n_y)
    )
    gx, gy = np.meshgrid(x, y, indexing="ij")
    lam, mu, rho = medium.material_at(gx, gy)
    lam2mu = lam + 2 * mu
    inv_rho = 1.0 / rho
    k_x = 2j * np.pi * np.fft.rfftfreq(n_x, d=1.0 / n_x)
    k_y = 2j * np.pi * np.fft.rfftfreq(n_y, d=1.0 / n_y)
    if n_x % 2 == 0:
        k_x[-1] = 0.0
    if n_y % 2 == 0:
        k_y[-1] = 0.0
    k_x = k_x[:, None]

    def d_x(a: np.ndarray) -> np.ndarray:
        return sfft.irfft(
            sfft.rfft(a, axis=1, workers=workers) * k_x, n=n_x, axis=1, workers=workers
        )

    def d_y(a: np.ndarray) -> np.ndarray:
        return sfft.irfft(
            sfft.rfft(a, axis=2, workers=workers) * k_y, n=n_y, axis=2, workers=workers
        )

    def rhs(s: np.ndarray) -> np.ndarray:
        u_x, v_x, f_x, g_x = d_x(s[[0, 1, 2, 3]])
        u_y, v_y, g_y, h_y = d_y(s[[0, 1, 3, 4]])
        return np.stack(
            [
                inv_rho * (f_x + g_y),
                inv_rho * (g_x + h_y),
                lam2mu * u_x + lam * v_y,
                mu * (u_y + v_x),
                lam * u_x + lam2mu * v_y,
            ]
        )

    times = list(snapshot_times) if snapshot_times is not None else [t_end]
    keep = snapshot_transform or (lambda s: s)
    if t_end <= 0:
        return [keep(GridState(x=x, y=y, t=0.0, fields=state))]
    dt_target = cfl / (medium.c_max * math.hypot(n_x, n_y))
    if dt is not None:
        dt_target = min(dt, dt_target)
    n_steps = _step_count(t_end, dt_target, times)
    step = t_end / n_steps
    wanted = {round(t / t_end * n_steps): t for t in times}
    out = []
    if 0 in wanted:
        out.append(keep(GridState(x=x, y=y, t=0.0, fields=state.copy())))
    for i in range(1, n_steps + 1):
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * step * k1)
        k3 = rhs(state + 0.5 * step * k2)
        k4 = rhs(state + step * k3)
        state = state + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if i in wanted:
            out.append(keep(GridState(x=x, y=y, t=wanted[i], fields=state.copy())))
    return out
