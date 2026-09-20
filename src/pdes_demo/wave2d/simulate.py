"""RK4 time stepping for the 2-D elastic system on scattered nodes.

The semi-discrete system is ``s_t = (L + gamma H) s`` with ``L`` the block
elastic operator and ``H`` the block hyperviscosity of
:mod:`pdes_demo.wave2d.operators`; one sparse matrix-vector product per RK
stage, exactly as ``executeRbfSimulation.m`` does for the RBF part.
"""

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from .domain import FIELDS, LayeredMedium2D, NodeSet, plane_p_wave
from .operators import Mode, Operators, build_operators, hyperviscosity_gamma


@dataclass(frozen=True)
class Snapshots2D:
    t: np.ndarray  # (n_snap,)
    state: np.ndarray  # (n_snap, 5, n) in FIELDS order
    dt: float
    gamma: float

    def field(self, name: str) -> np.ndarray:
        """All snapshots of one field, shape ``(n_snap, n)``."""
        return self.state[:, FIELDS.index(name)]


def hyperviscosity_extreme(hyper: sp.csr_array) -> float:
    """Magnitude of the most negative eigenvalue of the hyperviscosity matrix.

    For the stencils used here it is real and dominant (the matrix is
    numerically negative semi-definite with much smaller imaginary parts),
    so ARPACK's largest-magnitude mode finds it in a few matrix-vector
    products. That is checked rather than assumed: if ARPACK stalls, or
    returns something that is not a negative real number, fall back to the
    Gershgorin bound, which is safe and about twice too large.
    """
    try:
        lam = sla.eigs(
            hyper.astype(float), k=1, which="LM", return_eigenvectors=False, tol=1e-4
        )[0]
        if lam.real < 0 and abs(lam.imag) <= 1e-3 * abs(lam.real):
            return float(abs(lam.real))
    except sla.ArpackNoConvergence:
        pass
    return float(np.max(np.abs(hyper).sum(axis=1)))


def stable_dt(
    nodes: NodeSet,
    medium: LayeredMedium2D,
    cfl: float = 0.5,
    *,
    hyper: sp.csr_array | None = None,
    gamma: float = 0.0,
    real_axis_limit: float = 2.0,
) -> float:
    """Time step from a CFL number, capped by the hyperviscosity spectrum.

    The wave part gives ``dt <= cfl h / c_p,max`` (the MATLAB driver used
    ``dt = 0.25 h`` at ``c_p,max = sqrt(6)``, i.e. CFL 0.61). Hyperviscosity
    adds eigenvalues on the negative real axis reaching about ``gamma`` times
    the extreme eigenvalue of ``H``, and RK4 is stable on the real axis only
    to 2.8, so with ``hyper`` given the step is also capped at
    ``real_axis_limit / (gamma |lambda_H|)``. Without the cap, doubling
    gamma blows up uniform-medium runs, whose CFL step is the larger one.
    """
    dt = cfl * nodes.h / medium.c_max
    if hyper is not None and gamma > 0:
        dt = min(dt, real_axis_limit / (gamma * hyperviscosity_extreme(hyper)))
    return dt


def rk4(
    state0: np.ndarray,
    operator: sp.csr_array,
    dt: float,
    n_steps: int,
    store_every: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Classical RK4 for ``s_t = operator @ s``; returns ``(times, states)``."""
    n_fields, n = state0.shape
    # One slot for t = 0, one per stored step, and one for the final step
    # when the stride does not divide the step count (t_end must be kept).
    n_snap = n_steps // store_every + 1 + (1 if n_steps % store_every else 0)
    states = np.empty((n_snap, n_fields, n))
    times = np.empty(n_snap)
    s = state0.ravel().copy()
    states[0], times[0] = state0, 0.0
    snap = 1
    for step in range(1, n_steps + 1):
        k1 = operator @ s
        k2 = operator @ (s + 0.5 * dt * k1)
        k3 = operator @ (s + 0.5 * dt * k2)
        k4 = operator @ (s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if step % store_every == 0 or step == n_steps:
            states[snap] = s.reshape(n_fields, n)
            times[snap] = step * dt
            snap += 1
    return times[:snap], states[:snap]


def run(
    nodes: NodeSet,
    medium: LayeredMedium2D,
    *,
    mode: Mode = "naive",
    t_end: float = 0.3,
    cfl: float = 0.5,
    gamma_scale: float = 1.0,
    n_snapshots: int | None = None,
    pulse_center: float = 0.75,
    pulse_sharpness: float = 23.0,
    operators: Operators | None = None,
    align_snapshots: bool = False,
    initial: np.ndarray | None = None,
) -> Snapshots2D:
    """Build operators (unless given), set the plane pulse, integrate to ``t_end``.

    ``t_end`` is hit exactly by shrinking the step so an integer number of
    steps lands on it. ``gamma_scale`` multiplies the MATLAB hyperviscosity
    amplitude of :func:`hyperviscosity_gamma`. ``initial`` ``(5, n)``
    replaces the plane pulse (the oblique train of
    :func:`~.domain.oblique_p_wave`, for one).

    With ``align_snapshots`` the step count is rounded up to a multiple of
    ``n_snapshots``, so the stored times are exactly ``j t_end / n_snapshots``
    whatever the stable step: runs on different node sets then share their
    snapshot times and can be compared frame by frame.
    """
    ops = (
        operators
        if operators is not None
        else build_operators(nodes, medium, mode=mode)
    )
    gamma = gamma_scale * hyperviscosity_gamma(nodes.h, ops.hyper_power)
    operator = sp.csr_array(ops.elastic + gamma * ops.hyper_block)

    dt_target = stable_dt(nodes, medium, cfl, hyper=ops.hyper, gamma=gamma)
    n_steps = max(1, math.ceil(t_end / dt_target))
    if align_snapshots and n_snapshots is not None:
        n_steps = n_snapshots * math.ceil(n_steps / n_snapshots)
    dt = t_end / n_steps
    store_every = 1 if n_snapshots is None else max(1, n_steps // n_snapshots)

    if initial is None:
        state0 = plane_p_wave(nodes, medium, pulse_center, pulse_sharpness)
    else:
        state0 = np.asarray(initial, dtype=float)
        if state0.shape != (len(FIELDS), nodes.n):
            raise ValueError(f"initial state must have shape (5, {nodes.n})")
    times, states = rk4(state0, operator, dt, n_steps, store_every)
    return Snapshots2D(t=times, state=states, dt=dt, gamma=gamma)


def energy(state: np.ndarray, nodes: NodeSet, medium: LayeredMedium2D) -> float:
    """Total elastic energy, kinetic plus strain, on the quasi-uniform node set.

    Strain energy is ``sigma : epsilon / 2`` with the plane-strain inverse
    ``eps_xx = ((lam + 2 mu) f - lam h) / D``, ``eps_yy`` likewise and
    ``eps_xy = g / (2 mu)``, ``D = 4 mu (lam + mu)``. Each node carries area
    ``1/n``. That quadrature is only first-order accurate on scattered
    nodes, so the value drifts by a few percent as a sharp pulse moves over
    different node arrangements even for the exact solution; compare
    against the exact state on the same nodes, not against a constant.
    """
    u, v, f, g, h = state
    lam = medium.lam_at(nodes.x, nodes.y)
    mu = medium.mu_at(nodes.x, nodes.y)
    rho = medium.rho_at(nodes.x, nodes.y)
    d = 4 * mu * (lam + mu)
    kinetic = 0.5 * rho * (u**2 + v**2)
    strain = 0.5 * (((lam + 2 * mu) * (f**2 + h**2) - 2 * lam * f * h) / d + g**2 / mu)
    return float(np.sum(kinetic + strain) / nodes.n)
