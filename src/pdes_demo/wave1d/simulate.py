"""RK4 time stepping for the 1-D wave equation in first-order form."""

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .domain import Grid1D, LayeredMedium, right_going_pulse
from .operators import Mode, build_operators


@dataclass(frozen=True)
class Snapshots:
    t: np.ndarray  # (n_snap,)
    u: np.ndarray  # (n_snap, n)
    f: np.ndarray  # (n_snap, n)
    dt: float


def stable_dt(grid: Grid1D, medium: LayeredMedium, cfl: float = 0.4) -> float:
    """Time step from a CFL number relative to the fastest material.

    FD4 + RK4 is stable up to roughly c*dt/h ~ 2; 0.4 matches the original
    MATLAB driver (dt = 0.001 at N = 400, c_max = 2) with margin for the
    interface-aware stencils.
    """
    return cfl * grid.h / medium.c_max


def rk4_wave(
    u0: np.ndarray,
    f0: np.ndarray,
    du: sp.csr_array,
    df: sp.csr_array,
    rho: np.ndarray,
    c: np.ndarray,
    dt: float,
    n_steps: int,
    store_every: int = 1,
) -> Snapshots:
    """Integrate rho*u_t = f_x, f_t = rho*c**2*u_x with classical RK4."""
    inv_rho = 1.0 / rho
    rho_c2 = rho * c**2

    def rhs(u: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (df @ f) * inv_rho, (du @ u) * rho_c2

    # One slot for t = 0, one per stored step, and one for the final step
    # when the stride does not divide the step count.
    n_snap = n_steps // store_every + 1 + (1 if n_steps % store_every else 0)
    us = np.empty((n_snap, u0.size))
    fs = np.empty((n_snap, f0.size))
    ts = np.empty(n_snap)
    u, f = u0.copy(), f0.copy()
    us[0], fs[0], ts[0] = u, f, 0.0
    snap = 1
    for step in range(1, n_steps + 1):
        k1u, k1f = rhs(u, f)
        k2u, k2f = rhs(u + 0.5 * dt * k1u, f + 0.5 * dt * k1f)
        k3u, k3f = rhs(u + 0.5 * dt * k2u, f + 0.5 * dt * k2f)
        k4u, k4f = rhs(u + dt * k3u, f + dt * k3f)
        u = u + dt / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
        f = f + dt / 6 * (k1f + 2 * k2f + 2 * k3f + k4f)
        if step % store_every == 0 or step == n_steps:
            us[snap], fs[snap], ts[snap] = u, f, step * dt
            snap += 1
    return Snapshots(t=ts[:snap], u=us[:snap], f=fs[:snap], dt=dt)


def run(
    grid: Grid1D,
    medium: LayeredMedium,
    *,
    order: int = 4,
    mode: Mode = "aware",
    t_end: float = 1.5,
    cfl: float = 0.4,
    n_snapshots: int | None = None,
    pulse_center: float = -0.5,
    pulse_sharpness: float = 600.0,
) -> Snapshots:
    """Build operators, set the initial pulse, and integrate to ``t_end``.

    ``t_end`` is hit exactly: the step is shrunk from ``stable_dt`` so that
    an integer number of steps lands on it. With ``n_snapshots`` unset every
    step is stored.
    """
    dt_target = stable_dt(grid, medium, cfl)
    n_steps = max(1, math.ceil(t_end / dt_target))
    dt = t_end / n_steps
    store_every = 1 if n_snapshots is None else max(1, n_steps // n_snapshots)

    du, df = build_operators(grid, medium, order=order, mode=mode)
    u0, f0 = right_going_pulse(grid.x, medium, pulse_center, pulse_sharpness)
    return rk4_wave(
        u0,
        f0,
        du,
        df,
        medium.rho_at(grid.x),
        medium.c_at(grid.x),
        dt,
        n_steps,
        store_every,
    )
