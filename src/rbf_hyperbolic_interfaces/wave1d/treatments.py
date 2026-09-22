"""Coefficient treatments: the standard scheme on a changed medium (issue demo#69).

The standing alternative to an interface stencil in seismic and
electromagnetic finite differences changes the medium, not the stencil: the
coefficients are averaged or smoothed over a width tied to the grid spacing
``h`` and the unchanged standard scheme runs on the result. This module
makes that a :class:`~.domain.Medium1D`, so ``mode="naive"`` of
:func:`~.simulate.run` samples the treated medium exactly as it samples the
true one, and the error is measured against the true medium's reference.

Both treatments act on the compliance ``1/K = 1/(rho c**2)`` and the density
``rho``, the quantities the two sources agree on:

``kernel="box"``
    Cell averaging over ``[x - h/2, x + h/2]``: the harmonic mean of ``K``
    and the arithmetic mean of ``rho``, the Tikhonov–Samarskii line that
    Moczo et al. (2002) use on staggered grids. For a jump this is exactly
    the regularised coefficients of Tornberg & Engquist (2006), eq. (18)
    and (22): ``1/a`` linear across one cell centred on the jump, and
    ``1/b`` likewise, with ``a = -rho c**2`` and ``b = -1/rho``.
``kernel="sinc"``
    The low-pass filter of Mittet (2017) as run by Koene, Wittsten &
    Robertsson (2022, §3.3): an ideal low-pass at ``cutoff`` times the grid
    Nyquist wavenumber ``pi/h``, Hanning-windowed to a half-width of
    ``half_width`` cells (their 51 taps at ten-fold oversampling, centred at
    1.1 times the Nyquist). Their §3.2 anti-aliased step is the jump case.

Each treated value is a normalised weighted average, ``sum(w g) / sum(w)``
over the kernel's support, integrated piecewise by Gauss–Legendre with the
pieces split at the edge centres and at ``4 delta`` either side of them, so
a jump is integrated exactly and a tanh edge to rounding. Away from the
edges the average of a constant is that constant to rounding, so the
treated medium equals the true one where the pulse starts.

Widening the edge (the T0 comparator of demo#69) is not a kernel: it is the same
tanh profile with ``edge_width = max(delta, m h)``, which blends ``c`` and
``rho`` linearly, i.e. it regularises ``a`` rather than ``1/a``, the case
Tornberg & Engquist (2006, §4.1) predict stays first order.
"""

from dataclasses import dataclass, replace
from functools import cached_property
from typing import Literal

import numpy as np

from .domain import PERIOD, X_MIN, LayeredMedium

Kernel = Literal["box", "sinc"]


def widened(medium: LayeredMedium, h: float, cells: float) -> LayeredMedium:
    """The same layer with its edges no narrower than ``cells * h``."""
    return replace(medium, edge_width=max(medium.edge_width, cells * h))


@dataclass(frozen=True)
class TreatedMedium:
    """The naive scheme's medium after a coefficient treatment of width ``h``.

    Satisfies :class:`~.domain.Medium1D`; ``medium`` is the true medium the
    reference solution is computed on.
    """

    medium: LayeredMedium
    h: float
    kernel: Kernel = "box"
    cutoff: float = 1.1
    half_width: float = 2.5
    n_quad: int = 24

    def __post_init__(self) -> None:
        if self.h <= 0:
            raise ValueError("h must be positive")
        if self.kernel not in ("box", "sinc"):
            raise ValueError(f"unknown kernel {self.kernel!r}")
        if self.support > PERIOD / 2:
            raise ValueError("the kernel's support must fit in half a period")

    @property
    def support(self) -> float:
        """Half-width of the kernel's support."""
        return 0.5 * self.h if self.kernel == "box" else self.half_width * self.h

    def weights(self, z: np.ndarray) -> np.ndarray:
        """Unnormalised kernel at the offsets ``z`` (zero outside the support)."""
        z = np.asarray(z, dtype=float)
        inside = np.abs(z) <= self.support
        if self.kernel == "box":
            return inside.astype(float)
        k_c = self.cutoff * np.pi / self.h
        window = 0.5 * (1 + np.cos(np.pi * z / self.support))
        return np.where(inside, np.sinc(k_c * z / np.pi) * window, 0.0)

    def _quadrature(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gauss–Legendre points ``(x.size, q)`` and kernel-times-weight
        values for the average about each ``x``, normalised per point."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        a = self.support
        # Uniform pieces of at most h/2 keep the sinc resolved; the edge
        # pieces put every jump and every steep tanh region at a piece end.
        n_uniform = int(np.ceil(4 * a / self.h))
        breaks = [x[:, None] - a + 2 * a * np.arange(n_uniform + 1) / n_uniform]
        d = self.medium.edge_width
        for edge in (self.medium.layer_start, self.medium.layer_end):
            for image in (-PERIOD, 0.0, PERIOD):
                centre = edge + image
                for shift in (0.0,) if d == 0 else (-4 * d, 0.0, 4 * d):
                    breaks.append(np.clip(centre + shift, x - a, x + a)[:, None])
        b = np.sort(np.concatenate(breaks, axis=1), axis=1)
        nodes, wts = np.polynomial.legendre.leggauss(self.n_quad)
        lo, hi = b[:, :-1, None], b[:, 1:, None]
        pts = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
        w = 0.5 * (hi - lo) * wts * self.weights(pts - x[:, None, None])
        pts = pts.reshape(x.size, -1)
        w = w.reshape(x.size, -1)
        return pts, w / w.sum(axis=1, keepdims=True)

    def _treated(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(compliance, rho)`` at ``x``, the kernel averages of the true ones."""
        shape = np.shape(x)
        pts, w = self._quadrature(x)
        # The jump medium is not periodic in its own evaluation, so wrap the
        # quadrature points before sampling it.
        pts = (pts - X_MIN) % PERIOD + X_MIN
        rho = self.medium.rho_at(pts)
        compliance = 1.0 / (rho * self.medium.c_at(pts) ** 2)
        return (
            np.sum(w * compliance, axis=1).reshape(shape),
            np.sum(w * rho, axis=1).reshape(shape),
        )

    def rho_at(self, x: np.ndarray) -> np.ndarray:
        return self._treated(x)[1]

    def c_at(self, x: np.ndarray) -> np.ndarray:
        compliance, rho = self._treated(x)
        return 1.0 / np.sqrt(compliance * rho)

    def impedance_at(self, x: np.ndarray) -> np.ndarray:
        compliance, rho = self._treated(x)
        return np.sqrt(rho / compliance)

    @cached_property
    def c_max(self) -> float:
        """Largest treated speed. A cell mean lies between the true values,
        so the box kernel keeps the true ``c_max`` (and the same time step);
        the sinc kernel overshoots a jump (Gibbs), so it is sampled at 20
        points per cell over the period."""
        if self.kernel == "box":
            return self.medium.c_max
        n = int(np.ceil(20 * PERIOD / self.h))
        x = -1.0 + (np.arange(n) + 0.5) * PERIOD / n
        return float(np.max(self.c_at(x)))
