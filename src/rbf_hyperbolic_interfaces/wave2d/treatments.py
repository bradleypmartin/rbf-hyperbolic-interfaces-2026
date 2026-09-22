"""Coefficient treatments on scattered nodes (issue demo#69, T2).

The 2-D twin of :mod:`rbf_hyperbolic_interfaces.wave1d.treatments`: the coefficients the
naive RBF-FD operator samples at the nodes are replaced by kernel averages
of the true medium over a square of side tied to the nominal spacing ``h``,
and the unchanged naive operator runs on the result against the true
medium's reference. Isotropy is kept, so the treated medium stays a
``(lam, mu, rho)`` triple the five-field operator takes as it is:

- the two wave moduli ``K = lam + 2 mu`` and ``mu`` are averaged
  harmonically (their compliances ``1/K`` and ``1/mu`` arithmetically) and
  the density arithmetically, the Moczo et al. (2002) line and the 2-D
  reading of the 1-D compliance-and-density rule; ``lam`` is
  ``K - 2 mu`` of the averages;
- ``kernel="box"`` is the cell mean over a square of side ``h`` (the
  Tornberg–Engquist regularisation of a jump in 1-D) or ``2h``;
- ``kernel="sinc"`` is the product of two windowed sincs, the separable
  low-pass filter of Koene et al. (2022, §3.3) applied to the true
  profile: cutoff ``cutoff`` times the grid Nyquist ``pi/h`` along each
  axis, Hanning window of half-width ``half_width`` cells.

The anisotropic effective medium (Schoenberg–Muir, which Koene et al. find
best in elastic media) needs an anisotropic operator and is not built.

Each average is a composite Gauss–Legendre product rule on the square:
sub-squares of half-side at most ``1.5 delta`` (and at most ``h / 2``, so
the sinc stays resolved) with ``n_quad`` points per axis each, the true
medium evaluated at the wrapped quadrature points. The tanh edge's poles
lie ``pi delta / 2`` off the real axis, more than a sub-square's half-side,
where the rule converges geometrically: about ``1e-7`` relative with the
default 8 points at ``delta = h / 8`` (``tests/test_wave2d_treatments.py``),
four orders below the errors the treated media leave in a run. A jump has
no such scale and is integrated only to ``O(1 / n_quad)``; the treatments
are meant for the sub-grid smooth edges of Part 3, not for a jump.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np

from .domain import LayeredMedium2D

Kernel = Literal["box", "sinc"]


@dataclass(frozen=True)
class TreatedMedium2D:
    """The naive operator's medium after a coefficient treatment of width ``h``.

    Provides ``lam_at``, ``mu_at``, ``rho_at`` and ``material_at`` as
    :class:`~.domain.LayeredMedium2D` does, and passes the geometry
    (``background``, ``lower``, ``upper``) through so the plane pulse and
    the time step can be set from it.
    """

    medium: LayeredMedium2D
    h: float
    kernel: Kernel = "box"
    cutoff: float = 1.1
    half_width: float = 2.5
    n_quad: int = 8
    chunk: int = 256

    def __post_init__(self) -> None:
        if self.h <= 0:
            raise ValueError("h must be positive")
        if self.kernel not in ("box", "sinc"):
            raise ValueError(f"unknown kernel {self.kernel!r}")
        if self.support > 0.25:
            raise ValueError("the kernel's support must fit in a quarter period")

    @property
    def support(self) -> float:
        """Half-side of the kernel's square support."""
        return 0.5 * self.h if self.kernel == "box" else self.half_width * self.h

    @property
    def background(self):
        return self.medium.background

    @property
    def layer(self):
        return self.medium.layer

    @property
    def lower(self):
        return self.medium.lower

    @property
    def upper(self):
        return self.medium.upper

    @property
    def interfaces(self):
        return self.medium.interfaces

    def _weights_1d(self, z: np.ndarray) -> np.ndarray:
        """Unnormalised kernel along one axis at the offsets ``z``."""
        if self.kernel == "box":
            return np.ones_like(z)
        k_c = self.cutoff * np.pi / self.h
        window = 0.5 * (1 + np.cos(np.pi * z / self.support))
        return np.sinc(k_c * z / np.pi) * window

    def _rule_1d(self) -> tuple[np.ndarray, np.ndarray]:
        """Composite Gauss–Legendre offsets and kernel-times-weight values
        along one axis: pieces of half-side at most ``1.5 delta`` and
        ``h / 2`` over ``[-a, a]``."""
        a = self.support
        d = self.medium.edge_width
        half_max = 0.5 * self.h if d == 0 else min(1.5 * d, 0.5 * self.h)
        pieces = max(1, int(np.ceil(a / half_max)))
        half = a / pieces
        centres = -a + half * (2 * np.arange(pieces) + 1)
        nodes, wts = np.polynomial.legendre.leggauss(self.n_quad)
        offsets = (centres[:, None] + half * nodes[None, :]).ravel()
        w1 = np.tile(half * wts, pieces) * self._weights_1d(offsets)
        return offsets, w1

    def _treated(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(lam, mu, rho)`` of the treated medium at the points."""
        x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
        offsets, w1 = self._rule_1d()
        w2 = (w1[:, None] * w1[None, :]).ravel()  # (q*q,)
        w2 = w2 / w2.sum()
        dx = np.repeat(offsets, offsets.size)
        dy = np.tile(offsets, offsets.size)
        inv_k = np.empty(x.size)
        inv_mu = np.empty(x.size)
        rho = np.empty(x.size)
        for start in range(0, x.size, self.chunk):
            sl = slice(start, start + self.chunk)
            xq = (x[sl, None] + dx[None, :]) % 1.0
            yq = (y[sl, None] + dy[None, :]) % 1.0
            lam_q, mu_q, rho_q = self.medium.material_at(xq, yq)
            inv_k[sl] = (w2 / (lam_q + 2 * mu_q)).sum(axis=1)
            inv_mu[sl] = (w2 / mu_q).sum(axis=1)
            rho[sl] = (w2 * rho_q).sum(axis=1)
        k = 1.0 / inv_k
        mu = 1.0 / inv_mu
        return k - 2 * mu, mu, rho

    def material_at(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shape = np.shape(x)
        lam, mu, rho = self._treated(x, y)
        return lam.reshape(shape), mu.reshape(shape), rho.reshape(shape)

    def lam_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.material_at(x, y)[0]

    def mu_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.material_at(x, y)[1]

    def rho_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.material_at(x, y)[2]

    @cached_property
    def c_max(self) -> float:
        """Largest treated P speed. A cell mean lies between the true
        values, so the box kernel keeps the true ``c_max``; the sinc kernel
        overshoots (Gibbs) and is sampled on a grid of spacing ``h / 8``
        across the band's edges."""
        if self.kernel == "box":
            return self.medium.c_max
        step = self.h / 8
        span = self.support + 4 * self.medium.edge_width
        ys = []
        for ifc in self.medium.interfaces:
            ys.append(
                ifc.y0
                + np.arange(-span - abs(ifc.amplitude), span + abs(ifc.amplitude), step)
            )
        y = np.concatenate(ys)
        x = np.arange(0.0, 1.0, self.h / 2)
        xx, yy = np.meshgrid(x, y)
        lam, mu, rho = self._treated(xx.ravel(), yy.ravel())
        return float(np.sqrt((lam + 2 * mu) / rho).max())
