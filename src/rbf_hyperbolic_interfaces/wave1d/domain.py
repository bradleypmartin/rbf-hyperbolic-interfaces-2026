"""Problem setup for the 1-D wave equation on the periodic interval [-1, 1).

First-order form (dissertation eq. 1), with u = particle velocity and
f = stress:

    rho * u_t = f_x,        f_t = rho * c**2 * u_x

The medium is a background material everywhere except one layer
``[layer_start, layer_start + layer_width)`` with different (c, rho). The two
layer edges are the interfaces where naive finite differences break down.

With ``edge_width > 0`` the two edges are smooth tanh transitions of that
scale instead of jumps (issue demo#27): the same layer, swept from the
dissertation's discontinuity through the "twilight zone" of edges too steep
for a grid to resolve, to a gentle transition every scheme handles.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

PERIOD = 2.0
X_MIN = -1.0


class Medium1D(Protocol):
    """What the solvers ask of a medium: speed, density and impedance at points
    and the fastest speed for the time step. :class:`LayeredMedium` satisfies
    it, and so does the 1-D image of a flat 2-D medium that
    :mod:`rbf_hyperbolic_interfaces.wave2d.exact` hands to the spectral solver.
    """

    @property
    def c_max(self) -> float: ...

    def c_at(self, x: np.ndarray) -> np.ndarray: ...

    def rho_at(self, x: np.ndarray) -> np.ndarray: ...

    def impedance_at(self, x: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Material:
    c: float
    rho: float

    @property
    def impedance(self) -> float:
        return self.rho * self.c

    def taylor(self, p: int) -> tuple[np.ndarray, np.ndarray]:
        """Taylor coefficients (degree < p) of 1/rho and rho*c**2 about a point.

        Constant materials have only a constant term; kept as vectors so the
        operator construction also works for smoothly varying coefficients.
        """
        inv_rho = np.zeros(p)
        rho_c2 = np.zeros(p)
        inv_rho[0] = 1.0 / self.rho
        rho_c2[0] = self.rho * self.c**2
        return inv_rho, rho_c2


@dataclass(frozen=True)
class Interface:
    x: float
    left: Material
    right: Material


@dataclass(frozen=True)
class LayeredMedium:
    background: Material = Material(c=1.0, rho=1.0)
    layer: Material = Material(c=2.0, rho=1.0)
    layer_start: float = 0.0
    layer_width: float = 0.5
    edge_width: float = 0.0

    def __post_init__(self) -> None:
        if self.layer_width <= 0:
            raise ValueError("layer_width must be positive")
        if not (X_MIN < self.layer_start and self.layer_end < X_MIN + PERIOD):
            raise ValueError("layer must sit strictly inside (-1, 1)")
        if self.edge_width < 0:
            raise ValueError("edge_width must be non-negative (0 = jump)")

    @property
    def layer_end(self) -> float:
        return self.layer_start + self.layer_width

    @property
    def is_smooth(self) -> bool:
        return self.edge_width > 0

    @property
    def interfaces(self) -> tuple[Interface, Interface]:
        """The two edges as jumps; only meaningful for ``edge_width == 0``.

        With smooth edges the positions still mark the edge centres (the
        seed ODEs use them as breakpoints), but the materials on each side
        are the far-field values, not what a stencil sees.
        """
        return (
            Interface(self.layer_start, self.background, self.layer),
            Interface(self.layer_end, self.layer, self.background),
        )

    @property
    def c_max(self) -> float:
        return max(self.background.c, self.layer.c)

    def in_layer(self, x: np.ndarray) -> np.ndarray:
        return (x >= self.layer_start) & (x < self.layer_end)

    def layer_fraction(self, x: np.ndarray) -> np.ndarray:
        """Blend weight of the layer material: 1 inside, 0 outside.

        For ``edge_width > 0`` this is a tanh step up at ``layer_start`` and
        down at ``layer_end``, summed over periodic images so the profile is
        smooth and periodic to rounding (the images' tails are below
        ``exp(-2 * (2 * n_images - 1) / edge_width)``).
        """
        x = np.asarray(x, dtype=float)
        if not self.is_smooth:
            return self.in_layer(x).astype(float)
        d = self.edge_width
        n_images = 2 + int(10 * d)
        base = (x - X_MIN) % PERIOD + X_MIN
        s = np.zeros_like(base)
        for m in range(-n_images, n_images + 1):
            xm = base + m * PERIOD
            s += 0.5 * (
                np.tanh((xm - self.layer_start) / d)
                - np.tanh((xm - self.layer_end) / d)
            )
        return s

    def c_at(self, x: np.ndarray) -> np.ndarray:
        if not self.is_smooth:
            return np.where(self.in_layer(x), self.layer.c, self.background.c)
        s = self.layer_fraction(x)
        return self.background.c + (self.layer.c - self.background.c) * s

    def rho_at(self, x: np.ndarray) -> np.ndarray:
        if not self.is_smooth:
            return np.where(self.in_layer(x), self.layer.rho, self.background.rho)
        s = self.layer_fraction(x)
        return self.background.rho + (self.layer.rho - self.background.rho) * s

    def impedance_at(self, x: np.ndarray) -> np.ndarray:
        return self.rho_at(x) * self.c_at(x)

    def varies_over(self, xs: np.ndarray, rtol: float = 0.0) -> bool:
        """Whether the material differs between any two of the points ``xs``.

        With ``rtol == 0`` this is exact float inequality: for a tanh edge the
        tails round to the far-field value beyond about ``19 * edge_width``,
        so a stencil is "aware" of an edge out to that distance plus its own
        half-width. For a jump it is the same test as crossing an interface.
        """
        c = self.c_at(xs)
        rho = self.rho_at(xs)
        spread_c = np.ptp(c) / np.max(np.abs(c))
        spread_rho = np.ptp(rho) / np.max(np.abs(rho))
        return bool(max(spread_c, spread_rho) > rtol)


@dataclass(frozen=True)
class Grid1D:
    n: int
    x: np.ndarray
    h: float


def periodic_grid(n: int) -> Grid1D:
    """``n`` cell-centred nodes on [-1, 1); no node lands on x = -1 or x = 0."""
    h = PERIOD / n
    x = X_MIN + (np.arange(n) + 0.5) * h
    return Grid1D(n=n, x=x, h=h)


def gaussian(
    x: np.ndarray, center: float = -0.5, sharpness: float = 600.0
) -> np.ndarray:
    """Gaussian bump on the periodic interval (distance measured the short way)."""
    d = (x - center + PERIOD / 2) % PERIOD - PERIOD / 2
    return np.exp(-sharpness * d**2)


def right_going_pulse(
    x: np.ndarray,
    medium: Medium1D,
    center: float = -0.5,
    sharpness: float = 600.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Initial (u, f) for a Gaussian stress pulse travelling in the +x direction.

    A right-going wave satisfies f = -Z u with Z = rho * c, so u = -f / Z.
    With Z = 1 this is exactly the dissertation's initial condition (eq. 31).
    """
    f0 = gaussian(x, center, sharpness)
    u0 = -f0 / medium.impedance_at(x)
    return u0, f0
