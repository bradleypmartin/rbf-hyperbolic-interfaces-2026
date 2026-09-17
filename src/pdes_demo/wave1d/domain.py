"""Problem setup for the 1-D wave equation on the periodic interval [-1, 1).

First-order form (dissertation eq. 1), with u = particle velocity and
f = stress:

    rho * u_t = f_x,        f_t = rho * c**2 * u_x

The medium is a background material everywhere except one layer
``[layer_start, layer_start + layer_width)`` with different (c, rho). The two
layer edges are the interfaces where naive finite differences break down.
"""

from dataclasses import dataclass

import numpy as np

PERIOD = 2.0
X_MIN = -1.0


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

    def __post_init__(self) -> None:
        if self.layer_width <= 0:
            raise ValueError("layer_width must be positive")
        if not (X_MIN < self.layer_start and self.layer_end < X_MIN + PERIOD):
            raise ValueError("layer must sit strictly inside (-1, 1)")

    @property
    def layer_end(self) -> float:
        return self.layer_start + self.layer_width

    @property
    def interfaces(self) -> tuple[Interface, Interface]:
        return (
            Interface(self.layer_start, self.background, self.layer),
            Interface(self.layer_end, self.layer, self.background),
        )

    @property
    def c_max(self) -> float:
        return max(self.background.c, self.layer.c)

    def in_layer(self, x: np.ndarray) -> np.ndarray:
        return (x >= self.layer_start) & (x < self.layer_end)

    def c_at(self, x: np.ndarray) -> np.ndarray:
        return np.where(self.in_layer(x), self.layer.c, self.background.c)

    def rho_at(self, x: np.ndarray) -> np.ndarray:
        return np.where(self.in_layer(x), self.layer.rho, self.background.rho)

    def impedance_at(self, x: np.ndarray) -> np.ndarray:
        return self.rho_at(x) * self.c_at(x)


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
    return np.exp(-sharpness * (x - center) ** 2)


def right_going_pulse(
    x: np.ndarray,
    medium: LayeredMedium,
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
