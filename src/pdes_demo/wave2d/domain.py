"""Problem setup for the 2-D elastic wave equation on the periodic unit square.

First-order isotropic elastic system (dissertation eq. 32), with (u, v) the
particle velocities and (f, g, h) the stress components (xx, xy, yy):

    rho u_t = f_x + g_y            f_t = (lam + 2 mu) u_x + lam v_y
    rho v_t = g_x + h_y            g_t = mu (u_y + v_x)
                                   h_t = lam u_x + (lam + 2 mu) v_y

The medium is a background material everywhere except one band between two
interfaces ``y = y0 + amplitude sin(2 pi x)`` (dissertation §3.4.1 uses
flat interfaces at y = 0.25 and 0.5; §3.4.2 and the MATLAB use amplitude
0.02). Velocity and traction are continuous across an interface; the stress
component parallel to it is not.

With ``edge_width > 0`` the two interfaces are smooth tanh transitions of
that scale in the normal distance instead of jumps (issue #27, 2-D chain
from #36): lam, mu and rho are blended between the background and the band
material, so the Lame parameters vary smoothly and ``c_p``, ``c_s`` follow.
``edge_width = 0`` is the jump, bit for bit. Only flat interfaces take a
smooth edge for now; the curved case switches the vertical offset for the
true signed normal distance (#42).

Node sets follow ``EWE2DRbfPrep.m``: fixed hex-staggered rows that straddle
each interface orthogonally (Brad's empirical stability requirement), with
every other node relaxed by a simulated electrostatic repulsion. This is the
"all RBF" variant: no Cartesian far field, because the first 2-D milestone
is plain RBF-FD everywhere.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from .neighbors import minimal_image, periodic_knn, wrap

SQRT3 = math.sqrt(3.0)


@dataclass(frozen=True)
class ElasticMaterial:
    lam: float
    mu: float
    rho: float

    @property
    def c_p(self) -> float:
        return math.sqrt((self.lam + 2 * self.mu) / self.rho)

    @property
    def c_s(self) -> float:
        return math.sqrt(self.mu / self.rho)

    @property
    def p_impedance(self) -> float:
        return self.rho * self.c_p


@dataclass(frozen=True)
class SineInterface:
    """The curve ``y = y0 + amplitude * sin(2 pi x)``, periodic in x."""

    y0: float
    amplitude: float = 0.0

    def height(self, x: np.ndarray) -> np.ndarray:
        return self.y0 + self.amplitude * np.sin(2 * np.pi * np.asarray(x))

    def slope(self, x: np.ndarray) -> np.ndarray:
        return 2 * np.pi * self.amplitude * np.cos(2 * np.pi * np.asarray(x))

    def angle(self, x: np.ndarray) -> np.ndarray:
        """Tangent angle theta with tan(theta) = dy/dx (MATLAB ``curvedinterface``)."""
        return np.arctan(self.slope(x))

    def normal(self, x: np.ndarray) -> np.ndarray:
        """Unit normal pointing to +y, shape ``(..., 2)``."""
        th = self.angle(x)
        return np.stack([-np.sin(th), np.cos(th)], axis=-1)

    def vertical_offset(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Signed vertical distance ``y - height(x)`` wrapped to [-1/2, 1/2]."""
        return minimal_image(np.asarray(y) - self.height(x))


@dataclass(frozen=True)
class LayeredMedium2D:
    background: ElasticMaterial = ElasticMaterial(lam=1.0, mu=1.0, rho=1.0)
    layer: ElasticMaterial = ElasticMaterial(lam=4.0, mu=4.0, rho=2.0)
    lower: SineInterface = SineInterface(0.25)
    upper: SineInterface = SineInterface(0.5)
    edge_width: float = 0.0

    def __post_init__(self) -> None:
        gap = self.upper.y0 - self.lower.y0
        amp = abs(self.lower.amplitude) + abs(self.upper.amplitude)
        if not (0 < self.lower.y0 and self.upper.y0 < 1 and gap > amp):
            raise ValueError("interfaces must be ordered and inside (0, 1)")
        if self.edge_width < 0:
            raise ValueError("edge_width must be non-negative (0 = jump)")
        # Two tanh steps a distance g apart reach only tanh(g / (2 d)) of the
        # contrast between them: 99.6% at d = g / 8 (delta = 0.03 for the
        # default band), 96% at d = g / 4, where the band stops being one.
        if 4 * self.edge_width > gap - amp:
            raise ValueError(
                f"edge_width {self.edge_width:g} is too wide for a band of "
                f"width {gap - amp:g}: the two edges would merge (need 4 d <= width)"
            )
        if self.edge_width > 0 and not self.is_flat:
            raise NotImplementedError(
                "a smooth edge on a curved interface needs the signed normal "
                "distance (issue #42); only flat interfaces take edge_width > 0"
            )

    @property
    def interfaces(self) -> tuple[SineInterface, SineInterface]:
        return (self.lower, self.upper)

    @property
    def c_max(self) -> float:
        # Valid for smooth edges too: lam + 2 mu and rho are both linear in
        # the blend weight, so c_p**2 is a ratio of linear functions of it,
        # monotone, and takes its extremes at the two pure materials.
        return max(self.background.c_p, self.layer.c_p)

    @property
    def is_flat(self) -> bool:
        return self.lower.amplitude == 0.0 and self.upper.amplitude == 0.0

    @property
    def is_smooth(self) -> bool:
        return self.edge_width > 0

    def in_layer(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Which side of the edge centres a point is on (the band for a jump)."""
        x, y = np.asarray(x), np.asarray(y)
        return (y >= self.lower.height(x)) & (y < self.upper.height(x))

    def layer_fraction(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Blend weight of the band material: 1 inside, 0 outside.

        For ``edge_width > 0`` each edge is a tanh step of that scale in the
        signed normal distance to the interface, which for a flat interface
        is the vertical offset ``y - y0`` (the curved case, #42, will use
        the true signed distance). The band profile, a step up at the lower
        interface and down at the upper, is summed over its periodic images
        in y so the result is smooth and periodic to rounding: the first
        omitted image has both edges more than ``n_images`` from any point
        of [0, 1), where the tails are below ``2 exp(-2 n_images / d)``,
        which ``n_images > 19 d`` keeps under 1e-16. Inside the band the
        weight peaks at ``tanh(gap / (2 d))``, not 1, unless ``gap >> d``
        (99.6% of the contrast at d = 0.03 for the default band).
        """
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if not self.is_smooth:
            return self.in_layer(x, y).astype(float)
        d = self.edge_width
        n_images = 1 + int(19 * d)
        y0 = np.mod(y, 1.0)
        lower, upper = self.lower.height(x), self.upper.height(x)
        s = np.zeros(np.broadcast_shapes(x.shape, y.shape))
        for m in range(-n_images, n_images + 1):
            s += 0.5 * (np.tanh((y0 + m - lower) / d) - np.tanh((y0 + m - upper) / d))
        return s

    def _blend(self, x: np.ndarray, y: np.ndarray, attr: str) -> np.ndarray:
        bg, ly = getattr(self.background, attr), getattr(self.layer, attr)
        if not self.is_smooth:
            return np.where(self.in_layer(x, y), ly, bg)
        return bg + (ly - bg) * self.layer_fraction(x, y)

    def lam_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._blend(x, y, "lam")

    def mu_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._blend(x, y, "mu")

    def rho_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._blend(x, y, "rho")

    def material_at(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(lam, mu, rho)`` at the points, from one evaluation of the blend
        weight (the seed ODE march of :mod:`.seeds` calls this thousands of
        times per stencil)."""
        w = self.layer_fraction(x, y)
        bg, ly = self.background, self.layer
        return (
            bg.lam + (ly.lam - bg.lam) * w,
            bg.mu + (ly.mu - bg.mu) * w,
            bg.rho + (ly.rho - bg.rho) * w,
        )

    def varies_over(self, x: np.ndarray, y: np.ndarray, rtol: float = 0.0) -> bool:
        """Whether lam, mu or rho differ between any two of the points.

        The 2-D twin of ``wave1d.domain.LayeredMedium.varies_over``. With
        ``rtol == 0`` this is exact float inequality: for a tanh edge the
        tails round to the far-field value beyond about ``19 * edge_width``,
        so a stencil is "aware" of an edge out to that distance plus its own
        radius; for a jump it is the same test as straddling an interface.
        """
        if np.size(x) == 0 or np.size(y) == 0:
            return False
        spread = 0.0
        for attr in ("lam", "mu", "rho"):
            values = self._blend(x, y, attr)
            spread = max(spread, float(np.ptp(values) / np.max(np.abs(values))))
        return bool(spread > rtol)

    def distance_to_interfaces(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Smallest |vertical offset| to any interface (fine for mild curves)."""
        d = [np.abs(ifc.vertical_offset(x, y)) for ifc in self.interfaces]
        return np.min(d, axis=0)


# --- node sets -----------------------------------------------------------------


@dataclass(frozen=True)
class NodeSet:
    xy: np.ndarray  # (n, 2), all in [0, 1)
    h: float  # nominal spacing 1 / sqrt(n)
    fixed: np.ndarray = field(repr=False)  # (n,) bool: interface-straddling rows

    @property
    def n(self) -> int:
        return self.xy.shape[0]

    @property
    def x(self) -> np.ndarray:
        return self.xy[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.xy[:, 1]


def straddling_rows(
    interface: SineInterface, n_per_row: int, h: float, rows_per_side: int = 3
) -> np.ndarray:
    """Fixed nodes hugging one interface, shape ``(2 * rows_per_side * n_per_row, 2)``.

    Row ``j`` (0-based) sits at normal distance ``(1/2 + j sqrt(3)/2) h`` on
    each side and odd rows are shifted half a spacing along x, which is the
    hex packing of ``EWE2DRbfPrep.m`` (rows at 0.5, 0.5 + sqrt(3)/2 and
    0.5 + sqrt(3) times ``hInt``). Nodes are displaced along the local normal
    so the pairs straddle the curve orthogonally.
    """
    xs = np.arange(n_per_row) * (1.0 / n_per_row)
    out = []
    for j in range(rows_per_side):
        x_row = xs + (j % 2) * 0.5 / n_per_row
        base = np.stack([x_row, interface.height(x_row)], axis=-1)
        normal = interface.normal(x_row)
        offset = (0.5 + j * SQRT3 / 2) * h
        out.append(base - offset * normal)
        out.append(base + offset * normal)
    return wrap(np.concatenate(out))


def _repulsion_step(
    xy: np.ndarray,
    free: np.ndarray,
    delta: float,
    n_neighbors: int,
) -> np.ndarray:
    """Move every free node by ``delta`` along its net 1/r^4 repulsion direction.

    Port of ``mos2dsqperiodic7``: force from the ``n_neighbors`` nearest
    nodes, magnitude ``1/r^4`` (coded there as ``1/r^5`` on the raw
    displacement), then normalised, so only the direction matters and every
    node travels the same distance per step.
    """
    idx, dist = periodic_knn(xy, n_neighbors + 1)
    idx, dist = idx[free, 1:], dist[free, 1:]
    disp = minimal_image(xy[free][:, None, :] - xy[idx])
    force = np.sum(disp / dist[..., None] ** 5, axis=1)
    norm = np.linalg.norm(force, axis=1, keepdims=True)
    # Deliberate deviation from the MATLAB, which divides by zero here: a
    # node with no net force (a perfectly symmetric neighbourhood) stays put.
    norm[norm == 0] = 1.0
    new = xy.copy()
    new[free] = wrap(xy[free] + delta * force / norm)
    return new


def _reseed_intruders(
    xy: np.ndarray,
    free: np.ndarray,
    medium: LayeredMedium2D,
    h: float,
    band: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Throw free nodes that wandered into the fixed band back out to +-4h.

    Same rule as the MATLAB: random x, a random interface, a random side.
    """
    dist = medium.distance_to_interfaces(xy[:, 0], xy[:, 1])
    bad = np.flatnonzero(free & (dist <= band))
    if bad.size == 0:
        return xy
    new = xy.copy()
    x = rng.random(bad.size)
    which = rng.integers(0, len(medium.interfaces), bad.size)
    side = rng.choice([-1.0, 1.0], bad.size)
    y = np.empty(bad.size)
    for k, ifc in enumerate(medium.interfaces):
        m = which == k
        y[m] = ifc.height(x[m]) + side[m] * 4 * h
    new[bad] = wrap(np.stack([x, y], axis=-1))
    return new


def make_node_set(
    medium: LayeredMedium2D,
    n: int,
    *,
    rows_per_side: int = 3,
    repulsion_steps: int = 100,
    n_neighbors: int = 10,
    seed: int = 0,
) -> NodeSet:
    """Interface-fitted, repulsion-relaxed node set with exactly ``n`` nodes.

    ``n`` must be a perfect square (the MATLAB starts from an ``n1d x n1d``
    cell-centred grid). Steps:

    1. cell-centred Cartesian grid, ``h = 1 / n1d``;
    2. for each interface, ``2 * rows_per_side`` fixed rows of ``n1d`` nodes
       (:func:`straddling_rows`); the same number of grid nodes, the ones
       nearest the interfaces, are removed so the count stays ``n``;
    3. the remaining nodes are jittered and relaxed with ``repulsion_steps``
       steps of length ``delta = 2.5 h / step`` (MATLAB: ``0.05 sqrt(2500/N)
       / iteration``), wrapping periodically and re-seeding any free node
       that drifts within ``(1/2 + (rows_per_side - 1) sqrt(3)/2) h`` of an
       interface, i.e. inside the outermost fixed row.
    """
    n1d = math.isqrt(n)
    if n1d * n1d != n:
        raise ValueError(f"n must be a perfect square, got {n}")
    if rows_per_side < 1:
        raise ValueError("rows_per_side must be at least 1")
    h = 1.0 / n1d
    rng = np.random.default_rng(seed)

    g = (np.arange(n1d) + 0.5) * h
    gx, gy = np.meshgrid(g, g, indexing="xy")
    grid = np.stack([gx.ravel(), gy.ravel()], axis=-1)

    fixed_xy = np.concatenate(
        [straddling_rows(ifc, n1d, h, rows_per_side) for ifc in medium.interfaces]
    )
    n_fixed = fixed_xy.shape[0]
    if n_fixed >= n:
        raise ValueError("too few nodes for the requested interface rows")
    # Drop the grid nodes closest to the interfaces to make room for the rows.
    dist = medium.distance_to_interfaces(grid[:, 0], grid[:, 1])
    keep = np.argsort(dist, kind="stable")[n_fixed:]
    free_xy = grid[keep]
    free_xy = wrap(free_xy + (rng.random(free_xy.shape) - 0.5) * 0.05 * h)

    xy = np.concatenate([fixed_xy, free_xy])
    free = np.zeros(n, dtype=bool)
    free[n_fixed:] = True
    band = (0.5 + (rows_per_side - 1) * SQRT3 / 2) * h

    xy = _reseed_intruders(xy, free, medium, h, band, rng)
    for step in range(1, repulsion_steps + 1):
        xy = _repulsion_step(xy, free, 2.5 * h / step, n_neighbors)
        xy = _reseed_intruders(xy, free, medium, h, band, rng)
    return NodeSet(xy=xy, h=h, fixed=~free)


def nearest_spacing(nodes: NodeSet) -> np.ndarray:
    """Distance from every node to its nearest neighbour (periodic)."""
    _, dist = periodic_knn(nodes.xy, 2)
    return dist[:, 1]


# --- initial data ----------------------------------------------------------------


FIELDS = ("u", "v", "f", "g", "h")


def require_background_start(medium: LayeredMedium2D, center: float) -> None:
    """Raise unless a plane pulse centred at ``y = center`` starts in the
    background for every x (between the edge centres is the band, jump or
    smooth); the initial data and both references assume it."""
    lowest = medium.lower.y0 - abs(medium.lower.amplitude)
    highest = medium.upper.y0 + abs(medium.upper.amplitude)
    if lowest <= center % 1.0 < highest:
        raise ValueError("pulse must start in the background material for all x")


def plane_p_wave(
    nodes: NodeSet,
    medium: LayeredMedium2D,
    center: float = 0.75,
    sharpness: float = 23.0,
) -> np.ndarray:
    """Initial state ``(5, n)`` for a plane P-wave pulse travelling in -y.

    Dissertation eq. 48 with ``sharpness`` = 23: ``v = exp(-s^2 (y - c)^2)``,
    ``h = Z_p v`` and ``f = lam / (lam + 2 mu) h`` for the material the pulse
    starts in, ``u = g = 0``. With the default background (lam = mu = rho = 1)
    that is ``h = sqrt(3) v`` and ``f = v / sqrt(3)``.
    """
    mat = medium.background
    require_background_start(medium, center)
    v = np.exp(-(sharpness**2) * minimal_image(nodes.y - center) ** 2)
    state = np.zeros((len(FIELDS), nodes.n))
    state[1] = v
    state[4] = mat.p_impedance * v
    state[2] = mat.lam / (mat.lam + 2 * mat.mu) * state[4]
    return state
