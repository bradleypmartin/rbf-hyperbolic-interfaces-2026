"""Exact solution for a Gaussian pulse hitting a piecewise-constant layer.

In each constant-coefficient region the wave equation is solved exactly by
d'Alembert pulses travelling at the local speed. At an interface between
impedances Z_a (incident side) and Z_b, a stress pulse splits into a
transmitted pulse of amplitude ``T = 2 Z_b / (Z_a + Z_b)`` and a reflected one
of amplitude ``R = (Z_b - Z_a) / (Z_a + Z_b)``; ``1 + R == T`` keeps f
continuous and the velocity relation ``u = -direction * f / Z`` keeps u
continuous. Summing every such "ray" (including multiple bounces inside the
layer and wrap-around on the periodic domain) gives the exact solution, which
the numerical solvers are tested against.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from .domain import PERIOD, X_MIN, LayeredMedium

_LEFT, _LAYER, _RIGHT = 0, 1, 2
_SEAM_RIGHT = X_MIN + PERIOD


@dataclass(frozen=True)
class _Ray:
    region: int
    direction: int  # +1 right-going, -1 left-going
    amplitude: float  # stress amplitude relative to the initial pulse
    x_emit: float  # where the ray was born
    delay: float  # accumulated travel time of its phase
    seam_child: bool = True  # spawn a continuation when crossing the seam


def _trace_rays(
    medium: LayeredMedium,
    center_x: float,
    center_t: float,
    t_max: float,
    margin_t: float,
    tol: float,
) -> list[_Ray]:
    """Enumerate rays; ``center_t`` and ``margin_t`` are in travel-time units."""
    x1, x2 = medium.layer_start, medium.layer_end
    bg, ly = medium.background, medium.layer
    speed = {_LEFT: bg.c, _LAYER: ly.c, _RIGHT: bg.c}
    imped = {_LEFT: bg.impedance, _LAYER: ly.impedance, _RIGHT: bg.impedance}

    # (region, direction) -> (boundary x, next region, is_interface)
    next_hop = {
        (_LEFT, +1): (x1, _LAYER, True),
        (_LAYER, +1): (x2, _RIGHT, True),
        (_RIGHT, +1): (_SEAM_RIGHT, _LEFT, False),
        (_LEFT, -1): (X_MIN, _RIGHT, False),
        (_LAYER, -1): (x1, _LEFT, True),
        (_RIGHT, -1): (x2, _LAYER, True),
    }

    start_region = _LEFT if center_x < x1 else (_LAYER if center_x < x2 else _RIGHT)
    queue = deque([_Ray(start_region, +1, 1.0, 0.0, 0.0)])
    if start_region == _LEFT:
        # The initial pulse is periodic, so its left tail also sits just
        # below the right seam, in the right region. Rays only run forward,
        # so that pre-image has to be seeded explicitly; its own seam
        # crossing would duplicate the main ray, hence no child there.
        queue.append(_Ray(_RIGHT, +1, 1.0, PERIOD, 0.0, seam_child=False))
    rays: list[_Ray] = []
    while queue:
        ray = queue.popleft()
        rays.append(ray)
        x_b, region_b, is_interface = next_hop[(ray.region, ray.direction)]
        delay = ray.delay + ray.direction * (x_b - ray.x_emit) / speed[ray.region]
        # The phase at x_b is delay - t and the pulse peaks at phase == center_t,
        # so the peak reaches x_b at t = delay - center_t. Later than t_max (plus
        # the pulse half-width) means nothing downstream can matter yet.
        if delay - center_t > t_max + margin_t:
            continue
        if is_interface:
            z_a, z_b = imped[ray.region], imped[region_b]
            t_coef = 2 * z_b / (z_a + z_b)
            r_coef = (z_b - z_a) / (z_a + z_b)
            children = [
                _Ray(region_b, ray.direction, ray.amplitude * t_coef, x_b, delay),
                _Ray(ray.region, -ray.direction, ray.amplitude * r_coef, x_b, delay),
            ]
        elif ray.seam_child:
            x_wrap = X_MIN if x_b == _SEAM_RIGHT else _SEAM_RIGHT
            children = [_Ray(region_b, ray.direction, ray.amplitude, x_wrap, delay)]
        else:
            children = []
        queue.extend(child for child in children if abs(child.amplitude) > tol)
    return rays


def exact_solution(
    x: np.ndarray,
    t: float,
    medium: LayeredMedium,
    center: float = -0.5,
    sharpness: float = 600.0,
    tol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact (u, f) at time ``t`` for the right-going Gaussian pulse problem.

    The initial data are those of :func:`~pdes_demo.wave1d.domain.right_going_pulse`
    with the same ``center`` and ``sharpness``; the pulse must start in the
    background material.
    """
    x = np.asarray(x, dtype=float)
    if medium.in_layer(np.array([center]))[0]:
        raise ValueError("pulse must start outside the layer")
    # Rays carry their phase as a travel time, so the pulse centre and width
    # must be expressed in those units through the speed of the material the
    # pulse starts in: G(x - c t) = exp(-s c^2 (x/c - t - center/c)^2).
    c0 = medium.background.c
    center_t = center / c0
    sharpness_t = sharpness * c0**2
    margin_t = np.sqrt(-np.log(tol) / sharpness_t)
    rays = _trace_rays(medium, center, center_t, t, margin_t, tol)

    region = np.where(
        x < medium.layer_start,
        _LEFT,
        np.where(x < medium.layer_end, _LAYER, _RIGHT),
    )
    bg, ly = medium.background, medium.layer
    speed = {_LEFT: bg.c, _LAYER: ly.c, _RIGHT: bg.c}
    imped = {_LEFT: bg.impedance, _LAYER: ly.impedance, _RIGHT: bg.impedance}
    u = np.zeros_like(x)
    f = np.zeros_like(x)
    for ray in rays:
        mask = region == ray.region
        if not mask.any():
            continue
        travel = ray.direction * (x[mask] - ray.x_emit) / speed[ray.region]
        phase = travel - t + ray.delay
        pulse = ray.amplitude * np.exp(-sharpness_t * (phase - center_t) ** 2)
        f[mask] += pulse
        u[mask] += -ray.direction * pulse / imped[ray.region]
    return u, f
