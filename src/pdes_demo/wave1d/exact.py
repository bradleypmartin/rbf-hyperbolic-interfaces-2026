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


def _trace_rays(
    medium: LayeredMedium,
    center: float,
    t_max: float,
    margin: float,
    tol: float,
) -> list[_Ray]:
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

    start_region = _LEFT if center < x1 else (_LAYER if center < x2 else _RIGHT)
    queue = deque([_Ray(start_region, +1, 1.0, 0.0, 0.0)])
    rays: list[_Ray] = []
    while queue:
        ray = queue.popleft()
        rays.append(ray)
        x_b, region_b, is_interface = next_hop[(ray.region, ray.direction)]
        delay = ray.delay + ray.direction * (x_b - ray.x_emit) / speed[ray.region]
        # The pulse peak reaches x_b at time delay + center; later than t_max
        # (plus the pulse half-width) means nothing downstream can matter yet.
        if delay + center > t_max + margin:
            continue
        if is_interface:
            z_a, z_b = imped[ray.region], imped[region_b]
            t_coef = 2 * z_b / (z_a + z_b)
            r_coef = (z_b - z_a) / (z_a + z_b)
            children = [
                _Ray(region_b, ray.direction, ray.amplitude * t_coef, x_b, delay),
                _Ray(ray.region, -ray.direction, ray.amplitude * r_coef, x_b, delay),
            ]
        else:
            x_wrap = X_MIN if x_b == _SEAM_RIGHT else _SEAM_RIGHT
            children = [_Ray(region_b, ray.direction, ray.amplitude, x_wrap, delay)]
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
    margin = np.sqrt(-np.log(tol) / sharpness)
    rays = _trace_rays(medium, center, t, margin, tol)

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
        pulse = ray.amplitude * np.exp(-sharpness * (phase - center) ** 2)
        f[mask] += pulse
        u[mask] += -ray.direction * pulse / imped[ray.region]
    return u, f
