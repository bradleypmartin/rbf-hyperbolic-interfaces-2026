"""Finite-difference weights on arbitrary node sets (Fornberg's algorithm)."""

import numpy as np


def fornberg_weights(z: float, x: np.ndarray, m: int) -> np.ndarray:
    """Weights approximating derivatives 0..m at ``z`` from samples at nodes ``x``.

    Returns an array of shape ``(m + 1, len(x))`` whose row ``k`` holds the
    weights for the k-th derivative. Port of B. Fornberg, *Calculation of
    weights in finite difference formulas*, SIAM Review 40(3), 1998 (the
    ``weights.m`` shipped with the original MATLAB code).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    c = np.zeros((m + 1, n))
    c1 = 1.0
    c4 = x[0] - z
    c[0, 0] = 1.0
    for i in range(1, n):
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - z
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[k, i] = c1 * (k * c[k - 1, i - 1] - c5 * c[k, i - 1]) / c2
                c[0, i] = -c1 * c5 * c[0, i - 1] / c2
            for k in range(mn, 0, -1):
                c[k, j] = (c4 * c[k, j] - k * c[k - 1, j]) / c3
            c[0, j] = c4 * c[0, j] / c3
        c1 = c2
    return c
