from math import factorial

import numpy as np
import pytest
from scipy.special import eval_genlaguerre

from rbf_hyperbolic_interfaces.wave2d import LayeredMedium2D, make_node_set
from rbf_hyperbolic_interfaces.wave2d.operators import build_operators
from rbf_hyperbolic_interfaces.wave2d.rbf import (
    laplacian_power_of_gaussian,
    laplacian_power_of_monomials,
    monomial_exponents,
    rbf_fd_weights,
)


def _random_stencil(rng: np.random.Generator, n: int = 30, h: float = 0.02):
    off = rng.normal(size=(1, n, 2)) * 1.3 * h
    off[0, 0] = 0.0
    return off


def _monomials(off: np.ndarray, exps: np.ndarray) -> np.ndarray:
    """Values of x^a y^b at the stencil nodes, shape (n, m)."""
    return off[0, :, 0:1] ** exps[:, 0] * off[0, :, 1:2] ** exps[:, 1]


def test_monomial_exponents_order() -> None:
    np.testing.assert_array_equal(
        monomial_exponents(2), [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    )
    assert len(monomial_exponents(4)) == 15


@pytest.mark.parametrize("k", range(6))
def test_laplacian_power_of_gaussian_matches_laguerre_closed_form(k: int) -> None:
    # Delta^k exp(-s) in 2-D equals (-4)^k k! L_k^{(0)}(s) exp(-s).
    s = np.linspace(0, 8, 33)
    expected = (-4.0) ** k * factorial(k) * eval_genlaguerre(k, 0, s)
    np.testing.assert_allclose(laplacian_power_of_gaussian(k, s), expected, rtol=1e-10)


def test_laplacian_power_of_monomials() -> None:
    np.testing.assert_array_equal(
        laplacian_power_of_monomials(1, monomial_exponents(2)), [0, 0, 0, 2, 0, 2]
    )
    exps = monomial_exponents(6)
    out = laplacian_power_of_monomials(3, exps)
    lookup = {tuple(e): v for e, v in zip(exps, out, strict=True)}
    # Delta^3 x^6 = 720; Delta^3 x^4 y^2 = 3 * 4! * 2! = 144; odd powers vanish.
    assert lookup[(6, 0)] == 720 and lookup[(0, 6)] == 720
    assert lookup[(4, 2)] == 144 and lookup[(2, 4)] == 144
    assert lookup[(5, 1)] == 0 and lookup[(3, 3)] == 0 and lookup[(4, 0)] == 0


def test_weights_reproduce_polynomial_derivatives() -> None:
    rng = np.random.default_rng(7)
    off = _random_stencil(rng)
    w = rbf_fd_weights(off, poly_degree=4, hyper_power=3)
    exps = monomial_exponents(4)
    vals = _monomials(off, exps)
    is_x = (exps[:, 0] == 1) & (exps[:, 1] == 0)
    is_y = (exps[:, 0] == 0) & (exps[:, 1] == 1)
    np.testing.assert_allclose(w.dx[0] @ vals, is_x, atol=1e-9)
    np.testing.assert_allclose(w.dy[0] @ vals, is_y, atol=1e-9)
    # Delta^3 of a quartic is zero; the weights are ~1/h^6 so judge the
    # residual against the size of the terms that cancel.
    scale = np.abs(w.hyper[0])[:, None] * np.abs(vals)
    np.testing.assert_allclose(w.hyper[0] @ vals, 0.0, atol=1e-9 * scale.sum())
    # Higher augmentation degree makes Delta^2 exact on sextics.
    w2 = rbf_fd_weights(off, poly_degree=6, hyper_power=2)
    exps6 = monomial_exponents(6)
    vals6 = _monomials(off, exps6)
    scale6 = (np.abs(w2.hyper[0])[:, None] * np.abs(vals6)).sum()
    np.testing.assert_allclose(
        w2.hyper[0] @ vals6,
        laplacian_power_of_monomials(2, exps6),
        atol=1e-9 * scale6,
    )


def test_weights_transform_correctly_under_rotation() -> None:
    # Rotating the stencil by +90 degrees (x, y) -> (-y, x) turns the
    # x-derivative into -d/dy and the y-derivative into d/dx; the Laplacian
    # power is invariant. Gaussians and the full polynomial space are both
    # rotation invariant, so the weights must follow exactly.
    rng = np.random.default_rng(11)
    off = _random_stencil(rng)
    rot = np.stack([-off[..., 1], off[..., 0]], axis=-1)
    w = rbf_fd_weights(off)
    wr = rbf_fd_weights(rot)
    np.testing.assert_allclose(wr.dx, -w.dy, rtol=1e-8, atol=1e-6)
    np.testing.assert_allclose(wr.dy, w.dx, rtol=1e-8, atol=1e-6)
    np.testing.assert_allclose(wr.hyper, w.hyper, rtol=1e-6)


def test_derivatives_converge_at_fourth_order_on_relaxed_nodes() -> None:
    medium = LayeredMedium2D()
    errs = []
    ns = [900, 1600, 2500]
    for n in ns:
        nodes = make_node_set(medium, n, repulsion_steps=40)
        ops = build_operators(nodes, medium)
        f = np.sin(2 * np.pi * nodes.x) * np.cos(4 * np.pi * nodes.y)
        fx = 2 * np.pi * np.cos(2 * np.pi * nodes.x) * np.cos(4 * np.pi * nodes.y)
        fy = -4 * np.pi * np.sin(2 * np.pi * nodes.x) * np.sin(4 * np.pi * nodes.y)
        ex = np.linalg.norm(ops.dx @ f - fx) / np.linalg.norm(fx)
        ey = np.linalg.norm(ops.dy @ f - fy) / np.linalg.norm(fy)
        errs.append(max(ex, ey))
    h = 1 / np.sqrt(np.array(ns, dtype=float))
    rates = np.log(np.array(errs[:-1]) / np.array(errs[1:])) / np.log(h[:-1] / h[1:])
    assert np.all(rates > 3.0), (errs, rates)
    assert errs[-1] < 2e-4


def test_rejects_bad_inputs() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        rbf_fd_weights(rng.normal(size=(1, 10, 2)), poly_degree=4)  # 10 < 15 terms
    with pytest.raises(ValueError):
        rbf_fd_weights(rng.normal(size=(30, 2)))
    off = _random_stencil(rng)
    off[0, 5] = off[0, 6]  # coincident nodes
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        rbf_fd_weights(off)
