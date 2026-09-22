import numpy as np
import pytest
import scipy.sparse as sp

from rbf_hyperbolic_interfaces.wave2d import (
    ElasticMaterial,
    LayeredMedium2D,
    make_node_set,
    minimal_image,
    plane_p_wave,
)
from rbf_hyperbolic_interfaces.wave2d.operators import (
    build_operators,
    elastic_block,
    hyperviscosity_gamma,
)

UNIFORM = LayeredMedium2D(layer=ElasticMaterial(lam=1.0, mu=1.0, rho=1.0))


def test_sparsity_and_constant_annihilation() -> None:
    nodes = make_node_set(LayeredMedium2D(), 400, repulsion_steps=10)
    ops = build_operators(nodes, LayeredMedium2D(), stencil_size=30)
    for mat in (ops.dx, ops.dy, ops.hyper):
        assert mat.shape == (400, 400)
        assert np.all(np.diff(mat.indptr) == 30)
        # Exactness on the constant polynomial: every row sums to zero.
        np.testing.assert_allclose(mat @ np.ones(400), 0.0, atol=1e-6 * abs(mat).max())
    assert ops.elastic.shape == (2000, 2000)
    assert ops.hyper_block.shape == (2000, 2000)


def test_elastic_block_matches_the_equations() -> None:
    # Random derivative matrices and coefficients: check the block wiring
    # against a hand-written right-hand side of eq. 32.
    rng = np.random.default_rng(1)
    n = 12
    dx = sp.csr_array(rng.normal(size=(n, n)))
    dy = sp.csr_array(rng.normal(size=(n, n)))
    lam, mu, rho = rng.uniform(1, 3, n), rng.uniform(1, 3, n), rng.uniform(1, 3, n)
    u, v, f, g, h = rng.normal(size=(5, n))
    expected = np.concatenate(
        [
            (dx @ f + dy @ g) / rho,
            (dx @ g + dy @ h) / rho,
            (lam + 2 * mu) * (dx @ u) + lam * (dy @ v),
            mu * (dy @ u + dx @ v),
            lam * (dx @ u) + (lam + 2 * mu) * (dy @ v),
        ]
    )
    out = elastic_block(dx, dy, lam, mu, rho) @ np.concatenate([u, v, f, g, h])
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_plane_wave_time_derivative_in_uniform_medium() -> None:
    # v = V(y), h = sqrt(3) V, f = V / sqrt(3): the exact rates are
    # v_t = sqrt(3) V', h_t = 3 V', f_t = V', u_t = g_t = 0.
    nodes = make_node_set(UNIFORM, 2500, repulsion_steps=40)
    ops = build_operators(nodes, UNIFORM)
    sharp = 8.0
    state = plane_p_wave(nodes, UNIFORM, sharpness=sharp)
    rate = (ops.elastic @ state.ravel()).reshape(5, -1)
    # The pulse tail wraps through y = 1 -> 0, so measure y - 0.75 periodically.
    dy = minimal_image(nodes.y - 0.75)
    dv = -2 * sharp**2 * dy * np.exp(-(sharp**2) * dy**2)
    scale = np.abs(dv).max()
    np.testing.assert_allclose(rate[1], np.sqrt(3) * dv, atol=2e-3 * scale)
    np.testing.assert_allclose(rate[4], 3 * dv, atol=5e-3 * scale)
    np.testing.assert_allclose(rate[2], dv, atol=2e-3 * scale)
    np.testing.assert_allclose(rate[0], 0.0, atol=2e-3 * scale)
    np.testing.assert_allclose(rate[3], 0.0, atol=2e-3 * scale)


def test_coefficients_follow_the_layer() -> None:
    medium = LayeredMedium2D()  # lam + 2 mu: 12 in the band, 3 outside
    nodes = make_node_set(medium, 1600, repulsion_steps=40)
    ops = build_operators(nodes, medium)
    state = np.zeros((5, nodes.n))
    state[1] = np.sin(2 * np.pi * nodes.y)
    h_t = (ops.elastic @ state.ravel()).reshape(5, -1)[4]
    exact = 2 * np.pi * np.cos(2 * np.pi * nodes.y)
    far = medium.distance_to_interfaces(nodes.x, nodes.y) > 4 * nodes.h
    inside = medium.in_layer(nodes.x, nodes.y)
    np.testing.assert_allclose(h_t[far & inside], 12 * exact[far & inside], atol=0.05)
    np.testing.assert_allclose(h_t[far & ~inside], 3 * exact[far & ~inside], atol=0.02)


def test_hyperviscosity_moves_spectrum_into_rk4_region() -> None:
    # Dissertation Fig. 3-2 in numbers, on a small node set.
    medium = LayeredMedium2D()
    nodes = make_node_set(medium, 400, repulsion_steps=40)
    ops = build_operators(nodes, medium)
    gamma = hyperviscosity_gamma(nodes.h, ops.hyper_power)
    dense = ops.elastic.toarray()
    ev_plain = np.linalg.eigvals(dense)
    ev_hyper = np.linalg.eigvals(dense + gamma * ops.hyper_block.toarray())
    assert ev_plain.real.max() > 5.0
    assert ev_hyper.real.max() < 0.5
    assert ev_hyper.real.max() < ev_plain.real.max() / 20
    dt = 0.5 * nodes.h / medium.c_max
    z = ev_hyper * dt
    amp = np.abs(1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24)
    assert amp.max() < 1.005, amp.max()
    # The hyperviscosity operator alone is (numerically) negative semi-definite.
    ev_h = np.linalg.eigvals(ops.hyper.toarray())
    assert ev_h.real.max() < 1e-3 * abs(ev_h.real.min())
    assert ev_h.real.min() < -1e6


def test_gamma_scaling_and_modes() -> None:
    assert hyperviscosity_gamma(0.02) == pytest.approx(2.4e-11)
    assert hyperviscosity_gamma(0.01) == pytest.approx(2.4e-11 / 32)
    assert hyperviscosity_gamma(0.01, hyper_power=2) == pytest.approx(2.4e-11 / 8)
    nodes = make_node_set(LayeredMedium2D(), 400, repulsion_steps=5)
    with pytest.raises(NotImplementedError):
        build_operators(nodes, LayeredMedium2D(), mode="aware")
    with pytest.raises(ValueError):
        build_operators(nodes, LayeredMedium2D(), mode="bogus")  # type: ignore[arg-type]
