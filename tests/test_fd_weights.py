import numpy as np
import pytest

from pdes_demo.fd_weights import fornberg_weights


def test_centred_fd4_first_derivative() -> None:
    w = fornberg_weights(0.0, np.arange(-2, 3), 1)
    np.testing.assert_allclose(w[1] * 12, [1, -8, 0, 8, -1], atol=1e-13)
    np.testing.assert_allclose(w[0], [0, 0, 1, 0, 0], atol=1e-13)


def test_second_derivative_and_scaling() -> None:
    h = 0.1
    w = fornberg_weights(0.0, h * np.arange(-1, 2), 2)
    np.testing.assert_allclose(w[2] * h**2, [1, -2, 1], atol=1e-12)


@pytest.mark.parametrize("m", [1, 2, 3])
def test_exact_on_polynomials_up_to_degree_n_minus_1(m: int) -> None:
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(-1, 1, 6))
    z = 0.3
    w = fornberg_weights(z, x, m)
    for deg in range(x.size):
        coef = rng.normal(size=deg + 1)
        poly = np.polynomial.Polynomial(coef)
        approx = w[m] @ poly(x)
        np.testing.assert_allclose(approx, poly.deriv(m)(z), atol=1e-9)
