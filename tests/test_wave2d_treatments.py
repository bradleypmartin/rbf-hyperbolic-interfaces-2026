"""Coefficient treatments on scattered nodes (issue #69, T2)."""

import numpy as np
import pytest
from scipy.integrate import quad

from pdes_demo.wave2d import LayeredMedium2D, SineInterface, build_operators
from pdes_demo.wave2d.treatments import TreatedMedium2D

H = 0.02


def _flat(delta: float) -> LayeredMedium2D:
    return LayeredMedium2D(edge_width=delta)


def _profile_means(medium: LayeredMedium2D, y: float, width: float):
    """Reference cell means along y of a flat medium (its coefficients depend
    on y alone): harmonic K and mu, arithmetic rho, by adaptive quadrature."""

    def at(yy, f):
        lam, mu, rho = medium.material_at(np.array([0.3]), np.array([yy]))
        return f(lam[0], mu[0], rho[0])

    lo, hi = y - width / 2, y + width / 2
    inv_k = quad(lambda t: at(t, lambda la, m, r: 1 / (la + 2 * m)), lo, hi)[0] / width
    inv_mu = quad(lambda t: at(t, lambda la, m, r: 1 / m), lo, hi)[0] / width
    rho = quad(lambda t: at(t, lambda la, m, r: r), lo, hi)[0] / width
    return 1 / inv_k, 1 / inv_mu, rho


@pytest.mark.parametrize("delta", [H / 8, H / 2])
@pytest.mark.parametrize("cells", [1.0, 2.0])
def test_box_kernel_on_a_flat_edge_is_the_cell_mean_along_y(delta, cells) -> None:
    medium = _flat(delta)
    treated = TreatedMedium2D(medium, cells * H, kernel="box")
    ys = medium.lower.y0 + H * np.array([-1.0, -0.5, -0.1, 0.2, 0.7, 1.5])
    xs = np.linspace(0.1, 0.9, ys.size)
    lam, mu, rho = treated.material_at(xs, ys)
    for i, y in enumerate(ys):
        k_ref, mu_ref, rho_ref = _profile_means(medium, y, cells * H)
        assert lam[i] + 2 * mu[i] == pytest.approx(k_ref, rel=1e-5)
        assert mu[i] == pytest.approx(mu_ref, rel=1e-5)
        assert rho[i] == pytest.approx(rho_ref, rel=1e-5)


@pytest.mark.parametrize("kernel", ["box", "sinc"])
def test_far_from_the_edges_the_treated_medium_is_the_true_one(kernel) -> None:
    medium = _flat(0.0025)
    treated = TreatedMedium2D(medium, H, kernel=kernel)
    x = np.array([0.1, 0.5, 0.9, 0.3])
    y = np.array([0.05, 0.375, 0.8, 0.95])
    for got, true in zip(
        treated.material_at(x, y), medium.material_at(x, y), strict=True
    ):
        np.testing.assert_allclose(got, true, rtol=1e-13)


def test_treated_moduli_lie_between_the_materials_and_x_does_not_matter() -> None:
    medium = _flat(0.005)
    treated = TreatedMedium2D(medium, H, kernel="box")
    y = np.linspace(0.2, 0.55, 40)
    lam, mu, rho = treated.material_at(np.full_like(y, 0.2), y)
    lam2, mu2, rho2 = treated.material_at(np.full_like(y, 0.7), y)
    np.testing.assert_allclose(lam, lam2, rtol=1e-12)
    bg, ly = medium.background, medium.layer
    k = lam + 2 * mu
    assert np.all(k >= min(bg.lam + 2 * bg.mu, ly.lam + 2 * ly.mu) - 1e-12)
    assert np.all(k <= max(bg.lam + 2 * bg.mu, ly.lam + 2 * ly.mu) + 1e-12)
    assert np.all(mu >= min(bg.mu, ly.mu) - 1e-12) and np.all(mu <= max(bg.mu, ly.mu))
    assert np.all(rho >= min(bg.rho, ly.rho) - 1e-12)
    assert np.all(rho <= max(bg.rho, ly.rho) + 1e-12)


def test_sinc_kernel_is_symmetric_about_a_flat_edge_and_overshoots() -> None:
    medium = _flat(0.0025)
    treated = TreatedMedium2D(medium, H, kernel="sinc")
    z = H * np.linspace(0.05, 2.4, 12)
    y0 = medium.lower.y0
    x = np.full_like(z, 0.4)
    below = treated.rho_at(x, y0 - z)
    above = treated.rho_at(x, y0 + z)
    np.testing.assert_allclose(
        below + above, medium.background.rho + medium.layer.rho, rtol=1e-10
    )
    fine = y0 + H * np.linspace(-4, 4, 161)
    rho = treated.rho_at(np.full_like(fine, 0.4), fine)
    assert rho.max() > medium.layer.rho + 0.03
    assert rho.min() < medium.background.rho - 0.03
    assert treated.c_max > medium.c_max
    assert TreatedMedium2D(medium, H, kernel="box").c_max == medium.c_max


def test_curved_edge_average_follows_the_curve() -> None:
    """On the curved geometry the treated coefficient depends on x through
    the interface height: the cell mean at a fixed offset from the curve is
    the same at every x."""
    medium = LayeredMedium2D(
        lower=SineInterface(0.25, 0.02),
        upper=SineInterface(0.5, 0.02),
        edge_width=0.005,
    )
    treated = TreatedMedium2D(medium, H, kernel="box")
    x = np.linspace(0.0, 1.0, 9, endpoint=False)
    y = medium.lower.height(x) + 0.3 * H
    lam, mu, rho = treated.material_at(x, y)
    # Same normal offset up to the tilt (at most 7 degrees): equal to 2%.
    assert np.ptp(rho) < 0.02 * np.abs(rho).mean()
    lam_off, _, _ = treated.material_at(x, y + 3 * H)
    assert np.all(lam_off > lam)  # inside the band the moduli are larger


def test_treated_medium_builds_the_naive_operator() -> None:
    from pdes_demo.wave2d import make_node_set, run

    medium = _flat(0.01)
    nodes = make_node_set(medium, 400, seed=0)
    treated = TreatedMedium2D(medium, nodes.h, kernel="box")
    ops = build_operators(nodes, treated, mode="naive")
    snaps = run(nodes, treated, t_end=0.05, n_snapshots=1, operators=ops)
    assert np.all(np.isfinite(snaps.state[-1]))


def test_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="positive"):
        TreatedMedium2D(_flat(0.01), 0.0)
    with pytest.raises(ValueError, match="kernel"):
        TreatedMedium2D(_flat(0.01), H, kernel="gauss")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="quarter period"):
        TreatedMedium2D(_flat(0.01), 0.2, kernel="sinc")
