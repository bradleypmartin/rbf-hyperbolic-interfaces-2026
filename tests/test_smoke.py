"""Smoke test: the package imports and the numerical stack is present."""

import numpy as np
import scipy

import rbf_hyperbolic_interfaces


def test_imports() -> None:
    assert rbf_hyperbolic_interfaces.__name__ == "rbf_hyperbolic_interfaces"
    assert np.__version__
    assert scipy.__version__
