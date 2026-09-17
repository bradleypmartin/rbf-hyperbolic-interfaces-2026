"""Smoke test: the package imports and the numerical stack is present."""

import numpy as np
import scipy

import pdes_demo


def test_imports() -> None:
    assert pdes_demo.__name__ == "pdes_demo"
    assert np.__version__
    assert scipy.__version__
