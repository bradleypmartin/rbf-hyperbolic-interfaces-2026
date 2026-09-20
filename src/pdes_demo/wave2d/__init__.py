"""2-D elastic wave equation with curved interfaces on scattered nodes (ch. 3).

Same split as ``wave1d``:

- ``domain``: materials, interfaces, interface-fitted node sets, initial data
- ``neighbors``: periodic nearest-neighbour search
- ``rbf``: Gaussian RBF-FD stencil weights with polynomial augmentation
- ``operators``: sparse derivative, hyperviscosity and block elastic operators
- ``simulate``: RK4 time stepping
- ``exact``: references for the flat-interface plane-wave problem: the ray
  sum for a jump, the 1-D spectral solver for smooth edges
- ``spectral``: Fourier-in-x reference for oblique incidence on flat smooth
  edges (Part 3)
- ``resample``: one-sided interpolation to pixel grids and other node sets
- ``seeds``: ODE-continued elastic seed bases for stiff smooth edges (Part 3)
"""

from .domain import (
    FIELDS,
    ElasticMaterial,
    LayeredMedium2D,
    NodeSet,
    SineInterface,
    make_node_set,
    nearest_spacing,
    oblique_p_wave,
    plane_p_wave,
    straddling_rows,
)
from .exact import exact_plane_wave, spectral_plane_wave
from .neighbors import minimal_image, periodic_knn, stencil_offsets, wrap
from .operators import Operators, build_operators, hyperviscosity_gamma
from .resample import pixel_grid, resample_matrix
from .simulate import Snapshots2D, energy, run, stable_dt
from .spectral import ModeState, run_fourier

__all__ = [
    "FIELDS",
    "ElasticMaterial",
    "LayeredMedium2D",
    "ModeState",
    "NodeSet",
    "Operators",
    "SineInterface",
    "Snapshots2D",
    "build_operators",
    "energy",
    "exact_plane_wave",
    "hyperviscosity_gamma",
    "make_node_set",
    "minimal_image",
    "nearest_spacing",
    "oblique_p_wave",
    "periodic_knn",
    "pixel_grid",
    "plane_p_wave",
    "resample_matrix",
    "run",
    "run_fourier",
    "spectral_plane_wave",
    "stable_dt",
    "stencil_offsets",
    "straddling_rows",
    "wrap",
]
