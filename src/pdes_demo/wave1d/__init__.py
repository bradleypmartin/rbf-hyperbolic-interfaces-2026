"""1-D two-way wave equation with a heterogeneous layer (dissertation ch. 2).

Modules mirror the split we will reuse in 2-D:

- ``domain``: periodic grid, materials, interfaces, initial pulse
- ``operators``: spatial differentiation matrices (naive vs interface-aware)
- ``simulate``: RK4 time stepping
- ``exact``: closed-form reference solution (ray sum) for verification
"""

from .domain import Grid1D, LayeredMedium, Material, periodic_grid, right_going_pulse
from .exact import exact_solution
from .operators import build_operators
from .simulate import Snapshots, run

__all__ = [
    "Grid1D",
    "LayeredMedium",
    "Material",
    "Snapshots",
    "build_operators",
    "exact_solution",
    "periodic_grid",
    "right_going_pulse",
    "run",
]
