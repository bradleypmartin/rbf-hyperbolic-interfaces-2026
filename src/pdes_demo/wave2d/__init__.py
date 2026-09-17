"""2-D elastic wave equation with curved interfaces on scattered nodes (ch. 3).

Same split as ``wave1d``:

- ``domain``: materials, interfaces, interface-fitted node sets, initial data
- ``neighbors``: periodic nearest-neighbour search
- ``rbf`` / ``operators`` / ``simulate`` / ``exact``: later milestones
"""

from .domain import (
    FIELDS,
    ElasticMaterial,
    LayeredMedium2D,
    NodeSet,
    SineInterface,
    make_node_set,
    nearest_spacing,
    plane_p_wave,
    straddling_rows,
)
from .neighbors import minimal_image, periodic_knn, stencil_offsets, wrap

__all__ = [
    "FIELDS",
    "ElasticMaterial",
    "LayeredMedium2D",
    "NodeSet",
    "SineInterface",
    "make_node_set",
    "minimal_image",
    "nearest_spacing",
    "periodic_knn",
    "plane_p_wave",
    "stencil_offsets",
    "straddling_rows",
    "wrap",
]
