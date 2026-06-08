"""
scripts.mesh — regime-aware structured C-grid meshing for NACASurrogate.

Modules
-------
regime_parameters : per-regime parameter dict + classifier
geometry          : pure-Python aerofoil surface + farfield point placement
boundary_layer    : y+ / first-cell-height / progression-ratio math
topology          : gmsh transfinite C+H block construction
bl_field          : gmsh BoundaryLayer-field + frontal-quad mesh (Regime B)
wake_refinement   : wake block grading helpers
quality           : checkMesh parsing, OpenFOAM glue, metadata writer
"""

from .regime_parameters import REGIME_MESH, classify_regime
from .boundary_layer import first_cell_height, bl_thickness, solve_progression
from .geometry import naca_symmetric, farfield_points, load_polygon
from .topology import build_c_grid
from .bl_field import build_bl_mesh
from .quality import (
    parse_check_mesh,
    validate_quality,
    rewrite_boundary_types,
    run_openfoam_command,
    write_metadata,
)

__all__ = [
    "REGIME_MESH",
    "classify_regime",
    "first_cell_height",
    "bl_thickness",
    "solve_progression",
    "naca_symmetric",
    "farfield_points",
    "load_polygon",
    "build_c_grid",
    "build_bl_mesh",
    "parse_check_mesh",
    "validate_quality",
    "rewrite_boundary_types",
    "run_openfoam_command",
    "write_metadata",
]
