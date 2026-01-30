"""SPEEDY physics utilities for coordinate system creation.

Date: 2026-01-26
"""
from dinosaur.coordinate_systems import CoordinateSystem
from jcm.physics.speedy.physical_constants import SIGMA_LAYER_BOUNDARIES
from jcm.utils import get_coords


def get_speedy_coords(layers=8, spectral_truncation=31, nodal_shape=None, spmd_mesh=None) -> CoordinateSystem:
    """Create a CoordinateSystem with SPEEDY's standard sigma layers.

    This is a convenience wrapper around jcm.utils.get_coords() that uses
    SPEEDY's standard sigma layer boundaries.

    Args:
        layers: Number of vertical levels (7 or 8)
        spectral_truncation: Spectral truncation number (default 31)
        nodal_shape: Optional nodal shape (ix, il) to infer spectral_truncation
        spmd_mesh: Optional SPMD mesh for parallelization

    Returns:
        CoordinateSystem object with SPEEDY sigma levels
    """
    if layers not in SIGMA_LAYER_BOUNDARIES:
        raise ValueError(f"SPEEDY physics supports {list(SIGMA_LAYER_BOUNDARIES.keys())} layers, got {layers}")

    return get_coords(
        sigma_boundaries=SIGMA_LAYER_BOUNDARIES[layers],
        spectral_truncation=spectral_truncation,
        nodal_shape=nodal_shape,
        spmd_mesh=spmd_mesh
    )
