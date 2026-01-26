import jax.numpy as jnp
import tree_math
from dinosaur.coordinate_systems import CoordinateSystem
from jcm.constants import p0, grav, cp
from jcm.physics.speedy.physical_constants import SIGMA_LAYER_BOUNDARIES


@tree_math.struct
class SpeedyCoords:
    """SPEEDY-specific coordinate system data.

    This struct caches precomputed coordinate transformations needed by SPEEDY physics.
    All fields are constant during a simulation.

    Attributes:
        # Vertical coordinate data
        hsg: Sigma layer boundaries (kx+1,)
        fsg: Sigma layer midpoints (kx,)
        dhs: Sigma layer thicknesses (kx,)
        sigl: Log of sigma layer midpoints (kx,)
        grdsig: g/(d_sigma * p0) - converts fluxes to tendencies (kx,)
        grdscp: g/(d_sigma * p0 * c_p) - converts energy fluxes to temperature tendencies (kx,)
        wvi: Weights for vertical interpolation (kx, 2)

        # Horizontal coordinate data
        radang: Latitude in radians (il,)
        sia: Sin of latitude (il,)
        coa: Cos of latitude (il,)
    """
    
    # Vertical
    hsg: jnp.ndarray
    fsg: jnp.ndarray
    dhs: jnp.ndarray
    sigl: jnp.ndarray
    grdsig: jnp.ndarray
    grdscp: jnp.ndarray
    wvi: jnp.ndarray

    # Horizontal
    radang: jnp.ndarray
    sia: jnp.ndarray
    coa: jnp.ndarray


def compute_vertical_coords(kx: int):
    """Compute SPEEDY vertical coordinate transformations.

    Args:
        kx: Number of vertical levels

    Returns:
        Tuple of (hsg, fsg, dhs, sigl, grdsig, grdscp, wvi)

    Raises:
        ValueError: If kx is not a supported number of vertical levels
    """
    if kx not in SIGMA_LAYER_BOUNDARIES:
        raise ValueError(f"Invalid number of vertical levels: {kx}. Must be one of: {tuple(SIGMA_LAYER_BOUNDARIES.keys())}")

    # Layer boundaries and midpoints
    hsg = SIGMA_LAYER_BOUNDARIES[kx]
    fsg = (hsg[1:] + hsg[:-1]) / 2.
    dhs = jnp.diff(hsg)
    sigl = jnp.log(fsg)

    # Conversion factors for fluxes -> tendencies
    grdsig = grav / (dhs * p0)
    grdscp = grdsig / cp

    # Weights for vertical interpolation at half-levels(1,kx) and surface
    # Note that for phys.par. half-lev(k) is between full-lev k and k+1
    # Fhalf(k) = Ffull(k) + WVI(K,2) * (Ffull(k+1) - Ffull(k))
    # Fsurf = Ffull(kx) + WVI(kx,2) * (Ffull(kx) - Ffull(kx-1))
    wvi = jnp.zeros((kx, 2))
    wvi = wvi.at[:-1, 0].set(1. / jnp.diff(sigl))
    wvi = wvi.at[:-1, 1].set((jnp.log(hsg[1:-1]) - sigl[:-1]) * wvi[:-1, 0])
    wvi = wvi.at[-1, 1].set((jnp.log(0.99) - sigl[-1]) * wvi[-2, 0])

    return hsg, fsg, dhs, sigl, grdsig, grdscp, wvi


def speedy_coords_from_coordinate_system(coords: CoordinateSystem) -> SpeedyCoords:
    """Create SpeedyCoords from a dinosaur CoordinateSystem.

    This function extracts and transforms coordinate data from a CoordinateSystem
    into the specific form needed by SPEEDY physics parameterizations.

    Args:
        coords: dinosaur.coordinate_systems.CoordinateSystem object

    Returns:
        SpeedyCoords struct containing all coordinate transformations
    """
    # Compute vertical coordinates
    kx = coords.nodal_shape[0]
    hsg, fsg, dhs, sigl, grdsig, grdscp, wvi = compute_vertical_coords(kx)

    # Compute horizontal coordinates
    radang = coords.horizontal.latitudes
    sia = jnp.sin(radang)
    coa = jnp.cos(radang)

    return SpeedyCoords(
        hsg=hsg,
        fsg=fsg,
        dhs=dhs,
        sigl=sigl,
        grdsig=grdsig,
        grdscp=grdscp,
        wvi=wvi,
        radang=radang,
        sia=sia,
        coa=coa
    )
