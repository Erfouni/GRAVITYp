"""This module contains all the functions needed for the geodesic integration
routine
"""

import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
from bisect import bisect_right

from .compact_object import BaseCompactObject


__all__ = [
    "find_interval",
    "find_rmin",
    "compute_outgoing",
    "compute_geodesic",
]


DEFAULT_NPOINTS = 100000  # ODE integration resolution
DEFAULT_R0 = 1000.0  # Asymptotic boundary for initial conditions
MINIMUM_R = 1e-6  # Minimum value of r for integration


def find_interval(inverse_b2, photon_sph, local_maxima, lower_bound=1e-4, upper_bound=100):
    """Returns the bounds (r1,r2) where the rmin is found for a given 1/b^2"""
    # NOTE: photon_sph and local_maxima have to be ordered by co.potential_maxima
    if len(photon_sph) == 0 and len(local_maxima) == 0:  # Case 1: there is no local maximum
        return (lower_bound, upper_bound)
    # Case 2: 1/b^2 higher than all local maxima
    if inverse_b2 >= local_maxima[0]:
        return (lower_bound, photon_sph[0])
    # Case 3: 1/b^2 lower than all local maxima
    if inverse_b2 <= local_maxima[-1]:
        return (photon_sph[-1], upper_bound)

    # Case 4: there are more than 1 local maximum and 1/b^2 lies between two of them
    # We negate both the array elements and 1/b^2 so the logic mirrors an increasing array
    idx = bisect_right(local_maxima, -inverse_b2, key=lambda x: -x)
    return (photon_sph[idx - 1], photon_sph[idx])


def find_rmin(b, co: BaseCompactObject):
    """
    Finds the minimum radius of the trajectory using Brent's method. 

    Parameters:
        b (float): impact parameter of the photon trajectory
        co (BaseCompactObject): generic compact object defining the geometry

    Returns:
        i)  If b > bmin, then rmin is found by solving V(r)-1/b^2 = 0
        ii) If b < bmin, then rmin = inner_edge.
    """
    photon_sph, local_maxima = co.potential_maxima
    # Interval for Brent's method
    bounds = find_interval(1/b**2, photon_sph, local_maxima)

    def fun(r):
        return co.potential(r)-1/b**2
    if fun(bounds[0])*fun(bounds[1]) >= 0:  # No root if there is no sign change
        root = 0
    else:
        root, _ = optimize.brentq(fun, *bounds, full_output=True)
    return max(max(root, co.inner_edge), MINIMUM_R)


def dphi_dr(phi, r, b, co: BaseCompactObject, incoming: bool = True):
    """Expression for dphi/dr in the null geodesic equation"""
    R = co.areal(r)  # Store value for single function call
    sign = -2*int(incoming)+1
    return sign*(b / np.sqrt(R))/np.sqrt(R - b**2*co.radial(r))


def _integrator(phi0, rr, b, co: BaseCompactObject, incoming: bool):
    phi = integrate.odeint(dphi_dr, phi0, rr, (b, co, incoming)).reshape(-1,)
    finite = np.isfinite(phi)  # Finite values mask
    return rr[finite], phi[finite]


def compute_outgoing(r_phs, inner_edge, rmin) -> bool:
    """Returns a boolean that tells whether to compute outgoing geodesic after reaching rmin"""
    compute = False
    if (len(r_phs) > 0):  # There is at least 1 photon sphere
        innermost_photon_sphere = np.array(r_phs).min()
        if (innermost_photon_sphere < MINIMUM_R):
            if (rmin > MINIMUM_R):
                compute = True
        # if (rmin > inner_edge) and (innermost_photon_sphere > 0): # Probamos con inner_edge mejor
        else:
            # print('Hola')
            if (rmin > inner_edge) and (rmin > MINIMUM_R):  # Probamos con inner_edge mejor
                compute = True
    else:  # There is no photon sphere
        compute = True
    return compute


def compute_geodesic(b, co: BaseCompactObject,
                     Npoints=DEFAULT_NPOINTS, r0=DEFAULT_R0):
    """
    Computes null geodesics for a spacetime defined by a compact object

    Integrates the geodesic equations to trace light ray paths. For incoming rays,
    integrates from r0 down to rmin. For rays escaping to infinity (rmin > r_ph),
    also integrates the outgoing trajectory from rmin back to r0.

    Parameters:
        b (float): Impact parameter in units of M=1
        co (BaseCompactObject): generic compact object defining the geometry
        Npoints (int): Number of radial points for integration. Default: DEFAULT_NPOINTS
        r0 (float): Starting radius. Default: DEFAULT_R0

    Returns:
        tuple[np.ndarray, np.ndarray]: (rr, phi) where rr is the radial coordinate 
            array and phi is the azimuthal angle array along the geodesic
    """
    rmin = find_rmin(b, co)
    # We must ensure rmin > 0 in find_rmin for np.geomspace
    rr_in = np.geomspace(r0, rmin, Npoints)

    # phi0 = b/r0 is a good approximation for r0>>inner_edge
    rr, phi = _integrator(b/r0, rr_in, b, co, incoming=True)

    if compute_outgoing(co.r_phs, co.inner_edge, rmin):
        # Linear approx. for next angle. Required for strict monotonicity!!
        phi0 = 2*phi[-1]-phi[-2]
        rr_out, phi_out = _integrator(phi0, rr[::-1], b, co, incoming=False)
        rr = np.concatenate([rr, rr_out])
        phi = np.concatenate([phi, phi_out])

    return (rr, phi)


if __name__ == "__main__":
    from .compact_object import CompactObject

    def g_rr(r: float | np.ndarray) -> float | np.ndarray:
        """Radial function g^{rr} of the Schwarzschild metric in units of M=1"""
        return 1 - 2./r

    def g_thth(r: float | np.ndarray) -> float | np.ndarray:
        """Areal radius squared g^{theta theta} of the Schwarzschild in units of M=1"""
        return r**2

    schwarzschild = CompactObject(
        r_phs=[3.,],        # Photon sphere at r=3M
        inner_edge=2.,      # Inner edge of disk at horizon r=2M
        radial=g_rr,        # Above defined radial function
        areal=g_thth,       # Above defined areal radius squared
    )

    rr, phi = compute_geodesic(3, schwarzschild)
    print(rr, phi)
