"""This module contains all the functions needed for the geodesic integration
routine
"""

import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
from bisect import bisect_right

from .utils import to_tuple


__all__ = [
    "potential",
    "get_increasing_maxima",
    "find_interval",
    "find_rmin",
    "geodesic_in",
    "geodesic_out",
    "compute_outgoing",
    "compute_geodesic",
]

DEFAULT_NPOINTS = 100000  # ODE integration resolution
DEFAULT_R0 = 1000.0  # Asymptotic boundary for initial conditions
MINIMUM_R = 1e-6 # Minimum value of r for integration


def potential(r, radial_fun, radial_params, areal, areal_params):
    """ 
    Effective potential of a spherically symmetric metric of the form:

    ds^2 = - radial_fun(r; radial_params)dt^2 + 1/radial_fun(r; radial_params)dr^2 + areal(r; areal_params)dOmega^2

    Parameters:
        r (float): radial coordinate of the metric
        radial_fun (callable): -gtt metric function. Its first argument must be r
        radial_params (float | tuple): additional parameters for radial_fun
        areal (callable): areal radius squared.  Its first argument must be r
        areal_params (float | tuple): additional parameters for areal
    """
    return radial_fun(r,*to_tuple(radial_params)) / areal(r,*to_tuple(areal_params))
    
def get_increasing_maxima(r_phs, potential):
    """Returns increasing local maxima of the potential with their photon sphere values"""
    if len(r_phs) >= 1: # There has to be at least 1 local maximum
        r_phs = np.array(r_phs)
        r_phs = np.sort(r_phs)[::-1] # descending order of 
        photon_sph = r_phs[0].reshape(1,)
        local_maxima = potential(photon_sph)
        for r_ph in r_phs:
            maximum = potential(r_ph)
            if maximum > local_maxima.max(): # We take increasing local maxima
                local_maxima = np.append(local_maxima, maximum)
                photon_sph = np.append(photon_sph, r_ph)
        return photon_sph[::-1], local_maxima[::-1] # Swap to ascending order of photon_spheres
    else:
        return np.array([]), np.array([])

def find_interval(inverse_b2, photon_sph, local_maxima, lower_bound=1e-4, upper_bound=100):
    """Returns the bounds (r1,r2) where the rmin is found for a given 1/b^2"""
    # NOTE: photon_sph and local_maxima have to be ordered by get_increasing_maxima
    if len(photon_sph)==0 and len(local_maxima)==0: # Case 1: there is no local maximum
        return (lower_bound, upper_bound)
    if inverse_b2 >= local_maxima[0]: # Case 2: 1/b^2 higher than all local maxima
        return (lower_bound, photon_sph[0])
    if inverse_b2 <= local_maxima[-1]: # Case 3: 1/b^2 lower than all local maxima
        return (photon_sph[-1], upper_bound)

    # Case 4: there are more than 1 local maximum and 1/b^2 lies between two of them
    # We negate both the array elements and 1/b^2 so the logic mirrors an increasing array
    idx = bisect_right(local_maxima, -inverse_b2, key=lambda x: -x)
    return (photon_sph[idx - 1], photon_sph[idx])

def find_rmin(b, r_phs, inner_edge, radial_fun, radial_params, areal, areal_params):
    """
    Finds the minimum radius of the trajectory using Brent's method. 

    Parameters:
        b (float): impact parameter of the photon trajectory
        r_phs (array): array with the position of photon spheres (maxima) of the potential
        inner_edge (float): value of the r coordinate of the inner edge (event horizon, throat, etc.)
        potential (callable): effective potential

    Returns:
        i)  If b > bmin, then rmin is found by solving V(r)-1/b^2 = 0
        ii) If b < bmin, then rmin = inner_edge.
    """
    pot = lambda r: potential(r, radial_fun, radial_params, areal, areal_params) # V(r)
    photon_sph, local_maxima = get_increasing_maxima(r_phs, pot)
    bounds = find_interval(1/b**2, photon_sph, local_maxima) # Interval for Brent's method
    fun = lambda r: pot(r)-1/b**2
    if fun(bounds[0])*fun(bounds[1]) >= 0: # No root if there is no sign change
        root = 0
    else:
        root, results = optimize.brentq(fun, *bounds, full_output=True)
    return max( max(root, inner_edge) , MINIMUM_R )

def geodesic_in(phi, r, b, radial_fun, radial_params, areal, areal_params):
    """Expression for dphi/dr in the incoming null geodesic equation"""
    areal_params = to_tuple(areal_params)
    radial_params = to_tuple(radial_params)
    return -( b / np.sqrt(areal(r,*areal_params)) )/np.sqrt( areal(r,*areal_params) - b**2*radial_fun(r,*radial_params) )

def geodesic_out(phi, r, b, radial_fun, radial_params, areal, areal_params):
    """Expression for dphi/dr in the outgoing null geodesic equation"""
    areal_params = to_tuple(areal_params)
    radial_params = to_tuple(radial_params)
    return +( b / np.sqrt(areal(r,*areal_params)) )/np.sqrt( areal(r,*areal_params) - b**2*radial_fun(r,*radial_params) )

def compute_outgoing(r_phs, inner_edge, rmin) -> bool:
    """Returns a boolean that tells whether to compute outgoing geodesic after reaching rmin"""
    compute = False
    if (len(r_phs)>0): # There is at least 1 photon sphere
        innermost_photon_sphere = np.array(r_phs).min()
        if (innermost_photon_sphere < MINIMUM_R):
            if (rmin > MINIMUM_R):
                compute = True
        # if (rmin > inner_edge) and (innermost_photon_sphere > 0): # Probamos con inner_edge mejor
        else:
            # print('Hola')
            if (rmin > inner_edge) and (rmin > MINIMUM_R): # Probamos con inner_edge mejor
                compute = True
    else: # There is no photon sphere
        compute = True
    return compute

def compute_geodesic(b, r_phs, inner_edge, radial_fun, radial_params, areal, areal_params,
                     Npoints=DEFAULT_NPOINTS, r0=DEFAULT_R0):
    """
    Computes the null geodesic trajectory in Schwarzschild spacetime.
    
    Integrates the geodesic equations to trace light ray paths. For incoming rays,
    integrates from r0 down to rmin. For rays escaping to infinity (rmin > 3),
    also integrates the outgoing trajectory from rmin back to r0.

    Parameters:
        b (float): Impact parameter in units of M=1
        Npoints (int): Number of radial points for integration. Default: DEFAULT_NPOINTS
        r0 (float): Starting radius. Default: DEFAULT_R0
    
    Returns:
        tuple[np.ndarray, np.ndarray]: (rr, phi) where rr is the radial coordinate 
            array and phi is the azimuthal angle array along the geodesic
    """ 
    rmin = find_rmin(b, r_phs, inner_edge, radial_fun, radial_params, areal, areal_params)
    rr = np.geomspace(r0, rmin, Npoints) # We must ensure rmin > 0 in find_rmin for np.geomspace

    phi0 = b/rr[0] # good approximation for r0>>rSch and better than phi0=0
    phi = integrate.odeint(
        geodesic_in, phi0, rr, (b,radial_fun,radial_params,areal,areal_params)
    ).reshape(-1,)
    phi = phi[np.isfinite(phi)]
    rr = rr[:len(phi)]

    # NOTE: does not work properly with high r0 and low Npoints for outgoing geodesics
    # FIXME: si len(r_phs)=0, solo integra incoming.
    if compute_outgoing(r_phs, inner_edge, rmin):
        rout = rr[::-1]
        phi0 = 2*phi[-1]-phi[-2] # Linear approx. for next angle. Required for strict monotonicity!!
        phi_out = integrate.odeint(
            geodesic_out, phi0, rout, (b,radial_fun,radial_params,areal,areal_params)
        ).reshape(-1,)
        phi_out = phi_out[np.isfinite(phi_out)]
        rout = rout[:len(phi_out)]
        rr = np.concatenate([rr,rout])
        phi = np.concatenate([phi,phi_out])

    return (rr,phi)


if __name__=="__main__":

    def g_rr(r: float, dummy) -> float:
        """Radial function g^{rr} of the Schwarzschild metric in units of M=1"""
        return 1 - 2./r

    def g_thth(r: float, dummy) -> float:
        """Areal radius squared g^{theta theta} of the Schwarzschild in units of M=1"""
        return r**2

    # Dictionary codifying Schwarzschild metric and required params
    Schwarzschild_kwargs = {
        'r_phs': [3.,],         # Photon sphere at r=3M
        'inner_edge': 2.,       # Inner edge of disk at horizon r=2M
        'radial_fun': g_rr,     # Above defined radial function
        'radial_params': None,  # g_rr has no additional params
        'areal': g_thth,        # Above defined areal radius squared
        'areal_params': None,   # g_thth has no additional params
    }


    rr, phi = compute_geodesic(3, **Schwarzschild_kwargs)
    print(rr, phi)