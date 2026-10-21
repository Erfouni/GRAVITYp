"""Integration of null geodesics (light rays)."""

import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
from bisect import bisect_right

from .compact_object import BaseCompactObject


__all__ = ["find_rmin", "compute_geodesic"]


_DEFAULT_NPOINTS = 100000  # ODE integration resolution
_DEFAULT_R0 = 1000.0  # Asymptotic boundary for initial conditions
_MINIMUM_R = 1e-6  # Minimum value of r for integration


def _find_interval(inverse_b2, photon_sph, local_maxima, lower_bound=1e-4, upper_bound=100):
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
    """Minimum radius reached by a light ray with impact parameter `b`.

    Parameters
    ----------
    b : float
        Impact parameter of the light ray, in units of M.
    co : BaseCompactObject
        Compact object defining the geometry.

    Returns
    -------
    float
        Turning point of the ray, i.e. the root of :math:`V(r) = 1/b^2` found
        with Brent's method, if the ray is scattered. Otherwise the ray is
        captured and its inner edge `co.inner_edge` is returned.

    See Also
    --------
    compute_geodesic : Full trajectory of the light ray.
    BaseCompactObject.potential : Effective potential :math:`V(r)`.

    Notes
    -----
    The root is searched between the photon spheres of
    `co.potential_maxima` that enclose :math:`1/b^2`. The result is never
    smaller than `co.inner_edge`, nor than a small positive cutoff (1e-6)
    that keeps the radial grid of `compute_geodesic` well defined.

    Examples
    --------
    >>> from gravityp import Schwarzschild, find_rmin
    >>> bh = Schwarzschild()
    >>> print(round(find_rmin(6., bh), 4))  # scattered ray
    4.4534
    >>> find_rmin(3., bh)  # captured ray, b < sqrt(27)
    2.0
    """
    photon_sph, local_maxima = co.potential_maxima
    # Interval for Brent's method
    bounds = _find_interval(1/b**2, photon_sph, local_maxima)

    def fun(r):
        return co.potential(r)-1/b**2
    if fun(bounds[0])*fun(bounds[1]) >= 0:  # No root if there is no sign change
        root = 0
    else:
        root, _ = optimize.brentq(fun, *bounds, full_output=True)
    return max(max(root, co.inner_edge), _MINIMUM_R)


def _dphi_dr(phi, r, b, co: BaseCompactObject, incoming: bool = True):
    """Expression for dphi/dr in the null geodesic equation"""
    R = co.areal(r)  # Store value for single function call
    sign = -2*int(incoming)+1
    return sign*(b / np.sqrt(R))/np.sqrt(R - b**2*co.radial(r))


def _integrator(phi0, rr, b, co: BaseCompactObject, incoming: bool):
    """Integrates phi(r) over the grid rr, dropping non-finite values"""
    phi = integrate.odeint(_dphi_dr, phi0, rr, (b, co, incoming)).reshape(-1,)
    finite = np.isfinite(phi)  # Finite values mask
    return rr[finite], phi[finite]


def _compute_outgoing(r_phs, inner_edge, rmin) -> bool:
    """Decides whether to compute outgoing geodesic after reaching rmin"""
    if len(r_phs) == 0:  # There is no photon sphere
        return True
    if min(r_phs) < _MINIMUM_R:  # Innermost photon sphere very small
        return rmin > _MINIMUM_R  # Compute only if rmin is > _MINIMUM_R
    return (rmin > inner_edge) and (rmin > _MINIMUM_R)


def compute_geodesic(b, co: BaseCompactObject,
                     Npoints=_DEFAULT_NPOINTS, r0=_DEFAULT_R0):
    r"""Trajectory of a light ray with impact parameter `b`.

    The ray is integrated inwards from `r0` down to its minimum radius
    (see `find_rmin`). If the ray is scattered, the outgoing leg from the
    minimum radius back to `r0` is integrated as well. Captured rays end at the
    inner edge of `co`.

    Parameters
    ----------
    b : float
        Impact parameter of the light ray, in units of M.
    co : BaseCompactObject
        Compact object defining the geometry.
    Npoints : int, optional
        Number of radial points of each leg, logarithmically spaced. Default
        is 100000.
    r0 : float, optional
        Initial radius, which should lie far from the compact object. Default
        is 1000.

    Returns
    -------
    rr : ndarray
        Radial coordinate along the trajectory, with ``Npoints`` (captured
        rays) or ``2*Npoints`` (scattered rays) elements at most.
    phi : ndarray
        Azimuthal angle along the trajectory, with the same shape as `rr`.
        It starts at ``b/r0`` and grows monotonically in absolute value, so
        ``abs(phi[-1])`` is the total deflection of the ray.

    See Also
    --------
    find_rmin : Minimum radius of the light ray.
    gravityp.compute_nturns : Number of turns of the light rays.
    gravityp.transfer_function : Radii where light rays cross the disk.

    Notes
    -----
    The trajectory is obtained by integrating the orbit equation

    .. math::

        \frac{d\phi}{dr} = \mp \frac{b}{\sqrt{R(r)}\sqrt{R(r) - b^2 A(r)}}

    with ``scipy.integrate.odeint``, where the minus (plus) sign holds along
    the incoming (outgoing) leg. Points where the integration yields
    non-finite values, typically next to the turning point, are dropped.
    """
    rmin = find_rmin(b, co)
    # We must ensure rmin > 0 in find_rmin for np.geomspace
    rr_in = np.geomspace(r0, rmin, Npoints)

    # phi0 = b/r0 is a good approximation for r0>>inner_edge
    rr, phi = _integrator(b/r0, rr_in, b, co, incoming=True)

    if _compute_outgoing(co.r_phs, co.inner_edge, rmin):
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
