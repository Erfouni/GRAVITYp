r"""Number of turns and order of light rays.

For a face-on observer, the number of turns :math:`n = \phi/(2\pi)` of a light
ray, with :math:`\phi` its total deflection, determines how many times it
crosses the accretion disk. Light rays are thus classified by their order:
direct (0), lensed (1) or photon ring (2) [1]_.

References
----------
.. [1] S. E. Gralla, D. E. Holz and R. M. Wald, "Black hole shadows, photon
   rings, and lensing rings", Phys. Rev. D 100, 024018 (2019),
   arXiv:1906.00873.
"""

import numpy as np

from .compact_object import BaseCompactObject
from .geodesic_integration import compute_geodesic


__all__ = ["compute_nturns", "get_order"]


def _compute_nturns_scalar(b, co: BaseCompactObject, **geodesic_kwargs):
    """Returns the number of turns of a single light ray"""
    (_, phi) = compute_geodesic(b, co, **geodesic_kwargs)
    return np.abs(phi[-1])/(2*np.pi)


def compute_nturns(bs, co: BaseCompactObject, **geodesic_kwargs):
    r"""Number of turns of light rays with impact parameters `bs`.

    Parameters
    ----------
    bs : float or array_like
        Impact parameters of the light rays, in units of M.
    co : BaseCompactObject
        Compact object defining the geometry.

    Returns
    -------
    ndarray
        Number of turns :math:`n = |\phi|/(2\pi)` of each light ray, where
        :math:`\phi` is its total deflection. One-dimensional, with one element
        per impact parameter (also for a scalar `bs`).

    Other Parameters
    ----------------
    **geodesic_kwargs
        Keyword arguments passed to `gravityp.compute_geodesic`, such as
        `Npoints` or `r0`.

    See Also
    --------
    get_order : Order of a light ray from its deflection.
    gravityp.compute_geodesic : Trajectory of a light ray.

    Examples
    --------
    Direct, photon ring and lensed rays of the Schwarzschild black hole:

    >>> import numpy as np
    >>> from gravityp import Schwarzschild, compute_nturns
    >>> n = compute_nturns([3., 5.2, 6.], Schwarzschild())
    >>> print(np.round(n, 2))
    [0.27 1.56 0.77]
    """
    return np.array([_compute_nturns_scalar(b, co, **geodesic_kwargs)
                     for b in np.atleast_1d(bs)])


def get_order(deviation: float, correction: float = 0) -> int:
    r"""Order of a light ray from its total deflection.

    The number of turns :math:`n = |\phi - \phi_c|/(2\pi)`, with :math:`\phi`
    the deflection and :math:`\phi_c` the correction, is mapped to

    - 0 (direct image) if :math:`n \leq 3/4`,
    - 1 (lensed image) if :math:`3/4 < n \leq 5/4`,
    - 2 (photon ring) if :math:`n > 5/4`.

    Parameters
    ----------
    deviation : float
        Total deflection :math:`\phi` of the light ray, in radians. Negative
        values (rays with ``b < 0``) are also accepted.
    correction : float, optional
        Additional angle :math:`\phi_c` the light ray must travel to reach the
        accretion disk, which depends on the observer's inclination and on the
        polar angle on the observer's screen. Inclined observers are not
        supported yet, so only the face-on value 0 (default) is meaningful.

    Returns
    -------
    int
        Order of the light ray: 0, 1 or 2.

    Notes
    -----
    A light ray of order 0, 1 or 2 crosses the accretion disk once, twice, or
    three or more times, respectively.

    Examples
    --------
    >>> import numpy as np
    >>> from gravityp import get_order
    >>> get_order(np.pi), get_order(2*np.pi), get_order(3*np.pi)
    (0, 1, 2)
    """
    # We take absolute value for rays with b<0
    nturns = abs(deviation - correction) / (2*np.pi)
    return int(nturns > 0.75) + int(nturns > 1.25)


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

    (_, phi) = compute_geodesic(3, schwarzschild)
    order = get_order(phi[-1])
    nturns = phi[-1] / (2*np.pi)
    print(f"nturns = {nturns:.2f}. Order = {order}")
