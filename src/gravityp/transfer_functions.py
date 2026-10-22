"""Transfer functions of light rays crossing the accretion disk."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import CubicSpline

from .compact_object import BaseCompactObject
from .geodesic_integration import compute_geodesic


__all__ = ["transfer_function", "compute_inner_shadow"]


def transfer_function(bb: ArrayLike, order: int, co: BaseCompactObject,
                      correction: float = 0, **geodesic_kwargs
                      ) -> tuple[np.ndarray, np.ndarray]:
    r"""Coordinate :math:`r` where geodesics hit the disk the (order+1)-th time.

    For a face-on observer, a light ray with impact parameter :math:`b` crosses
    the disk for the :math:`(m+1)`-th time when its deflection reaches
    :math:`\phi_m = m\pi + \pi/2 + \phi_c`. The transfer function
    :math:`r_m(b)` is the radial coordinate at that point [1]_.

    Parameters
    ----------
    bb : float or array_like
        Positive impact parameters of the light rays, in units of M.
    order : int
        Order :math:`m` of the transfer function: 0 (direct), 1 (lensed) or
        2 (photon ring).
    co : BaseCompactObject
        Compact object defining the geometry.
    correction : float, optional
        Additional angle :math:`\phi_c` the light ray must travel to reach the
        accretion disk, which depends on the observer's inclination and on the
        polar angle on the observer's screen. Inclined observers are not
        supported yet, so only the face-on value 0 (default) is meaningful.

    Returns
    -------
    bs : ndarray
        Impact parameters of the light rays that reach the `order`-th
        crossing.
    rbs : ndarray
        Radial coordinates :math:`r_m(b)` of those crossings, with the same
        shape as `bs`.

    Other Parameters
    ----------------
    **geodesic_kwargs
        Keyword arguments passed to `gravityp.compute_geodesic`, such as
        `Npoints` or `r0`.

    See Also
    --------
    gravityp.get_order : Order of a light ray from its deflection.

    Notes
    -----
    Light rays that do not reach the `order`-th crossing, either because they
    are deflected too little or captured before, are discarded. So are those
    with negative impact parameter. The crossing radius is interpolated with a
    cubic spline of :math:`r(\phi)` along each trajectory.

    References
    ----------
    .. [1] S. E. Gralla, D. E. Holz and R. M. Wald, "Black hole shadows, photon
       rings, and lensing rings", Phys. Rev. D 100, 024018 (2019),
       arXiv:1906.00873.

    Examples
    --------
    The light ray with ``b = 1`` is captured before reaching the disk:

    >>> import numpy as np
    >>> from gravityp import Schwarzschild, transfer_function
    >>> bs, rbs = transfer_function([1., 3., 6.], 0, Schwarzschild())
    >>> bs
    array([3., 6.])
    >>> print(np.round(rbs, 2))
    [2.15 5.08]
    """
    bs = []
    rbs = []

    for b in np.atleast_1d(bb):
        (rr, phi) = compute_geodesic(b, co, **geodesic_kwargs)

        if abs(phi[-1]) > (order*np.pi + np.pi/2 + correction):
            try:
                f_hat = CubicSpline(phi, rr)
                r_b = f_hat(order*np.pi+np.pi/2+correction)
                rbs.append(r_b)
                bs.append(b)
            except ValueError:
                continue

    return (np.array(bs), np.array(rbs))


def compute_inner_shadow(bmin: float, bmax: float, co: BaseCompactObject,
                         samples: int = 10, **geodesic_kwargs) -> float:
    """Smallest impact parameter of the light rays crossing the accretion disk.

    For a face-on observer, light rays with smaller impact parameters are
    captured before crossing the disk, so this is the edge of the inner shadow:
    the dark region of the image within the direct image of the disk.

    Parameters
    ----------
    bmin, bmax : float
        Initial search interval of impact parameters, in units of M. It must
        contain the inner shadow edge.
    co : BaseCompactObject
        Compact object defining the geometry.
    samples : int, optional
        Number of impact parameters sampled in each iteration, at least 4 so
        that the search interval shrinks. Default is 10.

    Returns
    -------
    float
        Edge of the inner shadow, with an absolute tolerance of about 1e-5.

    Other Parameters
    ----------------
    **geodesic_kwargs
        Keyword arguments passed to `gravityp.compute_geodesic`, such as
        `Npoints` or `r0`.

    Raises
    ------
    ValueError
        If the edge is not found within 500 iterations.

    See Also
    --------
    transfer_function : Radii where light rays cross the accretion disk.

    Notes
    -----
    In each iteration, `samples` impact parameters are evenly spaced over the
    search interval, which then shrinks around the smallest of them whose
    light ray crosses the disk.

    Examples
    --------
    >>> from gravityp import Schwarzschild, compute_inner_shadow
    >>> print(round(compute_inner_shadow(2., 4., Schwarzschild()), 4))
    2.8477
    """
    maxiters = 500
    for i in range(maxiters):
        bs = np.linspace(bmin, bmax, samples)
        (b, _) = transfer_function(bs, order=0, co=co, **geodesic_kwargs)
        inner_shadow = b[0]
        diff = abs(b[1]-b[0])
        if 2*diff < 1e-5:
            break
        bmin = inner_shadow-diff
        bmax = inner_shadow+diff
    if 2*diff < 1e-5:
        return inner_shadow
    else:
        raise ValueError(
            f'Inner shadow has not been found after {maxiters} iterations.')
