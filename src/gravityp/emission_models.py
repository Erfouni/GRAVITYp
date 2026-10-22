"""Predefined emission profiles of the accretion disk."""

import numpy as np
import scipy.optimize as optimize


__all__ = [
    "intensity_at_ISCO",
    "intensity_at_photon_ring",
    "intensity_at_inner_edge",
    "standard_Unbound",
    "normalized_Standard_Unbound",
    "exponential",
    "uniform"
]


def intensity_at_ISCO(r: np.ndarray, r_ISCO: float) -> np.ndarray:
    r"""Emission profile starting at the innermost stable circular orbit.

    .. math::

        I(r) = \frac{1}{(r-r_{ISCO}+1)^2} \quad \text{for } r \geq r_{ISCO},

    and zero otherwise, so it is normalised to 1 at :math:`r_{ISCO}`.

    Parameters
    ----------
    r : ndarray
        Radial coordinate.
    r_ISCO : float
        Radius of the innermost stable circular orbit.

    Returns
    -------
    ndarray
        Emitted intensity, with the same shape as `r`.

    See Also
    --------
    intensity_at_photon_ring, intensity_at_inner_edge

    Notes
    -----
    For a Schwarzschild black hole (``r_ISCO = 6``), this is the first model of
    [1]_.

    References
    ----------
    .. [1] S. E. Gralla, D. E. Holz and R. M. Wald, "Black hole shadows, photon
       rings, and lensing rings", Phys. Rev. D 100, 024018 (2019),
       arXiv:1906.00873.
    """
    func = lambda x: (1./(x-(r_ISCO-1))**2) / (1./(r_ISCO-(r_ISCO-1))**2)
    return np.piecewise(r, [r>=r_ISCO,r<r_ISCO] , [func,0] )

def intensity_at_photon_ring(r: np.ndarray, r_ph: float) -> np.ndarray:
    r"""Emission profile starting at the photon sphere.

    .. math::

        I(r) = \frac{1}{(r - r_{ph} + 1)^3} \quad \text{for } r \geq r_{ph},

    and zero otherwise, so it is normalised to 1 at :math:`r_{ph}`.

    Parameters
    ----------
    r : ndarray
        Radial coordinate.
    r_ph : float
        Radius of the photon sphere.

    Returns
    -------
    ndarray
        Emitted intensity, with the same shape as `r`.

    See Also
    --------
    intensity_at_ISCO, intensity_at_inner_edge

    Notes
    -----
    For a Schwarzschild black hole (``r_ph = 3``), this is the second model of
    [1]_.

    References
    ----------
    .. [1] S. E. Gralla, D. E. Holz and R. M. Wald, "Black hole shadows, photon
       rings, and lensing rings", Phys. Rev. D 100, 024018 (2019),
       arXiv:1906.00873.
    """
    func = lambda x: (1./(x-(r_ph-1))**3) / (1./(r_ph-(r_ph-1))**3)
    return np.piecewise(r, [r>=r_ph,r<r_ph] , [func,0] )

def intensity_at_inner_edge(r: np.ndarray, inner_edge: float
                            ) -> np.ndarray:
    r"""Emission profile starting at the inner edge of the geometry.

    .. math::

        I(r) = \frac{\pi/2 - \arctan(r - 5)}{\pi/2 - \arctan(r_{in} - 5)}
        \quad \text{for } r \geq r_{in},

    and zero otherwise, so it is normalised to 1 at the inner edge
    :math:`r_{in}`.

    Parameters
    ----------
    r : ndarray
        Radial coordinate.
    inner_edge : float
        Radius :math:`r_{in}` of the inner edge, e.g. ``co.inner_edge``.

    Returns
    -------
    ndarray
        Emitted intensity, with the same shape as `r`.

    See Also
    --------
    intensity_at_ISCO, intensity_at_photon_ring

    Notes
    -----
    For a Schwarzschild black hole (``inner_edge = 2``), this is the third
    model of [1]_. The decay, centred at :math:`r = 5`, is kept fixed for any
    `inner_edge`.

    References
    ----------
    .. [1] S. E. Gralla, D. E. Holz and R. M. Wald, "Black hole shadows, photon
       rings, and lensing rings", Phys. Rev. D 100, 024018 (2019),
       arXiv:1906.00873.
    """
    func = lambda x: (np.pi/2-np.arctan(x-5))/(np.pi/2-np.arctan(inner_edge-5))
    return np.piecewise(r, [r>=inner_edge,r<inner_edge] , [func,0] )

def standard_Unbound(r: float | np.ndarray, mu: float = 8., sigma: float = 2.,
                     gamma: float = 2.) -> float | np.ndarray:
    r"""Emission profile following Johnson's SU distribution.

    .. math::

        I(r) = \frac{\exp\left[-\frac{1}{2}\left(\gamma +
        \operatorname{arcsinh}\frac{r - \mu}{\sigma}\right)^2\right]}
        {\sqrt{(r - \mu)^2 + \sigma^2}}

    Parameters
    ----------
    r : float or ndarray
        Radial coordinate.
    mu : float, optional
        Location parameter, shifting the profile. Default is 8.
    sigma : float, optional
        Scale parameter, setting the width of the profile. Default is 2.
    gamma : float, optional
        Shape parameter, setting the asymmetry of the profile. Default is 2.

    Returns
    -------
    float or ndarray
        Emitted intensity (not normalised), with the same shape as `r`.

    See Also
    --------
    normalized_Standard_Unbound : Same profile, normalised to a peak of 1.
    """
    return ( np.exp(-0.5*(gamma+np.arcsinh((r-mu)/sigma))**2) / 
            np.sqrt((r-mu)**2+sigma**2) )

def normalized_Standard_Unbound(r: float | np.ndarray, mu: float = 8.,
                                sigma: float = 2., gamma: float = 2.
                                ) -> float | np.ndarray:
    """Emission profile following Johnson's SU distribution, with peak 1.

    Parameters
    ----------
    r : float or ndarray
        Radial coordinate.
    mu : float, optional
        Location parameter, shifting the profile. Default is 8.
    sigma : float, optional
        Scale parameter, setting the width of the profile. Default is 2.
    gamma : float, optional
        Shape parameter, setting the asymmetry of the profile. Default is 2.

    Returns
    -------
    float or ndarray
        Emitted intensity, with the same shape as `r`.

    See Also
    --------
    standard_Unbound : Same profile, not normalised.

    Notes
    -----
    The peak is found numerically, starting the search at :math:`r = 0`.
    """
    # FIXME: for some values of params, it doesn't find the minimum
    neg_fun = lambda r: -standard_Unbound(r, mu, sigma, gamma)
    minimum = optimize.minimize(neg_fun, 0).fun
    return neg_fun(r)/minimum

def exponential(r: float | np.ndarray) -> float | np.ndarray:
    r"""Exponentially decaying emission profile, :math:`I(r) = e^{-r}`.

    Parameters
    ----------
    r : float or ndarray
        Radial coordinate.

    Returns
    -------
    float or ndarray
        Emitted intensity, with the same shape as `r`.
    """
    return np.exp(-r)

def uniform(r: np.ndarray) -> np.ndarray:
    """Uniform emission profile, :math:`I(r) = 1`.

    Parameters
    ----------
    r : ndarray
        Radial coordinate.

    Returns
    -------
    ndarray
        Array of ones, with the same shape as `r`.
    """
    return np.ones(shape=r.shape)