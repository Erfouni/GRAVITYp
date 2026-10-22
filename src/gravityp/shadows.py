"""Observed intensity and optical appearance of compact objects."""

import numpy as np
from collections.abc import Callable, Sequence

from .compact_object import BaseCompactObject
from .transfer_functions import transfer_function


__all__ = ["observed_intensity", "total_intensity", "compute_points"]


def _redshift(r, co: BaseCompactObject):
    """Gravitational redshift factor sqrt(A(r)) of static emitters"""
    return np.sqrt(co.radial(r))


def observed_intensity(eval_points: np.ndarray, bs: np.ndarray,
                       rbs: np.ndarray, emission_model: Callable,
                       co: BaseCompactObject) -> np.ndarray:
    r"""Observed intensity contributed by a single transfer function.

    The intensity emitted at the crossing radii :math:`r_m(b)` is redshifted
    as :math:`I_{ob} = A(r_m)^2 \, I_{em}(r_m)` and interpolated at
    `eval_points`.

    Parameters
    ----------
    eval_points : ndarray
        Impact parameters where the intensity is evaluated, of any shape.
    bs : ndarray
        Increasing impact parameters of the transfer function.
    rbs : ndarray
        Crossing radii :math:`r_m(b)` at `bs`, as returned together with them
        by `gravityp.transfer_function`.
    emission_model : callable
        Emitted intensity :math:`I_{em}(r)`, taking an ndarray of radii.
    co : BaseCompactObject
        Compact object defining the geometry.

    Returns
    -------
    ndarray
        Observed intensity, with the same shape as `eval_points`. It is zero
        outside the range of `bs`.

    See Also
    --------
    total_intensity : Observed intensity summed over all orders.
    """
    intensity = emission_model(rbs)*_redshift(rbs, co)**4
    return np.interp(eval_points, bs, intensity, left=0, right=0)


def total_intensity(bs: np.ndarray, bs_transfer_list: Sequence[np.ndarray],
                    emission_model: Callable, co: BaseCompactObject,
                    correction: float = 0, **geodesic_kwargs) -> np.ndarray:
    r"""Observed intensity profile of a face-on accretion disk.

    The observed intensity at impact parameter :math:`b` adds the
    contributions of all the crossings of the light ray with the disk [1]_,

    .. math::

        I_{ob}(b) = \sum_m A(r_m)^2 \, I_{em}(r_m), \qquad r_m = r_m(b),

    where :math:`r_m(b)` is the transfer function of order :math:`m` and
    :math:`A(r_m)^2` accounts for the gravitational redshift of static
    emitters.

    Parameters
    ----------
    bs : ndarray
        Impact parameters where the intensity is evaluated, of any shape.
    bs_transfer_list : sequence of ndarray
        Increasing impact parameters sampling the transfer function of each
        order, starting from order 0. Usually computed with
        ``compute_optimal_array(..., joint=False, fill=True)``.
    emission_model : callable
        Emitted intensity :math:`I_{em}(r)`, taking an ndarray of radii.
    co : BaseCompactObject
        Compact object defining the geometry.
    correction : float, optional
        Additional angle the light ray must travel to reach the accretion
        disk, which depends on the observer's inclination and on the polar
        angle on the observer's screen. Inclined observers are not supported
        yet, so only the face-on value 0 (default) is meaningful.

    Returns
    -------
    ndarray
        Observed intensity, with the same shape as `bs`.

    Other Parameters
    ----------------
    **geodesic_kwargs
        Keyword arguments passed to `gravityp.compute_geodesic`, such as
        `Npoints` or `r0`.

    See Also
    --------
    compute_points : Observed intensity on the observer's screen.
    gravityp.compute_optimal_array : Sampling of the transfer functions.
    gravityp.transfer_function : Transfer function of a given order.

    References
    ----------
    .. [1] S. E. Gralla, D. E. Holz and R. M. Wald, "Black hole shadows, photon
       rings, and lensing rings", Phys. Rev. D 100, 024018 (2019),
       arXiv:1906.00873.
    """
    # Important to use bs.shape instead of len(bs)
    intensity = np.zeros(shape=bs.shape)

    for order, bb in enumerate(bs_transfer_list):
        points = transfer_function(
            bb, order, co, correction, **geodesic_kwargs)
        intensity += observed_intensity(bs, *points, emission_model, co)

    return intensity


def compute_points(bs_transfer_list: Sequence[np.ndarray],
                   emission_model: Callable, co: BaseCompactObject,
                   Npixels: float, correction: float = 0,
                   **geodesic_kwargs) -> np.ndarray:
    r"""Observed intensity on the observer's screen.

    The screen is a square grid covering ``[-10, 10]`` in units of M along
    both axes, and each pixel at :math:`(x, y)` gets the observed intensity at
    impact parameter :math:`b = \sqrt{x^2 + y^2}`.

    Parameters
    ----------
    bs_transfer_list : sequence of ndarray
        Increasing impact parameters sampling the transfer function of each
        order, starting from order 0. Usually computed with
        ``compute_optimal_array(..., joint=False, fill=True)``.
    emission_model : callable
        Emitted intensity :math:`I_{em}(r)`, taking an ndarray of radii.
    co : BaseCompactObject
        Compact object defining the geometry.
    Npixels : int or float
        Total number of pixels. The grid has ``int(sqrt(Npixels))`` pixels
        per side.
    correction : float, optional
        Additional angle the light ray must travel to reach the accretion
        disk, which depends on the observer's inclination and on the polar
        angle on the observer's screen. Inclined observers are not supported
        yet, so only the face-on value 0 (default) is meaningful.

    Returns
    -------
    ndarray
        Square image of shape ``(n, n)``, with ``n = int(sqrt(Npixels))``.
        Rows run along the vertical axis :math:`y` and columns along :math:`x`,
        both increasing, as expected by ``imshow(..., origin='lower')``.

    Other Parameters
    ----------------
    **geodesic_kwargs
        Keyword arguments passed to `gravityp.compute_geodesic`, such as
        `Npoints` or `r0`.

    See Also
    --------
    total_intensity : Observed intensity profile.
    gravityp.make_shadow_plot : Plot of the resulting image.
    """
    x = np.linspace(-10, 10, np.sqrt(Npixels).astype(int))
    y = np.linspace(-10, 10, np.sqrt(Npixels).astype(int))
    X, Y = np.meshgrid(x, y)
    B = np.sqrt(X**2 + Y**2)
    return total_intensity(B, bs_transfer_list, emission_model, co,
                           correction, **geodesic_kwargs)
