"""This module contains the transfer function implementation and related
functionalities.
"""

import numpy as np
from scipy.interpolate import CubicSpline

from .compact_object import BaseCompactObject
from .geodesic_integration import compute_geodesic


__all__ = ["transfer_function", "compute_inner_shadow"]


def transfer_function(bb, order, co: BaseCompactObject, correction: float = 0,
                      **geodesic_kwargs):
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


def compute_inner_shadow(bmin, bmax, co: BaseCompactObject, samples=10,
                         **geodesic_kwargs):
    """Computes the minimum value of the impact parameter for which a light ray intersects the accretion disk"""
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
