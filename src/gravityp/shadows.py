"""This module contains the implementation of the simulation of the optical
appearance of a compact object and observed intensity profile.
"""

import numpy as np

from .compact_object import BaseCompactObject
from .transfer_functions import transfer_function


__all__ = ["redshift", "observed_intensity",
           "total_intensity", "compute_points"]


def redshift(r, co: BaseCompactObject):
    """Gravitational redshift"""
    return np.sqrt(co.radial(r))


def observed_intensity(eval_points, bs, rbs, emission_model,
                       co: BaseCompactObject):
    intensity = emission_model(rbs)*redshift(rbs, co)**4
    return np.interp(eval_points, bs, intensity, left=0, right=0)


def total_intensity(bs, bs_transfer_list, emission_model,
                    co: BaseCompactObject, correction=0, **geodesic_kwargs):
    # Important to use bs.shape instead of len(bs)
    intensity = np.zeros(shape=bs.shape)

    for order, bb in enumerate(bs_transfer_list):
        points = transfer_function(
            bb, order, co, correction, **geodesic_kwargs)
        intensity += observed_intensity(bs, *points, emission_model, co)

    return intensity


def compute_points(bs_transfer_list, emission_model, co: BaseCompactObject,
                   Npixels, correction=0, **geodesic_kwargs):
    x = np.linspace(-10, 10, np.sqrt(Npixels).astype(int))
    y = np.linspace(-10, 10, np.sqrt(Npixels).astype(int))
    X, Y = np.meshgrid(x, y)
    B = np.sqrt(X**2 + Y**2)
    return total_intensity(B, bs_transfer_list, emission_model, co,
                           correction, **geodesic_kwargs)
