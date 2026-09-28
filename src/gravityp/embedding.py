"""This module contains the integration procedures to calculate
the isometric Euclidean embedding of a given spacetime
"""

import numpy as np
import scipy.integrate as integrate

from .utils import to_tuple

__all__ = [
    "surface_domain",
    "areal_derivative",
    "make_embedding"
]

DEFAULT_NPOINTS = 100000  # ODE integration resolution
DEFAULT_3DPOINTS = 700 #3D radial plotting resolution
DEFAULT_POLAR_POINTS = 100 #3D polar plotting resolution
DEFAULT_R0 = 12.0  # Asymptotic boundary for initial conditions
MINIMUM_R = 1e-6 # Minimum value of r for integration

def surface_domain(inner_edge, areal, areal_params,
                   Npoints=DEFAULT_3DPOINTS, Npolar=DEFAULT_POLAR_POINTS, r0=DEFAULT_R0):
    """Sets the x and y points (in cartesian coordinates) of the 3D embedding
    calculated from the radial coordinate of the spacetime"""

    rr = np.geomspace(inner_edge + 1e-5, r0, Npoints)

    areal_params = to_tuple(areal_params)

    polar = np.linspace(0, 2*np.pi, Npolar)

    x_emb = np.sqrt(areal(rr, *areal_params))[:, None]*np.cos(polar)[None, :]
    y_emb = np.sqrt(areal(rr, *areal_params))[:, None]*np.sin(polar)[None, :]

    return x_emb, y_emb

def areal_derivative(inner_edge, areal, areal_params,
                     Npoints=DEFAULT_3DPOINTS, r0=DEFAULT_R0):
    """Computes the derivative of the areal radius of the metric"""

    rr = np.geomspace(inner_edge + 1e-5, r0, Npoints)

    areal_params = to_tuple(areal_params)

    h = 1/Npoints
    
    return (np.sqrt(areal(rr + h, *areal_params)) - np.sqrt(areal(rr - h, *areal_params))) / (2.0 * h)

def make_embedding(inner_edge, radial_fun, radial_params, areal, areal_params,
                   Npoints=DEFAULT_3DPOINTS, r0=DEFAULT_R0):
    """Computes the vertical coordinate of the 3D isometric embedding"""

    rr = np.geomspace(inner_edge + 1e-5, r0, Npoints)

    areal_params = to_tuple(areal_params)
    radial_params = to_tuple(radial_params)

    R_der = areal_derivative(inner_edge, areal, areal_params)

    integral_arg_squared = 1/radial_fun(rr, *radial_params) - R_der**2

    z_emb = integrate.cumulative_trapezoid(np.sqrt(integral_arg_squared), rr, initial=0)

    return rr, z_emb