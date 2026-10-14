"""This module contains some predefined emission profiles
"""

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


def intensity_at_ISCO(r, r_ISCO):
    func = lambda x: (1./(x-(r_ISCO-1))**2) / (1./(r_ISCO-(r_ISCO-1))**2)
    return np.piecewise(r, [r>=r_ISCO,r<r_ISCO] , [func,0] )

def intensity_at_photon_ring(r, r_ph):
    func = lambda x: (1./(x-(r_ph-1))**3) / (1./(r_ph-(r_ph-1))**3)
    return np.piecewise(r, [r>=r_ph,r<r_ph] , [func,0] )

def intensity_at_inner_edge(r, inner_edge):
    func = lambda x: ( np.pi/2-np.arctan(x-5) ) / ( np.pi/2-np.arctan(inner_edge-5) )
    return np.piecewise(r, [r>=inner_edge,r<inner_edge] , [func,0] )

def standard_Unbound(r, mu: float=8., sigma: float=2., gamma: float=2.):
    return ( np.exp(-0.5*(gamma+np.arcsinh((r-mu)/sigma))**2) / 
            np.sqrt((r-mu)**2+sigma**2) )

def normalized_Standard_Unbound(r, mu: float=8., sigma: float=2., gamma: float=2.):
    # FIXME: for some values of params, it doesn't find the minimum
    neg_fun = lambda r: -standard_Unbound(r, mu, sigma, gamma)
    minimum = optimize.minimize(neg_fun, 0).fun
    return neg_fun(r)/minimum

def exponential(r):
    return np.exp(-r)

def uniform(r):
    return np.ones(shape=r.shape)