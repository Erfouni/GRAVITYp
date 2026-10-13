"""This module contains very helpful builtin functions for finding relevant
impact parameter values.
"""

import numpy as np
import scipy.optimize as optimize

from .compact_object import BaseCompactObject
from .ray_tracing import compute_nturns_scalar

__all__ = [
    "find_ring_edge",
    "find_ring_bs",
    "get_minima",
    "find_rings_list",
    "compute_optimal_array_steps",
    "compute_optimal_array_Npoints"
]


def find_ring_edge(fun, lower_bound, upper_bound, to_upper=True) -> float:
    """Returns the impact parameter value for which the number of turns is 0.75 or 1.25 (depending on fun)"""
    b = (upper_bound+lower_bound)/2.
    if not to_upper:  # Bounds are reversed
        upper_bound, lower_bound = lower_bound, upper_bound
    a = lower_bound

    maxiter = 500
    success, iter = False, 0
    for iter in range(maxiter):
        success = fun(a)*fun(b) < 0
        if success:
            break
        a = b
        b = (upper_bound+b)/2.
    if success:
        return optimize.brentq(fun, a, b)  # type: ignore[index]
    else:
        raise ValueError(
            f'Invalid interval for ring edges. Iterations: {iter}')


def find_ring_bs(b_crit, ring_interval, epsilon, scalar_fun_nturns):
    """Returns the interval of impact parameters where lensed and photon rings
    are found around a single b_crit
    """
    # TODO: considerar el caso en el que los lensed/photon rings se solapan
    def fun_lensed(b):
        return scalar_fun_nturns(b)-0.75

    def fun_p_ring(b):
        return scalar_fun_nturns(b)-1.25

    # retro
    bounds = (ring_interval[0], b_crit-epsilon)
    retro_lensed = find_ring_edge(fun_lensed, *bounds, to_upper=True)
    retro_p_ring = find_ring_edge(fun_p_ring, *bounds, to_upper=True)

    # non-retro
    bounds = (b_crit+epsilon, ring_interval[1])
    lensed = find_ring_edge(fun_lensed, *bounds, to_upper=False)
    p_ring = find_ring_edge(fun_p_ring, *bounds, to_upper=False)

    return {
        'retro_lensed': (retro_lensed, retro_p_ring),
        'retro_p_ring': (retro_p_ring, b_crit),
        'p_ring': (b_crit, p_ring),
        'lensed': (p_ring, lensed)
    }


def get_minima(b_crits, epsilon, scalar_fun_nturns):
    """Returns the local minima of the nturns function between a list of critical impact parameters"""
    minima = []
    for i, b_crit in enumerate(b_crits[:-1]):
        bounds = (b_crits[i]+epsilon, b_crits[i+1]-epsilon)
        result = optimize.minimize_scalar(scalar_fun_nturns, bounds=bounds)
        if result.success:  # type: ignore[index]
            minima.append(result.x)  # type: ignore[index]

    return minima


def find_rings_list(b_crits, co: BaseCompactObject, **geodesic_kwargs):
    """Returns a list with all impact parameter intervals for each ring and for each b_crit"""
    epsilon = 1e-11  # Maybe toooo small
    max_value = max(b_crits) + 5

    def scalar_fun_nturns(b):
        return compute_nturns_scalar(b, co, **geodesic_kwargs)

    b_crits = sorted(b_crits)  # We ensure that they are in increasing order
    minima = get_minima(b_crits, epsilon, scalar_fun_nturns)
    rings = []
    for i, b_crit in enumerate(b_crits):
        a = epsilon if i == 0 else minima[i-1]
        b = max_value if i == len(b_crits)-1 else minima[i]
        rings.append(find_ring_bs(b_crit, (a, b), epsilon, scalar_fun_nturns))
    return rings


def compute_optimal_array_steps(bmin, bmax, rings,
                                steps=(0.1, 0.01, 0.001), joint=True, fill=False):
    """
    Sample points with specified steps. 
    If joint=True (default), direct, lensed and photon ring contributions are
    returned in an ordered single array.
    If joint=False, these three contributions are returned as a triplet:
    ( bs_direct, bs_lensed, bs_p_ring )
    """
    bs_direct = []
    bs_lensed = []
    bs_p_ring = []

    if not fill:
        for i, ring in enumerate(rings):
            # array of impact parameters for the direct image
            start = bmin if i == 0 else rings[i-1]["lensed"][1]
            stop = ring["retro_lensed"][0]
            bs_direct.append(np.arange(start, stop, steps[0]))

            # array of impact parameters for the lensed image
            bs_lensed.append(
                np.concatenate([
                    np.arange(ring["retro_lensed"][0],
                              ring["retro_lensed"][1], steps[1]),
                    np.arange(ring["lensed"][0], ring["lensed"][1], steps[1])
                ]).reshape(-1,)
            )

            # array of impact parameters for the ring image
            bs_p_ring.append(
                np.concatenate([
                    np.arange(ring["retro_p_ring"][0],
                              ring["retro_p_ring"][1], steps[2]),
                    np.arange(ring["p_ring"][0], ring["p_ring"][1], steps[2])
                ]).reshape(-1,)
            )

        # right tail of array for the direct image
        bs_direct.append(np.arange(rings[-1]["lensed"][1], bmax, steps[0]))
    else:
        bs_direct.append(np.arange(bmin, bmax, steps[0]))
        for i, ring in enumerate(rings):
            # array of impact parameters for the lensed image
            bs_lensed.append(
                np.arange(ring["retro_lensed"][0], ring["lensed"][1], steps[1])
            )

            # array of impact parameters for the ring image
            bs_p_ring.append(
                np.arange(ring["retro_p_ring"][0], ring["p_ring"][1], steps[2])
            )

    bs_direct = np.concatenate(bs_direct)
    bs_lensed = np.concatenate(bs_lensed)
    bs_p_ring = np.concatenate(bs_p_ring)

    if joint:
        total = np.sort(np.concatenate([bs_direct, bs_lensed, bs_p_ring]))
        return total[(total >= bmin) & (total <= bmax)]
    else:
        return [bs_direct, bs_lensed, bs_p_ring]


def compute_optimal_array_Npoints(bmin, bmax, rings,
                                  Npoints=100, joint=True, fill=False):
    """
    Sample points with specified number of points. 
    If joint=True (default), direct, lensed and photon ring contributions are
    returned in an ordered single array.
    If joint=False, these three contributions are returned as a triplet:
    ( bs_direct, bs_lensed, bs_p_ring )
    """
    bs_direct = []
    bs_lensed = []
    bs_p_ring = []

    if not fill:
        for i, ring in enumerate(rings):
            # array of impact parameters for the direct image
            start = bmin if i == 0 else rings[i-1]["lensed"][1]
            stop = ring["retro_lensed"][0]
            bs_direct.append(np.linspace(start, stop, Npoints))

            # array of impact parameters for the lensed image
            bs_lensed.append(
                np.concatenate([
                    np.linspace(ring["retro_lensed"][0],
                                ring["retro_lensed"][1], Npoints),
                    np.linspace(ring["lensed"][0], ring["lensed"][1], Npoints)
                ]).reshape(-1,)
            )

            # array of impact parameters for the ring image
            bs_p_ring.append(
                np.concatenate([
                    np.linspace(ring["retro_p_ring"][0],
                                ring["retro_p_ring"][1], Npoints),
                    np.linspace(ring["p_ring"][0], ring["p_ring"][1], Npoints)
                ]).reshape(-1,)
            )

        # right tail of array for the direct image
        bs_direct.append(np.linspace(rings[-1]["lensed"][1], bmax, Npoints))
    else:
        bs_direct.append(np.linspace(bmin, bmax, Npoints))
        for i, ring in enumerate(rings):
            # array of impact parameters for the lensed image
            bs_lensed.append(
                np.linspace(ring["retro_lensed"][0],
                            ring["lensed"][1], Npoints)
            )

            # array of impact parameters for the ring image
            bs_p_ring.append(
                np.linspace(ring["retro_p_ring"][0],
                            ring["p_ring"][1], Npoints)
            )

    bs_direct = np.concatenate(bs_direct)
    bs_lensed = np.concatenate(bs_lensed)
    bs_p_ring = np.concatenate(bs_p_ring)

    if joint:
        total = np.sort(np.concatenate([bs_direct, bs_lensed, bs_p_ring]))
        return total[(total >= bmin) & (total <= bmax)]
    else:
        return [bs_direct, bs_lensed, bs_p_ring]
