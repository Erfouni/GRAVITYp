"""This module contains very helpful builtin functions for finding relevant
impact parameter values.
"""

import numpy as np
import scipy.optimize as optimize

from .compact_object import BaseCompactObject
from .ray_tracing import _compute_nturns_scalar

__all__ = ["find_rings_list", "compute_optimal_array"]


def _find_ring_edge(fun, lower_bound, upper_bound, to_upper=True) -> float:
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


def _find_ring_bs(b_crit, ring_interval, epsilon, scalar_fun_nturns):
    """Returns the interval of impact parameters where lensed and photon rings
    are found around a single b_crit
    """
    # TODO: issue #32
    def fun_lensed(b):
        return scalar_fun_nturns(b)-0.75

    def fun_p_ring(b):
        return scalar_fun_nturns(b)-1.25

    # retro
    bounds = (ring_interval[0], b_crit-epsilon)
    retro_lensed = _find_ring_edge(fun_lensed, *bounds, to_upper=True)
    retro_p_ring = _find_ring_edge(fun_p_ring, *bounds, to_upper=True)

    # non-retro
    bounds = (b_crit+epsilon, ring_interval[1])
    lensed = _find_ring_edge(fun_lensed, *bounds, to_upper=False)
    p_ring = _find_ring_edge(fun_p_ring, *bounds, to_upper=False)

    return {
        'retro_lensed': (retro_lensed, retro_p_ring),
        'retro_p_ring': (retro_p_ring, b_crit),
        'p_ring': (b_crit, p_ring),
        'lensed': (p_ring, lensed)
    }


def _get_minima(b_crits, epsilon, scalar_fun_nturns):
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
        return _compute_nturns_scalar(b, co, **geodesic_kwargs)

    b_crits = sorted(b_crits)  # We ensure that they are in increasing order
    minima = _get_minima(b_crits, epsilon, scalar_fun_nturns)
    rings = []
    for i, b_crit in enumerate(b_crits):
        a = epsilon if i == 0 else minima[i-1]
        b = max_value if i == len(b_crits)-1 else minima[i]
        rings.append(_find_ring_bs(b_crit, (a, b), epsilon, scalar_fun_nturns))
    return rings


def _ring_intervals(bmin, bmax, rings, fill):
    """Returns the (start, stop) intervals of impact parameters for the direct,
    lensed and photon ring contributions as a triplet of lists
    """
    if fill:
        direct = [(bmin, bmax)]
        lensed = [(ring["retro_lensed"][0], ring["lensed"][1])
                  for ring in rings]
        p_ring = [(ring["retro_p_ring"][0], ring["p_ring"][1])
                  for ring in rings]
    else:
        # Direct image between consecutive rings, plus both tails
        starts = [bmin] + [ring["lensed"][1] for ring in rings]
        stops = [ring["retro_lensed"][0] for ring in rings] + [bmax]
        direct = list(zip(starts, stops))
        lensed = [ring[key]
                  for ring in rings for key in ("retro_lensed", "lensed")]
        p_ring = [ring[key]
                  for ring in rings for key in ("retro_p_ring", "p_ring")]
    return direct, lensed, p_ring


def compute_optimal_array(bmin, bmax, rings, *, steps=None, Npoints=None,
                          joint=True, fill=False):
    """
    Sample impact parameters around the rings, either with fixed step sizes
    (steps) or with a fixed number of points per interval (Npoints).
    Exactly one of them must be given. Each one is either a scalar or a triplet
    (direct, lensed, photon ring), e.g. steps=(0.1, 0.01, 0.001).
    If joint=True (default), direct, lensed and photon ring contributions are
    returned in an ordered single array.
    If joint=False, these three contributions are returned as a triplet:
    ( bs_direct, bs_lensed, bs_p_ring )
    If fill=True, each contribution is sampled over a single interval covering
    all of its regions instead of only within them.
    """
    if steps is not None and Npoints is None:
        sample, params = np.arange, steps
    elif Npoints is not None and steps is None:
        sample, params = np.linspace, Npoints
    else:
        raise ValueError("Pass exactly one of steps or Npoints")
    params = np.broadcast_to(params, 3)  # Scalar -> same value for all three

    bs_direct, bs_lensed, bs_p_ring = (
        np.concatenate([sample(start, stop, p) for start, stop in intervals])
        for intervals, p in zip(_ring_intervals(bmin, bmax, rings, fill), params)
    )

    if joint:
        total = np.sort(np.concatenate([bs_direct, bs_lensed, bs_p_ring]))
        return total[(total >= bmin) & (total <= bmax)]
    else:
        return [bs_direct, bs_lensed, bs_p_ring]
