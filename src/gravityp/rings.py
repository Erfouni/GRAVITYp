"""Impact parameters of the lensed and photon rings."""

import numpy as np
import scipy.optimize as optimize
from collections.abc import Iterable, Sequence
from typing import Literal, overload

from .compact_object import BaseCompactObject
from .ray_tracing import _compute_nturns_scalar

__all__ = ["find_rings_list", "compute_optimal_array"]


def _find_ring_edge(fun, lower_bound, upper_bound, to_upper=True) -> float:
    """Returns the impact parameter for which the nturns is 0.75 or 1.25"""
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
    """Returns the local minima of ``nturns`` between a list of ``b_crits``"""
    minima = []
    for i, b_crit in enumerate(b_crits[:-1]):
        bounds = (b_crits[i]+epsilon, b_crits[i+1]-epsilon)
        result = optimize.minimize_scalar(scalar_fun_nturns, bounds=bounds)
        if result.success:  # type: ignore[index]
            minima.append(result.x)  # type: ignore[index]

    return minima


def find_rings_list(b_crits: Iterable[float], co: BaseCompactObject,
                    **geodesic_kwargs) -> list[dict]:
    r"""Intervals of impact parameters of the lensed and photon rings.

    Around each critical impact parameter :math:`b_c`, the number of turns
    :math:`n(b)` of the light rays diverges. The lensed rings correspond to
    :math:`3/4 < n \leq 5/4` and the photon rings to :math:`n > 5/4`, each of
    them having an inner (:math:`b < b_c`) and an outer (:math:`b > b_c`) part.

    Parameters
    ----------
    b_crits : sequence of float
        Critical impact parameters :math:`b_c = V_{max}^{-1/2}` of the photon
        spheres, in any order.
    co : BaseCompactObject
        Compact object defining the geometry.

    Returns
    -------
    list of dict
        One dictionary per critical impact parameter, sorted in increasing
        order of :math:`b_c`. Each one maps the following keys to intervals
        ``(start, stop)`` of impact parameters:

        - ``'retro_lensed'``: inner part of the lensed ring.
        - ``'retro_p_ring'``: inner part of the photon ring, up to :math:`b_c`.
        - ``'p_ring'``: outer part of the photon ring, from :math:`b_c`.
        - ``'lensed'``: outer part of the lensed ring.

    Other Parameters
    ----------------
    **geodesic_kwargs
        Keyword arguments passed to `gravityp.compute_geodesic`, such as
        `Npoints` or `r0`.

    Raises
    ------
    ValueError
        If the edge of a ring cannot be found.

    See Also
    --------
    compute_optimal_array : Sample impact parameters around the rings.
    gravityp.compute_nturns : Number of turns of light rays.

    Notes
    -----
    Each ring is searched between the minima of :math:`n(b)` that separate
    consecutive critical impact parameters, from :math:`b \approx 0` for the
    first one up to ``max(b_crits) + 5`` for the last one.

    Examples
    --------
    >>> import numpy as np
    >>> from gravityp import Schwarzschild, find_rings_list
    >>> bh = Schwarzschild()
    >>> b_crits = 1 / np.sqrt(bh.potential_maxima[1])  # [sqrt(27)]
    >>> rings = find_rings_list(b_crits, bh)
    >>> len(rings), list(rings[0])
    (1, ['retro_lensed', 'retro_p_ring', 'p_ring', 'lensed'])
    >>> print(np.round(rings[0]['p_ring'], 2))
    [5.2  5.23]
    """
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


@overload
def compute_optimal_array(bmin: float, bmax: float, rings: list[dict], *,
                          steps: float | Sequence[float] | None = ...,
                          Npoints: int | Sequence[int] | None = ...,
                          joint: Literal[True] = ..., fill: bool = ...
                          ) -> np.ndarray: ...


@overload
def compute_optimal_array(bmin: float, bmax: float, rings: list[dict], *,
                          steps: float | Sequence[float] | None = ...,
                          Npoints: int | Sequence[int] | None = ...,
                          joint: Literal[False], fill: bool = ...
                          ) -> list[np.ndarray]: ...


def compute_optimal_array(bmin: float, bmax: float, rings: list[dict], *,
                          steps: float | Sequence[float] | None = None,
                          Npoints: int | Sequence[int] | None = None,
                          joint: bool = True, fill: bool = False
                          ) -> np.ndarray | list[np.ndarray]:
    """Sample impact parameters around the lensed and photon rings.

    The direct, lensed and photon ring contributions are sampled separately,
    either with fixed step sizes (`steps`) or with a fixed number of points per
    interval (`Npoints`), so the narrow rings can be sampled more densely.

    Parameters
    ----------
    bmin, bmax : float
        Range of impact parameters, in units of M.
    rings : list of dict
        Ring intervals, as returned by `find_rings_list`.
    steps : float or sequence of 3 floats, optional
        Step size of the samples, either common to all contributions or given
        as ``(direct, lensed, photon ring)``, e.g. ``(0.1, 0.01, 0.001)``.
    Npoints : int or sequence of 3 ints, optional
        Number of samples per interval, either common to all contributions or
        given as ``(direct, lensed, photon ring)``.
    joint : bool, optional
        If True (default), return all contributions in a single array.
        Otherwise, return them separately.
    fill : bool, optional
        If True, sample each contribution over a single interval covering all
        of its regions. If False (default), sample only within them.

    Returns
    -------
    ndarray or list of ndarray
        If `joint` is True, sorted array with all the samples between `bmin`
        and `bmax`. Otherwise, list ``[bs_direct, bs_lensed, bs_p_ring]``.

    Raises
    ------
    ValueError
        If not exactly one of `steps` and `Npoints` is given.

    See Also
    --------
    find_rings_list : Intervals of the lensed and photon rings.
    gravityp.total_intensity : Observed intensity, which takes the samples
        returned with ``joint=False`` and ``fill=True``.

    Notes
    -----
    With ``fill=False``, the impact parameters are split by image type: direct
    samples lie outside the rings, while lensed and photon ring samples lie in
    the inner and outer parts of their rings. With ``fill=True``, direct
    samples cover the whole range ``[bmin, bmax]``, lensed samples span from
    the inner to the outer edge of each lensed ring (photon ring included), and
    photon ring samples span both parts of each photon ring. The latter is the
    sampling required for the transfer functions, since a light ray crossing
    the disk several times contributes to all the lower orders as well.
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
