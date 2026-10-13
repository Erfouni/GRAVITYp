"""Module containing the implementation of a generic compact object"""

import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Iterable, Callable, Mapping
from functools import cached_property, partial


__all__ = ["BaseCompactObject", "CompactObject"]


def _get_increasing_maxima(r_phs, potential):
    """Returns increasing local maxima of the potential with their photon sphere values"""
    if len(r_phs) >= 1:  # There has to be at least 1 local maximum
        r_phs = np.array(r_phs)
        r_phs = np.sort(r_phs)[::-1]  # descending order of
        photon_sph = r_phs[0].reshape(1,)
        local_maxima = potential(photon_sph)
        for r_ph in r_phs:
            maximum = potential(r_ph)
            if maximum > local_maxima.max():  # We take increasing local maxima
                local_maxima = np.append(local_maxima, maximum)
                photon_sph = np.append(photon_sph, r_ph)
        # Swap to ascending order of photon_spheres
        return photon_sph[::-1], local_maxima[::-1]
    else:
        return np.array([]), np.array([])


class BaseCompactObject(ABC):
    """Static, spherically symmetric compact object with metric

    ds^2 = -A(r) dt^2 + dr^2/A(r) + R(r) dOmega^2

    where A(r) = `self.radial(r)` and R(r) = `self.areal(r)`.

    `r_phs` and `inner_edge` are read-only, so that cached quantities derived
    from them (e.g. `potential_maxima`) cannot go stale.
    """

    def __init__(self, r_phs: Iterable[float], inner_edge: float) -> None:
        self._r_phs = tuple(r_phs)
        self._inner_edge = inner_edge

    @property
    def r_phs(self) -> tuple[float, ...]:
        """Positions of the photon spheres of the effective potential"""
        return self._r_phs

    @property
    def inner_edge(self) -> float:
        """Value of the r coordinate of the inner edge"""
        return self._inner_edge

    @abstractmethod
    def radial(self, r: float | np.ndarray) -> float | np.ndarray:
        """Radial function A(r) = -g_tt = 1/g_rr"""

    @abstractmethod
    def areal(self, r: float | np.ndarray) -> float | np.ndarray:
        """Areal radius squared R(r) = g_thth"""

    def potential(self, r: float | np.ndarray) -> float | np.ndarray:
        """Effective potential V(r) = A(r)/R(r)"""
        return self.radial(r) / self.areal(r)

    @cached_property
    def potential_maxima(self) -> tuple[np.ndarray, np.ndarray]:
        """Photon spheres and their increasing potential maxima, outermost barrier last

        Computed once per instance. The returned arrays are read-only so the
        cached values cannot be modified in place.
        """
        photon_sph, local_maxima = _get_increasing_maxima(self.r_phs,
                                                          self.potential)
        photon_sph.flags.writeable = False
        local_maxima.flags.writeable = False
        return photon_sph, local_maxima


class CompactObject(BaseCompactObject):
    """Compact object defined from user-supplied metric functions

    The additional parameters of `radial` and `areal` are bound by keyword at
    construction, so the resulting methods only take the radial coordinate r.

    Parameters:
        r_phs (iterable of float): positions of the photon spheres (maxima) of
            the effective potential
        inner_edge (float): value of the r coordinate of the inner edge (event
            horizon, throat, etc.)
        radial (callable): radial function A(r). Its first argument must be r
        areal (callable): areal radius squared R(r). Its first argument must be r
        radial_params (mapping, optional): additional keyword arguments for
            `radial`. Default: None
        areal_params (mapping, optional): additional keyword arguments for
            `areal`. Default: None

    Note:
        Additional parameters are passed by keyword, so they cannot be
        positional-only nor be named r.

    Example:
        >>> def rn(r, M, Q):
        ...     return 1 - 2*M/r + Q**2/r**2
        >>> bh = CompactObject(
        ...     r_phs=[2.82], inner_edge=1.87,
        ...     radial=rn, radial_params={'M': 1., 'Q': 0.5},
        ...     areal=lambda r: r**2,
        ... )
        >>> bh.radial(2.)
        0.0625
    """

    def __init__(
        self, r_phs: Iterable[float], inner_edge: float,
        radial: Callable, areal: Callable,
        radial_params: Mapping[str, float] | None = None,
        areal_params: Mapping[str, float] | None = None
    ) -> None:
        super().__init__(r_phs, inner_edge)
        self._radial = partial(radial, **(radial_params or {}))
        self._areal = partial(areal, **(areal_params or {}))

    def radial(self, r: float | np.ndarray) -> float | np.ndarray:
        return self._radial(r)

    def areal(self, r: float | np.ndarray) -> float | np.ndarray:
        return self._areal(r)
