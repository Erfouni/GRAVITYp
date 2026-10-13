"""Module containing the implementation of a generic compact object"""

import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Iterable, Callable, Mapping
from functools import partial


__all__ = ["BaseCompactObject", "CompactObject"]


class BaseCompactObject(ABC):
    """Static, spherically symmetric compact object with metric

    ds^2 = -A(r) dt^2 + dr^2/A(r) + R(r) dOmega^2

    where A(r) = `self.radial(r)` and R(r) = `self.areal(r)`.
    """

    def __init__(self, r_phs: Iterable[float], inner_edge: float) -> None:
        self.r_phs = list(r_phs)
        self.inner_edge = inner_edge

    @abstractmethod
    def radial(self, r: float | np.ndarray) -> float | np.ndarray:
        """Radial function A(r) = -g_tt = 1/g_rr"""

    @abstractmethod
    def areal(self, r: float | np.ndarray) -> float | np.ndarray:
        """Areal radius squared R(r) = g_thth"""

    def potential(self, r: float | np.ndarray) -> float | np.ndarray:
        """Effective potential V(r) = A(r)/R(r)"""
        return self.radial(r) / self.areal(r)


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
