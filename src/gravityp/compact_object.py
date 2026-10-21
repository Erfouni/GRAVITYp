"""Base classes describing static, spherically symmetric compact objects.

Every compact object is defined by its line element

    ds^2 = -A(r) dt^2 + dr^2/A(r) + R(r) dOmega^2

together with its photon spheres and the inner edge of the accretion disk.
`BaseCompactObject` is the interface used by the rest of the package: subclass
it (as done in `gravityp.zoo`) or use `CompactObject` to build one directly from
plain Python functions.
"""

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
        for r_ph in r_phs[1:]:  # Skip first to avoid duplicated evaluation
            maximum = potential(r_ph)
            if maximum > local_maxima.max():  # Take increasing local maxima
                local_maxima = np.append(local_maxima, maximum)
                photon_sph = np.append(photon_sph, r_ph)
        # Swap to ascending order of photon_spheres
        return photon_sph[::-1], local_maxima[::-1]
    else:
        return np.array([]), np.array([])


class BaseCompactObject(ABC):
    r"""Abstract static, spherically symmetric compact object.

    The geometry is described by the line element

    .. math::

        ds^2 = -A(r)\,dt^2 + \frac{dr^2}{A(r)} + R(r)\,d\Omega^2

    where :math:`A(r)` is given by `radial` and :math:`R(r)` by `areal`.
    Subclasses must implement both methods and pass the photon spheres and the
    inner edge to ``super().__init__``.

    Parameters
    ----------
    r_phs : iterable of float
        Radial coordinates of the photon spheres, i.e. the local maxima of the
        effective potential `potential`. May be empty.
    inner_edge : float
        Radial coordinate of the inner edge of the geometry (event horizon,
        wormhole throat, etc.). Light rays reaching it are not continued.

    See Also
    --------
    CompactObject : Compact object built from user-supplied functions.
    gravityp.zoo : Predefined compact objects.

    Notes
    -----
    `r_phs` and `inner_edge` are read-only, so that cached quantities derived
    from them (e.g. `potential_maxima`) cannot go stale.

    Examples
    --------
    Reissner-Nordström black hole with charge `Q`, in units of M=1:

    >>> import numpy as np
    >>> from gravityp import BaseCompactObject
    >>> class ReissnerNordstrom(BaseCompactObject):
    ...     def __init__(self, Q):
    ...         super().__init__(
    ...             r_phs=[(3 + np.sqrt(9 - 8*Q**2)) / 2],
    ...             inner_edge=1 + np.sqrt(1 - Q**2),
    ...         )
    ...         self.Q = Q
    ...     def radial(self, r):
    ...         return 1 - 2/r + self.Q**2/r**2
    ...     def areal(self, r):
    ...         return r**2
    >>> rn = ReissnerNordstrom(Q=0.5)
    >>> print(round(rn.inner_edge, 4))
    1.866
    """

    def __init__(self, r_phs: Iterable[float], inner_edge: float) -> None:
        self._r_phs = tuple(r_phs)
        self._inner_edge = inner_edge

    @property
    def r_phs(self) -> tuple[float, ...]:
        """Radial coordinates of the photon spheres (read-only)."""
        return self._r_phs

    @property
    def inner_edge(self) -> float:
        """Radial coordinate of the inner edge of the geometry (read-only)."""
        return self._inner_edge

    @abstractmethod
    def radial(self, r: float | np.ndarray) -> float | np.ndarray:
        r"""Radial metric function :math:`A(r) = -g_{tt} = 1/g_{rr}`.

        Parameters
        ----------
        r : float or ndarray
            Radial coordinate.

        Returns
        -------
        float or ndarray
            Value of :math:`A(r)`, with the same shape as `r`.
        """

    @abstractmethod
    def areal(self, r: float | np.ndarray) -> float | np.ndarray:
        r"""Areal radius squared :math:`R(r) = g_{\theta\theta}`.

        Parameters
        ----------
        r : float or ndarray
            Radial coordinate.

        Returns
        -------
        float or ndarray
            Value of :math:`R(r)`, with the same shape as `r`.
        """

    def potential(self, r: float | np.ndarray) -> float | np.ndarray:
        r"""Effective potential for null geodesics, :math:`V(r) = A(r)/R(r)`.

        Parameters
        ----------
        r : float or ndarray
            Radial coordinate.

        Returns
        -------
        float or ndarray
            Value of :math:`V(r)`, with the same shape as `r`.
        """
        return self.radial(r) / self.areal(r)

    @cached_property
    def potential_maxima(self) -> tuple[np.ndarray, np.ndarray]:
        """Photon spheres acting as potential barriers, and their potentials.

        Tuple ``(photon_sph, local_maxima)`` of arrays sorted by increasing
        radius, hence `local_maxima` decreases along the arrays, and the
        outermost barrier comes last.

        Computed once per instance. The arrays are read-only, so the cached
        values cannot be modified in place.
        """
        photon_sph, local_maxima = _get_increasing_maxima(self.r_phs,
                                                          self.potential)
        photon_sph.flags.writeable = False
        local_maxima.flags.writeable = False
        return photon_sph, local_maxima


class CompactObject(BaseCompactObject):
    """Compact object defined from user-supplied metric functions.

    The additional parameters of `radial` and `areal` are bound by keyword at
    construction, so the resulting methods only take the radial coordinate.

    Parameters
    ----------
    r_phs : iterable of float
        Radial coordinates of the photon spheres, i.e. the local maxima of the
        effective potential. May be empty.
    inner_edge : float
        Radial coordinate of the inner edge of the geometry (event horizon,
        wormhole throat, etc.).
    radial : callable
        Radial metric function :math:`A(r)`, called as
        ``radial(r, **radial_params)``.
    areal : callable
        Areal radius squared :math:`R(r)`, called as
        ``areal(r, **areal_params)``.
    radial_params : mapping of str to float, optional
        Additional keyword arguments for `radial`. Default is None.
    areal_params : mapping of str to float, optional
        Additional keyword arguments for `areal`. Default is None.

    See Also
    --------
    BaseCompactObject : Abstract interface, to subclass instead.

    Notes
    -----
    Additional parameters are passed by keyword, so they can be neither
    positional-only nor named ``r``.

    Examples
    --------
    Reissner-Nordström black hole with mass `M` and charge `Q`:

    >>> from gravityp import CompactObject
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
