"""Predefined compact objects."""

import numpy as np
import scipy.optimize as optimize

from typing import cast

from .compact_object import BaseCompactObject


__all__ = ["Schwarzschild", "SimpsonVisser", "BardeenBH", "BardeenBB"]


class Schwarzschild(BaseCompactObject):
    r"""Schwarzschild black hole, in units of M=1.

    The metric functions are

    .. math::

        A(r) = 1 - \frac{2}{r}, \qquad R(r) = r^2.

    The photon sphere lies at :math:`r = 3`, with critical impact parameter
    :math:`b_c = \sqrt{27}`, and the inner edge is the event horizon at
    :math:`r = 2`.

    See Also
    --------
    BaseCompactObject : Interface implemented by this class.
    """

    def __init__(self) -> None:
        super().__init__(
            r_phs=[3.,],
            inner_edge=2.
        )

    def radial(self, r: float | np.ndarray) -> float | np.ndarray:
        return 1. - 2./r

    def areal(self, r: float | np.ndarray) -> float | np.ndarray:
        return r**2


class SimpsonVisser(BaseCompactObject):
    r"""Simpson-Visser geometry, in units of M=1.

    Regular geometry for :math:`a > 0`, with metric functions

    .. math::

        A(r) = 1 - \frac{2}{\sqrt{r^2 + a^2}}, \qquad R(r) = r^2 + a^2.

    Parameters
    ----------
    a : float
        Non-negative bounce parameter, interpolating between the
        Schwarzschild black hole (``a = 0``), a *black bounce*
        (``0 < a < 2``), a one-way wormhole with an extremal null throat
        (``a = 2``) and a two-way traversable wormhole (``a > 2``).

    Raises
    ------
    ValueError
        If `a` is negative.

    See Also
    --------
    BardeenBB : Bardeen-like black bounce.

    Notes
    -----
    The inner edge is the event horizon :math:`r = \sqrt{4 - a^2}` for
    :math:`a < 2`, and the throat :math:`r = 0` otherwise. The photon sphere
    lies at :math:`r = \sqrt{9 - a^2}` for :math:`a < 3`, and at the throat
    otherwise.

    References
    ----------
    .. [1] A. Simpson and M. Visser, "Black-bounce to traversable wormhole",
       arXiv:1812.07114.
    """

    def __init__(self, a: float) -> None:
        if a < 0:
            raise ValueError("a must be non-negative")
        super().__init__(
            r_phs=[np.sqrt(9. - a**2) if a < 3. else 0.,],
            inner_edge=np.sqrt(4. - a**2) if a < 2. else 0.
        )
        self._a = a

    @property
    def a(self) -> float:
        """Bounce parameter (read-only)."""
        return self._a

    def radial(self, r: float | np.ndarray) -> float | np.ndarray:
        return 1. - 2./np.sqrt(r**2+self.a**2)

    def areal(self, r: float | np.ndarray) -> float | np.ndarray:
        return r**2+self.a**2


def _depressed_cubic_root(p: float, q: float) -> float:
    """Returns the largest real root of a depressed cubic polynomial:
    t^3 + pt + q = 0
    """
    if p == 0:
        return -np.cbrt(q)
    disc = 4*p**3 + 27*q**2  # Discriminant
    if disc > 1e-10*(4*abs(p)**3 + 27*q**2):  # NOTE: still one real solution
        raise ValueError("There are complex roots")
    sqrt_p3 = np.sqrt(-p/3)
    arg = np.clip(3*q/(2*p*sqrt_p3), -1, 1)  # Ensure real-valued arccos
    return 2*sqrt_p3 * np.cos( np.arccos(arg)/3 )


def _general_cubic_root(a: float, b: float, c: float, d: float) -> float:
    """Returns the largest real root of a general cubic polynomial:
    ax^3 + bx^2 + cx + d = 0
    """
    if a == 0:
        raise ValueError("Leading coefficient a must be non-zero")
    t = _depressed_cubic_root(
        p=(3*a*c-b**2)/(3*a**2),
        q=(2*b**3-9*a*b*c+27*a**2*d)/(27*a**3)
    )
    return t - b/(3*a)
    

class BardeenBH(BaseCompactObject):
    r"""Non-rotating Bardeen regular black hole, in units of M=1.

    The metric functions are

    .. math::

        A(r) = 1 - \frac{2 r^2}{(r^2 + q^2)^{3/2}}, \qquad R(r) = r^2.

    Parameters
    ----------
    q : float
        Non-negative regularisation parameter, interpolating between the
        Schwarzschild black hole (``q = 0``), a regular black hole
        (``0 < q < Q_EXTREMAL``), an extremal regular black hole
        (``q = Q_EXTREMAL``) and a regular, horizonless geometry
        (``q > Q_EXTREMAL``).

    Attributes
    ----------
    Q_EXTREMAL : float
        Extremal value :math:`4/(3\sqrt{3})`: horizons exist only for
        ``q <= Q_EXTREMAL``.
    Q_PHOTON_SPHERE : float
        Critical value :math:`48/(25\sqrt{5})`: there is a single photon sphere
        for ``q < Q_PHOTON_SPHERE`` and none otherwise.

    Raises
    ------
    ValueError
        If `q` is negative.

    See Also
    --------
    BardeenBB : Bardeen-like black bounce.

    Notes
    -----
    The inner edge is the outer event horizon for ``q <= Q_EXTREMAL``, and
    :math:`r = 0` for horizonless geometries.

    References
    ----------
    .. [1] J. M. Bardeen, in Proceedings of GR5, Tbilisi, USSR (1968), p. 174.
    """

    Q_EXTREMAL = 4/(3*np.sqrt(3))
    Q_PHOTON_SPHERE = 48/(25*np.sqrt(5))

    def __init__(self, q: float) -> None:
        if q < 0:
            raise ValueError("q must be non-negative")
        super().__init__(
            r_phs=self._photon_spheres(q),
            inner_edge=self._outer_horizon(q)
        )
        self._q = q

    @classmethod
    def _outer_horizon(cls, q: float) -> float:
        """Largest root of A(r)=0, or 0 if the geometry is horizonless.

        With x=r^2: x^3 + (3q^2-4)x^2 + 3q^4 x + q^6 = 0
        """
        if q > cls.Q_EXTREMAL:
            return 0.
        return np.sqrt(_general_cubic_root(a=1, b=3*q**2-4, c=3*q**4, d=q**6))

    @classmethod
    def _photon_spheres(cls, q: float) -> list[float]:
        """Maximum of V(r)=A/r^2, i.e. largest root of
        (x+q^2)^(5/2) = 3x^2, with x=r^2.

        The root is bracketed in [4q^2, 9] for 0 < q < Q_PHOTON_SPHERE.
        """
        if q < 1e-8:  # r_ph = 3 - O(q^2) is exact to machine precision
            return [3.,]  # Schwarzschild limit is handled separately
        if q >= cls.Q_PHOTON_SPHERE:
            return []
        x_ph = optimize.brentq(
            lambda x: (x+q**2)**(5./2.) - 3*x**2, a=4*q**2, b=9.
        )
        return [np.sqrt(cast(float, x_ph)),]

    @property
    def q(self) -> float:
        """Regularisation parameter (read-only)."""
        return self._q

    def radial(self, r: float | np.ndarray) -> float | np.ndarray:
        return 1. - 2.*r**2/(r**2+self.q**2)**(3./2.)

    def areal(self, r: float | np.ndarray) -> float | np.ndarray:
        return r**2


class BardeenBB(BaseCompactObject):
    r"""Non-rotating Bardeen-like *black bounce*, in units of M=1.

    The metric functions are

    .. math::

        A(r) = 1 - \frac{2 r^2}{(r^2 + a^2)^{3/2}}, \qquad R(r) = r^2 + a^2.

    Parameters
    ----------
    a : float
        Non-negative bounce parameter, interpolating between the
        Schwarzschild black hole (``a = 0``), a *black bounce*
        (``0 < a < A_EXTREMAL``), an extremal *black bounce*
        (``a = A_EXTREMAL``) and a traversable wormhole (``a > A_EXTREMAL``).

    Attributes
    ----------
    A_EXTREMAL : float
        Extremal value :math:`4/(3\sqrt{3})`: horizons exist only for
        ``a <= A_EXTREMAL``.
    A_PHOTON_SPHERE : float
        Critical value :math:`2/\sqrt{5}`: the outer photon sphere exists only
        for ``a < A_PHOTON_SPHERE``.

    Raises
    ------
    ValueError
        If `a` is negative.

    See Also
    --------
    BardeenBH : Bardeen regular black hole.
    SimpsonVisser : Simpson-Visser black bounce.

    Notes
    -----
    The inner edge is the outer event horizon for ``a <= A_EXTREMAL``, and the
    throat :math:`r = 0` otherwise. There is a single photon sphere for
    ``a <= A_EXTREMAL``, two of them (the throat and an outer one) for
    ``A_EXTREMAL < a < A_PHOTON_SPHERE``, and only the throat for
    ``a >= A_PHOTON_SPHERE``.

    References
    ----------
    .. [1] Eq. (72) in arXiv:2009.12057.
    """

    A_EXTREMAL = 4/(3*np.sqrt(3))
    A_PHOTON_SPHERE = 2/np.sqrt(5)

    def __init__(self, a: float) -> None:
        if a < 0:
            raise ValueError("a must be non-negative")
        super().__init__(
            r_phs=self._photon_spheres(a),
            inner_edge=self._outer_horizon(a)
        )
        self._a = a

    @classmethod
    def _outer_horizon(cls, a: float) -> float:
        """Largest root of A(r)=0, or 0 if the geometry is horizonless.

        With x=r^2: x^3 + (3a^2-4)x^2 + 3a^4 x + a^6 = 0
        """
        if a > cls.A_EXTREMAL:
            return 0.
        return np.sqrt(_general_cubic_root(a=1, b=3*a**2-4, c=3*a**4, d=a**6))

    @classmethod
    def _photon_spheres(cls, a: float) -> list[float]:
        """Maxima of V(r)=A/(r^2+a^2): the throat r=0 (if not behind the
        horizon) and the largest root of

        y^3 - 3y^2 + 5a^2 = 0, with y=√(r^2+a^2),

        which exists only for a < A_PHOTON_SPHERE.
        """
        if a >= cls.A_PHOTON_SPHERE:
            return [0.,]
        y_ph = _general_cubic_root(a=1, b=-3, c=0, d=5*a**2)
        r_ph = np.sqrt(y_ph**2 - a**2)
        if a <= cls.A_EXTREMAL:
            return [r_ph,]  # Throat is behind the horizon
        return [0., r_ph]

    @property
    def a(self) -> float:
        """Bounce parameter (read-only)."""
        return self._a

    def radial(self, r: float | np.ndarray) -> float | np.ndarray:
        return 1. - 2.*r**2/(r**2+self.a**2)**(3./2.)

    def areal(self, r: float | np.ndarray) -> float | np.ndarray:
        return r**2 + self.a**2