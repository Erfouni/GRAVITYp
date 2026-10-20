"""Module containing predefined classes of different compact objects"""

import numpy as np
import scipy.optimize as optimize

from typing import cast

from .compact_object import BaseCompactObject


__all__ = ["Schwarzschild", "SimpsonVisser", "BardeenBH", "BardeenBB"]


class Schwarzschild(BaseCompactObject):
    """Schwarzschild black hole in units of M=1. The metric reads:

    ds^2 = -A(r) dt^2 + 1/A(r) dr^2 + R(r) dOmega^2

    with A(r)=1-2/r and R(r)=r^2 .
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
    """Simpson-Visser metric in units of M=1.

    It describes regular geometries (for `a>0`) and it reads:

    ds^2 = -A(r) dt^2 + 1/A(r) dr^2 + R(r) dOmega^2

    with A(r)=1-2/√R(r) and R(r)=r^2+a^2 . For more information, see
    <https://arxiv.org/abs/1812.07114>.

    Parameters:
        a (float): continuous parameter interpolating between Schwarzschild
            black hole (`a = 0`), a *black bounce* (`0 < a < 2`), a one-way wormhole
            with an extremal null throat (`a = 2`) and a two-way traversable
            wormhole (`a > 2`).
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
        """Bounce parameter"""
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
    """Non-rotating Bardeen regular black hole in units of M=1:

    ds^2 = -A(r) dt^2 + 1/A(r) dr^2 + r^2 dOmega^2

    with A(r)=1-2r^2/(r^2+q^2)^(3/2) . For more information, see
    (J. M. Bardeen, 1968).

    Parameters:
        q (float): continuous parameter interpolating between Schwarzschild
            black hole (`q = 0`), a regular black hole (`0 < q < 4/(3√3)`), an
            extremal regular black hole (`q = 4/(3√3)`) and a regular,
            horizonless geometry (`q > 4/(3√3)`). This metric presents a single
            photon sphere for `0 <= q < 48/(25√5)` and no photon sphere for
            `q >= 48/(25√5)`.
    """

    Q_EXTREMAL = 4/(3*np.sqrt(3))
    """Extremal value of q: horizons exist only for q <= Q_EXTREMAL"""
    Q_PHOTON_SPHERE = 48/(25*np.sqrt(5))
    """Critical value of q: photon spheres exist only for q < Q_PHOTON_SPHERE"""

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
        """Regularisation parameter"""
        return self._q

    def radial(self, r: float | np.ndarray) -> float | np.ndarray:
        return 1. - 2.*r**2/(r**2+self.q**2)**(3./2.)

    def areal(self, r: float | np.ndarray) -> float | np.ndarray:
        return r**2


class BardeenBB(BaseCompactObject):
    """Non-rotating Bardeen-like *black bounce* in units of M=1:

    ds^2 = -A(r) dt^2 + 1/A(r) dr^2 + R(r) dOmega^2

    with A(r)=1-2r^2/(r^2+a^2)^(3/2) and R(r) = r^2+a^2 .
    For more information, see Eq. (72) in <https://arxiv.org/abs/2009.12057>.

    Parameters:
        a (float): bouncing parameter interpolating between Schwarzschild
            black hole (`a = 0`), a *black bounce* (`0 < a < 4/(3√3)`), an extremal
            *black bounce* (`a = 4/(3√3)`) and a traversable wormhole
            (`a > 4/(3√3)`). This metric presents a single photon sphere for
            `0 <= a <= 4/(3√3)`, two photon spheres for `4/(3√3) < a < 2/√5` and
            again a single photon sphere for `a >= 2/√5`.
    """

    A_EXTREMAL = 4/(3*np.sqrt(3))
    """Extremal value of a: horizons exist only for a <= A_EXTREMAL"""
    A_PHOTON_SPHERE = 2/np.sqrt(5)
    """Critical value of a: outer photon sphere exists only for a < A_PHOTON_SPHERE"""

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
        """Bounce parameter"""
        return self._a

    def radial(self, r: float | np.ndarray) -> float | np.ndarray:
        return 1. - 2.*r**2/(r**2+self.a**2)**(3./2.)

    def areal(self, r: float | np.ndarray) -> float | np.ndarray:
        return r**2 + self.a**2