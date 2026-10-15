"""Module containing predefined classes of different compact objects"""

import numpy as np

from .compact_object import BaseCompactObject


__all__ = ["Schwarzschild", "SimpsonVisser"]


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
            black hole (`a=0`), a *black bounce* (`0<a<2`), a one-way wormhole
            with an extremal null throat (`a=2`) and a two-way traversable
            wormhole (`a>2`).
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
        """Regularisation parameter"""
        return self._a

    def radial(self, r: float | np.ndarray) -> float | np.ndarray:
        return 1. - 2./np.sqrt(r**2+self.a**2)

    def areal(self, r: float | np.ndarray) -> float | np.ndarray:
        return r**2+self.a**2
