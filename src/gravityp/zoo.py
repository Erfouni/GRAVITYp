"""Module containing predefined classes of different compact objects"""

import numpy as np

from .compact_object import BaseCompactObject


__all__ = ["Schwarzschild"]


class Schwarzschild(BaseCompactObject):
    """Schwarzschild black hole in units of M=1. The metric reads:

    ds^2 = -A(r) dt^2 + 1/A(r) dr^2 + R(r) dOmega^2

    with A(r)=1-2M/r (here M=1) and R(r)=r^2 .
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
