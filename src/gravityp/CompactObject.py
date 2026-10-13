"""Module containing the implementation of a generic compact object"""

from collections.abc import Callable


__all__ = [
    "CompactObject",
]


class CompactObject:
    def __init__(
        self, r_phs: list[float], inner_edge: float, radial_fun: Callable,
        radial_params: tuple[float, ...] | None, areal_fun: Callable,
        areal_params: tuple[float, ...] | None
    ) -> None:
        self.r_phs = r_phs
        self.inner_edge = inner_edge
        self.radial_fun = radial_fun
        self.radial_params = radial_params
        self.areal_fun = areal_fun
        self.areal_params = areal_params
