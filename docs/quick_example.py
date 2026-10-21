import numpy as np
import matplotlib.pyplot as plt
from gravityp import CompactObject, make_geodesics_plot


def g_rr(r: float) -> float:
    """Radial function g^{rr} of the Schwarzschild metric in units of M=1"""
    return 1 - 2./r


def g_thth(r: float) -> float:
    """Areal radius squared g^{theta theta} of the Schwarzschild in units of M=1"""
    return r**2


schwarzschild = CompactObject(
    r_phs=[3.,],        # Photon sphere at r=3M
    inner_edge=2.,      # Inner edge of disk at horizon r=2M
    radial=g_rr,        # Above defined radial function
    areal=g_thth,       # Above defined areal radius squared
)


if __name__ == '__main__':
    # Array of impact parameters from 0M to 10M
    impact_parameters = np.arange(0, 10, 0.2)
    fig, ax = make_geodesics_plot(impact_parameters, schwarzschild, figsize=(4, 4))
    plt.show()
