import numpy as np
from gravityp import CompactObject
from gravityp import intensity_at_inner_edge, compute_optimal_array, find_rings_list
from gravityp import plot_emission_model, plot_observed_intensity, make_shadow_plot


def g_rr(r: float) -> float:
    """Radial function g^{rr} of the Schwarzschild metric in units of M=1"""
    return 1 - 2./r


def g_thth(r: float) -> float:
    """Areal radius squared g^{theta theta} of the Schwarzschild in units of M=1"""
    return r**2


schw = CompactObject(
    r_phs=[3.,],        # Photon sphere at r=3M
    inner_edge=2.,      # Inner edge of disk at horizon r=2M
    radial=g_rr,        # Above defined radial function
    areal=g_thth,       # Above defined areal radius squared
)


if __name__ == '__main__':
    figsize = (4, 4)
    savepath = None

    b_crits = [np.sqrt(27)]
    rings_Schwarzschild = find_rings_list(b_crits, schw)

    savepath = 'images/README_emitted.png'  # Uncomment for saving file
    def emission_model(r): return intensity_at_inner_edge(r, schw.inner_edge)
    rs = np.linspace(0, 13, 1000)
    plot_emission_model(rs, emission_model, figsize=figsize, savepath=savepath)

    savepath = 'images/README_observed.png'  # Uncomment for saving file
    bs = np.linspace(0, 10*np.sqrt(2), 1000)
    bs_transfer_list = compute_optimal_array(
        0, 10*np.sqrt(2), rings_Schwarzschild, Npoints=100, joint=False, fill=True)
    plot_observed_intensity(bs, bs_transfer_list, emission_model, schw,
                            figsize=figsize, savepath=savepath)

    savepath = 'images/README_shadow.png'  # Uncomment for saving file
    make_shadow_plot(bs_transfer_list, emission_model, schw,
                     figsize=figsize, savepath=savepath)
