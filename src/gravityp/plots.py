"""This module contains utilities and predefined functions to plot figures
abount the different steps of the code.
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .compact_object import BaseCompactObject
from .geodesic_integration import compute_geodesic
from .ray_tracing import compute_nturns, get_order
from .transfer_functions import transfer_function
from .shadows import total_intensity, compute_points


__all__ = [
    "plot_metric_function",
    "make_metric_function_plot",
    "plot_half_graph",
    "plot_nturns",
    "plot_geodesic",
    "plot_BH",
    "make_geodesics_plot",
    "plot_transfer_function",
    "make_transfer_function_plot",
    "add_extra_tick",
    "plot_emission_model",
    "plot_observed_intensity",
    "plot_intensities",
    "plot_colorbar",
    "make_shadow_plot"
]


def plot_metric_function(ax, r_range, fun, param, name):
    ax.plot(r_range, fun(r_range, param), label=f'{name}={param:.2f}')


def make_metric_function_plot(fun, params, r_range, figsize=(9, 6),
                              savepath=None):
    ax = plt.figure(figsize=figsize).add_subplot()
    ax.set_title(f'{fun.__name__[2:].capitalize()} metric function')

    for name, param_list in params.items():
        for param in param_list:
            plot_metric_function(ax, r_range, fun, param, name)

    ax.set_xlim(r_range[0], r_range[-1])
    ax.set_ylim(-1, 1)
    ax.set_xlabel(r'$r/M$')
    ax.set_ylabel(r'$e^{2\nu}$')
    xticks = np.arange(r_range[0], r_range[-1]+1, 2, dtype=int)
    yticks = np.arange(-1, 1.5, 0.5)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xticks)
    ax.set_yticklabels(yticks)
    ax.hlines(0, r_range[0], r_range[-1], color='black', linewidth=1)
    ax.legend()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()


def plot_half_graph(ax, bs, co: BaseCompactObject, plot_label=False,
                    **geodesic_kwargs):
    """Auxiliary function for plot_nturns"""
    nturns = compute_nturns(bs, co, **geodesic_kwargs)
    upper = 1.25
    lower = 0.75

    lines_data = {
        'Direct': (np.ma.masked_where(nturns >= lower, nturns), 'black'),
        'Lensed': (
            np.ma.masked_where((nturns < lower) | (nturns >= upper), nturns),
            'orange'
        ),
        'Photon ring': (np.ma.masked_where(nturns < upper, nturns), 'red')
    }

    for label, (y_data, color) in lines_data.items():
        plot_kwargs = {'color': color}
        if plot_label:
            plot_kwargs['label'] = label
        ax.plot(bs, y_data, **plot_kwargs)


def plot_nturns(bs, b_crits, co: BaseCompactObject, figsize=(7, 7),
                savepath=None, **geodesic_kwargs):
    ax = plt.figure(figsize=figsize).add_subplot()
    xticks = [i for i in range(0, 11, 2)]
    xtick_labels = [f'{tick}' for tick in xticks]

    if len(b_crits) == 0:
        plot_half_graph(ax, bs, co, plot_label=True, **geodesic_kwargs)
    else:
        # Interval from 0 to first b_crit, plotting labels
        bs_interval = bs[bs <= b_crits[0]]
        plot_half_graph(ax, bs_interval, co, plot_label=True,
                        **geodesic_kwargs)
        ax.vlines(b_crits[0], 0, 2, colors='gray',
                  linestyles='dashed', zorder=0)
        xticks += [b_crits[0]]
        if len(b_crits) == 1:
            xtick_labels += [rf'$b_c$']
        else:
            xtick_labels += [rf'$b_c^1$']

        # Intermediate intervals [b1,b2], [b2,b3], ...
        if len(b_crits) > 1:
            for i, b_crit in enumerate(b_crits[1:], start=1):
                bs_interval = bs[(bs >= b_crits[i-1]) & (bs < b_crit)]
                plot_half_graph(ax, bs_interval, co, plot_label=False,
                                **geodesic_kwargs)
                ax.vlines(b_crit, 0, 2, colors='gray',
                          linestyles='dashed', zorder=0)
                xticks += [b_crit]
                xtick_labels += [rf'$b_c^{i+1}$']

        # Last interval from last b_crit to infinity
        bs_interval = bs[bs >= b_crits[-1]]
        plot_half_graph(ax, bs_interval, co,
                        plot_label=False, **geodesic_kwargs)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_yticks(np.arange(0, 2.1, 0.25))
    ax.set_xlabel(r'$b/M$')
    ax.set_ylabel(r'$n=\phi /(2\pi)$')
    ax.legend(loc='upper right')
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()


def plot_geodesic(axes, co: BaseCompactObject, rr, phi, color=None):
    # NOTE: the plotted coordinate refers to the areal radius
    if color is None:
        order = get_order(phi[-1])
        color = ['dodgerblue', 'orange', 'red'][order]

    areal_radius = np.sqrt(co.areal(rr))
    trajectory = (areal_radius*np.cos(phi), areal_radius*np.sin(phi))
    axes.plot(*trajectory, color=color, linewidth=0.5, alpha=1)


def plot_BH(axes, co: BaseCompactObject):
    # NOTE: Within this function, the variable r refers to the areal radius
    i_edge_kwargs = {'color': 'black', 'fill': (co.inner_edge != 0)}
    p_ring_kwargs = {'color': 'black', 'fill': False}

    r_i_edge = float(np.sqrt(co.areal(co.inner_edge)))
    patch_i_edge = Circle((0, 0), r_i_edge, **i_edge_kwargs, zorder=3)
    axes.add_patch(patch_i_edge)
    for r_ph in co.r_phs:
        r_photon = float(np.sqrt(co.areal(r_ph)))
        photon_ring = Circle((0, 0), r_photon, linestyle=(
            0, (5, 2)), linewidth=2/3, zorder=3, **p_ring_kwargs)
        axes.add_patch(photon_ring)


def make_geodesics_plot(bs_dict, co: BaseCompactObject, figsize=(7, 7),
                        savepath=None, **geodesic_kwargs):
    ax = plt.figure(figsize=figsize).add_subplot()

    if bs_dict.__class__ != dict:
        bs_dict = {'label': (None, bs_dict)}  # Cast to dict
    for label, (color, bs) in bs_dict.items():
        for b in np.atleast_1d(bs):
            coords = compute_geodesic(b, co, **geodesic_kwargs)
            plot_geodesic(ax, co, *coords, color)
    plot_BH(ax, co)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xticks(np.arange(-10, 11, 5))
    ax.set_yticks(np.arange(-10, 11, 5))
    ax2 = ax.twinx()
    ax2.set_ylim(-10, 10)                 # Keep limits synced with ax
    ax2.set_yticks(np.arange(0, 11, 2))   # Ticks: 0, 1, 2, ..., 10

    if savepath is not None:
        plt.savefig(savepath)
    plt.show()


def plot_transfer_function(axes, bs, order, co: BaseCompactObject,
                           correction=0, **geodesic_kwargs):
    # NOTE: bs in bs_list can also be lists for lensed and photon ring, which
    # will be plotted separately
    order_colors = ['dodgerblue', 'orange', 'red']
    image = ['Direct', 'Lensed', 'Photon ring']
    points = transfer_function(bs, order, co, correction, **geodesic_kwargs)
    axes.plot(*points, color=order_colors[order],
              label=f'{image[order]}', linewidth=1)


def make_transfer_function_plot(b_crits, co: BaseCompactObject, bs_list,
                                correction=0, figsize=(7, 7), savepath=None,
                                **geodesic_kwargs):
    fig, ax = plt.subplots(figsize=figsize)

    for order, bs in enumerate(bs_list):
        if order == 0:
            plot_transfer_function(
                ax, bs, order, co, correction, **geodesic_kwargs)
        else:
            bs_rings = bs if isinstance(bs, (list, tuple)) else [bs]
            for bs_ring in bs_rings:
                plot_transfer_function(
                    ax, bs_ring, order, co, correction, **geodesic_kwargs)

    xticks = [i for i in range(0, 11, 2)]
    xticklabels = [f'{t}' for t in xticks]
    # xticks += [b_crit]
    # xticklabels += [r'$b_c$']

    ax.set_xlabel(r'$b/M$')
    ax.set_ylabel(r'$r_m/M$')
    ax.set_xlim(0, 10)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylim(0, 15)
    ax.set_yticks(np.arange(0, 16, 5))

    if savepath is not None:
        plt.savefig(savepath)
    plt.show()

# Intensity and shadows


def add_extra_tick(xticks, xticklabels, extra_tick=None, extra_ticklabel=None):
    if (extra_tick is not None) and (extra_ticklabel is not None):
        threshold = 0.6  # Minimum allowed distance between labels
        xticks = [t for t in xticks if abs(t - extra_tick) > threshold]
        xticklabels = [f'{t}' for t in xticks]
        xticks += [extra_tick]
        xticklabels += [extra_ticklabel]
    return xticks, xticklabels


def plot_emission_model(rs, emission_model, text_string=None,
                        tick=None, ticklabel=None, figsize=(7, 7),
                        savepath=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(rs, emission_model(rs), label=r'$I_{\mathrm{em}}/I_0$')

    rmin = round(rs.min())
    rmax = round(rs.max())

    # Add an extra xtick placed at specified position
    xticks = [i for i in range(rmin, rmax, 2)]
    xticklabels = [f'{t}' for t in xticks]
    xticks, xticklabels = add_extra_tick(xticks, xticklabels, tick, ticklabel)

    ax.set_xlabel(r'$r/M$')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(rmin, rmax)
    # ax.set_ylabel(r'$I_\mathrm{em}/I_0$')
    ax.set_ylim(0, 1)
    if text_string is not None:
        ax.text(7.5, 0.8, s=text_string)
    ax.legend()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()


def plot_observed_intensity(bs, bs_transfer_list, emission_model,
                            co: BaseCompactObject, figsize=(7, 7),
                            savepath=None, y_range=None, **geodesic_kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(bs, total_intensity(bs, bs_transfer_list, emission_model, co,
                                **geodesic_kwargs),
            label=r'$I_{\mathrm{ob}}/I_0$')

    bmin = round(bs.min())
    bmax = round(bs.max())
    ax.set_xlabel(r'$b/M$')
    ax.set_xticks(range(bmin, bmax+2, 2))
    ax.set_xlim(bmin, bmax)
    ax.set_ylim(y_range)
    ax.legend()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()


def plot_intensities(ax, z, vmin=None, vmax=None):
    return ax.imshow(
        z,
        norm=colors.PowerNorm(1, vmin, vmax),
        cmap='inferno',
        extent=(-10, 10, -10, 10),
        origin='lower',
        interpolation='none',
    )


def plot_colorbar(fig, ax, im):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cb = fig.colorbar(im, cax=cax)


def make_shadow_plot(bs_transfer_list, emission_model, co: BaseCompactObject,
                     figsize=(7, 7), savepath=None, y_range=None, Npixels=1e6,
                     **geodesic_kwargs):

    Z = compute_points(bs_transfer_list, emission_model,
                       co, Npixels, **geodesic_kwargs)

    fig, ax = plt.subplots(figsize=figsize)
    vmin, vmax = y_range if y_range is not None else (None, None)
    im = plot_intensities(ax, Z, vmin, vmax)
    ax.set_xticks(np.arange(-10, 11, 5))
    ax.set_yticks(np.arange(-10, 11, 5))
    plot_colorbar(fig, ax, im)

    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
