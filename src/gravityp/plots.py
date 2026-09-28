"""This module contains utilities and predefined functions to plot figures
abount the different steps of the code.
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .utils import to_tuple, to_list
from .geodesic_integration import compute_geodesic
from .ray_tracing import compute_nturns, get_order
from .transfer_functions import transfer_function
from .shadows import total_intensity, compute_points
from .embedding import surface_domain, make_embedding


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
    "make_shadow_plot",
    "make_embedding_plot"
]


def plot_metric_function(ax, r_range, fun, param, name):
    ax.plot(r_range, fun(r_range,param), label=f'{name}={param:.2f}')

def make_metric_function_plot(fun, params, r_range, figsize=(9,6), savepath=None):
    ax = plt.figure(figsize=figsize).add_subplot()
    ax.set_title(f'{fun.__name__[2:].capitalize()} metric function')

    for name, param_list in params.items():
        for param in param_list:
            plot_metric_function(ax, r_range, fun, param, name)

    ax.set_xlim(r_range[0], r_range[-1])
    ax.set_ylim(-1,1)
    ax.set_xlabel(r'$r/M$')
    ax.set_ylabel(r'$e^{2\nu}$')
    xticks = np.arange(r_range[0],r_range[-1]+1,2,dtype=int)
    yticks = np.arange(-1,1.5,0.5)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xticks)
    ax.set_yticklabels(yticks)
    ax.hlines(0, r_range[0], r_range[-1],color='black',linewidth=1)
    ax.legend()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()

def plot_half_graph(ax, bs, plot_label=False, **kwargs):
    """Auxiliary function for plot_nturns"""
    nturns = compute_nturns(bs, **kwargs)
    upper = 1.25
    lower = 0.75
    
    lines_data = {
        'Direct': (np.ma.masked_where(nturns >= lower, nturns), 'black'),
        'Lensed': (np.ma.masked_where((nturns < lower) | (nturns >= upper), nturns), 'orange'),
        'Photon ring': (np.ma.masked_where(nturns < upper, nturns), 'red')
    }

    for label, (y_data, color) in lines_data.items():
        plot_kwargs = {'color': color}
        if plot_label:
            plot_kwargs['label'] = label
        ax.plot(bs, y_data, **plot_kwargs)

def plot_nturns(bs, b_crits, figsize=(7,7), savepath=None, **kwargs):
    ax = plt.figure(figsize=figsize).add_subplot()
    xticks = [i for i in range(0,11,2)]
    xtick_labels = [f'{tick}' for tick in xticks]

    if len(b_crits)==0:
        plot_half_graph(ax, bs, plot_label=True, **kwargs)
    else:
        # Interval from 0 to first b_crit, plotting labels
        bs_interval = bs[ bs <= b_crits[0]]
        plot_half_graph(ax, bs_interval, plot_label=True, **kwargs)
        ax.vlines(b_crits[0],0,2,colors='gray',linestyles='dashed',zorder=0)
        xticks += [b_crits[0]]
        if len(b_crits)==1:
            xtick_labels += [rf'$b_c$']
        else:
            xtick_labels += [rf'$b_c^1$']

        # Intermediate intervals [b1,b2], [b2,b3], ...
        if len(b_crits) > 1:
            for i, b_crit in enumerate(b_crits[1:], start=1):
                bs_interval = bs[ (bs >= b_crits[i-1]) & (bs < b_crit) ]
                plot_half_graph(ax, bs_interval, plot_label=False, **kwargs)
                ax.vlines(b_crit,0,2,colors='gray',linestyles='dashed',zorder=0)
                xticks += [b_crit]
                xtick_labels += [rf'$b_c^{i+1}$']

        # Last interval from last b_crit to infinity
        bs_interval = bs[ bs >= b_crits[-1] ]
        plot_half_graph(ax, bs_interval, plot_label=False, **kwargs)

    ax.set_xlim(0,10)
    ax.set_ylim(0,2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_yticks(np.arange(0,2.1,0.25))
    ax.set_xlabel(r'$b/M$')
    ax.set_ylabel(r'$n=\phi /(2\pi)$')
    ax.legend(loc='upper right')
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()

def plot_geodesic(axes, areal, areal_params, rr, phi, color=None):
    # NOTE: the plotted coordinate refers to the areal radius
    if color is None:
        order = get_order(phi[-1])
        color = ['dodgerblue','orange','red'][order]

    areal_radius = np.sqrt( areal(rr,*to_tuple(areal_params)) )
    trajectory = (areal_radius*np.cos(phi), areal_radius*np.sin(phi))
    axes.plot(*trajectory, color=color, linewidth=0.5, alpha=1)

def plot_BH(axes, r_phs, inner_edge, areal, areal_params):
    # NOTE: Within this function, the variable r refers to the areal radius
    i_edge_kwargs = {'color':'black', 'fill':(inner_edge!=0)}
    p_ring_kwargs = {'color':'black', 'fill':False}

    r_i_edge = np.sqrt( areal(inner_edge, *to_tuple(areal_params)) )
    patch_i_edge = Circle((0, 0), r_i_edge, **i_edge_kwargs, zorder=3)
    axes.add_patch(patch_i_edge)
    for r_ph in r_phs:
        r_photon = np.sqrt( areal(r_ph, *to_tuple(areal_params)) )
        photon_ring = Circle((0, 0), r_photon, linestyle=(0,(5,2)), linewidth=2/3, zorder=3, **p_ring_kwargs)
        axes.add_patch(photon_ring)

def make_geodesics_plot(bs_dict, r_phs, inner_edge, radial_fun, radial_params, areal, areal_params,
                        figsize=(7,7), savepath=None):
    ax = plt.figure(figsize=figsize).add_subplot()

    if bs_dict.__class__ != dict:
        bs_dict = {'label': (None, bs_dict)} # Cast to dict
    for label, (color, bs) in bs_dict.items():
        for b in np.atleast_1d(bs):
            coords = compute_geodesic(
                b, r_phs, inner_edge, radial_fun, radial_params, areal, areal_params
            )
            plot_geodesic(ax, areal, areal_params, *coords, color)
        
    plot_BH(ax, r_phs, inner_edge, areal, areal_params)

    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_xticks(np.arange(-10,11,5))
    ax.set_yticks(np.arange(-10,11,5))
    ax2 = ax.twinx() 
    ax2.set_ylim(-10, 10)                 # Keep limits synced with ax
    ax2.set_yticks(np.arange(0, 11, 2))   # Ticks: 0, 1, 2, ..., 10
    
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()

def plot_transfer_function(axes, bs, order, correction: float=0, **kwargs):
    # NOTE: bs in bs_list can also be lists for lensed and photon ring, which will be plotted separately
    colors = ['dodgerblue','orange','red']
    image = ['Direct','Lensed','Photon ring']
    points = transfer_function(bs, order, correction, **kwargs)
    axes.plot(*points, color=colors[order], label=f'{image[order]}', linewidth=1)

def make_transfer_function_plot(b_crits, kwargs, bs_list,
                                correction=0, figsize=(7,7), savepath=None):
    fig, ax = plt.subplots(figsize=figsize)

    for order, bs in enumerate(bs_list):
        if order==0:
            plot_transfer_function(ax, bs, order, correction, **kwargs)
        else:
            for bs_ring in to_list(bs):
                plot_transfer_function(ax, bs_ring, order, correction, **kwargs)
            

    xticks = [i for i in range(0,11,2)]
    xticklabels = [f'{t}' for t in xticks]
    # xticks += [b_crit]
    # xticklabels += [r'$b_c$']

    ax.set_xlabel(r'$b/M$')
    ax.set_ylabel(r'$r_m/M$')
    ax.set_xlim(0,10)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylim(0,15)
    ax.set_yticks(np.arange(0,16,5))
    
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()

## Intensity and shadows
def add_extra_tick(xticks, xticklabels, extra_tick=None, extra_ticklabel=None):
    if (extra_tick is not None) and (extra_ticklabel is not None):
        threshold = 0.6  # Minimum allowed distance between labels
        xticks = [t for t in xticks if abs(t - extra_tick) > threshold]
        xticklabels = [f'{t}' for t in xticks]
        xticks += [extra_tick]
        xticklabels += [extra_ticklabel]
    return xticks, xticklabels

def plot_emission_model(rs, emission_model, text_string=None,
                        tick=None, ticklabel=None, figsize=(7,7), savepath=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(rs, emission_model(rs), label=r'$I_{\mathrm{em}}/I_0$')
    
    rmin = round(rs.min())
    rmax = round(rs.max())

    # Add an extra xtick placed at specified position
    xticks = [i for i in range(rmin,rmax,2)]
    xticklabels = [f'{t}' for t in xticks]
    xticks, xticklabels = add_extra_tick(xticks, xticklabels, tick, ticklabel)

    ax.set_xlabel(r'$r/M$')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(rmin,rmax)
    # ax.set_ylabel(r'$I_\mathrm{em}/I_0$')
    ax.set_ylim(0,1)
    if text_string is not None:
        ax.text( 7.5, 0.8, s=text_string )
    ax.legend()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()

def plot_observed_intensity(bs, bs_transfer_list, emission_model, kwargs,
                            figsize=(7,7), savepath=None, y_range=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(bs, total_intensity(bs, bs_transfer_list, emission_model, kwargs),
            label=r'$I_{\mathrm{ob}}/I_0$')
    
    bmin = round(bs.min())
    bmax = round(bs.max())
    ax.set_xlabel(r'$b/M$')
    ax.set_xticks(range(bmin,bmax+2,2))
    ax.set_xlim(bmin,bmax)
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

def make_shadow_plot(bs_transfer_list, emission_model, kwargs,
                     figsize=(7,7), savepath=None, y_range=None, Npixels=1e6):
    
    Z = compute_points(bs_transfer_list, emission_model, kwargs, Npixels)

    fig, ax = plt.subplots(figsize=figsize)
    im = plot_intensities(ax, Z, *to_tuple(y_range))
    ax.set_xticks(np.arange(-10, 11, 5))
    ax.set_yticks(np.arange(-10, 11, 5))
    plot_colorbar(fig, ax, im)

    if savepath is not None:
        plt.savefig(savepath)
    plt.show()

def make_embedding_plot(bs, r_phs,inner_edge, radial_fun, radial_params, areal, areal_params,
                        figsize=(7,7), savepath=None):
    
    x_emb, y_emb = surface_domain(inner_edge, areal, areal_params)

    r_emb, z_emb = make_embedding(inner_edge, radial_fun, radial_params, areal, areal_params)
    z_surface = np.broadcast_to(z_emb[:, None], x_emb.shape)

    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": "3d"},
    )
    ax.plot_surface(x_emb, y_emb, z_surface, color="0.8",
                alpha=0.3, linewidth=0)
    ax.set_axis_off()

    coords = compute_geodesic(
                bs, r_phs, inner_edge, radial_fun, radial_params, areal, areal_params)
    r_ray, phi_ray = coords

    valid = (
        (r_ray >= r_emb[0])
        & (r_ray <= r_emb[-1])
    )# Only the region onto the embedding surface.

    r_ray = r_ray[valid]
    phi_ray = phi_ray[valid]

    z_ray = np.interp(r_ray, r_emb, z_emb)# Evaluate z(r) by interpolation on the embedding profile.

    R_ray = np.sqrt(
        areal(r_ray, *to_tuple(areal_params))
    ) # Convert (r, phi) into Cartesian embedding coordinates.

    x_ray = R_ray * np.cos(phi_ray)
    y_ray = R_ray * np.sin(phi_ray)

    ax.plot(
        x_ray,
        y_ray,
        z_ray,
        color="red",
        linewidth=1.5,
    )
    
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()