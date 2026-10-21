"""Plots of each step of the ray-tracing pipeline."""

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
    "make_metric_function_plot",
    "plot_nturns",
    "make_geodesics_plot",
    "make_transfer_function_plot",
    "plot_emission_model",
    "plot_observed_intensity",
    "make_shadow_plot"
]


def _get_fig_ax(ax, figsize):
    """Returns the figure and axes to draw on, creating them if ax is None"""
    if ax is None:
        return plt.subplots(figsize=figsize)
    return ax.figure, ax


def _plot_metric_function(ax, r_range, fun, param, name):
    """Plots fun(r, param) on ax, labelled as name=param"""
    ax.plot(r_range, fun(r_range, param), label=f'{name}={param:.2f}')


def make_metric_function_plot(fun, params, r_range, figsize=(9, 6), ax=None):
    """Plot a metric function for several values of its parameter.

    Parameters
    ----------
    fun : callable
        Metric function, called as ``fun(r, param)``. The title is taken
        from its name without the first two characters, e.g. ``A_bardeen``
        gives "Bardeen metric function".
    params : dict of str to sequence of float
        Maps the name of the parameter, shown in the legend, to the values to
        plot. Each value is passed as the second positional argument of `fun`.
    r_range : ndarray
        Increasing radial coordinates where `fun` is evaluated, which also set
        the limits of the horizontal axis.
    figsize : tuple of float, optional
        Size of the new figure, ignored if `ax` is given. Default is (9, 6).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None (default), a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot.
    ax : matplotlib.axes.Axes
        Axes containing the plot.

    Notes
    -----
    The vertical axis is fixed to ``[-1, 1]``.
    """
    fig, ax = _get_fig_ax(ax, figsize)
    ax.set_title(f'{fun.__name__[2:].capitalize()} metric function')

    for name, param_list in params.items():
        for param in param_list:
            _plot_metric_function(ax, r_range, fun, param, name)

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
    return fig, ax


def _plot_half_graph(ax, bs, co: BaseCompactObject, plot_label=False,
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


def plot_nturns(bs, b_crits, co: BaseCompactObject, figsize=(7, 7), ax=None,
                **geodesic_kwargs):
    """Plot the number of turns of light rays against their impact parameter.

    The curve is coloured by image type: direct (black), lensed (orange) and
    photon ring (red). Each critical impact parameter is marked with a dashed
    vertical line.

    Parameters
    ----------
    bs : ndarray
        Increasing impact parameters of the light rays, in units of M.
    b_crits : sequence of float
        Critical impact parameters of the photon spheres, in increasing order.
        May be empty.
    co : BaseCompactObject
        Compact object defining the geometry.
    figsize : tuple of float, optional
        Size of the new figure, ignored if `ax` is given. Default is (7, 7).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None (default), a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot.
    ax : matplotlib.axes.Axes
        Axes containing the plot.

    Other Parameters
    ----------------
    **geodesic_kwargs
        Keyword arguments passed to `gravityp.compute_geodesic`, such as
        `Npoints` or `r0`.

    See Also
    --------
    gravityp.compute_nturns : Number of turns of light rays.

    Notes
    -----
    The axes are fixed to ``0 <= b <= 10`` and ``0 <= n <= 2``.
    """
    fig, ax = _get_fig_ax(ax, figsize)
    xticks = [i for i in range(0, 11, 2)]
    xtick_labels = [f'{tick}' for tick in xticks]

    if len(b_crits) == 0:
        _plot_half_graph(ax, bs, co, plot_label=True, **geodesic_kwargs)
    else:
        # Interval from 0 to first b_crit, plotting labels
        bs_interval = bs[bs <= b_crits[0]]
        _plot_half_graph(ax, bs_interval, co, plot_label=True,
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
                _plot_half_graph(ax, bs_interval, co, plot_label=False,
                                 **geodesic_kwargs)
                ax.vlines(b_crit, 0, 2, colors='gray',
                          linestyles='dashed', zorder=0)
                xticks += [b_crit]
                xtick_labels += [rf'$b_c^{i+1}$']

        # Last interval from last b_crit to infinity
        bs_interval = bs[bs >= b_crits[-1]]
        _plot_half_graph(ax, bs_interval, co,
                         plot_label=False, **geodesic_kwargs)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_yticks(np.arange(0, 2.1, 0.25))
    ax.set_xlabel(r'$b/M$')
    ax.set_ylabel(r'$n=\phi /(2\pi)$')
    ax.legend(loc='upper right')
    return fig, ax


def _plot_geodesic(axes, co: BaseCompactObject, rr, phi, color=None):
    """Plots a trajectory, coloured by its order if color is None"""
    # NOTE: the plotted coordinate refers to the areal radius
    if color is None:
        order = get_order(phi[-1])
        color = ['dodgerblue', 'orange', 'red'][order]

    areal_radius = np.sqrt(co.areal(rr))
    trajectory = (areal_radius*np.cos(phi), areal_radius*np.sin(phi))
    axes.plot(*trajectory, color=color, linewidth=0.5, alpha=1)


def _plot_BH(axes, co: BaseCompactObject):
    """Draws the inner edge and the photon spheres of the compact object"""
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
                        ax=None, **geodesic_kwargs):
    r"""Plot the trajectories of light rays around a compact object.

    The trajectories are drawn in the plane of motion, taking the areal radius
    :math:`\sqrt{R(r)}` as radial coordinate. The inner edge is drawn as a
    black disk (an empty circle if ``co.inner_edge == 0``), and the photon
    spheres as dashed circles.

    Parameters
    ----------
    bs_dict : array_like or dict
        Impact parameters of the light rays, in units of M, coloured by order:
        direct (blue), lensed (orange) and photon ring (red). Alternatively, a
        dictionary mapping labels to ``(color, bs)`` pairs, to give each group
        of impact parameters its own colour. A None colour colours the group
        by order.
    co : BaseCompactObject
        Compact object defining the geometry.
    figsize : tuple of float, optional
        Size of the new figure, ignored if `ax` is given. Default is (7, 7).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None (default), a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot.
    ax : matplotlib.axes.Axes
        Axes containing the plot.

    Other Parameters
    ----------------
    **geodesic_kwargs
        Keyword arguments passed to `gravityp.compute_geodesic`, such as
        `Npoints` or `r0`.

    See Also
    --------
    gravityp.compute_geodesic : Trajectory of a light ray.

    Notes
    -----
    The axes are fixed to ``[-10, 10]``. Light rays come from the right, so a
    secondary vertical axis on the right marks their impact parameters from 0
    to 10. The labels of `bs_dict` are not used.
    """
    fig, ax = _get_fig_ax(ax, figsize)

    if bs_dict.__class__ != dict:
        bs_dict = {'label': (None, bs_dict)}  # Cast to dict
    for label, (color, bs) in bs_dict.items():
        for b in np.atleast_1d(bs):
            coords = compute_geodesic(b, co, **geodesic_kwargs)
            _plot_geodesic(ax, co, *coords, color)
    _plot_BH(ax, co)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xticks(np.arange(-10, 11, 5))
    ax.set_yticks(np.arange(-10, 11, 5))
    ax2 = ax.twinx()
    ax2.set_ylim(-10, 10)                 # Keep limits synced with ax
    ax2.set_yticks(np.arange(0, 11, 2))   # Ticks: 0, 1, 2, ..., 10
    return fig, ax


def _plot_transfer_function(axes, bs, order, co: BaseCompactObject,
                            correction=0, **geodesic_kwargs):
    """Plots the transfer function of the given order, labelled by image type"""
    # NOTE: bs in bs_list can also be lists for lensed and photon ring, which
    # will be plotted separately
    order_colors = ['dodgerblue', 'orange', 'red']
    image = ['Direct', 'Lensed', 'Photon ring']
    points = transfer_function(bs, order, co, correction, **geodesic_kwargs)
    axes.plot(*points, color=order_colors[order],
              label=f'{image[order]}', linewidth=1)


def make_transfer_function_plot(b_crits, co: BaseCompactObject, bs_list,
                                correction=0, figsize=(7, 7), ax=None,
                                **geodesic_kwargs):
    """Plot the transfer functions of each order.

    The curves are coloured and labelled by image type: direct (blue), lensed
    (orange) and photon ring (red), so a legend can be added with
    ``ax.legend()``.

    Parameters
    ----------
    b_crits : sequence of float
        Critical impact parameters of the photon spheres. Currently unused.
    co : BaseCompactObject
        Compact object defining the geometry.
    bs_list : sequence
        Impact parameters sampling the transfer function of each order,
        starting from order 0. For orders 1 and 2, an element can also be a
        list or tuple of arrays, plotted as separate curves (e.g. the inner
        and outer parts of a ring).
    correction : float, optional
        Additional angle the light ray must travel to reach the accretion
        disk, which depends on the observer's inclination and on the polar
        angle on the observer's screen. Inclined observers are not supported
        yet, so only the face-on value 0 (default) is meaningful.
    figsize : tuple of float, optional
        Size of the new figure, ignored if `ax` is given. Default is (7, 7).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None (default), a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot.
    ax : matplotlib.axes.Axes
        Axes containing the plot.

    Other Parameters
    ----------------
    **geodesic_kwargs
        Keyword arguments passed to `gravityp.compute_geodesic`, such as
        `Npoints` or `r0`.

    See Also
    --------
    gravityp.transfer_function : Transfer function of a given order.

    Notes
    -----
    The axes are fixed to ``0 <= b <= 10`` and ``0 <= r_m <= 15``.
    """
    fig, ax = _get_fig_ax(ax, figsize)

    for order, bs in enumerate(bs_list):
        if order == 0:
            _plot_transfer_function(
                ax, bs, order, co, correction, **geodesic_kwargs)
        else:
            bs_rings = bs if isinstance(bs, (list, tuple)) else [bs]
            for bs_ring in bs_rings:
                _plot_transfer_function(
                    ax, bs_ring, order, co, correction, **geodesic_kwargs)

    xticks = [i for i in range(0, 11, 2)]
    xticklabels = [f'{t}' for t in xticks]

    ax.set_xlabel(r'$b/M$')
    ax.set_ylabel(r'$r_m/M$')
    ax.set_xlim(0, 10)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylim(0, 15)
    ax.set_yticks(np.arange(0, 16, 5))
    return fig, ax

# Intensity and shadows


def _add_extra_tick(xticks, xticklabels, extra_tick=None, extra_ticklabel=None):
    """Adds a labelled tick, removing the regular ticks too close to it"""
    if (extra_tick is not None) and (extra_ticklabel is not None):
        threshold = 0.6  # Minimum allowed distance between labels
        xticks = [t for t in xticks if abs(t - extra_tick) > threshold]
        xticklabels = [f'{t}' for t in xticks]
        xticks += [extra_tick]
        xticklabels += [extra_ticklabel]
    return xticks, xticklabels


def plot_emission_model(rs, emission_model, text_string=None,
                        tick=None, ticklabel=None, figsize=(7, 7), ax=None):
    """Plot an emission profile of the accretion disk.

    Parameters
    ----------
    rs : ndarray
        Radial coordinates where the profile is evaluated, which also set the
        limits of the horizontal axis.
    emission_model : callable
        Emitted intensity :math:`I_{em}(r)`, taking an ndarray of radii.
    text_string : str, optional
        Text shown at the data coordinates ``(7.5, 0.8)``. Default is None.
    tick, ticklabel : float and str, optional
        Position and label of an extra tick on the horizontal axis, e.g. to
        mark where the emission starts. Both must be given, and regular ticks
        too close to it are removed. Default is None.
    figsize : tuple of float, optional
        Size of the new figure, ignored if `ax` is given. Default is (7, 7).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None (default), a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot.
    ax : matplotlib.axes.Axes
        Axes containing the plot.

    See Also
    --------
    gravityp.emission_models : Predefined emission profiles.

    Notes
    -----
    The vertical axis is fixed to ``[0, 1]``, suited to profiles normalised
    to a maximum of 1.
    """
    fig, ax = _get_fig_ax(ax, figsize)
    ax.plot(rs, emission_model(rs), label=r'$I_{\mathrm{em}}/I_0$')

    rmin = round(rs.min())
    rmax = round(rs.max())

    # Add an extra xtick placed at specified position
    xticks = [i for i in range(rmin, rmax, 2)]
    xticklabels = [f'{t}' for t in xticks]
    xticks, xticklabels = _add_extra_tick(xticks, xticklabels, tick, ticklabel)

    ax.set_xlabel(r'$r/M$')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(rmin, rmax)
    ax.set_ylim(0, 1)
    if text_string is not None:
        ax.text(7.5, 0.8, s=text_string)
    ax.legend()
    return fig, ax


def plot_observed_intensity(bs, bs_transfer_list, emission_model,
                            co: BaseCompactObject, figsize=(7, 7), ax=None,
                            y_range=None, **geodesic_kwargs):
    """Plot the observed intensity profile of a face-on accretion disk.

    Parameters
    ----------
    bs : ndarray
        Impact parameters where the intensity is evaluated, in units of M,
        which also set the limits of the horizontal axis.
    bs_transfer_list : sequence of ndarray
        Increasing impact parameters sampling the transfer function of each
        order, starting from order 0. Usually computed with
        ``compute_optimal_array(..., joint=False, fill=True)``.
    emission_model : callable
        Emitted intensity :math:`I_{em}(r)`, taking an ndarray of radii.
    co : BaseCompactObject
        Compact object defining the geometry.
    figsize : tuple of float, optional
        Size of the new figure, ignored if `ax` is given. Default is (7, 7).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None (default), a new figure is created.
    y_range : tuple of float, optional
        Limits ``(bottom, top)`` of the vertical axis. If None (default), they
        are set automatically.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot.
    ax : matplotlib.axes.Axes
        Axes containing the plot.

    Other Parameters
    ----------------
    **geodesic_kwargs
        Keyword arguments passed to `gravityp.compute_geodesic`, such as
        `Npoints` or `r0`.

    See Also
    --------
    gravityp.total_intensity : Observed intensity profile.
    make_shadow_plot : Image of the compact object.
    """
    fig, ax = _get_fig_ax(ax, figsize)
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
    return fig, ax


def _plot_intensities(ax, z, vmin=None, vmax=None):
    """Shows the image z on ax, covering [-10, 10] along both axes"""
    return ax.imshow(
        z,
        norm=colors.PowerNorm(1, vmin, vmax),
        cmap='inferno',
        extent=(-10, 10, -10, 10),
        origin='lower',
        interpolation='none',
    )


def _plot_colorbar(fig, ax, im):
    """Adds a colorbar for im on the right of ax"""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    return fig.colorbar(im, cax=cax)


def make_shadow_plot(bs_transfer_list, emission_model, co: BaseCompactObject,
                     figsize=(7, 7), ax=None, y_range=None, Npixels=1e6,
                     **geodesic_kwargs):
    """Plot the image of a compact object surrounded by a face-on disk.

    The observed intensity is shown on the observer's screen, covering
    ``[-10, 10]`` in units of M along both axes, together with a colorbar.

    Parameters
    ----------
    bs_transfer_list : sequence of ndarray
        Increasing impact parameters sampling the transfer function of each
        order, starting from order 0. Usually computed with
        ``compute_optimal_array(..., joint=False, fill=True)``.
    emission_model : callable
        Emitted intensity :math:`I_{em}(r)`, taking an ndarray of radii.
    co : BaseCompactObject
        Compact object defining the geometry.
    figsize : tuple of float, optional
        Size of the new figure, ignored if `ax` is given. Default is (7, 7).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None (default), a new figure is created.
    y_range : tuple of float, optional
        Limits ``(vmin, vmax)`` of the colour scale. If None (default), they
        are set from the data.
    Npixels : int or float, optional
        Total number of pixels of the image. Default is 1e6.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot.
    ax : matplotlib.axes.Axes
        Axes containing the image. The colorbar lies in separate axes, and
        can be accessed as ``ax.images[0].colorbar``.

    Other Parameters
    ----------------
    **geodesic_kwargs
        Keyword arguments passed to `gravityp.compute_geodesic`, such as
        `Npoints` or `r0`.

    See Also
    --------
    gravityp.compute_points : Observed intensity on the observer's screen.
    """

    Z = compute_points(bs_transfer_list, emission_model,
                       co, Npixels, **geodesic_kwargs)

    fig, ax = _get_fig_ax(ax, figsize)
    vmin, vmax = y_range if y_range is not None else (None, None)
    im = _plot_intensities(ax, Z, vmin, vmax)
    ax.set_xticks(np.arange(-10, 11, 5))
    ax.set_yticks(np.arange(-10, 11, 5))
    _plot_colorbar(fig, ax, im)
    return fig, ax
