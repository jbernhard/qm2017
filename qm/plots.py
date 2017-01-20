""" plots / visualizations / figures """

import itertools
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

from . import workdir, systems, expt, model


fontsmall, fontnormal, fontlarge = 5, 6, 7
offblack = '#262626'
aspect = 1/1.618
resolution = 72.27
textwidth = 307.28987/resolution
textheight = 261.39864/resolution
fullwidth = 350/resolution
fullheight = 270/resolution

plt.rcdefaults()
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Lato'],
    'mathtext.fontset': 'custom',
    'mathtext.default': 'it',
    'mathtext.rm': 'sans',
    'mathtext.it': 'sans:italic:medium',
    'mathtext.cal': 'sans',
    'font.size': fontnormal,
    'legend.fontsize': fontnormal,
    'axes.labelsize': fontnormal,
    'axes.titlesize': fontlarge,
    'xtick.labelsize': fontsmall,
    'ytick.labelsize': fontsmall,
    'font.weight': 400,
    'axes.labelweight': 400,
    'axes.titleweight': 400,
    'lines.linewidth': .5,
    'lines.markersize': 3,
    'lines.markeredgewidth': 0,
    'patch.linewidth': .5,
    'axes.linewidth': .4,
    'xtick.major.width': .4,
    'ytick.major.width': .4,
    'xtick.minor.width': .4,
    'ytick.minor.width': .4,
    'xtick.major.size': 1.2,
    'ytick.major.size': 1.2,
    'xtick.minor.size': .8,
    'ytick.minor.size': .8,
    'xtick.major.pad': 1.5,
    'ytick.major.pad': 1.5,
    'axes.labelpad': 3,
    'text.color': offblack,
    'axes.edgecolor': offblack,
    'axes.labelcolor': offblack,
    'xtick.color': offblack,
    'ytick.color': offblack,
    'legend.numpoints': 1,
    'legend.scatterpoints': 1,
    'legend.frameon': False,
    'image.interpolation': 'none',
    'pdf.fonttype': 42
})


plotdir = workdir / 'plots'
plotdir.mkdir(exist_ok=True)

plot_functions = {}


def plot(f):
    """
    Plot function decorator.  Calls the function, does several generic tasks,
    and saves the figure as the function name.

    """
    def wrapper(*args, **kwargs):
        logging.info('generating plot: %s', f.__name__)
        f(*args, **kwargs)

        fig = plt.gcf()

        if getattr(fig, 'despine', True):
            despine(*fig.axes)

        if not fig.get_tight_layout():
            set_tight(fig)

        plotfile = plotdir / '{}.pdf'.format(f.__name__)
        fig.savefig(str(plotfile))
        logging.info('wrote %s', plotfile)
        plt.close(fig)

    plot_functions[f.__name__] = wrapper

    return wrapper


def despine(*axes):
    """
    Remove the top and right spines.

    """
    if not axes:
        axes = plt.gcf().axes

    for ax in axes:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for xy, pos in [('x', 'bottom'), ('y', 'left')]:
            axis = getattr(ax, xy + 'axis')
            if axis.get_ticks_position() == 'default':
                axis.set_ticks_position(pos)


def set_tight(fig=None, **kwargs):
    """
    Set tight_layout with a better default pad.

    """
    if fig is None:
        fig = plt.gcf()

    kwargs.setdefault('pad', .1)
    fig.set_tight_layout(kwargs)


def remove_ticks(*axes):
    """
    Remove all tick marks (but not labels).

    """
    if not axes:
        axes = plt.gcf().axes

    for ax in axes:
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')


def auto_ticks(
        ax, xy=None, nbins=5, steps=[1, 2, 4, 5, 10],
        prune=None, minor=0
):
    """
    Convenient interface to matplotlib.ticker locators.

    """
    if xy == 'x':
        axes = ax.xaxis,
    elif xy == 'y':
        axes = ax.yaxis,
    else:
        axes = ax.xaxis, ax.yaxis

    for axis in axes:
        axis.set_major_locator(
            ticker.MaxNLocator(nbins=nbins, steps=steps, prune=prune)
        )
        if minor:
            axis.set_minor_locator(ticker.AutoMinorLocator(minor))


def getitems(obj, *keys):
    """
    Get items from a nested dict-like object.  Equivalent to

        obj[k0][k1]...[kn]

    for keys = [k0, k1, ..., kn].  Keys of None are ignored.

    """
    r = obj
    for k in keys:
        if k is not None:
            r = r[k]
    return r


@plot
def observables():
    """
    Model observables at all design points with experimental data points.

    """
    id_parts = [
        ('pion',   r'$\pi^\pm$', 'Blues'),
        ('kaon',   r'$K^\pm$', 'Greens'),
        ('proton', r'$p\bar p$', 'Reds')
    ]

    charged_parts = [(None, r'$N_\mathrm{ch}$', 'Greys')]

    flows = [
        (n, '$v_{}$'.format(n), c)
        for n, c in enumerate(['GnBu', 'Purples', 'Oranges'], start=2)
    ]

    plots = [
        ('Yields', r'$dN_\mathrm{ch}/d\eta,\ dN/dy$', (1., 2e4), [
            ('dNch_deta', charged_parts),
            ('dN_dy', id_parts)
        ]),
        ('Mean $p_T$', r'$p_T$ [GeV]', (0, 2.), [
            ('mean_pT', id_parts)
        ]),
        ('Flow cumulants', r'$v_n\{2\}$', (0, 0.15), [
            ('vn', flows)
        ]),
    ]

    fig, axes = plt.subplots(
        nrows=len(systems), ncols=len(plots),
        figsize=(fullwidth, .55*fullwidth)
    )

    for (system, (title, ylabel, ylim, observables)), ax in zip(
            itertools.product(systems, plots), axes.flat
    ):
        for obs, subplots in observables:
            factor = 5 if obs == 'dNch_deta' else 1
            for subobs, label, cmap in subplots:
                color = getattr(plt.cm, cmap)(.6)

                dset = getitems(model.data, system, obs, subobs)
                x = dset['x']
                Y = dset['Y'] * factor

                for y in Y:
                    ax.plot(x, y, color=color, alpha=.08, lw=.3)

                try:
                    dset = getitems(expt.data, system, obs, subobs)
                except KeyError:
                    continue

                x = dset['x']
                y = dset['y'] * factor
                yerr = np.sqrt(sum(
                    e**2 for e in dset['yerr'].values()
                )) * factor

                ax.errorbar(
                    x, y, yerr=yerr, fmt='o', ms=1.7,
                    capsize=0, color='.25', zorder=1000
                )

        if title == 'Yields':
            ax.set_yscale('log')
            ax.minorticks_off()
        else:
            auto_ticks(ax, 'y', nbins=4, minor=2)

        if ax.is_first_row():
            ax.set_title(title)
        elif ax.is_last_row():
            ax.set_xlabel('Centrality %')

        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)

    set_tight(fig, w_pad=-.2)


if __name__ == '__main__':
    import argparse

    choices = list(plot_functions)

    def arg_to_plot(arg):
        arg = Path(arg).stem
        if arg not in choices:
            raise argparse.ArgumentTypeError(arg)
        return arg

    parser = argparse.ArgumentParser(description='generate plots')
    parser.add_argument(
        'plots', nargs='*', type=arg_to_plot, metavar='PLOT',
        help='{} (default: all)'.format(', '.join(choices).join('{}'))
    )
    args = parser.parse_args()

    if args.plots:
        for p in args.plots:
            plot_functions[p]()
    else:
        for f in plot_functions.values():
            f()
