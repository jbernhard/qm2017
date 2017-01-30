""" plots / visualizations / figures """

import itertools
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

from . import workdir, systems, expt, model, mcmc


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


def _observables_plots():
    charged_parts = [('dNch_deta', None, r'$N_\mathrm{ch}$', 'Greys')]

    def id_parts(obs):
        return [
            (obs, 'pion',   r'$\pi^\pm$', 'Blues'),
            (obs, 'kaon',   r'$K^\pm$', 'Greens'),
            (obs, 'proton', r'$p\bar p$', 'Reds'),
        ]

    flows = [
        ('vn', n, '$v_{}$'.format(n), c)
        for n, c in enumerate(['GnBu', 'Purples', 'Oranges'], start=2)
    ]

    return [
        ('Yields', r'$dN_\mathrm{ch}/d\eta,\ dN/dy$', (1., 2e4),
         charged_parts + id_parts('dN_dy')),
        ('Mean $p_T$', r'$p_T$ [GeV]', (0, 2.), id_parts('mean_pT')),
        ('Flow cumulants', r'$v_n\{2\}$', (0, 0.15), flows),
    ]


def _observables(posterior=False):
    """
    Model observables at all design points or drawn from the posterior with
    experimental data points.

    """
    plots = _observables_plots()

    fig, axes = plt.subplots(
        nrows=len(systems), ncols=len(plots),
        figsize=(fullwidth, .55*fullwidth)
    )

    if posterior:
        samples = mcmc.Chain().samples(100)

    for (system, (title, ylabel, ylim, subplots)), ax in zip(
            itertools.product(systems, plots), axes.flat
    ):
        for obs, subobs, label, cmap in subplots:
            factor = 5 if obs == 'dNch_deta' else 1
            color = getattr(plt.cm, cmap)(.6)

            x = model.data[system][obs][subobs]['x']
            Y = (
                samples[system][obs][subobs]
                if posterior else
                model.data[system][obs][subobs]['Y']
            )

            for y in Y * factor:
                ax.plot(x, y, color=color, alpha=.08, lw=.3)

            try:
                dset = expt.data[system][obs][subobs]
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


@plot
def observables_design():
    _observables(posterior=False)


@plot
def observables_posterior():
    _observables(posterior=True)


@plot
def observables_map():
    """
    Model observables and ratio to experiment at the maximum a posteriori
    (MAP) estimate.

    """
    plots = _observables_plots()

    fig = plt.figure(figsize=(fullwidth, .75*fullwidth))
    gs = plt.GridSpec(3*len(systems), len(plots))

    for (nsys, system), (nplot, (title, ylabel, ylim, subplots)) in \
            itertools.product(enumerate(systems), enumerate(plots)):
        nrow = 3*nsys
        ax = fig.add_subplot(gs[nrow:nrow+2, nplot])
        ratio_ax = fig.add_subplot(gs[nrow+2, nplot])

        for obs, subobs, label, cmap in subplots:
            factor = 5 if obs == 'dNch_deta' else 1
            color = getattr(plt.cm, cmap)(.6)

            x = model.map_data[system][obs][subobs]['x']
            y = model.map_data[system][obs][subobs]['Y'] * factor

            ax.plot(x, y, color=color, lw=.5)

            try:
                dset = expt.data[system][obs][subobs]
            except KeyError:
                continue

            x = dset['x']
            yexp = dset['y'] * factor
            yerr = np.sqrt(sum(
                e**2 for e in dset['yerr'].values()
            )) * factor

            ax.errorbar(
                x, yexp, yerr=yerr, fmt='o', ms=1.7,
                capsize=0, color='.25', zorder=1000
            )

            ratio_ax.plot(x, y/yexp, color=color)

        if title == 'Yields':
            ax.set_yscale('log')
            ax.minorticks_off()
        else:
            auto_ticks(ax, 'y', nbins=4, minor=2)

        if ax.is_first_row():
            ax.set_title(title)
        elif ratio_ax.is_last_row():
            ratio_ax.set_xlabel('Centrality %')

        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)

        ratio_ax.axhline(1, lw=.5, color='0.5', zorder=-100)
        ratio_ax.axhspan(0.9, 1.1, color='0.95', zorder=-200)

        ratio_ax.set_ylim(0.8, 1.2)
        ratio_ax.set_yticks(np.arange(80, 121, 10)/100)
        ratio_ax.set_ylabel('Ratio')


def format_ci(samples, ci=.9):
    """
    Compute the median and a credible interval for an array of samples and
    return a TeX-formatted string.

    """
    cil, cih = mcmc.credible_interval(samples, ci=ci)
    median = np.median(samples)
    ul = median - cil
    uh = cih - median

    # decide precision for formatting numbers
    # this is NOT general but it works for the present data
    if abs(median) < .2 and ul < .02:
        precision = 3
    elif abs(median) < 1:
        precision = 2
    else:
        precision = 1

    fmt = str(precision).join(['{:#.', 'f}'])

    return ''.join([
        '$', fmt.format(median),
        '_{-', fmt.format(ul), '}',
        '^{+', fmt.format(uh), '}$'
    ])


@plot
def posterior():
    chain = mcmc.Chain()
    data = chain.load().T

    cmap = plt.cm.Blues
    line_color = cmap(.8)
    fill_color = cmap(.5, alpha=.1)

    fig, axes = plt.subplots(
        nrows=chain.ndim, ncols=chain.ndim,
        sharex='col', sharey='row',
        figsize=(fullheight, fullheight)
    )

    for ax, d, lim in zip(axes.diagonal(), data, chain.range):
        counts, edges = np.histogram(d, bins=200, range=lim)
        x = (edges[1:] + edges[:-1]) / 2
        y = .84 * (lim[1] - lim[0]) * counts / counts.max() + lim[0]
        ax.plot(x, y, lw=.5, color=line_color)
        ax.fill_between(x, lim[0], y, color=fill_color, zorder=-10)

        ax.set_xlim(lim)
        ax.set_ylim(lim)

        ticks = [lim[0], (lim[0] + lim[1])/2, lim[1]]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        ax.annotate(
            format_ci(d), (.2, .95), xycoords='axes fraction',
            ha='left', va='bottom', fontsize=3.5
        )

    for ny, nx in zip(*np.tril_indices_from(axes, k=-1)):
        H, xedges, yedges = np.histogram2d(
            data[nx], data[ny], bins=200,
            range=(chain.range[nx], chain.range[ny])
        )
        H[H == 0] = None
        axes[ny][nx].pcolorfast(xedges, yedges, H.T, cmap=cmap)
        axes[nx][ny].set_axis_off()

    for n, label in enumerate(chain.labels):
        for ax, xy in [(axes[-1, n], 'x'), (axes[n, 0], 'y')]:
            getattr(ax, 'set_{}label'.format(xy))(
                label.replace(r'\ [', '$\n$['), fontdict=dict(size=3)
            )
            ticklabels = getattr(ax, 'get_{}ticklabels'.format(xy))()
            for t in ticklabels:
                t.set_fontsize(2.5)
            if xy == 'x':
                ticklabels[0].set_horizontalalignment('left')
                ticklabels[-1].set_horizontalalignment('right')
            else:
                ticklabels[0].set_verticalalignment('bottom')
                ticklabels[-1].set_verticalalignment('top')

    set_tight(fig, pad=.05, h_pad=.2, w_pad=.2, rect=[0., 0., .98, .98])


@plot
def etas_estimate():
    plt.figure(figsize=(.75*textwidth, .75*aspect*textwidth))
    ax = plt.axes()

    Tc = .154

    def etas(T, m=0, s=0, c=0):
        return m + s*(T - Tc)*(T/Tc)**c

    chain = mcmc.Chain()

    rangedict = dict(zip(chain.keys, chain.range))
    ekeys = ['etas_' + k for k in ['min', 'slope', 'curv']]

    T = np.linspace(.154, .3, 50)

    prior = ax.fill_between(
        T, etas(T, *(rangedict[k][1] for k in ekeys)),
        color='.92'
    )

    eparams = chain.load(*ekeys).T
    intervals = np.array([
        mcmc.credible_interval(etas(t, *eparams))
        for t in T
    ]).T

    band = ax.fill_between(T, *intervals, color=plt.cm.Blues(.32))

    ax.plot(T, np.full_like(T, 1/(4*np.pi)), color='.6')
    ax.text(.299, .07, r'KSS bound $1/4\pi$', va='top', ha='right', color='.4')

    median, = ax.plot(
        T, etas(T, *map(np.median, eparams)),
        color=plt.cm.Blues(.77)
    )

    ax.set_xlim(xmin=.146)
    ax.set_ylim(0, .6)
    ax.set_xticks(np.arange(150, 301, 50)/1000)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    auto_ticks(ax, 'y', minor=2)

    ax.set_xlabel('Temperature [GeV]')
    ax.set_ylabel(r'$\eta/s$')

    ax.legend(*zip(*[
        (prior, 'Prior range'),
        (median, 'Posterior median'),
        (band, '90% CR'),
    ]), loc='upper left')


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
