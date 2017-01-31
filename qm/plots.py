""" plots / visualizations / figures """

import colorsys
import itertools
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import ticker
from scipy.interpolate import PchipInterpolator

from . import workdir, systems, parse_system, expt, model, mcmc


def darken(rgba, amount=.5):
    h, l, s = colorsys.rgb_to_hls(*rgba[:3])
    r, g, b = colorsys.hls_to_rgb(h, l*amount, s)

    try:
        return r, g, b, rgba[3]
    except IndexError:
        return r, g, b


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

            ax.text(
                x[-1] + 2.5,
                model.map_data[system][obs][subobs]['Y'][-1] * factor,
                label,
                color=darken(color), ha='left', va='center'
            )

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

        if ax.is_last_col():
            proj, energy = parse_system(system)
            ax.text(
                1.07, .5, '{} {:.2f} TeV'.format('+'.join(proj), energy/1000),
                transform=ax.transAxes, ha='left', va='center',
                size=plt.rcParams['axes.titlesize'], rotation=-90
            )

        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)

    set_tight(fig, w_pad=1, rect=[0, 0, .97, 1])


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

            ax.text(
                x[-1] + 2.5,
                model.map_data[system][obs][subobs]['Y'][-1] * factor,
                label,
                color=darken(color), ha='left', va='center'
            )

            try:
                dset = expt.data[system][obs][subobs]
            except KeyError:
                continue

            x = dset['x']
            yexp = dset['y'] * factor
            yerr = dset['yerr']

            ax.errorbar(
                x, yexp, yerr=yerr.get('stat'), fmt='o', ms=1.7,
                capsize=0, color='.25', zorder=1000
            )

            yerrsys = yerr.get('sys', yerr.get('sum'))
            ax.fill_between(
                x, yexp - yerrsys, yexp + yerrsys,
                color='.9', zorder=-10
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

        if ax.is_last_col():
            proj, energy = parse_system(system)
            ax.text(
                1.07, 0, '{} {:.2f} TeV'.format('+'.join(proj), energy/1000),
                transform=ax.transAxes, ha='left', va='bottom',
                size=plt.rcParams['axes.titlesize'], rotation=-90
            )

        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)

        ratio_ax.axhline(1, lw=.5, color='0.5', zorder=-100)
        ratio_ax.axhspan(0.9, 1.1, color='0.95', zorder=-200)
        ratio_ax.text(
            ratio_ax.get_xlim()[1], .9, '±10%',
            color='.6', zorder=-50,
            ha='right', va='bottom',
            size=plt.rcParams['xtick.labelsize']
        )

        ratio_ax.set_ylim(0.8, 1.2)
        ratio_ax.set_yticks(np.arange(80, 121, 10)/100)
        ratio_ax.set_ylabel('Ratio')

    set_tight(fig, w_pad=1, rect=[0, 0, .97, 1])


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


def _posterior(params=None, ignore=None, scale=1, padr=.99, padt=.98):
    """
    Triangle plot of posterior marginal and joint distributions.

    """
    chain = mcmc.Chain()

    if params is None and ignore is None:
        params = set(chain.keys)
    elif params is not None:
        params = set(params)
    elif ignore is not None:
        params = set(chain.keys) - set(ignore)

    keys, labels, ranges = map(list, zip(*(
        i for i in zip(chain.keys, chain.labels, chain.range)
        if i[0] in params
    )))
    ndim = len(params)

    data = chain.load(*keys).T

    cmap = plt.cm.Blues
    line_color = cmap(.8)
    fill_color = cmap(.5, alpha=.1)

    fig, axes = plt.subplots(
        nrows=ndim, ncols=ndim,
        sharex='col', sharey='row',
        figsize=2*(scale*fullheight,)
    )

    for ax, d, lim in zip(axes.diagonal(), data, ranges):
        counts, edges = np.histogram(d, bins=50, range=lim)
        x = (edges[1:] + edges[:-1]) / 2
        y = .85 * (lim[1] - lim[0]) * counts / counts.max() + lim[0]
        # smooth histogram with monotonic cubic interpolation
        interp = PchipInterpolator(x, y)
        x = np.linspace(x[0], x[-1], 10*x.size)
        y = interp(x)
        ax.plot(x, y, lw=.5, color=line_color)
        ax.fill_between(x, lim[0], y, color=fill_color, zorder=-10)

        ax.set_xlim(lim)
        ax.set_ylim(lim)

        ticks = [lim[0], (lim[0] + lim[1])/2, lim[1]]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        ax.annotate(
            format_ci(d), (.62, .92), xycoords='axes fraction',
            ha='center', va='bottom', fontsize=4.5
        )

    for ny, nx in zip(*np.tril_indices_from(axes, k=-1)):
        H, xedges, yedges = np.histogram2d(
            data[nx], data[ny], bins=100,
            range=(ranges[nx], ranges[ny])
        )
        H[H == 0] = None
        axes[ny][nx].pcolorfast(xedges, yedges, H.T, cmap=cmap)

        axes[nx][ny].set_axis_off()

    for n, label in enumerate(labels):
        for ax, xy in [(axes[-1, n], 'x'), (axes[n, 0], 'y')]:
            getattr(ax, 'set_{}label'.format(xy))(
                label.replace(r'\ [', '$\n$['), fontdict=dict(size=4)
            )
            ticklabels = getattr(ax, 'get_{}ticklabels'.format(xy))()
            for t in ticklabels:
                t.set_fontsize(3)
                if xy == 'x' and len(str(sum(ranges[n])/2)) > 4:
                    t.set_rotation(30)
            if xy == 'x':
                ticklabels[0].set_horizontalalignment('left')
                ticklabels[-1].set_horizontalalignment('right')
            else:
                ticklabels[0].set_verticalalignment('bottom')
                ticklabels[-1].set_verticalalignment('top')

    set_tight(fig, pad=.05, h_pad=.3, w_pad=.3, rect=[0., 0., padr, padt])


@plot
def posterior():
    _posterior(
        ignore={'norm {}'.format(s) for s in systems} | {'dmin3', 'etas_hrg'}
    )


@plot
def posterior_withnorm():
    _posterior(scale=1.2, ignore={'dmin3', 'etas_hrg'})


@plot
def posterior_etas():
    _posterior(
        scale=.45, padt=.97, padr=1.,
        params={'etas_min', 'etas_slope', 'etas_curv'}
    )


@plot
def posterior_zetas():
    _posterior(
        scale=.33, padt=.96, padr=1.,
        params={'zetas_max', 'zetas_width'}
    )


@plot
def posterior_p():
    """
    Distribution of trento p parameter with annotations for other models.

    """
    plt.figure(figsize=(.65*textwidth, .25*textwidth))
    ax = plt.axes()

    data = mcmc.Chain().load('trento_p').ravel()

    counts, edges = np.histogram(data, bins=50)
    x = (edges[1:] + edges[:-1]) / 2
    y = counts / counts.max()
    interp = PchipInterpolator(x, y)
    x = np.linspace(x[0], x[-1], 10*x.size)
    y = interp(x)
    ax.plot(x, y, color=plt.cm.Blues(0.8))
    ax.fill_between(x, y, color=plt.cm.Blues(0.15), zorder=-10)

    ax.set_xlabel('$p$')

    for spine in ax.spines.values():
        spine.set_visible(False)

    for label, x, err in [
            ('KLN', -.67, .01),
            ('EKRT /\nIP-Glasma', 0, .1),
            ('Wounded\nnucleon', 1, None),
    ]:
        args = ([x], [0], 'o') if err is None else ([x - err, x + err], [0, 0])
        ax.plot(*args, lw=4, ms=4, color=offblack, alpha=.58, clip_on=False)

        if label.startswith('EKRT'):
            x -= .275

        ax.text(x, .05, label, va='bottom', ha='center')

    ax.text(.1, .8, format_ci(data))
    ax.set_xticks(np.arange(-10, 11, 5)/10)
    ax.set_xticks(np.arange(-75, 76, 50)/100, minor=True)

    for t in ax.get_xticklabels():
        t.set_y(-.03)

    xm = 1.2
    ax.set_xlim(-xm, xm)
    ax.add_artist(
        patches.FancyArrowPatch(
            (-xm, 0), (xm, 0),
            linewidth=.6,
            arrowstyle=patches.ArrowStyle.CurveFilledAB(
                head_length=3, head_width=1.5
            ),
            facecolor=offblack, edgecolor=offblack,
            clip_on=False, zorder=100
        )
    )

    ax.set_yticks([])
    ax.set_ylim(0, 1.01*y.max())

    set_tight(pad=0)


@plot
def etas_estimate():
    """
    Estimate of the temperature dependence of shear viscosity eta/s.

    """
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


@plot
def zetas_estimate():
    """
    Estimate of the temperature dependence of bulk viscosity zeta/s.

    """
    plt.figure(figsize=(.6*textwidth, .6*aspect*textwidth))
    ax = plt.axes()

    Tc = .154

    def zetas(T, zetas_max=0, zetas_width=1):
        return zetas_max / (1 + ((T - Tc)/zetas_width)**2)

    chain = mcmc.Chain()

    keys, ranges = map(list, zip(*(
        i for i in zip(chain.keys, chain.range)
        if i[0].startswith('zetas')
    )))

    T0 = .5*Tc
    T1 = .95*Tc
    T2 = 2*Tc - T1
    T3 = 2*Tc - T0

    # higher density of points near Tc needed to resolve peak
    T = np.concatenate([
        np.linspace(T0, T1, 50, endpoint=False),
        np.linspace(T1, T2, 100, endpoint=False),
        np.linspace(T2, T3, 50, endpoint=True),
    ])

    maxdict = {k: r[1] for k, r in zip(keys, ranges)}
    prior = ax.fill_between(T, zetas(T, **maxdict), color='.92')

    params = dict(zip(keys, chain.load(*keys).T))
    intervals = np.array([
        mcmc.credible_interval(zetas(t, **params))
        for t in T
    ]).T

    band = ax.fill_between(T, *intervals, color=plt.cm.Blues(.32))

    median, = ax.plot(
        T, zetas(T, **{k: np.median(p) for k, p in params.items()}),
        color=plt.cm.Blues(.77)
    )

    ax.set_xlim(T[0], T[-1])
    ax.set_ylim(0, 1.05*maxdict['zetas_max'])
    auto_ticks(ax, minor=2)

    ax.set_xlabel('Temperature [GeV]')
    ax.set_ylabel(r'$\zeta/s$')

    ax.legend(*zip(*[
        (prior, 'Prior range'),
        (median, 'Posterior median'),
        (band, '90% CR'),
    ]), loc='upper left')


@plot
def flow_corr():
    """
    Symmetric cumulants SC(m, n) at the MAP point compared to experiment.

    """
    plots, width_ratios = zip(*[
        (('sc_central', 9e-8), 2),
        (('sc', 2.8e-6), 3),
    ])

    fig, axes = plt.subplots(
        figsize=(textwidth, .4*textwidth),
        ncols=len(plots), gridspec_kw=dict(width_ratios=width_ratios)
    )

    sys = 'PbPb2760'
    labelfmt = r'$\mathrm{{SC}}({}, {})$'

    for (obs, ylim), ax in zip(plots, axes):
        for mn, cmap in [
                ((4, 2), 'Blues'),
                ((3, 2), 'Oranges'),
        ]:
            color = getattr(plt.cm, cmap)(.7)

            x = model.map_data[sys][obs][mn]['x']
            y = model.map_data[sys][obs][mn]['Y']

            ax.plot(x, y, color=color, lw=.75, label=labelfmt.format(*mn))

            x = expt.extra_data[sys][obs][mn]['x']
            y = expt.extra_data[sys][obs][mn]['y']
            yerr = expt.extra_data[sys][obs][mn]['yerr']

            ax.errorbar(
                x, y, yerr=yerr['stat'],
                fmt='o', ms=2, capsize=0, color='.25', zorder=100
            )

            ax.fill_between(
                x, y - yerr['sys'], y + yerr['sys'],
                color='.9', zorder=-10
            )

        ax.axhline(
            0, color='.75', lw=plt.rcParams['xtick.major.width'],
            zorder=-100
        )

        ax.set_xlabel('Centrality %')
        ax.set_ylim(-ylim, ylim)

        auto_ticks(ax, 'y', nbins=6, minor=2)
        ax.ticklabel_format(scilimits=(-5, 5))

        if ax.is_first_col():
            ax.set_ylabel(labelfmt.format('m', 'n'))
        else:
            ax.legend(loc='upper left')


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
