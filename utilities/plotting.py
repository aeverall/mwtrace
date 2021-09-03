import numpy as np
import emcee, tqdm, corner

import matplotlib
import matplotlib.pyplot as plt
# Plotting modules
font = {'family' : 'serif', 'weight' : 'normal',
        'size'   : 13}
legend = {'fontsize': 13}
matplotlib.rc('font', **font)
matplotlib.rc('legend', **legend)

from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from matplotlib.ticker import ScalarFormatter
from scipy.ndimage import gaussian_filter



def plot_hist(data, bins, ax=None, fill_poisson=None, plot_kwargs={}, fill_kwargs={}):

    hist, bins = np.histogram(data, bins=bins)
    ax.plot(np.repeat(bins,2), np.insert(np.repeat(hist/(bins[1:]-bins[:-1]),2), (0,len(hist)*2), (0,0)),**plot_kwargs)
    if fill_poisson:
        y = hist/(bins[1:]-bins[:-1])
        y_lower = (hist-np.sqrt(hist))/(bins[1:]-bins[:-1])
        y_upper = (hist+np.sqrt(hist))/(bins[1:]-bins[:-1])
        ax.fill_between(np.repeat(bins,2), np.insert(np.repeat(y_lower,2),(0,len(y)*2),(0,0)),
                                            np.insert(np.repeat(y_upper,2),(0,len(y)*2),(0,0)),
                                            **fill_kwargs)


    return None

def plot_corner(sampler, functions=None, nburnt=None, clean_chains=np.inf, **kwargs):

    if type(sampler) is np.ndarray:
        chains=sampler.copy()
        good_chains = np.ones(sampler.shape[0]).astype(bool)
    else:
        chains=sampler.chain.copy()
        good_chains = sampler.lnprobability[:,-1].copy()>np.max(sampler.lnprobability)-clean_chains

    if functions is None:
        functions = [None for i in range(chains.shape[2])]

    if nburnt is None: flatchain = chains[good_chains,int(chains.shape[1]/2):,:].reshape(-1,chains.shape[2])
    else: flatchain = chains[good_chains,-nburnt:,:].reshape(-1,chains.shape[2])
    for j in range(chains.shape[2]):
        if functions[j] is not None:
            flatchain[:,j]=functions[j](flatchain[:,j])
            kwargs['truths'] = kwargs['truths'].copy()
            try:
                kwargs['truths'][j] = functions[j](kwargs['truths'][j])
            except KeyError:
                pass
    _=corner.corner(flatchain, **kwargs)

def plot_chains(sampler, truths=None, labels=None, functions=None, clean_chains=np.inf, lnprob=None, plot_bad=True):

    if type(sampler) is np.ndarray:
        chains=sampler.copy()
        if lnprob is not None: good_chains = lnprob[:,-1]>np.max(lnprob)-clean_chains
        else: good_chains = np.ones(chains.shape[0]).astype(bool)
    else:
        chains=sampler.chain.copy()
        good_chains = sampler.lnprobability[:,-1].copy()>np.max(sampler.lnprobability)-clean_chains
    if truths is not None: true_pars = truths.copy()

    nwalkers, niter, ndim = chains.shape
    if functions is not None:
        for j in range(len(functions)):
            if functions[j] is not None:
                chains[:,:,j]=functions[j](chains[:,:,j])
                if truths is not None:
                    true_pars[j]=functions[j](true_pars[j])
    fig, axes = plt.subplots(ndim,1,figsize=(10,ndim*5), sharex=True)
    for idim in range(ndim):
        if ndim>1:plt.sca(axes[idim])
        for iw in range(nwalkers):
            if good_chains[iw]: plt.plot(chains[iw,:,idim], c='k', alpha=0.5)
            else:
                if plot_bad: plt.plot(chains[iw,:,idim], c='r', alpha=0.5)
                else: pass
        if truths is not None:
            plt.plot([0, chains.shape[1]], [true_pars[idim],true_pars[idim]], '--r', linewidth=3)
        if labels is not None: plt.ylabel(labels[idim])


    plt.subplots_adjust(hspace=0.01)

def layered_corners(samplers, labels=None, index=None, savefolder=None, savefile=None,
                functions=None, truths=None, fig=None, ax=None, alphas=[1.0]*10,
                colors=plt.rcParams['axes.prop_cycle'].by_key()['color'], legend_kwargs={},
                linestyles=['-']*10, pad=0.1, legend_on=True, rng=None, **corner_kwargs):

    if type(samplers[0])==np.ndarray: chains=[sampler.copy() for sampler in samplers]
    else: chains=[sampler.chain.copy() for sampler in samplers]

    if functions is None:
        functions = [None for i in range(chains[0].shape[2])]

    ndim = chains[0].shape[2]

    if ax is None: fig, ax = plt.subplots(ndim,ndim, figsize=(8,8), sharex='col')

    corner_kwargs = dict({'max_n_ticks':4, 'title_kwargs':{"fontsize": 15}, 'label_kwargs':{'fontsize':15},  'label_coords':{'x':(0.5,-0.5), 'y':(-0.5,0.5)},
                                            'plot_contours':True, 'fill_contours':True, 'smooth':2, 'bins':50,
                                            'data_kwargs':{'alpha':0.}, 'truth_color':'k'}, **corner_kwargs)

    for i, chain in enumerate(chains):
        flatchain = chain[:,int(chain.shape[1]/2)::5,:].reshape(-1,chain.shape[2])
        for j in range(chain.shape[2]):
            if functions[j] is not None:
                flatchain[:,j]=functions[j](flatchain[:,j])
        if i==0:
            data_rng = np.vstack((np.min(flatchain, axis=0), np.max(flatchain, axis=0))).T
        else:
            min_chain = np.min(flatchain, axis=0)
            max_chain = np.max(flatchain, axis=0)
            data_rng = np.vstack((np.min(np.vstack((data_rng[:,0], min_chain)), axis=0),
                             np.max(np.vstack((data_rng[:,1], max_chain)), axis=0))).T
    diff = data_rng[:,1]-data_rng[:,0]
    data_rng[:,1] += diff*pad
    data_rng[:,0] -= diff*pad
    if rng is None:
        rng = data_rng.copy()
    else:
        rng = np.vstack(( np.max(np.vstack((data_rng[:,0], rng[:,0])), axis=0),
                          np.min(np.vstack((data_rng[:,1], rng[:,1])), axis=0) )).T
    # rng = [[max(x.min(), range[0][0]), min(x.max(), range[0][1])],
    #        [max(y.min(), range[1][0]), min(y.max(), range[1][1])]]

    if truths is not None:
        truths_f = truths.copy()
        for j in range(chain.shape[2]):
            if functions[j] is not None:
                truths_f[j] = functions[j](truths_f[j])
    else: truths_f = None#truths.copy()

    for i in range(ndim):
        for j in range(ndim):
            if j>i: plt.sca(ax[i,j]); plt.axis('off')

    for i, chain in enumerate(chains):
        flatchain = chain[:,int(chain.shape[1]/2):,:].reshape(-1,chain.shape[2])
        for j in range(chain.shape[2]):
            if functions[j] is not None:
                flatchain[:,j]=functions[j](flatchain[:,j])

        corner_new(flatchain, fig=fig, range=rng, color=colors[i], linestyle=linestyles[i], truths=truths_f,
                            labels=labels, index=index[i], alpha=alphas[i], **corner_kwargs);

        plt.sca(ax[0,0])
        if legend_on: plt.legend(frameon=False, **legend_kwargs)

        if savefolder is not None:
            if i==0:
                if not os.path.exists(savefolder): os.mkdir(savefolder)
            plt.savefig(os.path.join(savefolder, savefile+'_layer%d.png'%i), bbox_inches='tight')



def corner_new(xs, bins=20, range=None, weights=None, color="k", linestyle='-', alpha=1.0,
           smooth=None, smooth1d=None,
           labels=None, label_kwargs=None, label_coords=None, index=None,
           show_titles=False, title_fmt=".2f", title_kwargs=None,
           truths=None, truth_color="#4682b4",
           scale_hist=False, quantiles=None, verbose=False, fig=None,
           max_n_ticks=5, top_ticks=False, use_math_text=False,
           hist_kwargs=None, contourf_kwargs=None, functions=None, **hist2d_kwargs):


    if quantiles is None:
        quantiles = []
    if title_kwargs is None:
        title_kwargs = dict()
    if label_kwargs is None:
        label_kwargs = dict()

    # Try filling in labels from pandas.DataFrame columns.
    if labels is None:
        try:
            labels = xs.columns
        except AttributeError:
            pass

    # Deal with 1D sample lists.
    xs = np.atleast_1d(xs)
    if len(xs.shape) == 1:
        xs = np.atleast_2d(xs)
    else:
        assert len(xs.shape) == 2, "The input sample array must be 1- or 2-D."
        xs = xs.T
    assert xs.shape[0] <= xs.shape[1], "I don't believe that you want more " \
                                       "dimensions than samples!"

    # Parse the parameter ranges.
    if range is None:
        if "extents" in hist2d_kwargs:
            logging.warn("Deprecated keyword argument 'extents'. "
                         "Use 'range' instead.")
            range = hist2d_kwargs.pop("extents")
        else:
            range = [[x.min(), x.max()] for x in xs]
            # Check for parameters that never change.
            m = np.array([e[0] == e[1] for e in range], dtype=bool)
            if np.any(m):
                raise ValueError(("It looks like the parameter(s) in "
                                  "column(s) {0} have no dynamic range. "
                                  "Please provide a `range` argument.")
                                 .format(", ".join(map(
                                     "{0}".format, np.arange(len(m))[m]))))

    else:
        # If any of the extents are percentiles, convert them to ranges.
        # Also make sure it's a normal list.
        range = list(range)
        for i, _ in enumerate(range):
            try:
                emin, emax = range[i]
            except TypeError:
                q = [0.5 - 0.5*range[i], 0.5 + 0.5*range[i]]
                range[i] = quantile(xs[i], q, weights=weights)

    if len(range) != xs.shape[0]:
        raise ValueError("Dimension mismatch between samples and range")

    # Parse the bin specifications.
    try:
        bins = [int(bins) for _ in range]
    except TypeError:
        if len(bins) != len(range):
            raise ValueError("Dimension mismatch between bins and range")

    # Some magic numbers for pretty axis layout.
    K = len(xs)
    factor = 2.0           # size of one side of one panel
    lbdim = 0.5 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.05         # w/hspace size
    plotdim = factor * K + factor * (K - 1.) * whspace
    dim = lbdim + plotdim + trdim

    # Create a new figure if one wasn't provided.
    if fig is None:
        fig, axes = plt.subplots(K, K, figsize=(dim, dim))
    else:
        try:
            axes = np.array(fig.axes).reshape((K, K))
        except:
            raise ValueError("Provided figure has {0} axes, but data has "
                             "dimensions K={1}".format(len(fig.axes), K))

    # Format the figure.
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                        wspace=whspace, hspace=whspace)

    # Set up the default histogram keywords.
    if hist_kwargs is None:
        hist_kwargs = dict()
    hist_kwargs["color"] = color#hist_kwargs.get("color", color)
    if smooth1d is None:
        hist_kwargs["histtype"] = hist_kwargs.get("histtype", "step")

    for i, x in enumerate(xs):
        # Deal with masked arrays.
        if hasattr(x, "compressed"):
            x = x.compressed()

        if np.shape(xs)[0] == 1:
            ax = axes
        else:
            ax = axes[i, i]
        # Plot the histograms.
        if smooth1d is None:
            n, _, _ = ax.hist(x, bins=bins[i], weights=weights, density=True,
                              range=np.sort(range[i]), label=index, linestyle=linestyle, **hist_kwargs)
        else:
            if gaussian_filter is None:
                raise ImportError("Please install scipy for smoothing")
            n, b = np.histogram(x, bins=bins[i], weights=weights,
                                range=np.sort(range[i]))
            n = gaussian_filter(n, smooth1d)
            x0 = np.array(list(zip(b[:-1], b[1:]))).flatten()
            y0 = np.array(list(zip(n, n))).flatten()
            ax.plot(x0, y0, linestyle=linestyle, **hist_kwargs)

        if truths is not None and truths[i] is not None:
            ax.axvline(truths[i], color=truth_color)

        # Plot quantiles if wanted.
        if len(quantiles) > 0:
            qvalues = quantile(x, quantiles, weights=weights)
            for q in qvalues:
                ax.axvline(q, ls="dashed", color=color)

            if verbose:
                print("Quantiles:")
                print([item for item in zip(quantiles, qvalues)])

        if show_titles:
            title = None
            if title_fmt is not None:
                # Compute the quantiles for the title. This might redo
                # unneeded computation but who cares.
                q_16, q_50, q_84 = quantile(x, [0.16, 0.5, 0.84],
                                            weights=weights)
                q_m, q_p = q_50-q_16, q_84-q_50

                # Format the quantile display.
                fmt = "{{0:{0}}}".format(title_fmt).format
                title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                title = title.format(fmt(q_50), fmt(q_m), fmt(q_p))

                # Add in the column name if it's given.
                if labels is not None:
                    title = "{0} = {1}".format(labels[i], title)

            elif labels is not None:
                title = "{0}".format(labels[i])

            if title is not None:
                ax.set_title(title, **title_kwargs)

        ax.set_yticklabels([])
        ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
        ax.set_xlim(range[i])

        if i < K - 1:
            if top_ticks:
                ax.xaxis.set_ticks_position("top")
                [l.set_rotation(45) for l in ax.get_xticklabels()]
            else:
                ax.set_xticklabels([])
        else:
            [l.set_rotation(45) for l in ax.get_xticklabels()]
            if labels is not None:
                ax.set_xlabel(labels[i], **label_kwargs)
                ax.xaxis.set_label_coords(*label_coords['x'])#0.5, -0.5)

            # use MathText for axes ticks
            ax.xaxis.set_major_formatter(
                ScalarFormatter(useMathText=use_math_text))

        for j, y in enumerate(xs):
            if np.shape(xs)[0] == 1:
                ax = axes
            else:
                ax = axes[i, j]
            if j > i:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            elif j == i:
                continue

            # Deal with masked arrays.
            if hasattr(y, "compressed"):
                y = y.compressed()

            hist2d(y, x, ax=ax, range=[range[j], range[i]], weights=weights,
                   color=color, linestyle=linestyle, smooth=smooth, bins=[bins[j], bins[i]], alpha=alpha,
                   countourf_kwargs=contourf_kwargs, **hist2d_kwargs)

            if truths is not None:
                if truths[i] is not None and truths[j] is not None:
                    ax.plot(truths[j], truths[i], "s", color=truth_color)
                if truths[j] is not None:
                    ax.axvline(truths[j], color=truth_color)
                if truths[i] is not None:
                    ax.axhline(truths[i], color=truth_color)

            ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
            ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))

            if i < K - 1:
                ax.set_xticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                if labels is not None:
                    ax.set_xlabel(labels[j], **label_kwargs)
                    ax.xaxis.set_label_coords(*label_coords['x'])#0.5, -0.5)

                # use MathText for axes ticks
                ax.xaxis.set_major_formatter(
                    ScalarFormatter(useMathText=use_math_text))

            if j > 0:
                ax.set_yticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_yticklabels()]
                if labels is not None:
                    ax.set_ylabel(labels[i], **label_kwargs)
                    ax.yaxis.set_label_coords(*label_coords['y'])#(-0.5, 0.5)

                # use MathText for axes ticks
                ax.yaxis.set_major_formatter(
                    ScalarFormatter(useMathText=use_math_text))

    return fig


def hist2d(x, y, bins=20, range=None, weights=None, levels=None, smooth=None,
           ax=None, color=None, linestyle='-', plot_datapoints=True, plot_density=True,
           plot_contours=True, no_fill_contours=False, fill_contours=False,
           contour_kwargs=None, contourf_kwargs=None, data_kwargs=None, alpha=1.0,
           **kwargs):
    """
    Plot a 2-D histogram of samples.

    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.

    y : array_like[nsamples,]
       The samples.

    levels : array_like
        The contour levels to draw.

    ax : matplotlib.Axes
        A axes instance on which to add the 2-D histogram.

    plot_datapoints : bool
        Draw the individual data points.

    plot_density : bool
        Draw the density colormap.

    plot_contours : bool
        Draw the contours.

    no_fill_contours : bool
        Add no filling at all to the contours (unlike setting
        ``fill_contours=False``, which still adds a white fill at the densest
        points).

    fill_contours : bool
        Fill the contours.

    contour_kwargs : dict
        Any additional keyword arguments to pass to the `contour` method.

    contourf_kwargs : dict
        Any additional keyword arguments to pass to the `contourf` method.

    data_kwargs : dict
        Any additional keyword arguments to pass to the `plot` method when
        adding the individual data points.

    """
    if ax is None:
        ax = plt.gca()

    # Set the default range based on the data range if not provided.
    if range is None:
        if "extent" in kwargs:
            logging.warn("Deprecated keyword argument 'extent'. "
                         "Use 'range' instead.")
            range = kwargs["extent"]
        else:
            range = [[x.min(), x.max()], [y.min(), y.max()]]

    # Set up the default plotting arguments.
    if color is None:
        color = "k"

    # Choose the default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)
        levels = 1.0 - np.exp(-0.5 * np.arange(1.0, 2.1, 1.0) ** 2)

    # This is the color map for the density plot, over-plotted to indicate the
    # density of the points near the center.
    density_cmap = LinearSegmentedColormap.from_list(
        "density_cmap", [color[:3]+(color[3]*0.5,), (1, 1, 1, 0)])

    # This color map is used to hide the points at the high density areas.
    white_cmap = LinearSegmentedColormap.from_list(
        "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2)

    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [list(rgba_color) for l in levels] + [list(rgba_color)]
    for i, l in enumerate(levels):
        contour_cmap[i][-1] *= 0.5 * float(i) / (len(levels)+1) * alpha
    contour_cmap[i+1][-1] *= 0.5 * alpha

    # We'll make the 2D histogram to directly estimate the density.
    try:
        H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=bins,
                                 range=list(map(np.sort, range)),
                                 weights=weights)
    except ValueError:
        raise ValueError("It looks like at least one of your sample columns "
                         "have no dynamic range. You could try using the "
                         "'range' argument.")

    if smooth is not None:
        if gaussian_filter is None:
            raise ImportError("Please install scipy for smoothing")
        H = gaussian_filter(H, smooth)

    # Compute the density levels.
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]
    V.sort()
    m = np.diff(V) == 0
    if np.any(m):
        logging.warning("Too few points to create valid contours")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(V) == 0
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate([
        X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
        X1,
        X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
    ])
    Y2 = np.concatenate([
        Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
        Y1,
        Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
    ])

    if plot_datapoints:
        if data_kwargs is None:
            data_kwargs = dict()
        data_kwargs["color"] = data_kwargs.get("color", color)
        data_kwargs["ms"] = data_kwargs.get("ms", 2.0)
        data_kwargs["mec"] = data_kwargs.get("mec", "none")
        data_kwargs["alpha"] = data_kwargs.get("alpha", 0.5)
        ax.plot(x, y, "o", zorder=-1, rasterized=True, **data_kwargs)

    # Plot the base fill to hide the densest data points.
    #if (plot_contours or plot_density) and not no_fill_contours:
    #    ax.contourf(X2, Y2, H2.T, [V.min(), H.max()],
    #                cmap=white_cmap, antialiased=False)

    if plot_contours and fill_contours:
        if contourf_kwargs is None:
            contourf_kwargs = dict()
        contourf_kwargs["colors"] = contourf_kwargs.get("colors", contour_cmap)
        contourf_kwargs["antialiased"] = contourf_kwargs.get("antialiased",
                                                             False)
        ax.contourf(X2, Y2, H2.T, np.concatenate([[0], V, [H.max()*(1+1e-4)]]),
                    **contourf_kwargs)

    # Plot the density map. This can't be plotted at the same time as the
    # contour fills.
    elif plot_density:
        ax.pcolor(X, Y, H.max() - H.T, cmap=density_cmap)

    # Plot the contour edge colors.
    if plot_contours:
        if contour_kwargs is None:
            contour_kwargs = dict()
        #contour_kwargs["colors"] = [contour_kwargs.get("colors", color)]
        #print(color)
        ax.contour(X2, Y2, H2.T, V, linestyles=linestyle, colors=[color], **contour_kwargs)

    ax.set_xlim(range[0])
    ax.set_ylim(range[1])
