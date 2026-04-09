import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt


# From https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

def heatmap(data, row_labels, col_labels, ax: plt.Axes = None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(range(data.shape[1]), labels=col_labels,
                  rotation=-30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def mark_special_cases(ax, cases, color: str = "red", marker: str = "*"):
    # Adds a marker to the left of a label and changes its color.
    import re
    pattern = re.compile(r"\W")
    yticklabels = [item.get_text() for item in ax.get_yticklabels()]
    for i, ytick in enumerate(ax.get_yticklabels()):
        if pattern.sub("", ytick.get_text()) in cases: # Drops non-alphanumerics to filter out markers
            ytick.set_color(color)
            yticklabels[i] = marker + ytick.get_text()

    ax.set_yticklabels(yticklabels)

def plotting():
    labels = np.load(estimation_dir / 'GWTC-3_labels.npy')
    events2 = np.load(estimation_dir / 'GWTC-2.1_events.npy')
    events3 = np.load(estimation_dir / 'GWTC-3_events.npy')
    divs2 = np.load(estimation_dir / 'GWTC-2.1_divergences.npy')
    divs3 = np.load(estimation_dir / 'GWTC-3_divergences.npy')

    means = np.concatenate([np.mean(divs2, axis=1), np.mean(divs3, axis=1)])*100
    print(f"GWTC-2.1-confident mean JSD: {np.mean(divs2, axis=1):.1e}")
    print(f"GWTC-3.0-confident mean JSD: {np.mean(divs3, axis=1):.1e}")
    print(f"Combined min (max): {means.min():.1e}, ({means.max():.1e})")

    # Send "weird" events to the low end of the table
    mask = np.isin(events3, tuple(weird_events))
    events3 = np.concat((np.delete(events3, mask), events3[mask]))
    divs3 = np.concat((np.delete(divs3, mask, axis=0), divs3[mask]), axis=0)

    fontsize = 13
    aspect = 2/3

    plt.style.use(estimation_dir / 'draft.mplstyle')

    ###################--GWTC-2.1--##################################

    fig, ax = plt.subplots(figsize=(10, 8))
    im, cbar = heatmap(divs2*100, events2, labels, ax=ax, vmin=0, vmax=70,
                       cmap="Blues", cbarlabel=r"$\text{JS Divergence} [\times 10^{-2} \text{ nat}]$",
                       cbar_kw={
                           'shrink': 0.8,
                            'aspect': 20 * 6 / 4
                            },
                       aspect=aspect
                       )
    texts = annotate_heatmap(im, valfmt="{x:.1f}", fontsize=fontsize)

    cbar.ax.yaxis.get_label().set_fontsize('x-large')
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.grid(False)

    mark_special_cases(ax, low_mass_events, "royalblue", "*")


    fig.tight_layout()
    fig.savefig(estimation_dir/'GWTC-2.1_GP15.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()

    ###################--GWTC-3.0--##################################

    fig, ax = plt.subplots(figsize=(10, 5))

    im, cbar = heatmap(divs3 * 100, events3, labels, ax=ax, vmin=0, vmax=70,
                       cmap="Blues", cbarlabel=r"$\text{JS Divergence} [\times 10^{-2} \text{ nat}]$",
                       cbar_kw={'shrink': 0.74},
                       aspect=aspect
                       )
    texts = annotate_heatmap(im, valfmt="{x:.1f}", fontsize=fontsize)

    cbar.ax.yaxis.get_label().set_fontsize('x-large')
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.grid(False)

    mark_special_cases(ax, weird_events, "goldenrod", r"\textsuperscript{\textdagger}")

    fig.tight_layout()
    fig.savefig(estimation_dir / 'GWTC-3_GP15.pdf', bbox_inches='tight')


low_mass_events = ("GW190408_181802", "GW190412_053044", "GW190512_180714", "GW190828_065509") # Near 15 solar masses in chirp_mass
weird_events = {"GW191113_071753", "GW200208_222617"} # Weird event with bimodal chirp_mass or sub 15 solar masses

if __name__ == '__main__':
    """
    Annotated tiled heatmap plot of previously calculated JSD divergences.
    """

    name = 'GP15_example'
    params = ['chirp_mass', 'mass_ratio',
              'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_jl', 'phi_12',
              'luminosity_distance', 'theta_jn', 'ra', 'dec',
              'phase', 'psi', 'geocent_time']

    estimation_dir = Path("")

    nsamples = 10000
    plotting()