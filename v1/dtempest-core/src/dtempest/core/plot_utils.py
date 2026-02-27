"""
Basic plot-related utilities.
"""

import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt

from .config import no_jargon
from .data_utils import make_image_array


def plot_image(data: dict | np.ndarray, fig=None, figsize=None, title_maker=None, jargon: dict = None,
               plot_layout=(1, 1, 1), title_kwargs=None, *imshow_args, **imshow_kwargs):
    """
        Plots an image of a single element of a dataset.

        Parameters
        ----------
        data : dict or array_like
            element of the Raw_dataset.
        fig : matplotlib.pyplot.figure, optional
            Matplotlib.pyplot figure to be plotted on. Especially useful to paint
            various plots manually. The default is None.
        figsize : tuple, optional
            'figsize' parameter for fig. The default is None.
        title_maker : callable, optional
            funtion to return image_title base on injection. The default is None.
        jargon :
            Configuration dictionary for each functionality of dtempest.
        plot_layout : tuple, optional
            Plot layout. Useful mostly to paint
            various plots manually. The default is (1,1,1).
        title_kwargs :
            Keyword arguments to pass to plt.title.
        *imshow_args : iterable
            Arguments to be passed to plt.imshow().
        **imshow_kwargs : dict
            Keyword arguments to be passed to plt.imshow().

        Returns
        -------
        fig : matplotlib.pyplot.figure
            updated figure with the image now plotted.

        """
    if title_kwargs is None:
        title_kwargs = {}
    if jargon is None:
        jargon = no_jargon

    if fig is None:
        fig = plt.figure(figsize=figsize)

    image = make_image_array(data, jargon=jargon)

    ax = fig.add_subplot(*plot_layout)
    ax.imshow(image, *imshow_args, **imshow_kwargs)
    if title_maker is None:
        if isinstance(data, dict):
            ax.set_title(jargon['default_title_maker'](data), **title_kwargs)
        else:
            pass
    else:
        ax.set_title(title_maker(data), **title_kwargs)
    return fig


def _set_corner_limits(fig: plt.Figure, lims: tuple[float, float], pos: int):
    """

    Parameters
    ----------
    fig :
        Matplotlib Figure containing the corner.
    lims :
        Limits for rescaling the axis.
    pos :
        Position (index) of the parameter in the corner plot.
    """
    axes = fig.get_axes()
    nparams = int(np.sqrt(len(axes)))
    axes = np.array(axes).reshape(nparams, nparams)
    for i in range(nparams):
        for j in range(nparams):
            if pos == 0:
                if j == 0:
                    axes[i, j].set_xlim(lims)
            elif pos == nparams - 1:
                if i == pos:
                    axes[i, j].set_xlim(lims) if j == pos else axes[i, j].set_ylim(lims)
            else:
                if j == pos:
                    axes[i, j].set_xlim(lims)
                if i == pos and j != pos:
                    axes[i, j].set_ylim(lims)


def set_corner_limits(fig: plt.Figure, limits: dict[int, tuple[float, float]]):
    """
    Updates limits of corner plot for a given parameter (specified by index).

    Parameters
    ----------
    fig :
        Matplotlib Figure containing the corner.
    limits :
        Dictionary of the form {index: (lower_lim, upper_lim)}.
    """
    for pos, lims in limits.items():
        _set_corner_limits(fig, lims, pos)


def redraw_legend(artist, *args, pos: int = 0, **kwargs):
    """

    Parameters
    ----------
    artist :
        Holder of the legend.
    args :
        Arguments of plt.legend.
    pos :
        If there are multiple legends, specify position on list.
    kwargs :
        Keyword arguments of plt.legend.
    """

    from matplotlib.legend import Legend
    if isinstance(artist, plt.Axes):
        l: Legend = artist.get_legend()
    elif isinstance(artist, plt.Figure):
        l: Legend = artist.legends[pos]  # A figure can have multiple legends
    else:
        raise NotImplementedError(f"Change of legend location is not implemented for objects of type '{type(artist)}'")
    handles = l.legend_handles

    linewidth = kwargs.pop('linewidth', None)
    # Do a more robust implementation for all handler properties if needed
    if linewidth is not None:
        for handle in handles:
            handle.set_linewidth(linewidth)

    l.remove()
    artist.legend(handles=handles, *args, **kwargs)


def change_legend_loc(artist, loc: str | int, pos: int = 0):
    """
    Relocates legend. Unlike 'redraw_legend', it is non-destructive.

    Parameters
    ----------
    artist :
        Holder of the legend.
    loc :
        New position.
    pos :
        If there are multiple legends, specify position on list.
    """
    if isinstance(artist, plt.Axes):
        legend = artist.get_legend()
    elif isinstance(artist, plt.Figure):
        legend = artist.legends[pos]  # A figure can have multiple legends
    else:
        raise NotImplementedError(f"Change of legend location is not implemented for objects of type '{type(artist)}'")

    if isinstance(loc, str):
        from matplotlib.offsetbox import AnchoredOffsetbox
        try:
            legend._loc = AnchoredOffsetbox.codes[loc]
        except KeyError as e:
            raise KeyError(f"Location '{loc}' not valid").with_traceback(e.__traceback__)
    elif isinstance(loc, int):
        legend._loc = loc
    else:
        raise ValueError(f"Paramemer loc can only be of class 'int' or 'str', not '{type(loc)}'")

class Pallet:

    def __init__(self, n: int = 10, cmap: str = 'jet'):
        self.cycler = colour_cycler(n, cmap)

    def colour(self):
        return next(self.cycler())

    def colours(self):
        return self.cycler

    # def merge(self, other):
    #     ex = []
    #     for b, o in blues, oranges:
    #         ex.append(b)
    #         ex.append(o)
    #     cyc = cycle


def colour_cycler(n: int = 10, cmap: str = 'jet'):
    return cycler('color', [plt.get_cmap(cmap)(1. * i / n) for i in range(1, n)])