import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

"""
Various useful functions, maily loss and drawing related.
"""

# Look for more colors just in case
class PrintStyle:
    black = '\033[30m'
    red = '\033[31m'
    green = '\033[32m'
    orange = '\033[33m'
    blue = '\033[34m'
    purple = '\033[35m'
    cyan = '\033[36m'
    lightgrey = '\033[37m'
    darkgrey = '\033[90m'
    lightred = '\033[91m'
    lightgreen = '\033[92m'
    yellow = '\033[93m'
    lightblue = '\033[94m'
    pink = '\033[95m'
    lightcyan = '\033[96m'

    reset = '\033[0m'
    bold = '\033[01m'
    disable = '\033[02m'
    underline = '\033[04m'
    reverse = '\033[07m'
    strikethrough = '\033[09m'
    invisible = '\033[08m'


def change_legend_loc(artist, loc: str | int, pos: int = 0) -> None:
    """
    Changes location of a pyplot legend.

    Parameters
    ----------
    artist : holder of the legend
    loc : new position
    pos : if there al multiple legends, specify position on list

    Unlike redraw_legend, it is non-destructive
    -------

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


def redraw_legend(artist, *args, pos: int = 0, **kwargs) -> None:
    """
    Redraws a pyplot legend

    Parameters
    ----------
    artist : holder of the legend
    args : new arguments for the legend
    pos : if there al multiple legends, specify position on list
    kwargs : new keyword arguments for the legend

    Returns
    -------

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


def identity(x):
    """
    Convenience functions that returns the argument it is passed
    """
    return x


def get_extractor(name):
    """
    Obtain specific extractor with its weights and preprocessing function.

    Parameters
    ----------
    name : Model code with no symbols or uppercase (e.g. resnet18 for ResNet-18)

    Returns
    -------
    the model, the pre-trained weights and the preprocessing pipeline.

    """
    import torchvision.models as models
    model = getattr(models, name)
    weights = getattr(getattr(models, f'{models_dict[name]}_Weights'), 'DEFAULT')
    pre_process = weights.transforms(antialias=True)  # True for internal compatibility reasons

    return model, weights, pre_process


models_dict = {
    'resnet18': 'ResNet18',
    'resnet50': 'ResNet50'
}


def process_loss(arr: np.ndarray, points: int = None) -> np.ndarray:
    """

    Parameters
    ----------
    arr : loss array
    points : number of points of the averaged loss. Currently, supports 1 or number of epochs

    Returns
    -------
    Averaged loss array

    """
    if points is None:
        return arr.flatten()
    elif points == arr.shape[0]:
        return np.mean(arr, axis=1)
    else:
        raise NotImplementedError("Only per-epoch averages are supported")


def process_epochs(arr: np.ndarray, points: int = None) -> np.ndarray:
    """

    Parameters
    ----------
    arr : epochs array
    points : number of points to average over

    Returns
    -------

    """
    if points is None:
        return arr.flatten()
    elif points == arr.shape[0]:
        return arr[:, -1]
    else:
        return arr.flatten()[0::len(arr.flatten())//points]

def get_loss(path: Path, points: int = None, validation: bool = False,):
    """

    Parameters
    ----------
    path : path to loss file
    points : points to average over. Currently, supports 1 or number of epochs
    validation

    Returns
    -------
    Epochs and average loss arrays

    """
    if validation:
        epochs, loss = torch.load(path/'validation_data.pt', weights_only=False)
    else:
        epochs, loss = torch.load(path/'loss_data.pt', weights_only=False)
    return process_epochs(epochs, points), process_loss(loss, points)