import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from cycler import cycler
import matplotlib.pyplot as plt
from collections import OrderedDict


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


class Pallete:

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


def change_legend_loc(artist, loc: str | int, pos: int = 0):
    """

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


def redraw_legend(artist, *args, pos: int = 0, **kwargs):
    from matplotlib.legend import Legend
    # from matplotlib.font_manager import FontProperties
    # from matplotlib.legend import legend_handler
    if isinstance(artist, plt.Axes):
        l: Legend = artist.get_legend()
    elif isinstance(artist, plt.Figure):
        l: Legend = artist.legends[pos]  # A figure can have multiple legends
    else:
        raise NotImplementedError(f"Change of legend location is not implemented for objects of type '{type(artist)}'")
    # l.prop = FontProperties(size=fontsize)
    # l._fontsize = l.prop.get_size_in_points()
    # for text in l.texts:
    #     text.set_fontsize(fontsize)
    # texts = [text.get_text() for text in l.texts]
    handles = l.legend_handles

    linewidth = kwargs.pop('linewidth', None)
    # Do a more robust implementation for all handler properties if needed
    if linewidth is not None:
        for handle in handles:
            handle.set_linewidth(linewidth)

    l.remove()
    artist.legend(handles=handles, *args, **kwargs)


def identity(x):
    return x


def get_missing_args(type_excep):
    crumbs = type_excep.args[0].split(' ')
    n_args = int(crumbs[2])

    return [crumbs[-2 * i + 1][1:-1] for i in reversed(range(1, n_args + 1))]


def get_extractor(name):
    import torchvision.models as models
    model = getattr(models, name)
    weights = getattr(getattr(models, f'{models_dict[name]}_Weights'), 'DEFAULT')
    pre_process = weights.transforms(antialias=True)  # True for internal compatibility reasons

    return model, weights, pre_process


models_dict = {
    'resnet18': 'ResNet18'
}


def load_losses(directory: str | Path,
                model: str = None,
                stages: int | list[int] = None,
                validation: bool = False,
                verbose: bool = True) -> tuple[dict, dict] | tuple[dict, dict, dict, dict]:
    subdirs = [f.name for f in os.scandir(directory) if f.is_dir()]
    if stages is None:
        chosen_subs = {int(subdir.split('_')[-1]): subdir for subdir in subdirs}
    else:
        if not hasattr(stages, '__iter__'):
            stages = [stages, ]
        chosen_subs = {int(subdir.split('_')[-1]): subdir for subdir in subdirs  # add '#' to ignore directory
                       if int(subdir.split('_')[-1]) in stages and subdir.split('_')[0] != '#'}

    if model is not None:
        chosen_subs = {stage: subdir for stage, subdir in chosen_subs.items() if subdir.split('_')[-3] == model}

    # Sort the stages to avoid issues down the line
    chosen_subs = {stage: chosen_subs[stage] for stage in sorted(chosen_subs.keys())}

    epochs = {}
    losses = {}
    vali_epochs = {}
    validations = {}
    last_epoch = 0  # Should never be needed, but suppresses static warning
    for i, (stage, sub) in enumerate(chosen_subs.items()):
        epoch, loss = torch.load(Path(directory) / sub / 'loss_data.pt')

        if len(epoch.shape) == 1:  # To deal with legacy loss format
            nepochs = int(epoch[-1]) + 1
            epoch = epoch.reshape((nepochs, -1))
            loss = loss.reshape((nepochs, -1))

        if i == 0:
            epochs[stage] = epoch

        else:
            epochs[stage] = epoch + last_epoch * np.ones_like(epoch)

        losses[stage] = loss
        if validation:
            try:
                vali_epoch, valid = torch.load(Path(directory) / sub / 'validation_data.pt')
                if len(vali_epoch.shape) == 1:
                    nepochs = int(vali_epoch[-1]) + 1
                    vali_epoch = vali_epoch.reshape((nepochs, -1))
                    valid = valid.reshape((nepochs, -1))
                if i == 0:
                    vali_epochs[stage] = vali_epoch
                else:
                    vali_epochs[stage] = vali_epoch + last_epoch * np.ones_like(vali_epoch)
                validations[stage] = valid
            except FileNotFoundError as exc:
                print(f'Validation not found in {sub}')
                print(exc)
        if verbose:
            print(f'Loaded {sub}')
        last_epoch = epochs[stage].flatten()[-1]
    if validation:
        return epochs, losses, vali_epochs, validations
    return epochs, losses