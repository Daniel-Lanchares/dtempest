import matplotlib.pyplot as plt
import numpy as np

from .config import no_jargon
from .common_utils import get_missing_args

'''
In this file the raw dataset (list of dictionaries) is transformed
into a dataset compatible with PyTorch [[image0, labels0], ...]

Here regression parameters are also chosen 
'''

def make_image(data: dict | np.ndarray, jargon: dict = None):
    """
    Creates image array (compatible with plt.imshow()) 
    from injection's dictionary
    """
    if jargon is None:
        jargon = no_jargon
    if isinstance(data, dict):
        image = data[jargon['image']]
        image_arr = np.dstack((image[jargon['R']],
                               image[jargon['G']],
                               image[jargon['B']]))
    else:
        image_arr = np.dstack((data[0], data[1], data[2]))
    return image_arr


def make_array(data: dict, jargon: dict = None):
    """
    Creates array apt to be fed to models
    Parameters
    ----------
    data : 
    jargon : 

    Returns
    -------

    """
    if jargon is None:
        jargon = no_jargon

    image = data[jargon['image']]
    image_arr = np.array((image[jargon['R']],  # Different shape from make_image
                          image[jargon['G']],
                          image[jargon['B']]))
    return image_arr


def extract_parameters(dataset, params_list, jargon: dict = None):
    """
    Extracts an array of specified parameters
    from a dataset (or its path).
    """
    if jargon is None:
        raise RuntimeError('You have no jargon defined: '
                           'To properly convert parameters a jargon["parameter_pool"] is required')
    param_pool = jargon['param_pool']

    # dataset = check_format(dataset)

    label_list = []

    for index, inj in dataset.iterrows():
        params_dict = inj[jargon['parameters']]

        new_params_dict = {}
        labels = []  # same info as new_params_dict but in an ordered container
        for param in params_list:
            if param in params_dict:
                new_params_dict[param] = params_dict[param]
            else:  # if not among the base params compute parameter from them
                new_params_dict[param] = param_pool[param](**params_dict)
            labels.append(new_params_dict[param])

        label_list.append(np.array(labels))
    return np.array(label_list)

def calc_parameter(param, pool, params_dict):
    try:
        return pool[param](**params_dict)
    except TypeError as error:
        missing_args = get_missing_args(error)
        missing_dict = {arg: calc_parameter(arg, pool, params_dict) for arg in missing_args}
        return pool[param](**missing_dict, **params_dict)

def get_param_alias(parameter, jargon: dict = None):
    """
    Returns alias of given parameter. Used for plotting.
    """
    if jargon is None:
        raise RuntimeError('You have no jargon defined: '
                           'To properly convert parameters a jargon["alias_dict"] is required')
    if jargon['labels'] is None:
        return 'unknown unit'
    try:
        label = jargon['labels'][parameter]
    except KeyError:
        print(f'Parameter "{parameter}" misspelled or unit not yet implemented')
        return 'unknown alias'
    split = label.split(' ')
    if len(split) == 1:
        alias = split[0][1:-1]
    elif len(split) == 2:
        alias = split[0][1:]
    else:
        alias = ''  # If this executes label format is not right ($alias [unit]$)
    return r'${}$'.format(alias)


def get_param_units(parameter, jargon: dict = None):
    """
    Returns units of given parameter. Used for plotting.
    """
    if jargon is None:
        raise RuntimeError('You have no jargon defined: '
                           'To properly convert parameters a jargon["unit_dict"] is required')
    if jargon['labels'] is None:
        return 'unknown unit'
    try:
        label = jargon['labels'][parameter]
    except KeyError:
        print(f'Parameter "{parameter}" misspelled or unit not yet implemented')
        return 'unknown unit'

    try:
        unit = label.split(' ')[1][1:-2]
    except IndexError:
        return r'$ø$'
    return r'${}$'.format(unit)


def plot_image(data, fig=None, figsize=None, title_maker=None, jargon: dict = None,
               plot_layout=(1, 1, 1), title_kwargs=None, *imshow_args, **imshow_kwargs):
    """
        Plots a histogram of a given parameter list on a single subplot

        Parameters
        ----------
        data : dict
            element of the Raw_dataset.
        fig : matplotlib.pyplot.figure, optional
            Matplotlib.pyplot figure to be plotted on. Especially useful to paint
            various plots manually. The default is None.
        figsize : tuple, optional
            'figsize' parameter for fig. The default is None.
        title_maker : callable, optional
            funtion to return image_title base on injection. The default is None.
        plot_layout : tuple, optional
            Plot layout. Useful mostly to paint
            various plots manually. Only The default is (1,1,1).
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

    image = make_image(data, jargon=jargon)

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


def plot_images(dataset,
                index_array: np.ndarray,
                fig=None,
                figsize=None,
                title_maker=None,
                jargon: dict = None,
                *imshow_args, **imshow_kwargs):
    """
        Plots histograms of the given parameter array on one or more subplots

        Parameters
        ----------
        dataset : TYPE
            Raw_dataset.
        index_array : np.ndarray
            Array of indexes of the dataset to plot. Dictates figure layout
        fig : matplotlib.pyplot.figure, optional
            Matplotlib.pyplot figure to be plotted on. Especially useful to paint
            various plots manually. The default is None.
        figsize : tuple, optional
            'figsize' parameter for fig. The default is None.
        title_maker : callable, optional
            funtion to return image_title base on injection. The default is None.
        *imshow_args : iterable
            Arguments to be passed to plt.imshow().
        **imshow_kwargs : dict
            Keyword arguments to be passed to plt.imshow().

        Returns
        -------
        fig : matplotlib.pyplot.figure
            updated figure with the images now plotted.

        """
    # Study way of having different args and kwargs for each hist

    if fig is None:
        fig = plt.figure(figsize=figsize)

    # dataset = check_format(dataset)
    layout = index_array.shape
    flat_array = index_array.flatten()
    for i, indx in enumerate(flat_array):
        fig = plot_image(dataset[indx], fig=fig, figsize=figsize, title_maker=title_maker, jargon=jargon,
                         plot_layout=(*layout, i + 1), *imshow_args, **imshow_kwargs)
    plt.tight_layout()
    return fig
