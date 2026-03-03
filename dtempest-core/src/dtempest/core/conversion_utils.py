import matplotlib.pyplot as plt
import numpy as np

from .config import no_jargon, Pool_map

'''
Utilities related to conversion between parameters and data structures.
'''

def make_image(data: dict | np.ndarray, jargon: dict = None):
    """
    Creates image array (compatible with plt.imshow()) from data.
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
    Creates array apt to be fed to model's parameters.
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
    Extracts an array of specified parameters from a dataset.
    """
    if jargon is None:
        raise RuntimeError('You have no jargon defined: '
                           'To properly convert parameters a jargon["parameter_pool"] is required')
    param_pool = jargon['param_pool']

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

def get_missing_args(type_excep: TypeError):
    """Extract missing parameters on a caught type-related exception."""
    crumbs = type_excep.args[0].split(' ')
    n_args = int(crumbs[2])
    return [crumbs[-2 * i + 1][1:-1] for i in reversed(range(1, n_args + 1))]

def calc_parameter(param: str, pool: Pool_map, params_dict):
    """
    Recursively calculates a given parameter from a pool of conversion functions

    Parameters
    ----------
    param : name of the parameter to calculate
    pool : mapping of conversion functions to draw from for calculation
    params_dict : mapping of parameters at our disposal.

    Returns
    -------
    Requested parameter

    """
    try:
        return pool[param](**params_dict)
    except TypeError as error:
        # If some of the required parameters are missing, it throws a TypeError we can inspect
        # and recursively calculate the missing arguments of pool[param].
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
        data : np.array
            element of the Dataset.
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
