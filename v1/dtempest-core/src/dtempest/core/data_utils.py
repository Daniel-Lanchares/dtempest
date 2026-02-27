"""
Utilities for data structuring, redefinition and representation.
"""

import numpy as np
from typing import Callable

from .config import no_jargon


def make_image_array(data: dict | np.ndarray, jargon: dict = None) -> np.ndarray:
    """
    Creates array compatible with plt.imshow, of shape (M, N, 3),
    from injection's dictionary or model array directly.
    """
    if jargon is None:
        jargon = no_jargon
    if isinstance(data, dict):
        image = data[jargon['image']]
        print(image)
        image_arr = np.dstack([image[jargon[channel]]
                               if jargon[channel] in image.keys() else np.zeros_like(list(image.values())[0])
                               for channel in ('R', 'G', 'B')]
                              )
    else:
        image_arr = np.dstack([data[i] for i in range(data.shape[0])])  # Todo: test this option with variable channels
    return image_arr


def make_model_array(data: dict, jargon: dict = None) -> np.ndarray:
    """
    Creates array apt to be fed to models, of shape (3, M, N), from injection's dictionary.

    """  # TODO: study n-channel cases, update for image arrays
    if jargon is None:
        jargon = no_jargon

    image = data[jargon['image']]
    image_arr = np.array((image[jargon['R']],  # Different shape from make_image
                          image[jargon['G']],
                          image[jargon['B']]))
    return image_arr


def get_missing_args(type_excep: Exception) -> list[str]:
    """
    Recover the arguments that raised an Exception.

    Parameters
    ----------
    type_excep :
        Exception raised.

    Returns
    -------
    out: list
    Arguments parsed from exception.

    Examples
    -------
    >>> def m_mean(m_1, m_2, **kwargs):
    ...     return (m_1 + m_2)/2
    >>> try:
    ...     mean = m_mean(m_0=30.0, m_2=40.0)
    ...     missing_args = None
    ... except TypeError as error:
    ...     missing_args = get_missing_args(error)
    >>> missing_args
    ['m_1']
    """
    crumbs = type_excep.args[0].split(' ')
    n_args = int(crumbs[2])

    return [crumbs[-2 * i + 1][1:-1] for i in reversed(range(1, n_args + 1))]


def calc_parameter(param: str, pool: dict[str, Callable], params_dict) -> float:
    """
    Recursively calculates parameter from a given set of relations and arguments.

    Parameters
    ----------
    param :
        Parameter to be calculated.
    pool :
        Map of parameter names to functions that compute them. Used also to calculate missing arguments.
    params_dict :
        Dictionary of available parameters and their value.

    Returns
    -------
    out :
        Value of the specified parameter

    Examples
    -------
    >>> def empty(**kwargs):
    ...     raise NotImplementedError
    >>> def m_1(m_0, **kwargs):
    ...     return 2*m_0
    >>> def m_mean(m_1, m_2, **kwargs):
    ...     return (m_1 + m_2)/2

    >>> example_pool = {'m_0': empty, 'm_1': m_1, 'm_2': empty, 'm_mean': m_mean}
    >>> example_dict = {'m_0': 30.0, 'm_2': 40.0}
    >>> calc_parameter('m_mean', example_pool, example_dict)
    50.0
    """
    try:
        return pool[param](**params_dict)
    except TypeError as error:
        missing_args = get_missing_args(error)
        missing_dict = {arg: calc_parameter(arg, pool, params_dict) for arg in missing_args}
        return pool[param](**missing_dict, **params_dict)


def get_parameter_alias(parameter, jargon: dict = None) -> str:
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


def get_parameter_units(parameter, jargon: dict = None) -> str:
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
