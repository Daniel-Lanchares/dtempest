"""
GW-oriented utilities.
"""

import numpy as np
from pathlib import Path
from typing import Iterable
from functools import partial

from gwpy.timeseries import TimeSeries

import dtempest.core.data_utils as core
import dtempest.core.plot_utils as p_core
from .config import cbc_jargon

make_image = partial(core.make_image_array, jargon=cbc_jargon)
make_array = partial(core.make_model_array, jargon=cbc_jargon)
get_parameter_alias = partial(core.get_parameter_alias, jargon=cbc_jargon)
get_parameter_units = partial(core.get_parameter_units, jargon=cbc_jargon)

plot_image = partial(p_core.plot_image, jargon=cbc_jargon)


def query_noise(t, ifos: Iterable[str], path: Path | str, duration=500, **fetch_kwargs):
    """
    Collects noise samples from the internet and writes them to an hdf5
    Wrapper utility function for gwpy Timeseries' fetch_open_data and write methods.

    Parameters
    ----------
    t :
        Initial time for the noise sample to query.

    ifos :
        List of interferometers to query. It will report missing data.

    path :
        Target directory for noise files.

    duration :
        duration of each noise sample.

    fetch_kwargs :
        extra keyword arguments for gwpy Timeseries' fetch_open_data.

    Returns
    -------

    """
    path = Path(path)
    path.mkdir(exist_ok=True)
    for ifo in ifos:
        try:
            strain = TimeSeries.fetch_open_data(ifo, t, t + duration, **fetch_kwargs)
            strain.write(target=path / f'noise_{t}_{ifo}', format='hdf5')
        except ValueError:
            print(f'GWOSC has no data for time {t} on every detector requested (missing at least {ifo})')
            continue


def prepare_array(arr: np.ndarray, resol: tuple[float, float] = (128, 128)):
    """
    Flip upright, normalize to maximum and resize array.

    Parameters
    ----------
    arr :
        Array to prepare.
    resol :
        Resizing resolution.

    Returns
    -------
    out:
        Prepared array of size 'resol'.

    """
    from skimage.transform import resize
    arr = np.abs(np.flip(arr, axis=1).T / np.max(arr))
    arr = resize(arr, resol)
    return arr

