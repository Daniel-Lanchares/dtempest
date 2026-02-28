import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

import bilby
from skimage.transform import resize
from gwpy.timeseries.timeseries import TimeSeries as gwpy_TS

from dtempest.core.conversion_utils import calc_parameter
from ..config import cbc_jargon

def data_convert(params_dict, param_pool, params_list):
    return np.array([params_dict[param] if param in params_dict else calc_parameter(param, param_pool, params_dict)
                     for param in params_list])


def fill_hdf5(batch_num, h_file, params_list, data, valid_fraction: float = 0.0):
    data_length = len(data)

    images, labels = zip(*data)
    images, labels = np.array(images), np.array(labels)

    # print(labels[:2]) # Need to be converted from dict to array with specific param_list

    if '_snr' in labels[0].keys():
        snr = np.array([data['_snr'] for data in labels])
    else:
        snr = None

    convert_func = partial(data_convert, params_list=params_list, param_pool=cbc_jargon['param_pool'])
    # row_dicts = labels #[dict(id=index, **dict(row)) for index, row in dataset.iterrows()]

    label_arr = np.zeros((len(labels), len(params_list)))
    i = 0

    with Pool() as pool:
        with tqdm(desc='Label Conversion', total=len(labels)) as p_bar:
            for label in pool.imap(convert_func, labels):
                p_bar.update(1)
                label_arr[i][:] = label
                i += 1

    labels = label_arr

    datasets = ['training']
    if valid_fraction > 0.0:
        datasets.append('validation')
        valid_length = int(data_length * valid_fraction)
        # print(valid_length)
        valid_images = images[:valid_length]
        valid_labels = labels[:valid_length]

        train_images = images[valid_length:]
        train_labels = labels[valid_length:]

        if snr is not None:
            valid_snr = snr[:valid_length]
            train_snr = snr[valid_length:]
            snrs = (train_snr, valid_snr)
        else:
            snrs = (None, None)

        images = (train_images, valid_images)
        labels = (train_labels, valid_labels)

    for s, image_dat, label_dat, snr_dat in zip(datasets, images, labels, snrs):
        low = batch_num * len(image_dat)
        high = low + len(image_dat)

        h_file[f'{s}']['images'][low:high] = image_dat
        h_file[f'{s}']['labels'][low:high] = label_dat
        h_file[f'{s}']['snrs'][low:high] = snr_dat


# Current approach - requires manual calculation
def format_timedelta(td):
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if days:
        return f"{days}d, {hours:02}:{minutes:02}:{seconds:02}"
    else:
        return f"{hours:02}:{minutes:02}:{seconds:02}"

def getsnr(ifos):
    """Calculates the signal-to-noise ratio (SNR) of the injected signal in two interferometers.

    Args:
        ifos: A list of two bilby.gw.detector.Interferometer objects.

    Returns:
        A float, representing the SNR of the injected signal.
    """

    # Calculate the matched filter SNR of the injected signal in each interferometer.
    matched_filters_sq = [abs(ifo.meta_data['matched_filter_SNR']) ** 2 for ifo in ifos]

    # Calculate the total SNR of the injected signal by combining the matched filter SNRs of the two interferometers.
    snr = np.sqrt(np.sum(matched_filters_sq))
    return np.nan_to_num(snr)

def ifo_q_transform(tseries: np.ndarray, resol=(128, 128), duration=2, sampling_frequency=1024, dtype=np.float32, **qtrans_kwargs):
    gw_tseries = gwpy_TS(tseries, t0=-duration / 2, sample_rate=sampling_frequency)

    qtrans = gw_tseries.q_transform(whiten=False, **qtrans_kwargs)
    return prepare_array(qtrans.real, resol, dtype)


def prepare_array(arr, resol=(128, 128), dtype=np.float32):
    """
    Transform q-transform's real part into a 128 x 128 channel of the image
    """
    # Might make use of the fact that it is still a Spectrogram before casting it
    arr = np.abs(np.flip(arr, axis=1).T / np.max(arr))
    # print(arr.shape)  # Default is (606, 1000)!!
    # arr = arr[:, 340:900]
    arr = resize(arr, resol)
    # arr = interpolate(torch.as_tensor(arr), size=resol)
    return np.asarray(arr, dtype=dtype)

def query_noise(t, ifos, path, segment_len=500, **fetch_kwargs):

    for ifo in ifos:
        try:
            strain = gwpy_TS.fetch_open_data(ifo, t, t + segment_len, **fetch_kwargs)
            strain.write(target=path / f'noise_{t}_{ifo}', format='hdf5')
        except ValueError:
            print(f'GWOSC has no data for time {t} on every detector requested (missing at least {ifo})')
            continue


def set_metadata(file, metadata: dict, master_key=""):
    for key, val in metadata.items():
            if master_key:
                key = f'{master_key}/{key}'

            if isinstance(val, dict):
                set_metadata(file, val, master_key=key)
            else:
                # print(key)
                file.attrs[key] = val

def setup_ifos(ifos=None, asds=None, duration=None, sampling_frequency=None, time=0.):
    """Sets up the two interferometers, H1 and L1. Uses ALIGO power spectral density by default.

    Args:
        ifos: A list of interferometer names
        asds: An optional list of amplitude spectral densities for each interferometer
        duration: The duration of the injected signal.
        sampling_frequency: The sampling frequency of the injected signal.

    Returns:
        A list of two bilby.gw.detector.Interferometer objects.
    """
    if ifos is None:
        ifos = ["H1", "L1"]

    # Create a list of two bilby.gw.detector.Interferometer objects.
    ifos = bilby.gw.detector.InterferometerList(ifos)

    # Set the amplitude spectral density of each interferometer if given
    if asds is not None:
        from bilby.gw.detector.psd import PowerSpectralDensity
        for i, ifo in enumerate(ifos):
            ifo_frequency_array = np.fft.rfftfreq(sampling_frequency, 1 / sampling_frequency)
            ifo_frequency_array = ifo_frequency_array[ifo_frequency_array > ifo.minimum_frequency]
            ifo_frequency_array = ifo_frequency_array[ifo_frequency_array < ifo.maximum_frequency]

            # ifo.power_spectral_density = \
            #     PowerSpectralDensity.from_amplitude_spectral_density_array(frequency_array=ifo_frequency_array,
            #                                                                asd_array=asds[i])

            # Depends on whether sampling_rate/2 is larger than the interferometer's max frequency
            max_freq = min([ifo.maximum_frequency, ifo_frequency_array[-1]])
            asd_array = asds[i][int(ifo.minimum_frequency):int(max_freq)]
            ifo.power_spectral_density = PowerSpectralDensity(frequency_array=ifo_frequency_array,
                                                              asd_array=asd_array)
    # Set the strain data for each interferometer from a power spectral density.
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=time - duration / 2,
    )

    return ifos