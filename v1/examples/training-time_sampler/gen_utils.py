# Base imports
import torch
import numpy as np

# from zuko.distributions import Distribution
from bilby.gw.prior import Prior

# GW-modules
import bilby
from gwpy.timeseries.timeseries import TimeSeries as gwpy_TS

def ifo_q_transform(tseries: np.ndarray,
                    resol=(128, 128),
                    duration=2,
                    sampling_frequency=1024,
                    **qtrans_kwargs) -> np.ndarray:
    """
    Converts a timeseries into a one-channel image of shape 'resol'.

    Parameters
    ----------
    tseries :
        Timeseries created in 'generate_timeseries'.
    resol :
        Image resolution (height, width) in pixels.
    duration :
        Duration for the intermediate timeseries.
    sampling_frequency :
        Sampling frequency for the (new) gwpy TimeSeries.
    qtrans_kwargs :
        Keyword arguments passed to gwpy's TimeSeries.q_transform.

    Returns
    -------
    out :
        numpy array of shape resol representing one channel of an image.
    """
    gw_tseries = gwpy_TS(tseries, t0=-duration / 2, sample_rate=sampling_frequency)

    qtrans = gw_tseries.q_transform(whiten=False, **qtrans_kwargs)
    image_channel = prepare_array(qtrans.real, resol)
    return np.array(image_channel)


def handle_time(theta: torch.Tensor, parameter_labels: np.array) -> float:
    """
    Handles different possible time parameters.

    Parameters
    ----------
    prior_sample :
        Sample of the parameters.

    Returns
    -------
    out : float
        time of coalescence for generation
    """
    if 'geocent_time' in parameter_labels:
        time = theta[parameter_labels == 'geocent_time']
    elif 'normalized_time' in parameter_labels:
        time = theta[parameter_labels == 'normalized_time'] * 24 * 3600
    else:
        time = 0.0
    return time

def handle_snr(snr_prior: Prior | float):
    if isinstance(snr_prior, float):
        return snr_prior
    return snr_prior.sample()


def setup_ifos(ifos=None, asds=None, duration=None, sampling_frequency=None, time=0.):
    """Sets up the two interferometers, H1 and L1. Uses ALIGO power spectral density by default.

    Args:
        ifos: A collection of interferometer names.
        asds: An optional list of amplitude spectral densities for each interferometer.
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


def getsnr(ifos):
    """Calculates the signal-to-noise ratio (SNR) of the injected signal in two interferometers.

    Args:
        ifos: A list of two bilby.gw.detector.Interferometer objects.

    Returns:
        A float, representing the SNR of the injected signal.
    """

    matched_filters_sq = [abs(ifo.meta_data['matched_filter_SNR']) ** 2 for ifo in ifos]

    # Calculate the total SNR of the injected signal by combining the matched filter SNRs of the two interferometers.
    snr = np.sqrt(np.sum(matched_filters_sq))

    return np.nan_to_num(snr)

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