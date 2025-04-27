"""
Simplified generation pipeline intended for the Artemisa computation cluster  (Large Datasets)
"""
# Base imports
import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from joblib import Parallel, delayed
from collections.abc import Callable

# GW-modules
import bilby
from bilby.core.prior import PriorDict, Uniform
from gwpy.timeseries.timeseries import TimeSeries as gwpy_TS

from ..config import cbc_jargon
from ..utils import prepare_array
from dtempest.core.data_utils import calc_parameter

def artemisa_gen(filepath: Path | str = 'Dataset.h5',
                 n_images: int = 1,
                 n_batches: int = 1,
                 valid_fraction: float = 0.0,
                 parameters: list = None,
                 prior: PriorDict = None,
                 snr_tol: float = 0.5,
                 seed: int = 0,
                 joblib_kwargs: dict = None,
                **injection_kwargs):
    """
    Generation pipeline tested in the Artemisa Computer Cluster.

    Parameters
    ----------
    filepath :
        Path to the created dataset.

    n_images :
        Number of images to generate.

    n_batches :
        Number of batches in which to split n_images.

    valid_fraction :
        Fraction of dataset to reserve for validation.

    parameters :
        Label categories for the generated dataset.

    prior :
        Bilby's PriorDict object from which to sample events.

    snr_tol :
        Tolerance for the SNR adjusting algorithm. Accepts SNR \\in (SNR-snr_tol, SNR+snr_tol).

    seed :
        Seed for reproducibility.

    joblib_kwargs :
        Arguments for joblib's Parallel.

    injection_kwargs :
        Arguments for create_image.

    Returns
    -------
    out :
        None
    """
    
    assert n_batches <= n_images, f"You need more images ({n_images}) than batches ({n_batches})"
    filepath = Path(filepath)
    filepath.parent.mkdir(exist_ok=True)
    bilby.utils.random.seed(seed)
    
    # prior must include 'snr' key. Otherwise, taken to be uniform 5 to 30.
    # Will rescale luminosity distance accordingly until within tolerance.
    
    if prior is None:
        prior = bilby.gw.prior.BBHPriorDict()
    if snr_tol in (0.0, -1.0):
        # Special cases: Return as is or with zero noise
        prior['_snr'] = snr_tol
    if '_snr' not in prior.keys():
        prior['_snr'] = Uniform(5, 30, name='snr', latex_label=r'$\textrm{SNR}$')
        
    if joblib_kwargs is None:
        joblib_kwargs = {}
    
    metadata = {
                # 'prior': prior._get_json_dict(),  # Later to be recovered with "def _get_from_json_dict(cls, prior_dict)"
                'seed': seed,
                'n_images': n_images,
                'snr_tol': snr_tol,
                'injection_kwargs': {key: value for key, value in injection_kwargs.items() 
                                    if key not in ('waveform_generator', 'asds')}
            }
    injection_kwargs['snr_tol'] = snr_tol
    
    prior_samples = np.array_split(
        [prior.sample() for _ in tqdm(range(n_images), desc='Sampling prior')], 
        n_batches)
    prior.to_json(filepath.parent, filepath.stem)
    
    datasets = ['training',]
    lengths = [n_images,]
    resol = injection_kwargs['img_res']
    if valid_fraction != 0.0:
        datasets.append('validation')
        lengths = [int(n_images*(1.0-valid_fraction)), int(n_images*valid_fraction)]
    
    
    with h5py.File(filepath, 'w') as h_file:
            
        for s, leng in zip(datasets, lengths):

            h_file.create_dataset(f'{s}/images',  
                                    compression=None,
                                    shape=(leng, 3, *resol))
            h_file.create_dataset(f'{s}/labels', 
                                    compression=None,
                                    shape=(leng, len(parameters)))
            h_file.create_dataset(f'{s}/snrs',
                                    compression=None,
                                    shape=(leng,))
        set_metadata(h_file, metadata)
        h_file.attrs['parameters'] = parameters
    
        for i, batch in enumerate(prior_samples):
            data = Parallel(**joblib_kwargs)(
                    delayed(create_image)(prior_s, **injection_kwargs)
                    for prior_s in tqdm(batch, desc=f'Creating Dataset {seed}, batch {i}'))
            
            fill_hdf5(i, h_file, parameters, data, valid_fraction)


def create_image(prior_sample,
                 sampling_frequency=1024,
                 waveform_generator=None,
                 ifos: tuple[str,...] = None,
                 asds=None,
                 duration=None,
                 img_res: tuple[int, int] = (128, 128),
                 snr_tol: float = 0.5,
                 qtrans_kwargs: dict = None):
    """
    Generates image from individual sample of the parameters.

    Parameters
    ----------
    prior_sample :
        Sample of the parameters.

    sampling_frequency :
        Sampling frequency for the intermediate timeseries.

    waveform_generator :
        Waveform generator to model the signal.

    ifos :
        Collection of interferometer names.

    asds :
        Collection of ASDs for noise generation.

    duration :
        Duration for the intermediate timeseries.

    img_res :
        Image resolution (height, width) in pixels.

    snr_tol :
        Tolerance for the SNR adjusting algorithm. Accepts SNR \\in (SNR-snr_tol, SNR+snr_tol).

    qtrans_kwargs :
        Keyword arguments passed to gwpy's TimeSeries.q_transform.

    Returns
    -------
    out :
        (image, parameters)

    """
    
    assert waveform_generator is not None, 'Cannot inject signals without a waveform generator'
    assert duration == waveform_generator.duration, "Specified duration does not match waveform_generator.duration"
    
    if qtrans_kwargs is None:
        qtrans_kwargs = {}
    ts_data, params = generate_timeseries(prior_sample,
                                          sampling_frequency=sampling_frequency,
                                          waveform_generator=waveform_generator,
                                          snr_tol=snr_tol,
                                          ifos=ifos,
                                          asds=asds)
    
    
    qt_data = [ifo_q_transform(tseries, img_res, duration, **qtrans_kwargs) for tseries in ts_data]
    return qt_data, params


def generate_timeseries(prior_sample,
                        ifos,
                        asds,
                        sampling_frequency=1024,
                        waveform_generator=None,
                        snr_tol: float = 0.5
                        ):
    """
    Generate timeseries from parameters.

    Parameters
    ----------
    prior_sample :
        Sample of the parameters.
    ifos :
        Collection of interferometer names.
    asds :
        Collection of ASDs for noise generation.
    sampling_frequency :
        Sampling frequency for the intermediate TimeSeries.
    waveform_generator :
        Waveform generator to model the signal.
    snr_tol :
        Tolerance for the SNR adjusting algorithm. Accepts SNR \\in (SNR-snr_tol, SNR+snr_tol).

    Returns
    -------
    out :
        Timeseries with injected signal.

    """
    
    duration = waveform_generator.duration
    time = handle_time(prior_sample)

    # Set up the two interferometers.
    ifos = setup_ifos(duration=duration, sampling_frequency=sampling_frequency, ifos=ifos, asds=asds,
                      time=time)

    # Save the background frequency domain strain data of the two interferometers if you always want the same noise.
    ifos_bck = [ifo.frequency_domain_strain for ifo in ifos]
    
    targ_snr = prior_sample['_snr']

    if targ_snr == -1:
        # Inject the signal into the noise-free background and return the resulting strain data.
        for ifo in ifos:
            ifo.set_strain_data_from_zero_noise(sampling_frequency=sampling_frequency,
                                                duration=duration,
                                                start_time=time - duration / 2)
        # prior_sample['geocent_time'] = 0.
        ifos.inject_signal(prior_sample, waveform_generator=waveform_generator)
        return [np.fft.irfft(ifo.whitened_frequency_domain_strain) for ifo in ifos], prior_sample
    else:
        if targ_snr == 0:  # Return as is
            ifos.inject_signal(prior_sample, waveform_generator=waveform_generator)
            return [np.fft.irfft(ifo.whitened_frequency_domain_strain) for ifo in ifos], prior_sample
        # Iteratively inject the signal and adjust the luminosity distance until the
        # injected signal has a SNR close to the target SNR.
        while True:
            # Reverted to old concept
            
            # Inject the signal into the background strain data and update the geocentric time of the signal.
            ifos_ = setup_ifos(duration=duration, sampling_frequency=sampling_frequency, ifos=ifos, asds=asds)
            for i, ifo in enumerate(ifos_):
                ifo.set_strain_data_from_frequency_domain_strain(ifos_bck[i], sampling_frequency=sampling_frequency,
                                                                 duration=duration, start_time=duration / 2)
            prior_sample['geocent_time'] = 0.
            ifos_.inject_signal(prior_sample, waveform_generator=waveform_generator)

            # Check if the SNR of the injected signal is close to the target SNR.
            new_snr = getsnr(ifos_)
            if abs(new_snr - targ_snr) < snr_tol:
                # Return the injected strain data for each interferometer.
                prior_sample['_snr'] = new_snr
                return [np.fft.irfft(ifo.whitened_frequency_domain_strain) for ifo in ifos_], prior_sample

            # Adjust the luminosity distance of the signal by the ratio of the current SNR to the target SNR.
            snr_ratios = np.array(getsnr(ifos_) / targ_snr)
            prior_sample['luminosity_distance'] = prior_sample['luminosity_distance'] * np.prod(snr_ratios)


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


def fill_hdf5(batch_num: int, h_file: h5py.File, params_list, data, valid_fraction: float = 0.0):
    """
    Fills the given HDF5 with a batch of data. Performs parameter conversion if necessary.

    Parameters
    ----------
    batch_num :
        Batch number. Used to calculate dataset indexes to fill.
    h_file :
        Dataset file.
    params_list :
        Desired parameters. Will perform conversion if necessary.
    data :
        List of (image, parameters) return by parallel calls to 'create_image'.
    valid_fraction :
        Fraction of dataset to reserve for validation.

    Returns
    -------
    out :
        None
    """
    data_length = len(data)

    images, labels = zip(*data)
    images, labels = np.array(images), np.array(labels)

    # print(labels[:2]) # Need to be converted from dict to array with specific estimation_parameters

    if '_snr' in dict(labels[0]).keys():
        snr = np.array([data['_snr'] for data in labels])
    else:
        snr = None

    convert_func = partial(data_convert, params_list=params_list, param_pool=cbc_jargon['param_pool'])

    label_arr = np.zeros((len(labels), len(params_list)))
    i = 0

    with Pool() as pool:  # TODO: configure
        with tqdm(desc='Label Conversion', total=len(labels)) as p_bar:
            for label in pool.imap(convert_func, labels):
                p_bar.update(1)
                label_arr[i][:] = label
                i += 1
    
    
    labels = label_arr

    datasets = ['training']
    if valid_fraction > 0.0: 
        datasets.append('validation')
        valid_length = int(data_length*valid_fraction)
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
    else:
        snrs = (None, None)
    
    for s, image_dat, label_dat, snr_dat in zip(datasets, images, labels, snrs):
        
        low = batch_num*len(image_dat)
        high = low + len(image_dat)
        
        h_file[f'{s}']['images'][low:high,:,:,:] =  image_dat
        h_file[f'{s}']['labels'][low:high,:] = label_dat
        h_file[f'{s}']['snrs'][low:high] = snr_dat


def data_convert(params_dict, param_pool: dict[str, Callable], params_list: list[str]) -> np.ndarray:
    """
    Converts the data to specific parameters using a given set of relations and arguments.

    Parameters
    ----------
    params_dict :
        Dictionary of available parameters and their value.
    param_pool :
        Map of parameter names to functions that compute them. Used also to calculate missing arguments.
    params_list :
        List of parameters to conserve or obtain.

    Returns
    -------
    out :
        Array of converted parameters in the order of 'params_list'.
    """
    return np.array([params_dict[param] if param in params_dict else calc_parameter(param, param_pool, params_dict)
                     for param in params_list])

def set_metadata(file: h5py.File, metadata: dict, master_key=""):
    """
    Codifies a metadata dictionary as attrs of a HDF5 file.

    Parameters
    ----------
    file :
        Dataset in which to inscribe the metadata.
    metadata :
        Dataset metadata as a HDF5 compatible dictionary (see https://docs.h5py.org/en/stable/high/attr.html).
    master_key :
        Crude way of storing dicts within dicts. Should never be passed by the user.

    Returns
    -------

    """
    for key, val in metadata.items():
            if master_key:
                key = f'{master_key}/{key}'

            if isinstance(val, dict):
                set_metadata(file, val, master_key=key) 
            else:
                file.attrs[key] = val


def handle_time(prior_sample: dict[str, float]) -> float:
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
    if 'geocent_time' in prior_sample.keys():
        time = prior_sample['geocent_time']
    elif 'normalized_time' in prior_sample.keys():
        time = prior_sample['normalized_time'] * 24 * 3600
        prior_sample['geocent_time'] = time
    else:
        time = 0.0
    return time


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
