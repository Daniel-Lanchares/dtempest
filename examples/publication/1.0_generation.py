"""
Basic generation script
"""
from pathlib import Path

import bilby
from gwpy.timeseries import TimeSeries
from bilby.core.utils.log import logger as bilby_logger

from dtempest.gw.generation import generate

if __name__ == '__main__':

    bilby_logger.setLevel('ERROR')

    # Arguments for parallelization utilities.
    # See joblib's Parallel for more info.
    parallel_kwargs = {
            'n_jobs': -1,
            'backend': 'multiprocessing'
        }

    duration = 4.0  # seconds  # change from around 2 to 8 depending on masses
    sampling_frequency = 1024  # Hz  DON'T KNOW WHY (yet), BUT DO NOT CHANGE!
    min_freq = 20.0  # Hz

    # List of interferometers (R, G, B)
    ifolist = ('L1', 'H1', 'V1')
    

    seed = 0
    n_batch = 20
    images = 1000 * 1000 #1000k
    valid_fraction = 0.1 # 10% validation
    dataset_name = f'Data12_{int(images/1000)}k_{int(valid_fraction*100)}%.h5'
    

    files_dir = Path(__file__).parent / 'files' # Main directory
    dataset_dir = files_dir / 'Datasets' # Data directory
    noise_dir = files_dir / 'Noise'      # Noise directory

    # Each chunk of the dataset is generated with noise (ASD) from one of these times
    times = [ 	1268431194.1, ]

    # Noise dict construction
    asd_dict = {
        t: [TimeSeries.read(noise_dir / f'noise_{t}_{ifo}').asd(fftlength=4).interpolate(df=1).value for ifo in ifolist]
        for t in times}

    waveform_arguments = dict(
        waveform_approximant="IMRPhenomXPHM",
        minimum_frequency=min_freq,

    )

    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,

    )

    bilby.core.utils.random.seed(seed)
    t = bilby.core.utils.random.rng.choice(times)
    print(f'Chosen time: {t}')
    asds = asd_dict[t]
    
    injection_kwargs = {
        # GW-related
        'ifos': ifolist,
        'asds': asds,
        
        # Timeseries-related
        'sampling_frequency': sampling_frequency,
        'waveform_generator': waveform_generator,

        # Image-related
        'img_res': (128, 128),  # (height, width) in pixels
        'duration': duration,

        'qtrans_kwargs': {
            'frange': (min_freq, 300),
            'qrange': (4, 8),
            'outseg': (-0.1, 0.1)
        }
    }

    prior = bilby.gw.prior.BBHPriorDict()
    prior['_snr'] = bilby.gw.prior.PowerLaw(alpha=-2, minimum=7, maximum=30, name='snr', latex_label=r'$\textrm{SNR}$')

    if 'geocent_time' not in prior.keys():
        prior['geocent_time'] = 0.0


    parameter_list = [
        'chirp_mass',
        'mass_ratio',
        'a_1',
        'a_2',
        'tilt_1',
        'tilt_2',
        'phi_jl',
        'phi_12',
        'luminosity_distance',
        'theta_jn',
        # 'ra',
        # 'dec',
        'phase',
        'psi',
        # 'geocent_time'
    ]

    generate(filepath=dataset_dir / f'{dataset_name}',
             n_images=images,
             n_batches=n_batch,
             valid_fraction=valid_fraction,
             parameters=parameter_list,

             snr_tol=0.5,
             prior=prior,
             seed=seed,
             joblib_kwargs=parallel_kwargs,
             **injection_kwargs)
