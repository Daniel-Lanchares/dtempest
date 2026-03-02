import numpy as np
from pathlib import Path
import bilby
from gwpy.timeseries import TimeSeries

import logging
logging.getLogger('bilby').disabled = True

from dtempest.gw.generation import make_dataset

if __name__ == '__main__':
    """
    Basic generation script.
    """

    # Arguments for parallelization utilities.
    # See joblib's Parallel for more info.
    parallel_kwargs = {
            'n_jobs': -1,
            'backend': 'multiprocessing'
        }

    duration = 6.0  # seconds  # change from around 2 to 8 depending on masses
    sampling_frequency = 4096  # Hz 
    min_freq = 20.0  # Hz

    # List of interferometers (R, G, B)
    ifolist = ('L1', 'H1', 'V1')
    

    seed = 1 #0
    valid_fraction = 0.1 # So 10% validation
    snr_range = (8, 60) # Network wide SNR

    test = True
    if test:
        images = 1e3
        n_batch = 1
        dtype = np.float32
    else:
        images = int(3e6)
        n_batch = int(images/2e4) # To have ~ 20k images per batch
        dtype = np.float16 # 32 for test, 16 for real
        
    img_text = f'{int(images/1e6)}M' if images >= 1e6 else f'{int(images/1e3)}k'
    dataset_name = f'Data15_{img_text}_{int(valid_fraction*100)}%.h5'
    

    files_dir = Path('') # Main directory
    project_dir = Path("") # Storage-heavy directory
    rawset_dir = project_dir / 'Datasets' # Data directory
    noise_dir = project_dir / 'Noise'         # Noise directory

    # Each chunk of the dataset is generated with noise (ASD) from one of these times
    times = [
        1240194342,
        1249784742,
        1267928742,
        1268431194.1
        ] 

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
    
    injection_kwargs = {
        # GW-related
        'ifolist': ifolist,
        
        # Timeseries-related
        'sampling_frequency': sampling_frequency,
        'waveform_generator': waveform_generator,

        # Image-related
        'img_res': (96, 170), #(128, 128) # (height, width) in pixels
        'duration': duration,

        'qtrans_kwargs': {
            'frange': (min_freq, 300),
            'qrange': (6, 6),
            'outseg': (-0.15, 0.15) # Important to tweak with quick_show and small datasets to ensure signal is in frame
        }
    }

    max_dL = 5e3 # 5000 Mpc for normal dataset, 10 Gpc for long distance.
    prior = bilby.gw.prior.BBHPriorDict()
    prior["chirp_mass"] = bilby.gw.prior.UniformInComponentsChirpMass(name='chirp_mass', minimum=15, maximum=120)
    prior["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(name='luminosity_distance', minimum=1e2, maximum=max_dL)
    prior['geocent_time'] = bilby.gw.prior.Uniform(name="geocent_time", minimum=-0.1, maximum=0.1)

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
        'ra',
        'dec',
        'phase',
        'psi',
        'geocent_time' 
    ]


    make_dataset(filepath=rawset_dir / f'{dataset_name}',
                        n_images=int(images),
                        n_batches=n_batch,
                        valid_fraction=valid_fraction,
                        parameters=parameter_list,
                    
                        snr_range=snr_range,
                        prior=prior,
                        seed=seed,
                        image_dtype=dtype,
                        image_compression="gzip",
                        joblib_kwargs=parallel_kwargs,
                        asd_dict=asd_dict,
                        **injection_kwargs)
