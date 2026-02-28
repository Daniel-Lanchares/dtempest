import h5py
import datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed


import bilby
from bilby.core.prior import PriorDict

from .utils import set_metadata, fill_hdf5, format_timedelta, ifo_q_transform, setup_ifos, getsnr

def make_dataset(filepath: Path = Path('Dataset.h5'),
                n_images: int = 1,
                n_batches: int = 1,
                valid_fraction: float = 0.0,
                parameters: list = None,
                prior: PriorDict = None,
                snr_range: tuple[float, float] = (8, 40),
                seed: int = 0,
                image_dtype=np.float32,
                image_compression="gzip",
                joblib_kwargs: dict = None,
                asd_dict=None,
                **injection_kwargs):
    assert n_batches <= n_images, f"You need more images ({n_images}) than batches ({n_batches})"
    len_batch, rest = divmod(n_images, n_batches)

    bilby.core.utils.random.seed(seed)

    if prior is None:
        prior = bilby.gw.prior.BBHPriorDict()
    if joblib_kwargs is None:
        joblib_kwargs = {}

    # For better reconstruction of custom priors
    prior.to_json(filepath.parent, filepath.stem)

    # Preparation for h5 file creation
    datasets = ['training', ]
    lengths = [n_images, ]
    resol = injection_kwargs['img_res']
    if valid_fraction != 0.0:
        datasets.append('validation')
        lengths = [int(n_images * (1.0 - valid_fraction)), int(n_images * valid_fraction)]

    metadata = {
        # 'prior': prior._get_json_dict(),  # Later to be recovered with "def _get_from_json_dict(cls, prior_dict)"
        'seed': seed,
        'n_images': n_images,
        "n_batches": n_batches,
        "image_compression": image_compression,
        "image_dtype": str(image_dtype),
        'snr_range': snr_range,
        'injection_kwargs': {key: value for key, value in injection_kwargs.items()
                             if key not in ('waveform_generator', 'asds')}
    }

    # h5 file creation
    with h5py.File(filepath, 'w') as h_file:

        for s, leng in zip(datasets, lengths):
            h_file.create_dataset(f'{s}/images',
                                  compression=image_compression,
                                  shape=(leng, 3, *resol),  # TODO DO NOT HARDCODE CHANNELS
                                  dtype=image_dtype,
                                  )
            h_file.create_dataset(f'{s}/labels',
                                  compression=None,
                                  shape=(leng, len(parameters)))
            h_file.create_dataset(f'{s}/snrs',
                                  compression=None,
                                  shape=(leng,))
        set_metadata(h_file, metadata)
        h_file.attrs['parameters'] = parameters

    ti = datetime.datetime.now()
    print(f'{ti:%Y-%m-%d %H:%M:%S}: H5 datasets allocated.\n')

    for batch in range(1, n_batches + 1):
        if batch == n_batches:
            n_calls = len_batch + rest
        else:
            n_calls = len_batch

        t = bilby.core.utils.random.rng.choice(list(asd_dict.keys()))
        print(f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S}: Batch {batch}/{n_batches}. Chosen noise time: {t}')
        injection_kwargs["asds"] = asd_dict[t]

        data = Parallel(**joblib_kwargs)(
            delayed(create_image)(prior,
                                  snr_range,
                                  (seed, batch, call_idx),
                                  img_dtype=image_dtype,
                                  **injection_kwargs)
            for call_idx in tqdm(range(n_calls), desc=f'Creating Raw Dataset {seed}, batch {batch}'))

        print(f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S}: Batch {batch}. Filling h5 file.')
        with h5py.File(filepath, 'r+') as h_file:
            fill_hdf5(batch - 1, h_file, parameters, data, valid_fraction)
        tf = datetime.datetime.now()
        print(f'{tf:%Y-%m-%d %H:%M:%S}: Batch {batch}. Filled h5 file.')
        print(f'Estimated remaining time: {format_timedelta((tf - ti) * (n_batches / batch - 1))}\n')


def create_image(prior_dict,
                 snr_range,
                 seed,
                 sampling_frequency=1024,
                 waveform_generator=None,
                 ifolist=None,
                 asds=None,
                 duration=None,
                 img_res: tuple[int, int] = (128, 128),
                 img_dtype=np.float32,
                 qtrans_kwargs: dict = None):
    bilby.core.utils.random.seed(seed)

    assert waveform_generator is not None, 'Cannot inject signals without a waveform generator'
    assert duration == waveform_generator.duration, "Specified duration does not match waveform_generator.duration"

    if qtrans_kwargs is None:
        qtrans_kwargs = {}
    ts_data, params = generate_timeseries(prior_dict,
                                          snr_range,
                                          sampling_frequency=sampling_frequency,
                                          waveform_generator=waveform_generator,
                                          ifolist=ifolist,
                                          asds=asds)

    qt_data = [ifo_q_transform(tseries, img_res, duration, sampling_frequency, img_dtype, **qtrans_kwargs) for tseries
               in ts_data]
    return qt_data, params


def generate_timeseries(prior_dict,
                        snr_range,
                        ifolist,
                        asds,
                        sampling_frequency=1024,
                        waveform_generator=None,
                        max_iter=100
                        ):
    duration = waveform_generator.duration

    # Set up the two interferometers.
    ifos = setup_ifos(duration=duration, sampling_frequency=sampling_frequency, ifos=ifolist, asds=asds)

    counter = 0

    while True:
        samples = prior_dict.sample()

        if "geocent_time" not in samples.keys():
            samples['geocent_time'] = 0.0

        ifos.set_strain_data_from_power_spectral_densities(  # New noise each time
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=- duration / 2,
        )

        # Generate and inject the signal on the new noise
        ifos.inject_signal(samples, waveform_generator=waveform_generator)

        # Check if the SNR of the injected signal is close to the target SNR.
        snr = getsnr(ifos)
        if snr_range[0] < snr < snr_range[1] or counter == max_iter:

            if counter == max_iter:
                print(f"WARNING: Reached max iteration ({max_iter}). Settling for SNR {snr:.2f}")
            # Return the injected strain data for each interferometer.
            samples['_snr'] = snr
            return [np.fft.irfft(ifo.whitened_frequency_domain_strain) for ifo in ifos], samples

        counter += 1