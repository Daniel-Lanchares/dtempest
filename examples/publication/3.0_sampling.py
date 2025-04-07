import torch
import numpy as np
from pathlib import Path

from dtempest.gw import CBCEstimator
from dtempest.gw.catalog import Merger
from dtempest.gw.utils import prepare_array


def sample_and_save(event: str,
                    nsamples: int,
                    path: Path | str,
                    cat: str = 'gwtc-3',
                    add_noise: bool = False):
    path.mkdir(parents=True, exist_ok=True)
    resol = (128, 128)
    img_window = (-0.1, 0.1)
    frange = (20, 300)
    duration = 4

    merger = Merger(event, 
                    source=cat, 
                    img_res=resol, 
                    image_window=img_window,
                    frange=frange,
                    duration=duration)
    
    image = merger.make_array()
    if add_noise:
        from gwpy.timeseries import TimeSeries

        noise_dir = files_dir / 'Noise'  # Noise directory
        t = 1268431194.1
        ifo = 'V1'
        qtrans_kwargs = {
            'frange': (20, 300),
            'qrange': (4, 8),
            'outseg': (-0.1, 0.1) # Important to tweak with quick_show and small datasets to ensure signal is in frame
        }

        noise = TimeSeries.read(noise_dir / f'noise_{t}_{ifo}').crop(t+1, t+5)
        noise.x0 = -noise.duration/2
        image[-1] = prepare_array(noise.q_transform(whiten=False, **qtrans_kwargs), resol)

    with torch.no_grad():
        sdict = flow.sample_dict(nsamples, context=image)
    np.savez(path / f'{event}_{flow.name}_{nsamples}', **sdict)


if __name__ == '__main__':
    '''
    Sampling from trained model and comparison to GWTC data
    '''

    name = 'GP12' # Model name
    files_dir = Path(__file__).parent / 'files'  # Main directory
    dataset_dir = files_dir / 'Datasets'
    train_dir = files_dir / 'Model'
    outdir = train_dir / 'toy_model'
    catalog_3 = files_dir / 'Samples'

    sample_dir = outdir / 'samples'

    flow = CBCEstimator.load_from_file(outdir / f'{name}.pt', device ='cpu')
    flow.eval()

    print(f'{flow.name} metadata')
    flow.pprint_metadata()

    event = 'GW191230_180458'
    nsamples = 10000

    sample_and_save(event, nsamples=nsamples, path=sample_dir)
