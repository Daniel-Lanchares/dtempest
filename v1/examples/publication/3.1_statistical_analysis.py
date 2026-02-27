from pathlib import Path

import numpy as np

from dtempest.gw import CBCEstimator
from dtempest.gw.sample_utils import CBCSampleDict
from dtempest.gw.catalog import Merger

from pesummary.gw.conversions import convert
from glob import glob

from scipy.spatial.distance import jensenshannon


if __name__ == '__main__':
    '''
    Sampling from trained model and comparison to GWTC data.
    '''
    name = 'GP12'  # Model name
    files_dir = Path(__file__).parent / 'files'  # Main directory
    dataset_dir = files_dir / 'Datasets'
    train_dir = files_dir / 'Model'
    outdir = train_dir / 'toy_model'
    sample_dir = outdir / 'samples'
    plot_dir = outdir / 'plots'
    catalog_3 = files_dir / 'Samples'

    flow = CBCEstimator.load_from_file(outdir / f'{name}.pt', device ='cpu') # Ensure that it leaves on the cpu to avoid cuda-availability problems
    flow.eval()

    event = 'GW191230_180458'
    # event = 'GW191222_033537'
    dropping = False
    nsamples = 10000


    npfile = np.load(sample_dir / f'{event}_{name}_{nsamples}.npz')

    flow_samples = np.array([npfile[key] for key in npfile.files])

    ev_path = glob(str(catalog_3 / f'*{event}*_cosmo.h5'))[0]
    gwtc = convert(CBCSampleDict.from_file(ev_path)['C01:IMRPhenomXPHM'])

    cat_samples = np.array([list(gwtc[param]) for param in flow.estimation_parameters])


    if dropping:
        flow_samples = np.delete(flow_samples, np.any(flow_samples < 0, axis=0), axis=1)

        max_array = np.array([100, 1, 1, 1, np.pi, np.pi, 2*np.pi, 2*np.pi, 10000, np.pi, 2*np.pi, np.pi])
        flow_samples = np.delete(flow_samples, np.any(np.array([list(sample > max_array) for sample in flow_samples.T]).T, axis=0), axis=1)


    js_dist = np.zeros(12)
    for i in range(flow_samples.shape[0]):
        a = flow_samples[i, :]
        b = cat_samples[i, :]

        nbins = 20
        mini = min(a.min(), b.min())*0.9999
        maxi = max(a.max(), b.max())*1.0001
        bins = np.linspace(mini, maxi, nbins)

        h = np.histogram(a, bins=bins, density=True)[0]
        hcat = np.histogram(cat_samples[i, :], bins=bins, density=True)[0]

        print(flow.estimation_parameters[i])
        print(f'    {jensenshannon(h, hcat, base=2):.3f}')

        # Computing in base 2 bounds the metric between 0 and 1
        js_dist[i] = jensenshannon(h, hcat, base=2)
    print(js_dist**2)
    print(list(np.round(js_dist**2, 3)))
