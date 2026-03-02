import numpy as np
from pathlib import Path
from functools import partial
from scipy.spatial.distance import jensenshannon

import gwosc.api.v2 as apiv2

from dtempest.core.conversion_utils import get_param_alias
from dtempest.gw.config import cbc_jargon

all_events = {"GWTC-3-confident": ['GW200311_115853','GW200224_222234','GW200219_094415','GW200216_220804',
                                   'GW200209_085452','GW200208_222617','GW200208_130117','GW200129_065458',
                                   'GW191230_180458','GW191215_223052','GW191127_050227','GW191113_071753',],
            "GWTC-2.1-confident": ['GW190929_012149','GW190926_050336','GW190916_200658','GW190915_235702',
                                   'GW190828_065509','GW190828_063405','GW190805_211137','GW190803_022701',
                                   'GW190727_060333','GW190706_222641','GW190701_203306','GW190602_175927',
                                   'GW190519_153544','GW190517_055101','GW190513_205428','GW190512_180714',
                                   'GW190503_185404','GW190413_134308','GW190413_052954','GW190412_053044',
                                   'GW190408_181802',]}

def obtain_references(event: dict|str, reference_samples_path: Path):
    from pesummary.gw.conversions import convert
    from dtempest.gw.sampling import CBCSampleDict
    name = event["name"] if isinstance(event, dict) else event
    return convert(CBCSampleDict.from_file(reference_samples_path / f'{name}_samples.h5')['C01:APPROXIMANT'])

def analysis(cat):
    events = sorted(all_events[cat])
    divergences = np.zeros((len(events), len(params)))
    short_cat = cat[:-10]
    np.save(estimation_dir/f'{short_cat.upper()}_events', np.array(events))
    np.save(estimation_dir/f'{short_cat.upper()}_labels', np.array([get_param_alias(param, cbc_jargon) for param in params]))

    for idx, event in enumerate(events):
        npfile = np.load(sample_dir / f'{event}_{nsamples}.npz')
        flow_samples = np.array([npfile[key] for key in npfile.files if key in params])

        gwtc_getter = partial(obtain_references, reference_samples_path=reference_dir / cat)
        gwtc = gwtc_getter(event)

        if "geocent_time" in params:
            event_dict = list(apiv2.fetch_event_versions(event))[-1] # VERY IMPORTANT: GWTC-2.1 IS LAST VERSION
            gwtc["geocent_time"] -= event_dict["gps"]

        cat_samples = np.array([list(gwtc[param]) for param in params])

        js_dist = np.zeros(len(params))
        for i in range(flow_samples.shape[0]):
            a = flow_samples[i, :]
            b = cat_samples[i, :]

            nbins = 20
            mini = min(a.min(), b.min()) * 0.9999
            maxi = max(a.max(), b.max()) * 1.0001
            bins = np.linspace(mini, maxi, nbins)

            h = np.histogram(a, bins=bins, density=True)[0]
            hcat = np.histogram(cat_samples[i, :], bins=bins, density=True)[0]
            js_dist[i] = jensenshannon(h, hcat)
        divergences[idx, :] = js_dist**2
    np.save(estimation_dir / f'{short_cat.upper()}_divergences', divergences)


if __name__ == '__main__':
    """
    JSD analysis script comparing to (given) reference samples.
    """
    files_dir = Path('') # Main directory

    name = 'GP15_example'
    params = ['chirp_mass', 'mass_ratio',
              'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_jl', 'phi_12',
              'luminosity_distance', 'theta_jn', 'ra', 'dec',
              'phase', 'psi', 'geocent_time']


    estimation_dir = files_dir / 'Estimation Data' / f'{name}'
    sample_dir = estimation_dir / 'refined samples'
    reference_dir = files_dir / "Public_Samples"

    nsamples = 10000
    analysis("GWTC-2.1-confident")
    analysis("GWTC-3-confident")