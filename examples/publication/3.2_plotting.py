import numpy as np
from glob import glob
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt

from pesummary.gw.conversions import convert

from dtempest.core.plot_utils import set_corner_limits, redraw_legend
from dtempest.gw.catalog import Merger
from dtempest.gw.utils import plot_image
from dtempest.gw.utils import prepare_array
from dtempest.gw.sample_utils import CBCSampleDict, CBCComparisonSampleDict




def comparison_corner_plot(samples, event, name, path: Path | str, cat='gwtc-3', params=None):
    path.mkdir(parents=True, exist_ok=True)

    from_internet = False
    if from_internet:
        # Needs the specific address for each event, but it is enough for a quick test
        gwtc = convert(CBCSampleDict.from_file("https://dcc.ligo.org/public/0157/P1800370/005/GW150914_GWTC-1.hdf5"))
    else:
        if cat == 'gwtc-3':
            ev_path = glob(str(catalog_3 / f'*{event}*_cosmo.h5'))[0]
            gwtc = convert(CBCSampleDict.from_file(ev_path)['C01:IMRPhenomXPHM'])#['C01:Mixed'])
        else:
            gwtc = None
    

    resol = (128, 128)  # (32, 48)  # (48, 72)
    img_window = (-0.1, 0.1) # (-0.065, 0.075)
    frange = (20, 300)
    duration = 4

    merger = Merger(event, 
                    source=cat, 
                    img_res=resol, 
                    image_window=img_window,
                    frange=frange,
                    duration=duration)
    image = merger.make_array()
    if add_virgo_noise:
        from gwpy.timeseries import TimeSeries

        # rng = np.random.default_rng(42) # 0.05 is very good, so is 0.025
        # noise = np.clip(rng.normal(loc=0.5, scale=0.025, size=image[-1, :].shape), 0, 1)
        # image[-1, :] += noise
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
        image[-1] = prepare_array(noise.q_transform(whiten=False, **qtrans_kwargs), (128, 128))




    multi = CBCComparisonSampleDict({cat.upper(): gwtc,
                                     f"Estimator GP12": samples,})
                                     # f"Estimator {flow2.name}": sdict2})  # More comparisons are possible

    # fig = plt.figure(figsize=(12, 10))
    select_params = params if params is not None else list(samples.keys())
    del gwtc, samples

    kwargs = {
        'medians': 'all',
        'hist_bin_factor': 1,
        'bins': 20,
        'title_quantiles': [0.16, 0.5, 0.84],
        'smooth': 1.4,
        'label_kwargs': {'fontsize': 25},  # 25 for GWTC-1, 25 for GWTC-2/3?
        # 'labelpad': 0.2,
        'title_kwargs': {'fontsize': 20},  # 20 for GWTC-1, 20 for GWTC-2/3?

        'kde': stats.gaussian_kde
        # 'kde': bounded_1d_kde,
        # 'kde_kwargs': multi.default_bounds(),
    }

    fig = multi.plot(type='corner', parameters=select_params, **kwargs)
    del multi

    parameter_limits = {

        2: (-0.01, 1),
        3: (-0.01, 1),
        4: (-0.05, 3.3),
        5: (-0.05, 3.3),
        6: (-0.01, 6.3),
        7: (-0.01, 6.3),

        9: (-0.05, 3.3),
        10: (-0.01, 6.3),
        11: (-0.05, 3.3)
    }

    if event == 'GW191222_033537':
        parameter_limits.update({0: (30, 66), 8: (-200, 6000)})
    elif event == 'GW191230_180458':
        parameter_limits.update({0: (20, 100), 8: (-200, 9000)})

    set_corner_limits(fig, parameter_limits)



    plt.tight_layout(h_pad=-4, w_pad=-0.8) # w_pad -0.8  # h_pad -4.5      -1 for 1 line title, -3 for 2 lines
    # fig = sdict.plot(type='corner', parameters=select_params, truths=sdict.select_truths(select_params),
    #                  smooth=smooth, smooth1d=smooth, medians=True, fig=fig)
    fig = plot_image(image, fig=fig,
                     title_maker=lambda data: f'{event} Q-Transform image\n(RGB = (L1, H1, V1))',
                     title_kwargs={'fontsize': 40},  # 40 for GWTC-1, 40 for GWTC-2/3?
                     aspect=resol[1] / resol[0])
    fig.get_axes()[-1].set_position(pos=[0.62, 0.55, 0.38, 0.38])

    redraw_legend(fig,
                  fontsize=35,  # 25 for GWTC-1, 30 for GWTC-2/3 approx
                  loc='upper center',
                  bbox_to_anchor=(0.4, 0.98),
                  handlelength=2,
                  linewidth=5)

    # To remove gridlines
    for ax in fig.get_axes():
        ax.grid(False)

    # Remove unnecessary ticks
    axes = fig.get_axes()[:-1]
    nparams = int(np.sqrt(len(axes)))
    axes = np.array(axes).reshape(nparams, nparams)
    for i in range(nparams):
        for j in range(nparams):
            if i != nparams-1:
                axes[i, j].xaxis.set_ticks_position('none')
            if j != 0:
                axes[i, j].yaxis.set_ticks_position('none')

    fig.savefig(path / f'{event}_{name}_corner.pdf', bbox_inches='tight')


if __name__ == '__main__':
    '''
    Plotting comparison to GWTC data.
    '''
    plt.style.use('publication.mplstyle')

    name = 'GP12'  # Model name
    files_dir = Path(__file__).parent / 'files'  # Main directory
    dataset_dir = files_dir / 'Datasets'
    train_dir = files_dir / 'Model'
    outdir = train_dir / 'toy_model'
    sample_dir = outdir / 'samples'
    plot_dir = outdir / 'plots'
    catalog_3 = files_dir / 'Samples'


    event = 'GW191230_180458'
    nsamples = 10000

    rejecting = True
    add_virgo_noise = False


    npfile = np.load(sample_dir / f'{event}_{name}_{nsamples}.npz')
    if rejecting:
        flow_samples = np.array([npfile[key] for key in npfile.files])
        print(flow_samples.shape)
        flow_samples = np.delete(flow_samples,
                                 np.any(flow_samples < 0, axis=0),
                                 axis=1)
        print(flow_samples.shape)

        max_array = np.array([100, 1, 1, 1, np.pi, np.pi, 2 * np.pi, 2 * np.pi, 10000, np.pi, 2 * np.pi, np.pi])
        flow_samples = np.delete(flow_samples,
                                 np.any(np.array([list(sample > max_array) for sample in flow_samples.T]).T, axis=0),
                                 axis=1)
        print(flow_samples.shape)
        filedict = {key: flow_samples[i, :] for i, key in enumerate(npfile.files)}
    else:
        filedict = {key: npfile[key] for key in npfile.files}
    sdict = CBCSampleDict(parameters=list(filedict.keys()))
    for key in sdict.keys():
        sdict[key] = filedict[key]

    
    comparison_corner_plot(sdict, event, name, plot_dir)

