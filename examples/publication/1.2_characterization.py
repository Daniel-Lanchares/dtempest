from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from dtempest.core.train_utils import H5Dataset
from dtempest.gw.utils import get_parameter_alias, get_parameter_units


def load_and_plot(axs, dataset: H5Dataset, max_events: int = None, label:str = None, **hist_kwargs):
    if max_events is None:
        max_events = len(dataset)
    else:
        max_events = int(max_events)
    data = np.zeros((12, max_events))

    for i, labels in enumerate(tqdm(dataset.iter_column('labels'), desc='Characterization', total=max_events)):
        if max_events is not None and i == max_events:
                break
        data[:, i] = labels

    for ax, dat, param in zip(axs.flatten(), data, params):
        ax.hist(dat, label=label, **hist_kwargs)
        ax.set_title(rf'{get_parameter_alias(param)} [{get_parameter_units(param)}]')
        if param == 'luminosity_distance':
            ax.legend()

if __name__ == '__main__':
    files_dir = Path(__file__).parent / 'files' # Main directory
    dataset_dir = files_dir / 'Datasets'
    images_dir = dataset_dir / 'images'
    images_dir.mkdir(exist_ok=True)

    plt.style.use('publication.mplstyle')

    imgs_per_seed = 1 * 1000
    valid_fraction = 0.1

    dataset_name = f'Data12_{int(imgs_per_seed/1000)}k_{int(valid_fraction*100)}%.h5'

    dataset = H5Dataset(dataset_dir/dataset_name, 'training')

    meta = dataset.get_metadata()
    params = meta['parameters']

    figsize = (15, 7)
    shape = (3, len(params)//3)

    fig, axs = plt.subplots(*shape, figsize=figsize)

    load_and_plot(axs, dataset, max_events=None,
                  density=True, bins=20, # 100
                  histtype='step', label='training')

    valiset = H5Dataset(dataset_dir/dataset_name, 'validation')
    load_and_plot(axs, valiset, max_events=None,
                  density=True, bins=10,  # 50
                  histtype='step', label='validation')

    fig.tight_layout(h_pad=2)
    plt.savefig(images_dir / 'characterization_Data12.pdf', bbox_inches='tight')
