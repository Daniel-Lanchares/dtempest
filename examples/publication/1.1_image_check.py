"""
Check that images are generating correctly.

Should be done with a small dataset before sending an HTC job
"""
import h5py
from pathlib import Path
import matplotlib.pyplot as plt

from dtempest.core.train_utils import H5Dataset
from dtempest.gw.utils import plot_image

imgs_per_seed = 1000 * 1000
valid_fraction = 0.1 # So 10% validation
dataset_name = f'Data12_{int(imgs_per_seed/1000)}k_{int(valid_fraction*100)}%.h5'

files_dir = Path(__file__).parent / 'files' # Main directory
dataset_dir = files_dir / 'Datasets'
images_dir = dataset_dir / 'images'
images_dir.mkdir(exist_ok=True)

with h5py.File(dataset_dir / dataset_name, 'r') as h_file:
    
    dataset = H5Dataset(dataset_dir / dataset_name, 'training')
    valiset = H5Dataset(dataset_dir / dataset_name, 'validation')
    
    for i in range(10):
    
        image, labels = dataset[i]
        
        snr = dataset.get_snr(i)

        plot_image(image, title_maker=lambda _: f'SNR: {snr:.2f}, RGB = (L1, H1, V1)')
        plt.grid(False)
        plt.savefig(images_dir / f'check_{i}.pdf')