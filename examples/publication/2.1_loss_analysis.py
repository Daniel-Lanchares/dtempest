import torch
import numpy as np
from pathlib import Path
from itertools import product
import matplotlib.pyplot as plt


def process_loss(arr: np.ndarray, points: int = None) -> np.ndarray:
    if points is None:
        return arr.flatten()
    elif points == arr.shape[0]:
        return np.mean(arr, axis=1)
    else:
        from skimage.measure import block_reduce
        return block_reduce(arr.flatten(), block_size=len(arr.flatten())//points, func=np.mean, cval=np.mean(arr[-1]))
    # 

def process_epochs(arr: np.ndarray, points: int = None) -> np.ndarray:
    if points is None:
        return arr.flatten()
    elif points == arr.shape[0]:
        return arr[:, -1]
    else:
        return arr.flatten()[0::len(arr.flatten())//points]

def get_loss(path: Path, points: int = None, validation: bool = False,):
    if validation:
        epochs, loss = torch.load(path/'validation_data.pt', weights_only=False)
    else:
        epochs, loss = torch.load(path/'loss_data.pt', weights_only=False)
    return process_epochs(epochs, points), process_loss(loss, points)


def plot_example(ax, point_multip=1, log=True):
    points0 = 10*point_multip if point_multip is not None else None
    points1 = 15*point_multip if point_multip is not None else None

    epochs_0, loss_0 = get_loss(path_0, points0)
    v_epochs_0, v_loss_0 = get_loss(path_0, points0, validation=True)

    epochs_1, loss_1 = get_loss(path_1, points1)
    v_epochs_1, v_loss_1 = get_loss(path_1, points1, validation=True)

    epochs = np.concatenate((epochs_0, epochs_1 + epochs_0[-1]))
    v_epochs = np.concatenate((v_epochs_0, v_epochs_1 + v_epochs_0[-1]))
    loss = np.concatenate((loss_0, loss_1))
    v_loss = np.concatenate((v_loss_0, v_loss_1))

    if log:
        lowest_loss = min(np.min(loss), np.min(v_loss))

        ax.semilogy(epochs, loss-lowest_loss+1, label='trianing data')
        ax.semilogy(v_epochs, v_loss-lowest_loss+1, label='validation data')

        title_str = f'Pt every {1/point_multip:.2f} epoch, semilog' if points0 is not None else 'All data, semilog'
    else:
        ax.plot(epochs, loss, label='training data')
        ax.plot(v_epochs, v_loss, label='validation data')

        title_str = f'Pt every {1/point_multip:.2f} epoch, linear' if points0 is not None else 'All data, linear'
    ax.set_title(title_str)
    ax.axvline(10, color='k', linestyle='--', label='stage change')
    ax.legend()



name = 'GP12v3' # Model name
files_dir = Path('/lhome/ext/uv098/uv0982/scratch/MSc-project/')
# rawdat_dir = files_dir / 'Raw Datasets'
train_dir = files_dir / 'Model'
outdir = train_dir / 'training_test_5'

path_0 = outdir / f'loss_data_{name}_stage_000'
path_1 = outdir / f'loss_data_{name}_stage_001'


def options_test():
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))

    point_muls = [1, 2, None]
    plot_form = [True, False]

    for ax, (log, p_mul) in zip(axs.flatten(), product(plot_form, point_muls)):
        plot_example(ax, p_mul, log)

    plt.savefig(files_dir/'plots'/f'{name}_loss_plot_options.pdf')

fig = plt.figure(figsize=(12, 3))
ax = plt.gca()
plot_example(ax, 1, True)
ax.set_title("")
ax.set_xlabel("Epoch")
ax.set_ylabel("Normalized logarithmic loss")
plt.savefig(files_dir/'plots'/f'{name}_loss_plot.pdf', bbox_inches='tight')