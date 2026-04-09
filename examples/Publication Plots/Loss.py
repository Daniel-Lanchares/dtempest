import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from dtempest.core.common_utils import get_loss


def plot_loss(ax, paths: tuple[Path,...]):
    """
    Individual loss plot consisting of training and validation curves over multiple stages.
    """
    points = 15

    epochs, losses = [], []
    v_epochs, v_losses = [], []
    for path in paths:
        epochs_i, loss_i = get_loss(path, points)
        v_epochs_i, v_loss_i = get_loss(path, points, validation=True)
        if len(epochs) > 0:
            epochs_i += epochs[-1][-1]
        if len(v_epochs) > 0:
            v_epochs_i += v_epochs[-1][-1]
        epochs.append(epochs_i)
        losses.append(loss_i)
        v_epochs.append(v_epochs_i)
        v_losses.append(v_loss_i)

    epochs = np.concatenate(epochs)
    v_epochs = np.concatenate(v_epochs)
    losses = np.concatenate(losses)
    v_losses = np.concatenate(v_losses)


    ax.plot(epochs, losses, label='training data')
    ax.plot(v_epochs, v_losses, label='validation data')
    ax.axvline(15, color='k', linestyle='--', label='stage change')
    ax.axvline(30, color='k', linestyle='--')
    ax.legend()


def loss_plot_with_break():
    """
    Plots a graph with a break on the y-axis.
    """
    # Adapted from https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, height_ratios=[0.25, 0.75], figsize=(6, 4))
    fig.subplots_adjust(hspace=0.05)  # adjust space between Axes

    plot_loss(ax1, paths)
    plot_loss(ax2, paths)

    ax2.get_legend().remove()

    # zoom-in / limit the view to different portions of the data
    ax1.set_ylim(-5.5, -2.4)  # outliers only
    ax2.set_ylim(-20, -10.5)  # most of the data

    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.set_ticks_position('none')
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # Cut-out slanted lines.
    # We create line objects in axes coordinates, in which (0,0), (0,1),
    # (1,0), and (1,1) are the four corners of the Axes.
    # The slanted lines themselves are markers at those locations, such that the
    # lines keep their angle and position, independent of the Axes size or scale
    # Finally, we need to disable clipping.

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    ax1.set_title("")
    ax2.set_title("")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss (logprob)")
    plt.savefig(f'GP15_loss_plot.pdf', bbox_inches='tight')


if __name__ == '__main__':
    """
    Quick loss plot similar to that of the paper REFERENCE
    """
    # The three actual stages of the publicated model
    paths = (Path(f'loss_data_GP15_ST1_stage_000'),
             Path(f'loss_data_GP15_ST2_stage_001'),
             Path(f'loss_data_GP15_ST3_stage_002'))
    loss_plot_with_break()