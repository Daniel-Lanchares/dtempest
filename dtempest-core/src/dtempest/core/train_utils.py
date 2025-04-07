import numpy as np
from pathlib import Path

import h5py
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, StepLR, CosineAnnealingLR

sched_dict = {'Plateau': ReduceLROnPlateau,
              'StepLR': StepLR,
              'MultiStepLR': MultiStepLR,
              'cosine': CosineAnnealingLR}
opt_dict = {'SGD': SGD, 'Adam': Adam}
loss_dict = {'MSE': nn.MSELoss, 'CE': nn.CrossEntropyLoss}


class H5Dataset(torch.utils.data.Dataset):
    """
    HDF5 based dataset for RAM efficiency.

    Adapted from:
    https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16?fbclid=IwAR2jFrRkKXv4PL9urrZeiHT_a3eEn7eZDWjUaQ-zcLP6BRtMO7e0nMgwlKU

    """

    def __init__(self, path, kind: str = 'training', name=None):
        self.file_path = Path(path)
        self.dataset = None
        self.kind = kind
        if name is None:
            self.name = self.file_path.name + '-' + self.kind
        else:
            self.name = name
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file[self.kind]['images'])

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')[self.kind]
        return self.dataset['images'][index], self.dataset['labels'][index]

    def get_metadata(self, dict_type=True):
        file = h5py.File(self.file_path, 'r')
        if dict_type:
            # TODO: Return proper dicts within dicts
            return dict(file.attrs.items())
        return file.attrs

    def get_snr(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')[self.kind]
        return self.dataset['snrs'][index]

    def iter_column(self, column):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')[self.kind]
        return (self.dataset[column][index] for index in range(self.dataset_len))

    def __len__(self):
        return self.dataset_len
    
    # def rescale(self, model, override=False):
    #     np.divide(x, self.scales.numpy()) - self.shifts.numpy()

def loss_print(epoch: int, losses: list, code='train', fmt='.3'):
    temp_loss = np.array(losses).mean(axis=1)
    deviation = np.array(losses).std(axis=1)
    if epoch > 0:
        msg = (f'\nAverage {code}: {temp_loss[-1]:{fmt}}±{deviation[-1]:{fmt}}, '
               f'Delta: {(temp_loss[-1] - temp_loss[-2]):{fmt}} '
               f'({(temp_loss[-1] - temp_loss[-2]) / temp_loss[-2] * 100:{fmt}}%)\n')
    else:
        msg = f'\nAverage {code}: {temp_loss[0]:{fmt}}±{deviation[0]:{fmt}}\n'
    # Consider logging to file
    print(msg)


def train_model(model, dataloader, train_config, valiloader, data_transforms, device):
    n_epochs = train_config['num_epochs']
    lr = train_config['learning_rate']
    opt = opt_dict[train_config['optim_type']](model.parameters(), lr)

    if 'sched_kwargs' in train_config and train_config['sched_kwargs'] is not None:
        sched_type = train_config['sched_kwargs'].pop('type')
        sched = sched_dict[sched_type](opt, **(train_config['sched_kwargs']))
    else:
        sched = None

    if 'checkpoint_every_x_epochs' in train_config and train_config['checkpoint_every_x_epochs'] is not None:
        checkpt = train_config['checkpoint_every_x_epochs']

    preprocess, scales, shifts = data_transforms

    # Train model
    losses = []
    epochs = []
    vali_losses = []
    vali_epochs = []
    for epoch in range(n_epochs):
        print(f'Epoch {epoch + 1}')
        n_batches = len(dataloader)
        batch_epochs = []
        batch_losses = []
        for i, (x, y) in enumerate(dataloader):
            # Update the weights of the network
            
            y = (np.divide(y, scales) - shifts).to(device)
            x = preprocess(x).to(device)

            loss_value = -model.log_prob(inputs=y.float(), context=x).mean()
            print(f'Epoch {epoch + 1:3d}, batch {i:3d}: {loss_value.item():.4}')

            loss_value.backward()
            if 'grad_clip' in train_config and train_config['grad_clip'] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               train_config['grad_clip'])
            opt.step()
            opt.zero_grad()
            # Store training data
            batch_epochs.append(epoch + i / n_batches)
            batch_losses.append(loss_value.item())

        epochs.append(batch_epochs)
        losses.append(batch_losses)

        loss_print(epoch, losses, code='train')

        if valiloader is not None:
            n_batches = len(valiloader)
            print(f'Validation of epoch {epoch + 1:3d}')
            batch_epochs = []
            batch_losses = []
            for i, (x, y) in enumerate(valiloader):
                y = np.divide(y, scales) - shifts
                x = preprocess(x)

                loss_value = -model.log_prob(inputs=y.float(), context=x).mean()

                print(f'Epoch {epoch + 1:3d}, batch {i:3d}: {loss_value.item():.4}')
                batch_epochs.append(epoch + i / n_batches)
                batch_losses.append(loss_value.item())

            vali_epochs.append(batch_epochs)
            vali_losses.append(batch_losses)

            loss_print(epoch, vali_losses, code='valid')

        # Update Scheduler
        if sched is not None:
            sched.step()  # TODO Implement possibility for schedulers that take loss values. Not urgent

    # For manually variable lr
    # for g in optim.param_groups:
    #     g['lr'] = 0.001

    return np.array(epochs), np.array(losses), np.array(vali_epochs), np.array(vali_losses)
