import sys
import h5py
from pathlib import Path
import numpy as np

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
    Adapted from:
    https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16?fbclid=IwAR2jFrRkKXv4PL9urrZeiHT_a3eEn7eZDWjUaQ-zcLP6BRtMO7e0nMgwlKU

    HDF5 based dataset for RAM efficiency
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
    
    def __getitems__(self, indexes):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')[self.kind]
        return self.dataset['images'][indexes], self.dataset['labels'][indexes]
    
    def get_metadata(self):
        file = h5py.File(self.file_path, 'r')
        return get_metadata(file)

    # For a GW subclass
    # def get_snr(self, index):
    #     if self.dataset is None:
    #         self.dataset = h5py.File(self.file_path, 'r')[self.kind]
    #     return self.dataset['snrs'][index]
    
    def iter_column(self, column):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')[self.kind]
        return (self.dataset[column][index] for index in range(self.dataset_len))

    def __len__(self):
        return self.dataset_len

def get_metadata(file):
    # https://stackoverflow.com/questions/16547643/convert-a-list-of-delimited-strings-to-a-tree-nested-dict-using-python
    metadata = {}
    for keys, val in file.attrs.items():
        t = metadata
        parts = keys.split("/")
        for part in parts[:-1]:
            t = t.setdefault(part, {})
        t[parts[-1]] = val
    return metadata

def h5_collate_fn(x):
    """Needed for the Dataloader to work properly"""
    return [torch.as_tensor(elem) for elem in x]

# count = 0
def reset_all_weights(model: nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """
    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it is callable call it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)


def loss_print(epoch: int, losses: list, code='train', fmt='.3'):
    """Print loss in a standard format across training"""
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
    """
    Main training loop
    """
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
            
            y = (torch.divide(y.to(device, dtype=torch.float32), scales) - shifts)
            x = preprocess(x).to(device, dtype=torch.float32)

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
            
            # To ensure printouts
            sys.stdout.flush()

        epochs.append(batch_epochs)
        losses.append(batch_losses)

        loss_print(epoch, losses, code='train')
        sys.stdout.flush()

        if valiloader is not None:
            n_batches = len(valiloader)
            print(f'Validation of epoch {epoch + 1:3d}')
            batch_epochs = []
            batch_losses = []
            for i, (x, y) in enumerate(valiloader):
                y = (torch.divide(y.to(device, dtype=torch.float32), scales) - shifts)
                x = preprocess(x).to(device, dtype=torch.float32)

                loss_value = -model.log_prob(inputs=y.float(), context=x).mean()

                print(f'Epoch {epoch + 1:3d}, batch {i:3d}: {loss_value.item():.4}')
                batch_epochs.append(epoch + i / n_batches)
                batch_losses.append(loss_value.item())
                
                # To ensure printouts
                sys.stdout.flush()

            vali_epochs.append(batch_epochs)
            vali_losses.append(batch_losses)

            loss_print(epoch, vali_losses, code='valid')
            sys.stdout.flush()

        # Update Scheduler
        if sched is not None:
            sched.step()
            print(f'Changing learning rate to: {sched.get_last_lr()}')

    return np.array(epochs), np.array(losses), np.array(vali_epochs), np.array(vali_losses)
