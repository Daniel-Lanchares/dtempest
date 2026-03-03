import torch
import numpy as np
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict
from typing import Callable
from typing_extensions import Self


from .config import no_jargon
from .common_utils import identity
from .flow_utils import create_flow
from .net_utils import create_feature_extractor, create_full_net
from .train_utils import train_model, H5Dataset
from .sampling import SampleDict

class_dict = {
    'sample_dict': SampleDict,
}


class Estimator:
    """
    Main class of the library. Holds the model and can train and sample.

    Parameters
    ----------
    param_list:
        Iterable of parameters to study.
    flow_config:
        Configuration dict for normalizing flow.
    net_config:
        Configuration dict for neural network.
    workdir:
        Path to the directory that will hold sample_related files.
    mode:
        String to interpret desired architecture ('net+flow', 'extractor (pretrained net)+flow' or just 'flow').
    preprocess:
        preprocess function for the data. It's stored as metadata and can be easily accessed.
        Context is preprocessed under the hood, but can still be done explicitly by the user
        (see preprocess method).
    device:
        Device in which to put the model (cpu/cuda).
    jargon:
        A dict that contains various task-specific info to be defined in each package.
    """
    def __init__(self,
                 param_list: list | np.ndarray | torch.Tensor,
                 flow_config: dict,
                 net_config: dict = None,
                 train_history: dict = None,
                 workdir: str | Path = Path(''),
                 name: str = '',
                 mode: str = 'extractor+flow',
                 device: str = 'cpu',
                 preprocess: Callable = None,
                 jargon: dict = no_jargon
                 ):

        if train_history is None:
            train_history = OrderedDict()

        if preprocess is None:
            preprocess = identity

        self.metadata = {
            'name': name,
            'jargon': jargon,
            'param_list': param_list,

            'net_config': net_config,
            'flow_config': flow_config,
            'train_history': train_history,

            'mode': mode,
            'preprocess': preprocess,
            'device': device
        }

        self.workdir = Path(workdir)
        self.device = device

        self._preprocess = preprocess
        self.param_list = param_list

        if 'scales' in flow_config.keys():
            self.scales = self._get_scales(flow_config['scales']).to(device)
        else:
            self.scales = torch.ones(len(self.param_list), device=device)

        if 'shifts' in flow_config.keys():
            self.shifts = self._get_shifts(flow_config['shifts']).to(device)
        else:
            self.shifts = torch.zeros(len(self.param_list), device=device)

        if mode == 'extractor+flow':
            self.model = create_flow(emb_net=create_feature_extractor(**net_config), **flow_config)
        elif mode == 'net+flow':
            self.model = create_flow(emb_net=create_full_net(**net_config), **flow_config)
        elif mode == 'flow':
            if net_config not in [None, {}]:
                raise ValueError(f'Mode {mode} requires no net: net_config must be None or {dict()}')
            self.model = create_flow(emb_net=None, **flow_config)
        else:
            raise ValueError(f'Either mode {mode} was misspelled or is not implemented')

        self.model_to_device(device)

    def __getattr__(self, item: 'str'):
        if item in self.metadata.keys() and item != 'preprocess':
            return self.metadata[item]
        else:
            return self.__dict__[item]

    def _get_scales(self, scales_config):
        if type(scales_config) is list:
            assert len(scales_config) == len(self.param_list), ('Scales need to be of same length as '
                                                                'list of parameters: '
                                                                f'{len(scales_config)} != {len(self.param_list)}')
            return torch.tensor(scales_config)
        elif type(scales_config) is dict:
            return torch.tensor(
                [scales_config[param] if param in scales_config.keys() else 1 for param in self.param_list]
            )
        else:
            return torch.ones(len(self.param_list))

    def _get_shifts(self, shifts_config):
        if type(shifts_config) is list:
            assert len(shifts_config) == len(self.param_list), ('Scales need to be of same length as '
                                                                'list of parameters: '
                                                                f'{len(shifts_config)} != {len(self.param_list)}')
            return torch.tensor(shifts_config)
        elif type(shifts_config) is dict:
            return torch.tensor(
                [shifts_config[param] if param in shifts_config.keys() else 0 for param in self.param_list]
            )
        else:
            return torch.zeros(len(self.param_list))

    def model_to_device(self, device):
        """
        Put model to device, and set self.device accordingly, adapted from dingo.
        """
        if device not in ("cpu", "cuda"):
            raise ValueError(f"Device should be either cpu or cuda, got {device}.")
        self.device = torch.device(device)
        # Commented below so that code runs on first cuda device in the case of multiple.
        # if device == 'cuda' and torch.cuda.device_count() > 1:
        #     print("Using", torch.cuda.device_count(), "GPUs.")
        #     raise NotImplementedError('This needs testing!')
        #     # dim = 0 [512, ...] -> [256, ...], [256, ...] on 2 GPUs
        #     self.model = torch.nn.DataParallel(self.model)
        print(f"Putting posterior model to device {self.device}.")
        self.model.to(self.device)

    @classmethod
    def load_from_file(cls, savefile_path: str | Path, get_metadata: bool = False, **kwargs) -> Self:
        """
        Savefile should be a .pt file containing the struct: tuple(state_dict, metadata)
        """
        state_dict, metadata = torch.load(savefile_path, weights_only=False)
        if 'workdir' not in kwargs.keys():
            kwargs['workdir'] = Path('')
        metadata.update(kwargs)
        
        if metadata['device'] == 'cuda' and not torch.cuda.is_available():
            print("This model has device 'cuda', but there are no such devices available. Setting model device to 'cpu' instead")
            metadata['device'] = 'cpu'
        
        estimator = cls(**metadata)
        estimator.model.load_state_dict(state_dict)

        if get_metadata:
            return estimator, metadata
        return estimator
    
    @classmethod
    def load_metadata_from_file(cls, savefile_path: str | Path):
        """
        Get metadata without initializing model.
        """
        _, metadata = torch.load(savefile_path, weights_only=False)
        return metadata
    
    @classmethod
    def load_dataset_metadata_from_file(cls, savefile_path: str | Path, stage: int = 0):
        return cls.load_metadata_from_file(savefile_path)["train_history"][f"stage {stage}"]["dataset_info"]

    def save_to_file(self, savefile: str | Path):
        """
        Saves model to a .pt file of the form: tuple(state_dict, metadata)
        """
        torch.save((self.model.state_dict(), self.metadata), self.workdir / savefile)

    @property
    def name(self):
        return self.metadata['name']

    def rename(self, new_name):
        self.metadata['name'] = new_name

    def change_parameter_name(self, old_name, to):  # to: new name
        self.metadata['param_list'] = [to if name == old_name else name for name in self.metadata['param_list']]

    def eval(self):
        """Set model to evaluation mode"""
        self.model.eval()

    def sample(self, num_samples, context, preprocess: bool = True, batch_size: int = None):
        """
        Produces samples from the model given some context and then scales and shifts them accordingly.
        """
        if preprocess:
            context = self.preprocess(context)
        elif len(context.shape) == 3:
            context = context.expand(1, *context.shape)
        samples = self.model.sample(num_samples, context, batch_size=batch_size)
        samples[:] = torch.mul(samples[:], self.scales.to(self.device)) + self.shifts.to(self.device)
        return samples

    def log_prob(self, inputs, context, preprocess: bool = True):
        """
        Obtains the logarithm of the probability that some inputs belong to the posterior distribution
        generated by some context.
        """
        if preprocess:
            context = self.preprocess(context)
        elif len(context.shape) == 3:
            context = context.expand(1, *context.shape)
        return self.model.log_prob(inputs, context)

    def sample_and_log_prob(self, num_samples, context, preprocess: bool = True):
        """
        Samples and calculates the logprob of said samples.
        """
        if preprocess:
            context = self.preprocess(context)
        elif len(context.shape) == 3:
            context = context.expand(1, *context.shape)
        samples, logprobs = self.model.sample_and_log_prob(num_samples, context)
        samples[:] = torch.mul(samples[:], self.scales) + self.shifts
        return samples, logprobs

    def pprint_metadata(self, except_keys=None):
        """
        Pretty prints model metadata, with the possibility of key exclusion up to second level.
        """
        from pprint import pprint
        from copy import deepcopy

        if except_keys is None:
            except_keys = ['jargon', ]

        data = deepcopy(self.metadata)
        for key in except_keys:
            if isinstance(key, str):
                data[key] = 'Not shown'
            elif isinstance(key, tuple):
                assert len(key) == 2, "Not showing keys is only implemented up to second level"
                data[key[0]][key[1]] = 'Not shown'
        pprint(data)
    
    def get_dataset_metadata(self, stage: int = 0, _deepcopy: bool = True):
        if _deepcopy:
            return deepcopy(self.metadata["train_history"][f"stage {stage}"]["dataset_info"])
        else:
            return self.metadata["train_history"][f"stage {stage}"]["dataset_info"]

    def turn_net_grad(self, code: str | bool):
        """Control whether the gradient of the embedding is calculated, or their parameters are considered fixed."""
        if code == 'on':
            code = True
        elif code == 'off':
            code = False
        elif type(code) is str:
            raise ValueError(f"Argument 'code' = {code} not understood")
        for parameter in self.model._embedding_net.parameters():
            parameter.requires_grad = code
    
    def numel(self, only_trainable: bool = False, only_net: bool = False):
        """
        Returns the total number of parameters used by `m` (only counting
        shared parameters once); if `only_trainable` is True, then only
        includes parameters with `requires_grad = True`
        """
        m = self.model if not only_net else self.model._embedding_net

        parameters = list(m.parameters())
        if only_trainable:
            parameters = [p for p in parameters if p.requires_grad]
        unique = {p.data_ptr(): p for p in parameters}.values()
        return sum(p.numel() for p in unique)

    def preprocess(self, trainset: np.ndarray | torch.Tensor):
        """Apply the model's preprocessing to the given dataset."""
        if isinstance(trainset, np.ndarray):
            trainset = self._preprocess(torch.tensor(trainset))
            trainset = trainset.expand(1, *trainset.shape)
            return trainset

        return trainset  # If it is a tensor, it has been processed already.

    def _append_training_stage(self, train_config):
        n = len(self.metadata['train_history'])
        self.metadata['train_history'].update({f'stage {n}': train_config})
        return n

    def _append_stage_training_time(self, stage_num, train_time):
        self.metadata['train_history'][f'stage {stage_num}'].update({'training_time': train_time})
        print(f'train_time: {train_time} hours')

    def train(self,
               trainset: H5Dataset,
               traindir: str | Path,
               train_config: dict,
               validation: H5Dataset = None,
               save_loss: bool = True,
               make_plot: bool = True) \
            -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Training routine. Thin wrapper for dtempest.core.train_utils.train_model.

        Parameters
        ----------
        trainset :
            dataset for training.
        traindir :
            train directory.
        train_config :
            configuration of the training stage.
        validation :
            dateset for validation.
        save_loss :
            whether to save the loss data on a file or not.
        make_plot :
            whether to make a preliminary loss plot or not.

        Returns
        -------
        tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Epoch and loss information for provided datasets

        """
        import time
        import matplotlib.pyplot as plt
        from torch.utils.data import DataLoader

        traindir = Path(traindir)
        # Check if traindir exists and make if it does not
        traindir.mkdir(parents=True, exist_ok=True)

        train_config.update({'dataset': trainset.name})
        n = self._append_training_stage(train_config)
        
        if 'loader_kwargs' not in train_config.keys():
            # Backwards compatibility
            train_config['loader_kwargs'] = {'batch_size': train_config.pop('batch_size')}
        
        trainset = DataLoader(trainset, **train_config['loader_kwargs'])  # TODO: explore arguments

        if validation is not None:
            validation = DataLoader(validation, **train_config['loader_kwargs'])
        
        t1 = time.time()
        epochs, losses, vali_epochs, vali_losses = train_model(self.model, trainset, train_config, validation, 
                                                               (self._preprocess, self.scales, self.shifts),
                                                               self.device)
        t2 = time.time()
        self._append_stage_training_time(n, (t2 - t1) / 3600)

        zero_pad = 3  # Hard coded for now

        if save_loss:
            lossdir = traindir / f'loss_data_{self.name}_stage_{n:0{zero_pad}}'
            lossdir.mkdir(parents=True, exist_ok=True)

            torch.save((epochs, losses), lossdir / 'loss_data.pt')
            if validation is not None:
                torch.save((vali_epochs, vali_losses), lossdir / 'validation_data.pt')

        if make_plot:
            # Redundancy in lossdir creation to please Pytorch's checker
            lossdir = traindir / f'loss_data_{self.name}_stage_{n:0{zero_pad}}'
            lossdir.mkdir(parents=True, exist_ok=True)

            epoch_data_avgd = epochs.mean(axis=1)
            loss_data_avgd = losses.mean(axis=1)

            plt.figure(figsize=(10, 8))
            plt.plot(epoch_data_avgd, loss_data_avgd, 'o--', label='loss')

            if validation is not None:
                vali_epochs_data_avgd = vali_epochs.mean(axis=1)
                validation_data_avgd = vali_losses.mean(axis=1)
                plt.plot(vali_epochs_data_avgd, validation_data_avgd, 'o--', label='validation')

            plt.xlabel('Epoch Number')
            plt.ylabel('Log Probability')
            plt.title('Log Probability (avgd per epoch)')
            plt.legend()
            plt.savefig(lossdir / f'loss_plot_{self.name}_stage_{n}.pdf', format='pdf')

        if validation is not None:
            return epochs, losses, vali_epochs, vali_losses
        else:
            return epochs, losses


    def sample_dict(self,
                    num_samples: int,
                    context: torch.Tensor,
                    params: torch.Tensor = None,
                    name: str = None,
                    reference: torch.Tensor = None,
                    preprocess: bool = True,
                    batch_size: int = None,
                    _class_dict: dict = None) -> SampleDict:
        """Creates a SampleDict object, a dictionary of samples with ploting and statistics functionality."""
        from copy import copy

        samples = self.sample(num_samples, context, preprocess, batch_size=batch_size).detach()
        if params is None:
            params = self.param_list
        if _class_dict is None:
            _class_dict = class_dict

        # Define the corresponding SampleDict child object
        sdict_obj = _class_dict['sample_dict']
        sdict = sdict_obj(params, name, jargon=self.jargon)
        for i in range(len(self.param_list)):
            if self.param_list[i] in params:
                sdict[self.param_list[i]] = copy(samples[0, :, i])
                if reference is not None:
                    sdict.truth[self.param_list[i]] = reference[i].item()
        return sdict