"""
Module containing the main class: Estimator
"""

import torch
import numpy as np
from pathlib import Path
from collections import OrderedDict
from typing import Callable
from typing_extensions import Self

from tqdm import tqdm

from .config import no_jargon
from .common_utils import identity
from .flow_utils import create_flow, get_transform_hidden_layers
from .net_utils import create_feature_extractor, create_full_net, numel
from .train_utils import train_model, H5Dataset
from .sample_utils import SampleSet, SampleDict, MSEDataFrame, MSESeries

class_dict = {
    'sample_set': SampleSet,
    'sample_dict': SampleDict,
    'data_frame': MSEDataFrame,
    'series': MSESeries
}


class Estimator:
    """
    Main class of the library. Holds the model and can train and sample.

    Parameters
    ----------
    estimation_parameters :
        Iterable of parameters to study.

    flow_config :
        Configuration dict for normalizing flow.

    net_config :
        Configuration dict for neural network.

    workdir :
        Path to the directory that will hold sample_related files.

    mode :
        String to interpret desired architecture ('net+flow', 'extractor (pretrained net)+flow' or just 'flow').

    preprocess :
        preprocess function for the data. It's stored as metadata and can be easily accessed.
        Context is preprocessed under the hood, but can still be done explicitly by the user
        (see preprocess method).

    device :
        Device in which to put the model (cpu/cuda).

    jargon :
        A dict that contains various task-specific info to be defined in each package.
    """
    def __init__(self,
                 estimation_parameters: list | np.ndarray | torch.Tensor,
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
            'estimation_parameters': estimation_parameters,

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
        self.estimation_parameters = np.asarray(estimation_parameters)

        if 'scales' in flow_config.keys():
            self.scales = self._get_scales(flow_config.pop('scales'))
        else:
            self.scales = torch.ones(len(self.estimation_parameters))

        if 'shifts' in flow_config.keys():
            self.shifts = self._get_shifts(flow_config.pop('shifts'))
        else:
            self.shifts = torch.zeros(len(self.estimation_parameters))

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

    def __getattr__(self, item: str):
        if item in self.metadata.keys() and item != 'preprocess':
            return self.metadata[item]
        else:
            return self.__dict__[item]

    def _get_scales(self, scales_config):
        """
        Constructs the model's scale tensor.
        Parameters
        ----------
        scales_config :
            Scale configuration structure. Can be a list (ordered) or a dictionary.

        Returns
        -------
        out :
            Scales tensor.
        """
        if type(scales_config) is list:
            assert len(scales_config) == len(self.estimation_parameters), ('Scales need to be of same length as '
                                                                'list of parameters: '
                                                                f'{len(scales_config)} != {len(self.estimation_parameters)}')
            return torch.tensor(scales_config)
        elif type(scales_config) is dict:
            return torch.tensor(
                [scales_config[param] if param in scales_config.keys() else 1 for param in self.estimation_parameters]
            )
        else:
            return torch.ones(len(self.estimation_parameters))

    def _get_shifts(self, shifts_config):
        """
        Constructs the model's shift tensor.
        Parameters
        ----------
        shifts_config :
            Shift configuration structure. Can be a list (ordered) or a dictionary.

        Returns
        -------
        out :
            Shifts tensor.
        """
        if type(shifts_config) is list:
            assert len(shifts_config) == len(self.estimation_parameters), ('Shifts need to be of same length as '
                                                                'list of parameters: '
                                                                f'{len(shifts_config)} != {len(self.estimation_parameters)}')
            return torch.tensor(shifts_config)
        elif type(shifts_config) is dict:
            return torch.tensor(
                [shifts_config[param] if param in shifts_config.keys() else 0 for param in self.estimation_parameters]
            )
        else:
            return torch.zeros(len(self.estimation_parameters))

    def model_to_device(self, device):
        """
        Puts model to device, and set self.device accordingly, adapted from dingo.
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
        Parameters
        ----------
        savefile_path :
            Savefile should be a torch.save() of a tuple(state_dict, metadata)
        get_metadata :
            Whether to return the metadata as well.
        kwargs :
            Arguments to update the saved metadata before model reconstruction.

        Returns
        -------
        out :
            Self
        """
        state_dict, metadata = torch.load(savefile_path, weights_only=False)
        if 'workdir' not in kwargs.keys():
            kwargs['workdir'] = Path('')
        metadata.update(kwargs)
        estimator = cls(**metadata)
        estimator.model.load_state_dict(state_dict)

        if get_metadata:
            return estimator, metadata
        return estimator

    def save_to_file(self, savefile: str | Path, send_to_cpu: bool = True):
        """
        Parameters
        ----------
        savefile :
            Saves to a '.pt' file.
        send_to_cpu :
            Whether to send model to CPU before saving. Facilitates later loading on GPU-less systems.

        Returns
        -------
        out :
            None
        """
        if send_to_cpu:
            # To ensure models trained on a GPU can be loaded on a CPU later
            self.model_to_device('cpu')
        torch.save((self.model.state_dict(), self.metadata), self.workdir / savefile)

    def numel(self, component: str = 'all', only_trainable: bool = False) -> int:
        """
        Measures the number of neural parameters of a component of the model

        Parameters
        ----------
        component :
            Module to measure. Either the embedding net, the flow, or both.
        only_trainable :
            Whether to ignore non-trainable parameters. The default is False.

        Returns
        -------
        out :
            Number (integer) of parameters
        """

        if component == 'all':
            return numel(self.model, only_trainable)
        elif component == 'net':
            assert self.model._embedding_net is not None, 'The model has no embedding net'
            return numel(self.model._embedding_net, only_trainable)
        elif component == 'flow':
            net_params = 0 if self.model._embedding_net is None else numel(self.model._embedding_net, only_trainable)
            return numel(self.model, only_trainable) - net_params
        else:
            raise ValueError("Selected component non understood. Posible values are 'net', 'flow' or 'all'.")

    def numel_hidden_flow_layers(self):
        """Get the number (approx.) of hidden layers. Only works for some transforms."""
        nsteps, base_t, base_t_kwargs = self.metadata['num_flow_steps'], self.metadata['base_transform'], self.metadata['base_transform_kwargs']
        return nsteps*get_transform_hidden_layers(base_t, base_t_kwargs)

    @property
    def name(self):
        return self.metadata['name']

    def rename(self, new_name):
        self.metadata['name'] = new_name

    def change_parameter_name(self, old_name, to):
        """
        Rewrite parameter name.
        Parameters
        ----------
        old_name :
            Name supposedly already in 'self.estimation_parameters'.
        to :
            New name.

        Returns
        -------

        """
        self.metadata['estimation_parameters'] = [to if name == old_name else name for name in self.metadata['estimation_parameters']]

    # @is_documented_by(torch.nn.Module.eval)
    def eval(self):
        r"""
        Set the module in evaluation mode.

        This has an effect only on certain modules. See the documentation of
        particular modules for details of their behaviors in training/evaluation
        mode, i.e. whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

        See :ref:`locally-disable-grad-doc` for a comparison between
        `.eval()` and several similar mechanisms that may be confused with it.

        Returns:
            Module: self
        """
        self.model.eval()

    def sample(self, num_samples, context, preprocess: bool = True):
        """
        Generates samples from the distribution. Samples can be generated in batches.

        Args:
            num_samples: int, number of samples to generate.
            context: Tensor or None, conditioning variables. If None, the context is ignored.
            preprocess: bool or None, whether to preprocess the context.

        Returns:
            A Tensor containing the samples, with shape [num_samples, ...] if context is None, or
            [context_size, num_samples, ...] if context is given.
        """
        if preprocess:
            context = self.preprocess(context)
        elif len(context.shape) == 3:
            context = context.expand(1, *context.shape)
        samples = self.model.sample(num_samples, context)
        samples[:] = torch.mul(samples[:], self.scales) + self.shifts
        return samples

    def log_prob(self, inputs, context, preprocess: bool = True):
        """
        Calculate log probability under the distribution.

        Args:
            inputs: Tensor, input variables.
            context: Tensor or None, conditioning variables. If a Tensor, it must have the same
                number of rows as the inputs. If None, the context is ignored.
            preprocess: bool or None, whether to preprocess the context.

        Returns:
            A Tensor of shape [input_size], the log probability of the inputs given the context.
        """
        if preprocess:
            context = self.preprocess(context)
        elif len(context.shape) == 3:
            context = context.expand(1, *context.shape)
        return self.model.log_prob(inputs, context)

    def sample_and_log_prob(self, num_samples, context, preprocess: bool = True):
        """Generates samples from the flow, together with their log probabilities.

        For flows, this is more efficient that calling `sample` and `log_prob` separately.
        """
        if preprocess:
            context = self.preprocess(context)
        elif len(context.shape) == 3:
            context = context.expand(1, *context.shape)
        samples, logprobs = self.model.sample_and_log_prob(num_samples, context)
        samples[:] = torch.mul(samples[:], self.scales) + self.shifts
        return samples, logprobs

    def pprint_metadata(self, except_keys=None, **kwargs):
        """
        Print the model's metadata in an organized fashion using the pprint module.
        Parameters
        ----------
        except_keys :
            Metadata keys to ignore. By default, it won't print jargon.
        kwargs :
            Keyword arguments to be passed to pprint.

        Returns
        -------

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
        pprint(data, **kwargs)

    def turn_net_grad(self, code: str | bool):
        """
        Turn feature extractor on or off.
        Parameters
        ----------
        code :
            Either 'on', 1 or 'off', 0.

        Returns
        -------

        """
        if code == 'on':
            code = True
        elif code == 'off':
            code = False
        elif type(code) is str:
            raise ValueError(f"Argument 'code' = {code} not understood")
        for parameter in self.model._embedding_net.parameters():
            parameter.requires_grad = code

    def preprocess(self, model_array: np.ndarray | torch.Tensor):
        """
        Apply the model's pre-process function to a model array and convert to tensor.
        Parameters
        ----------
        model_array :
            Array of shape (3, M, N), format compatible with ResNets.

        Returns
        -------
        out :
            Processed torch.Tensor.
        """
        if isinstance(model_array, np.ndarray):
            # if it is an array its context from the sampling methods, not an entire set
            model_array = self._preprocess(torch.tensor(model_array))
            model_array = model_array.expand(1, *model_array.shape)
            return model_array

        elif isinstance(model_array, torch.Tensor):
            return model_array  # Has been preprocessed already

        else:
            raise TypeError(f"Argument 'model_array' is of type '{type(model_array)}'."
                            "Only permitted types are np.ndarray and torch.Tensor")

    def _append_training_stage(self, train_config):
        """Update metadata with the training configuration at the beginning of a stage."""
        n = len(self.metadata['train_history'])
        self.metadata['train_history'].update({f'stage {n}': train_config})
        return n

    def _append_stage_training_time(self, stage_num, train_time):
        """Update metadata with the training time at the end of a stage."""
        self.metadata['train_history'][f'stage {stage_num}'].update({'training_time': train_time})
        print(f'train_time: {train_time} hours')

    def h5train(self,
                dataset: H5Dataset,
                traindir: str | Path,
                train_config: dict,
                validation: H5Dataset = None,
                preprocess: bool = True,
                save_loss: bool = True,
                make_plot: bool = True):
        """
        Training routine for HDF5 datasets (Currently the only one supported).
        Parameters
        ----------
        dataset :
            HDF5-based training dataset.
        traindir :
            Directory to store training results.
        train_config :
            Training configuration
        validation :
            HDF5-based validation dataset.
        preprocess :
            Deprecated. Only here for backwards compatibility.
        save_loss :
            Whether to save loss information.
        make_plot :
            Whether to plot loss information.

        Returns
        -------
        out :
            None
        """
        import time
        import matplotlib.pyplot as plt
        from torch.utils.data import DataLoader

        # model_array = self.rescale_trainset(model_array) # For now

        traindir = Path(traindir)
        # Check if traindir exists and make if it does not
        traindir.mkdir(parents=True, exist_ok=True)

        train_config.update({'dataset': dataset.name})
        n = self._append_training_stage(train_config)


        if 'loader_kwargs' not in train_config.keys():
            # Backwards compatibility
            train_config['loader_kwargs'] = {'batch_size': train_config.pop('batch_size')}

        dataset = DataLoader(dataset, **train_config['loader_kwargs'])

        if validation is not None:
            validation = DataLoader(validation, **train_config['loader_kwargs'])

        t1 = time.time()
        epochs, losses, vali_epochs, vali_losses = train_model(self.model, dataset, train_config, validation,
                                                               (self._preprocess, self.scales.numpy(),
                                                                self.shifts.numpy()),
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

    def train(self, dataset: H5Dataset, *args, **kwargs):
        """Main training entry point. Allows for extension of training methodologies."""
        if isinstance(dataset, H5Dataset):
            return self.h5train(dataset, *args, **kwargs)


    def sample_dict(self,
                    num_samples: int,
                    context: torch.Tensor,
                    params: torch.Tensor = None,
                    name: str = None,
                    reference: torch.Tensor = None,
                    preprocess: bool = True,
                    _class_dict: dict = None) -> SampleDict:
        """
        Creates a SampleDict object to store samples
        Parameters
        ----------
        num_samples :
            Number of samples to generate.
        context :
            Conditioning context.
        params :
            Labels of the samples. Subset of 'self.estimation_parameters'.
        name :
            SampleDict's name.
        reference :
            Reference data for injections.
        preprocess :
            Whether to preprocess the context.
        _class_dict :
            Internal argument. Controls family of subclasses.

        Returns
        -------
        out:
            SampleDict object or subclass.

        """
        from copy import copy

        samples = self.sample(num_samples, context, preprocess).detach()
        if params is None:
            params = self.estimation_parameters
        if _class_dict is None:
            _class_dict = class_dict

        # Define the corresponding SampleDict child object
        sdict_obj = _class_dict['sample_dict']
        sdict = sdict_obj(params, name, jargon=self.jargon)
        for i in range(len(self.estimation_parameters)):
            if self.estimation_parameters[i] in params:
                sdict[self.estimation_parameters[i]] = copy(samples[0, :, i])
                if reference is not None:
                    sdict.truth[self.estimation_parameters[i]] = reference[i].item()
        return sdict

    def sample_set(self,
                   num_samples: int,
                   data: H5Dataset,
                   params: torch.Tensor = None,
                   name: str = None,
                   preprocess: bool = True,
                   _class_dict: dict = None) -> SampleSet:
        """Sample over an entire dataset.

        TODO: Needs update, currently broken"""
        if params is None:
            params = self.estimation_parameters
        if _class_dict is None:
            _class_dict = class_dict

        # Define the corresponding SampleSet child object
        sset_obj = _class_dict['sample_set']
        sset = sset_obj(params, name)
        if hasattr(data, 'name'):
            sset.data_name = data.name
        with tqdm(total=len(data.index), desc=f'Creating SampleSet {name}', ncols=100) as p_bar:
            for event in data.index:
                sset[str(event)] = Estimator.sample_dict(self,
                                                         num_samples,
                                                         data['images'][event],
                                                         params,
                                                         str(event),
                                                         data['labels'][event],
                                                         # If real event: either None or estimation
                                                         preprocess,
                                                         _class_dict)
                p_bar.update(1)
        return sset