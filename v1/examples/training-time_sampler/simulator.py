import torch
import numpy as np
from functools import partial
from torch import Size, Tensor
from torch.utils.data import DataLoader, IterableDataset
import bilby.gw.prior as bilby_priors
from typing import Callable, Iterator, Tuple, Iterable

from generation import create_image, generate_timeseries

class BilbyPriorWrapper(bilby_priors.PriorDict):
    def __init__(self, bilby_prior, params):
        super(BilbyPriorWrapper, self).__init__(bilby_prior)
        self.theta_labels = params

    def sample(self, size = None):
        if size == () or size is None: # Patch to deal with differing approaches
            size = None
        else:
            size = int(size)
        return torch.tensor(
            list(self.sample_subset_constrained(keys=list(self.theta_labels), size=size).values()),
            dtype=torch.float32)

class IterableJointDataset(IterableDataset):
    r"""Creates an iterable dataset of batched pairs :math:`(\theta, x)`."""

    def __init__(
        self,
        prior: BilbyPriorWrapper,
        simulator: Callable,
        batch_shape: Size = (),
        numpy: bool = False,
    ):
        super().__init__()

        self.prior = prior
        self.simulator = simulator
        self.batch_shape = batch_shape
        self.numpy = numpy

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        while True:
            theta = self.prior.sample(self.batch_shape)

            if self.numpy:
                x = self.simulator(theta.detach().cpu().numpy().astype(np.double))
                x = torch.from_numpy(x).to(theta)
            else:
                x = self.simulator(theta)

            yield theta, x


class JointLoader(DataLoader):
    r"""Creates an infinite data loader of batched pairs :math:`(\theta, x)` generated
    by a prior distribution :math:`p(\theta)` and a simulator.

    The simulator is a stochastic function taking (a vector of) parameters
    :math:`\theta`, in the form of a NumPy array or a PyTorch tensor, as input and
    returning an observation :math:`x` as output, which implicitly defines a
    likelihood distribution :math:`p(x | \theta)`. Together with the prior, they form
    a joint distribution :math:`p(\theta, x) = p(\theta) p(x | \theta)` from which
    pairs :math:`(\theta, x)` are independently drawn.

    Arguments:
        prior: A prior distribution :math:`p(\theta)`.
        simulator: A callable simulator.
        batch_size: The batch size of the generated pairs.
        vectorized: Whether the simulator accepts batched inputs or not.
        numpy: Whether the simulator requires NumPy or PyTorch inputs.
        kwargs: Keyword arguments passed to :class:`torch.utils.data.DataLoader`.

    Example:
        >>> loader = JointLoader(prior, simulator, numpy=True, num_workers=4)
        >>> for theta, x in loader:
        ...     theta, x = theta.cuda(), x.cuda()
        ...     something(theta, x)
    """

    def __init__(
        self,
        prior: BilbyPriorWrapper,
        simulator: Callable,
        batch_size: int = 2**8,  # 256
        vectorized: bool = False,
        numpy: bool = False,
        **kwargs,
    ):
        super().__init__(
            IterableJointDataset(
                prior,
                simulator,
                batch_shape=(batch_size,) if vectorized else (),
                numpy=numpy,
            ),
            batch_size=None if vectorized else batch_size,
            **kwargs,
        )



class CBCSimulator:

    def __init__(self,
                 representation: str = 'timeseries',
                 **injection_kwargs):
        self.representation = representation

        ## Handle configuration

        self.timeseries_config = {
            key: value for key, value in injection_kwargs.items()
            if key in ['ifos', 'asds', 'sampling_frequency', 'waveform_generator',
                       'snr_tol', 'parameter_labels', 'target_snr_prior','default_prior']
        }

        self.generate_timeseries = partial(generate_timeseries, **self.timeseries_config)
        self.generate_image = partial(create_image, **injection_kwargs)
        self.generate_both = partial(create_image, return_both=True, **injection_kwargs)

    def __call__(self, theta: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.representation == 'timeseries':
            return torch.tensor(np.array(self.generate_timeseries(theta)[0]))
        elif self.representation == 'image':
            return torch.tensor(self.generate_image(theta))
        elif self.representation == 'dual':
            ts_data, qt_data = self.generate_both(theta)
            return torch.tensor(np.array(ts_data), dtype=torch.float32), torch.tensor(qt_data, dtype=torch.float32)
        else:
            raise NotImplementedError



def set_universal_seed(seed):
    import torch
    import bilby
    import numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)
    bilby.utils.random.seed(seed)

def loader_from_bilby_prior(parameter_list: list,
                            representation: str = 'timeseries',
                            prior: bilby_priors.PriorDict = None,
                            loader_kwargs: dict = None,
                            **injection_kwargs):
    assert 'snr_tol' in injection_kwargs.keys(), 'Missing argument in Injection arguments: snr_tol'
    snr_tol = injection_kwargs['snr_tol']
    if loader_kwargs is None:
        loader_kwargs = {}

    ## Handle SNR
    if prior is None:
        prior = bilby_priors.BBHPriorDict()
    if snr_tol in (0.0, -1.0):
        # Special cases: Return as is or with zero noise
        _snr = snr_tol
    elif '_snr' not in prior.keys():
        assert 'snr_lims' in injection_kwargs.keys(), 'Missing argument in Injection arguments: snr_lims'
        snr_low, snr_high = injection_kwargs.pop('snr_lims')

        _snr = bilby_priors.Uniform(snr_low, snr_high, name='snr', latex_label=r'$\textrm{SNR}$')
    else:
        _snr = prior.pop('_snr')

    injection_kwargs['target_snr_prior'] = _snr
    injection_kwargs['parameter_labels'] = parameter_list
    injection_kwargs['default_prior'] = prior


    prior = BilbyPriorWrapper(prior, parameter_list)

    simulator = CBCSimulator(representation, **injection_kwargs)

    if loader_kwargs:
        return JointLoader(prior, simulator, **loader_kwargs)
    else:
        return prior, simulator


def default_injection_config(durat=None, sampl=None, min_fr=None, approx=None):
    import bilby
    duration = 4.0 \
        if durat is None else durat  # seconds  # change from around 2 to 8 depending on masses
    sampling_frequency = 1024 \
        if sampl is None else sampl # Hz
    min_freq = 20.0  \
        if min_fr is None else min_fr # Hz
    approximant = "IMRPhenomXPHM" \
        if approx is None else approx
    # The other arguments can be updated at a later time


    waveform_arguments = dict(
        waveform_approximant=approximant,
        minimum_frequency=min_freq,

    )

    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,

    )

    # List of interferometers (R, G, B)
    ifolist = ('L1', 'H1', 'V1')
    asds = None

    injection_kwargs = {
        # GW-related
        'ifos': ifolist,
        'asds': asds,
        'snr_tol': 0.5,
        'snr_lims': (5, 30),

        # Timeseries-related
        'sampling_frequency': sampling_frequency,
        'waveform_generator': waveform_generator,

        # Image-related
        'img_res': (128, 128),  # (height, width) in pixels
        'duration': duration,

        'qtrans_kwargs': {
            'frange': (min_freq, 300),
            'qrange': (4, 8),
            'outseg': (-0.1, 0.1)
        }
    }
    return injection_kwargs