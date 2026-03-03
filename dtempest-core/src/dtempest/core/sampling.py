from collections import OrderedDict, namedtuple
import numpy as np
from pprint import pprint
import torch

from ._pesum_deps.samples_dict import SamplesDict, MultiAnalysisSamplesDict

from .config import no_jargon


average_dict = {
    'median': torch.median,
    'mean': torch.mean
}


class SampleDict(SamplesDict):

    def __init__(self, parameters, name: str = None, jargon: dict = no_jargon):
        super().__init__({param: 0 for param in parameters}, jargon=jargon)
        self._truth = OrderedDict()
        if name is None:
            self.name = type(self).__name__
        else:
            self.name = name

    def __setitem__(self, key, value):
        SamplesDict.__setitem__(self, key, value)

    @property
    def truth(self):
        """Truth values for data where it is known (e.g. synthetic injections in noise)"""
        return self._truth

    @truth.setter
    def truth(self, new_truth):
        if len(self.truth) != 0:
            print(f'This {type(self).__name__} already has truth values')
            pprint(self.truth)
            print('Are you sure you want to overwrite some / all of them? [y/n]')
            if input() not in ['y', 'yes']:
                print('Aborting operation then')
                return
        if isinstance(new_truth, dict):
            self._truth.update(new_truth)
        elif isinstance(new_truth, torch.Tensor):
            for i, param in enumerate(self.parameters):
                try:
                    self._truth[param] = new_truth[i].item()
                except IndexError:
                    self._truth[param] = new_truth.item()
        elif isinstance(new_truth, list | np.ndarray):
            for i, param in enumerate(self.parameters):
                self._truth[param] = new_truth[i]

    def select_truths(self, labels, order: str = 'internal'):
        if order == 'internal':
            truths = [self.truth[param] for param in self.truth.keys() if param in labels]
        elif order == 'external':
            truths = [self.truth[param] for param in labels if param in self.truth.keys()]
        else:
            raise ValueError(f"Couldn't understand order argument '{order}'. "
                             "Order can be either 'internal' to follow the SampleDict ordering of params "
                             "or 'external' to follow ordering of given iterable.")
        if len(truths) == 0:
            return None
        else:
            return truths

    @classmethod
    def from_file(cls, filepath):
        """Loading should be handled by its parent class"""
        raise NotImplementedError

    @classmethod
    def from_samplesdict(cls, samplesdict: SamplesDict, name: str = None, jargon: dict = no_jargon):
        sdict = cls(samplesdict.parameters, name, jargon)
        for param in sdict.parameters:
            sdict[param] = samplesdict[param]
        return sdict

    def save_samples(self, filepath):
        np.savez(filepath, **self)


    def get_one_dimensional_median_and_error_bar(self,
                                                 key,
                                                 fmt='.2f',
                                                 quantiles: tuple = None,
                                                 **extra_title_kwargs):
        # Credit: https://git.ligo.org/lscsoft/bilby/-/blob/master/bilby/core/result.py
        """ Calculate the median and error bar for a given key

        Parameters
        ==========
        key: str
            The parameter key for which to calculate the median and error bar
        fmt: str, ('.2f')
            A format string
        quantiles: list, tuple
            A length-2 tuple of the lower and upper-quantiles to calculate
            the errors bars for.

        Returns
        =======
        summary: namedtuple
            An object with attributes, median, lower, upper and string

        """
        summary = namedtuple('summary', ['median', 'lower', 'upper', 'string'])

        if quantiles is None:
            quantiles = (0.16, 0.84)
        if len(quantiles) != 2:
            raise ValueError("quantiles must be of length 2")

        quants_to_compute = np.array([quantiles[0], 0.5, quantiles[1]])
        quants = np.percentile(self[key], quants_to_compute * 100)
        summary.median = quants[1]
        summary.plus = quants[2] - summary.median
        summary.minus = summary.median - quants[0]

        fmt = "{{0:{0}}}".format(fmt).format
        string_template = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
        summary.string = string_template.format(
            fmt(summary.median), fmt(summary.minus), fmt(summary.plus))
        return summary

    def plot(self, *args,
             type: str = 'marginalized_posterior',
             values: bool = True,
             truth_fmt: str = '.2f', **kwargs):
        """Plotting map of its parent class with extra provisions for corner plots"""
        fig = super().plot(type=type, *args, **kwargs)

        if type == 'corner' and values:
            import corner
            if kwargs.get('parameters', None) is not None:
                # Avoids plotting parameters we don't have and maintains correct order.
                params = [param for param in self.parameters if param in kwargs['parameters']]
            else:
                params = self.parameters
            # Give means and quantiles in bilby fashion.
            axes = fig.get_axes()
            medians = []
            truths = kwargs.get('truths', None)
            #  Add the titles
            for i, par in enumerate(params):
                median_data = self.get_one_dimensional_median_and_error_bar(
                    par, quantiles=kwargs.get('quantiles', None), **kwargs.get('title_kwargs', {}))
                ax = axes[i + i * len(params)]
                if ax.title.get_text() == '':
                    if truths is not None and truth_fmt is not None:
                        truth = f'{truths[i]:{truth_fmt}}' + '\n'
                    else:
                        truth = ''
                    ax.set_title(truth + median_data.string, **kwargs.get('title_kwargs', {}))
                medians.append(median_data.median)
            medians = np.array(medians)
            if kwargs.get('medians', None) is not None:
                corner.overplot_lines(fig, medians, color=kwargs.get('median_colour', "tab:blue"))
                corner.overplot_points(fig, medians[None], color=kwargs.get('median_colour', "tab:blue"),
                                       marker=kwargs.get('median_marker', "s"))
        return fig


def get_one_dimensional_median_and_error_bar(cls,
                                             key,
                                             fmt='.2f',
                                             quantiles: tuple = None,
                                             **extra_title_kwargs):
    # Credit: https://git.ligo.org/lscsoft/bilby/-/blob/master/bilby/core/result.py
    """ Calculate the median and error bar for a given key

    Parameters
    ==========
    key: str
        The parameter key for which to calculate the median and error bar
    fmt: str, ('.2f')
        A format string
    quantiles: list, tuple
        A length-2 tuple of the lower and upper-quantiles to calculate
        the errors bars for.

    Returns
    =======
    summary: namedtuple
        An object with attributes, median, lower, upper and string

    """
    summary = namedtuple('summary', ['median', 'lower', 'upper', 'string'])

    if quantiles is None:
        quantiles = (0.16, 0.84)
    if len(quantiles) != 2:
        raise ValueError("quantiles must be of length 2")

    quants_to_compute = np.array([quantiles[0], 0.5, quantiles[1]])
    quants = np.percentile(cls[key], quants_to_compute * 100)
    summary.median = quants[1]
    summary.plus = quants[2] - summary.median
    summary.minus = summary.median - quants[0]

    fmt = "{{0:{0}}}".format(fmt).format
    string_template = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
    summary.string = string_template.format(
        fmt(summary.median), fmt(summary.minus), fmt(summary.plus))
    return summary


class ComparisonSampleDict(MultiAnalysisSamplesDict):
    @property
    def unord_param_intersec(self):
        # The parameters the various analysis have in common
        inter = frozenset(self.parameters[self.labels[0]])
        for params in self.parameters.values():
            inter &= frozenset(params)
        return list(inter)

    def get_given_analysis(self, analysis: str | list[str] = 'all'):
        """

        Handles samples requests from other methods

        Parameters
        ----------
        analysis : The requested samples

        Returns The analysis that match the request
        -------

        """
        if analysis == "all":
            analysis = self.labels
        elif analysis in self.labels:
            analysis = [analysis, ]
        elif isinstance(analysis, list):
            for label in analysis:
                if label not in self.labels:
                    raise ValueError(
                        "'{}' is not a stored analysis. The available analyses "
                        "are: '{}'".format(label, ", ".join(self.labels))
                    )
        else:
            raise ValueError(
                "Please provide a list of analyses that you wish to plot"
            )
        return analysis

    def get_given_parameters(self, parameters: list = None, analysis: str | list[str] = 'all'):
        """

        Returns parameters if available in all analysis in order

        Parameters
        ----------
        parameters : The requested parameters
        analysis : The requested samples

        Returns All parameters requested that are contained in all the specified analysis
        -------

        """

        analysis = self.get_given_analysis(analysis)

        _samples = {label: self[label] for label in analysis}
        _parameters = parameters
        if _parameters is not None:
            params = [
                param for param in _parameters if all(
                    param in posterior for posterior in _samples.values()
                )
            ]
            if not len(params):
                raise ValueError(
                    "None of the chosen parameters are in all of the posterior "
                    "samples tables. Please choose other parameters to plot"
                )

        else:
            _parameters = [list(_samples.keys()) for _samples in _samples.values()]
            params = [
                i for i in _parameters[0] if all(i in _params for _params in _parameters)
            ]
        return params

    def get_median_data(self, parameters: list = None, analysis: str = 'all', as_dict: bool = False, **kwargs):
        """Get median and credible intervals from all the analysis"""

        analysis = self.get_given_analysis(analysis)
        parameters = self.get_given_parameters(parameters, analysis)
        data = dict()

        for n, label in enumerate(analysis):
            data[label] = dict()
            for i, par in enumerate(parameters):
                ntuple = get_one_dimensional_median_and_error_bar(self[label],
                                                                  par,
                                                                  quantiles=kwargs.get('quantiles', None),
                                                                  **kwargs.get('title_kwargs', {}))
                if not as_dict:
                    data[label][par] = ntuple
                else:
                    data[label][par] = dict(median=ntuple.median,
                                            lower=ntuple.minus,
                                            upper=ntuple.plus,
                                            string=ntuple.string)

        return data

    def plot(self,
             *args,
             type: str = 'marginalized_posterior',
             medians: list | str = None,
             **kwargs):
        """Plotting map of its parent class with extra provisions for corner plots"""
        fig = super().plot(type=type, *args, **kwargs)

        if type == 'corner' and medians is not None:
            import corner
            from dtempest.core._pesum_deps.configuration import colorcycle

            colors = kwargs.get('colors', None)

            if colors is None:
                colors = list(colorcycle)
                while len(colors) < len(medians):
                    colors += colors

            medians = self.get_given_analysis(medians)
            params = self.get_given_parameters(kwargs.get('parameters', []), medians)
            # print(f'{params}, {self.parameters=}, {self.labels=}')
            # Give means and quantiles in bilby fashion.
            axes = fig.get_axes()
            #  Add the titles
            for n, label in enumerate(medians):
                median_list = []
                for i, par in enumerate(params):
                    median_data = get_one_dimensional_median_and_error_bar(self[label],
                                                                           par, quantiles=kwargs.get('quantiles', None),
                                                                           **kwargs.get('title_kwargs', {}))
                    ax = axes[i + i * len(params)]
                    previous_title = ax.title.get_text()
                    if previous_title == '':
                        ax.set_title(median_data.string, **kwargs.get('title_kwargs', {}))
                    else:
                        ax.set_title(previous_title + '\n' + median_data.string, **kwargs.get('title_kwargs', {}))

                    median_list.append(median_data.median)
                median_list = np.array(median_list)

                corner.overplot_lines(fig, median_list, color=colors[n])
                corner.overplot_points(fig, median_list[None], color=colors[n],
                                       marker=kwargs.get('median_marker', "s"))
        return fig