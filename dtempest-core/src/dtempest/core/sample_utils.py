from collections import OrderedDict, namedtuple
import numpy as np
import pandas as pd
from pprint import pprint
import torch
from tqdm import tqdm
from scipy import stats as st

from ._pesummary_dependencies.samples_dict import SamplesDict, MultiAnalysisSamplesDict

from .config import no_jargon
from .common_utils import PrintStyle, handle_multi_index_format, merge_headers
from .data_utils import get_parameter_units

# class SampleArray(np.ndarray):
#     # read this, might be useful
#     # https://numpy.org/doc/stable/user/basics.subclassing.html
#     def

average_dict = {
    'median': torch.median,
    'mean': torch.mean
}


class MSESeries:
    """
    Wrapper class for pandas Series. Stores Mean Squared Error for a SampleDict
    """

    def __init__(self,
                 name: str = None,
                 relative: bool = False,
                 sqrt: bool = True,
                 jargon: dict = None, *args, **kwargs):
        if 'data' in kwargs.keys() and isinstance(kwargs['data'], pd.Series):
            self._df = kwargs['data']
        else:
            self._df = pd.Series(name=name, *args, **kwargs)
        if name is None:
            name = type(self).__name__
        self.name = name
        self.relative = relative
        self.sqrt = sqrt
        self.jargon = jargon

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._df, attr)

    def __getitem__(self, item):
        if isinstance(self._df[item], pd.Series):
            return type(self)(data=self._df[item], name=self.name, sqrt=self.sqrt)
        return self._df[item]

    def __setitem__(self, item, data):
        self._df[item] = data

    def __repr__(self):
        temp_df = self._df.to_frame(name=self.name)
        if self.relative:
            temp_df['units'] = [r'$\%$' for _ in self._df.index]

        else:
            temp_df['units'] = [get_parameter_units(param, self.jargon) for param in
                                self._df.index]
            if not self.sqrt:
                temp_df['units'] = [temp_df['units'][param] if temp_df['units'][param] == r'$ø$'
                                    else temp_df['units'][param] + r'^2'
                                    for param in self._df.index]
        return temp_df.to_markdown()


class MSEDataFrame:
    """
    Wrapper class for pandas DataFrame. Stores Mean Squared Error for a SampleSet
    """

    def __init__(self,
                 name: str = None,
                 relative: bool = False,
                 sqrt: bool = True,
                 verbose: dict = None,
                 jargon: dict = None,
                 _series_class=MSESeries, *args, **kwargs):

        if 'data' in kwargs.keys() and isinstance(kwargs['data'], pd.DataFrame):
            self._df = kwargs['data']
        else:
            self._df = pd.DataFrame(*args, **kwargs)
        if name is None:
            name = type(self).__name__
        self.name = name
        self.relative = relative
        self.sqrt = sqrt
        self.jargon = jargon
        self._series_class = _series_class

        if verbose is None:
            self.verbose = {}
        else:
            self.verbose = verbose

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._df, attr)

    def __getitem__(self, item):
        result = self._df[item]
        if isinstance(result, pd.Series):
            return self._series_class(data=result, name=self.name, sqrt=self.sqrt, relative=self.relative)
        if isinstance(result, pd.DataFrame):
            return type(self)(data=result, name=self.name, sqrt=self.sqrt,
                              relative=self.relative, verbose=self.verbose)
        return self._df[item]

    def __setitem__(self, item, data):
        self._df[item] = data

    def __repr__(self):
        return self._df.__repr__()

    def select_parameter(self, parameter):

        result = self._df.loc[(slice(None), parameter), :]

        return type(self)(data=result, name=self.name, sqrt=self.sqrt,
                          relative=self.relative, verbose=self.verbose)

    def _temp_unit_df(self) -> pd.DataFrame:
        temp_df = pd.DataFrame.copy(self._df)
        if self.relative:
            temp_df['units'] = [r'$\%$' for _ in self._df.index]

        else:
            temp_df['units'] = [get_parameter_units(param, self.jargon)
                                if type(self._df.index) in [pd.Index, pd.RangeIndex]
                                else get_parameter_units(param[1], self.jargon)
                                for param in self._df.index]
            if not self.sqrt:
                temp_df['units'] = [temp_df['units'][param] if temp_df['units'][param] == r'$ø$'
                                    else temp_df['units'][param] + r'^2'
                                    for param in self._df.index]
        return temp_df

    def handle_verbose(self, temp_df, language):
        # if temp_df.index.name in self.verbose:
        #     temp_df.index.name += break_char[language] + str(self.verbose[temp_df.index.name])

        if language == 'markdown':
            names_verbose = [name + '<br>' + str(self.verbose[name])
                             if name in self.verbose else name for name in temp_df.columns]
        elif language == 'latex':
            names_verbose = [r'\makecell{' + name + r"\\" + str(self.verbose[name]) + '}'
                             if name in self.verbose else name for name in temp_df.columns]
        else:
            print(PrintStyle.red + 'Warning: language {language} not understood. '
                                   'Resorting to non verbose names' + PrintStyle.reset)
            names_verbose = temp_df.colums

        temp_df.columns = pd.Index(names_verbose)
        return temp_df

    def to_table(self, language: str, units: bool = True, **kwargs):

        if units and self._df.index.name != 'events':
            # Second condition excludes mono-parameter cross-sections. I cannot recover parameter unit
            temp_df = self._temp_unit_df()
        else:
            temp_df = pd.DataFrame.copy(self._df)

        # title = 'Title_here'
        # tuples = [('title', temp_df.columns[0]),] + [('', column) for column in temp_df.columns[1:]]
        # print(tuples)
        # temp_df.columns = pd.MultiIndex.from_tuples(tuples, names=['title', 'headers'])
        # temp_df = handle_multi_column_format(temp_df)

        if type(temp_df.index) is pd.MultiIndex:
            temp_df, kwargs = handle_multi_index_format(temp_df, **kwargs)

        temp_df = self.handle_verbose(temp_df, language)

        # print('|'+self.name.center(len(temp_df.to_markdown(**kwargs).split('\n')[0])-2)+'|')
        if language == 'markdown':
            return temp_df.to_markdown(**kwargs)
        elif language == 'latex':
            # Double replace tries to avoid miss-formatting of underscores. Not bulletproof, but functional
            string_table = (temp_df.to_latex(**kwargs)
                            .replace('_', r'\_').replace(r'\_{', '_{').replace('$ø$', r'ø').replace('tabular', 'tblr'))
            return merge_headers(string_table)

    def to_latex(self, units: bool = True, **kwargs):
        return self.to_table('latex', units, **kwargs)

    def to_markdown(self, units: bool = True, **kwargs):
        return self.to_table('markdown', units, **kwargs)

    # def __repr__(self):
    #     temp_df = self._df.to_frame()
    #     temp_df['units'] = [get_param_units(param)[1:-1] for param in self._df.index]
    #     if not self.sqrt:
    #         temp_df['units'] = [temp_df['units'][param] if temp_df['units'][param] == r'ø'
    #                             else temp_df['units'][param] + r'^2'
    #                             for param in self._df.index]
    #     return temp_df.to_markdown()

    def xs(self, *args, **kwargs):
        result = self._df.xs(*args, **kwargs)
        if isinstance(result, pd.Series):
            return self._series_class(data=result, name=self.name, sqrt=self.sqrt, relative=self.relative)
        if isinstance(result, pd.DataFrame):
            return type(self)(data=result, name=self.name, sqrt=self.sqrt,
                              relative=self.relative, verbose=self.verbose)

    def groupby(self, retain_pandas: bool = True, *args, **kwargs):
        result = self._df.groupby(*args, **kwargs)
        if retain_pandas:
            return result
        return type(self)(data=result, name=self.name, sqrt=self.sqrt,
                          relative=self.relative, verbose=self.verbose)

    def mean(self, *args, **kwargs):
        # May resort to implement these one by one (at least the most useful)
        return self._series_class(data=self._df.mean(*args, **kwargs), verbose=self.verbose,
                                  name=self.name, sqrt=self.sqrt, relative=self.relative)

    def pp_mean(self, sort: bool = False):
        '''
        Per-parameter mean
        '''
        return type(self)(data=self._df.groupby(level='parameters', sort=sort).mean(),
                          name=self.name, sqrt=self.sqrt, relative=self.relative, verbose=self.verbose)

    def pp_median(self, sort: bool = False):
        '''
        Per-parameter median
        '''
        return type(self)(data=self._df.groupby(level='parameters', sort=sort).median(),
                          name=self.name, sqrt=self.sqrt, relative=self.relative, verbose=self.verbose)


class SampleDict(SamplesDict):  # TODO: load_from_file/write for core classes. Consider h5py?

    def __init__(self, parameters, name: str = None,
                 _series_class=MSESeries, _dataframe_class=MSEDataFrame, jargon: dict = no_jargon):
        super().__init__({param: 0 for param in parameters}, jargon=jargon)
        self._truth = OrderedDict()
        if name is None:
            self.name = type(self).__name__
        else:
            self.name = name

        self._series_class = _series_class
        self._dataframe_class = _dataframe_class

    def __setitem__(self, key, value):
        SamplesDict.__setitem__(self, key, value)

    @property
    def truth(self):
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
        raise NotImplementedError

    @classmethod
    def from_samplesdict(cls, samplesdict: SamplesDict, name: str = None,
                         _series_class=MSESeries, _dataframe_class=MSEDataFrame, jargon: dict = no_jargon):
        sdict = cls(samplesdict.parameters, name, _series_class, _dataframe_class, jargon)
        for param in sdict.parameters:
            sdict[param] = samplesdict[param]
        return sdict

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

    def accuracy_test(self, relative: bool = False, sqrt: bool = True) -> MSESeries | None:
        if len(self.truth) == 0:
            print(f'{PrintStyle.red}{type(self).__name__} has no truth dictionary (reference values) to compare to.')
            print(f'Accuracy test aborted.{PrintStyle.reset}')
            return None
        error = self._series_class(index=self.parameters, name=f'MSE from {self.name}', relative=relative, sqrt=sqrt)
        for param in self.parameters:
            avg = self.mean[param]
            ref = self.truth[param]
            if sqrt:
                error[param] = np.abs(avg - ref)
                if relative:
                    error[param] *= 100 / np.abs(ref)
            else:
                error[param] = (avg - ref) ** 2
                if relative:  # This one I cannot even interpret, might just avoid
                    error[param] *= 100 / ref ** 2
        return error

    def precision_test(self,
                       relative: bool = False,
                       average: bool = False,
                       median: bool = True, **quantile_kwargs
                       ) -> MSESeries | tuple[MSESeries, MSESeries] | tuple[MSESeries, MSESeries, MSESeries]:
        '''

        Gives left and right deviation from median

        Parameters
        ----------
        relative : Whether it gives precision divided by reference
        average: Whether to return the average deviation or both individually
        median: Whether to return the median

        Returns
        -------

        '''

        if median:
            median = self._series_class(index=self.parameters, name=f'Median from {self.name}',
                                        relative=False, sqrt=True)

        if average:
            error = self._series_class(index=self.parameters, name=f'Deviation from {self.name}',
                                       relative=relative, sqrt=True)

            for param in self.parameters:
                summary = self.get_one_dimensional_median_and_error_bar(param, **quantile_kwargs)
                error[param] = (summary.minus + summary.plus) / 2
                if relative:
                    error[param] *= 100 / np.abs(summary.median)

                if median:
                    median[param] = summary.median

            if median:
                return error, median

            return error

        else:
            error_left = self._series_class(index=self.parameters,
                                            name=f'Left deviation from {self.name}', relative=relative, sqrt=True)
            error_right = self._series_class(index=self.parameters,
                                             name=f'Right Deviation from {self.name}', relative=relative, sqrt=True)

            for param in self.parameters:
                summary = self.get_one_dimensional_median_and_error_bar(param, **quantile_kwargs)
                error_left[param] = summary.minus
                error_right[param] = summary.plus
                if relative:
                    error_left[param] *= 100 / np.abs(summary.median)
                    error_right[param] *= 100 / np.abs(summary.median)
                if median:
                    median[param] = summary.median

            if median:
                return error_left, error_right, median
            return error_left, error_right

    def full_test(self, relative: bool = False, average: bool = False, median: bool = True, **quantile_kwargs):
        data = {'accuracy': self.accuracy_test(relative=relative, sqrt=True)}
        if not average:
            precisions = self.precision_test(relative=relative, average=average, median=median, **quantile_kwargs)
            if median:
                data.update({'median': precisions[2]})
            data.update({'precision_left': precisions[0], 'precision_right': precisions[1]})
        elif average and median:
            precision, median = self.precision_test(
                relative=relative, average=average, median=median, **quantile_kwargs)
            data.update({'median': median, 'precision': precision})
        else:
            data.update(
                {'precision': self.precision_test(relative=relative, average=average, median=median, **quantile_kwargs)}
            )
        return self._dataframe_class(data=data, name=f'deviations from {self.name}', sqrt=True, relative=relative)

    def plot(self, *args,
             type: str = 'marginalized_posterior',
             values: bool = True,
             truth_fmt: str = '.2f', **kwargs):
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


class SampleSet(SampleDict):

    def __init__(self, parameters, name: str = None,
                 _series_class=MSESeries, _dataframe_class=MSEDataFrame, jargon: dict = no_jargon):
        super().__init__(parameters, name, _series_class, jargon=jargon)
        # pesummary.SamplesDict requires parameters to be logged in __init__,
        # but I then need to get rid of them to include mine after initialization
        for key in parameters:
            del self[key]

        self._series_class = _series_class
        self._dataframe_class = _dataframe_class
        self.jargon = jargon

    def accuracy_test(self, relative: bool = False, sqrt: bool = True):
        # data = {event: sdict.accuracy_test(sqrt) for event, sdict in self.items()}
        data = OrderedDict()
        with tqdm(total=len(self.keys()), desc=f'Running accuracy test for {self.name}', ncols=100) as p_bar:
            for event, sdict in self.items():
                data[event] = sdict.accuracy_test(relative, sqrt)
                p_bar.update(1)

        return self._dataframe_class(data=pd.DataFrame(data=data).T,
                                     name=f'MSE from {self.name}', sqrt=sqrt, relative=relative)

    def precision_test(self,
                       relative: bool = False,
                       average: bool = True,
                       median: bool = True,
                       **quantile_kwargs):
        # data = {event: sdict.accuracy_test(sqrt) for event, sdict in self.items()}
        data = OrderedDict()
        with tqdm(total=len(self.keys()), desc=f'Running precision test for {self.name}', ncols=100) as p_bar:
            for event, sdict in self.items():
                data[event] = sdict.precision_test(relative, average, median, **quantile_kwargs)
                p_bar.update(1)

        df_median = None
        if median:
            medians = {event: data[event][-1] for event in data.keys()}
            df_median = self._dataframe_class(data=pd.DataFrame(data=medians).T,
                                              name=f'Median from {self.name}', sqrt=True, relative=False)
        if average:
            precision = {event: data[event][0] for event in data.keys()}
            df_precision = self._dataframe_class(data=pd.DataFrame(data=precision).T,
                                                 name=f'Deviation from {self.name}', sqrt=True, relative=relative)
            if median:
                return df_precision, df_median
            return df_precision
        else:
            data_left = {event: data[event][0] for event in data.keys()}
            data_right = {event: data[event][1] for event in data.keys()}

            df_left = self._dataframe_class(data=pd.DataFrame(data=data_left).T,
                                            name=f'Left deviation from {self.name}', sqrt=True, relative=relative)
            df_right = self._dataframe_class(data=pd.DataFrame(data=data_right).T,
                                             name=f'Right deviation from {self.name}', sqrt=True, relative=relative)
            if median:
                return df_left, df_right, df_median
            return df_left, df_right

    def full_test(self,
                  relative: bool = False,
                  average: bool = False,
                  median: bool = True,
                  truth: bool = True,
                  verbose: bool = True,
                  **quantile_kwargs):

        if quantile_kwargs is None:
            quantile_kwargs = {}
        if 'quantiles' not in quantile_kwargs:
            quantile_kwargs['quantiles'] = (0.16, 0.84)

        verbose_bool = verbose

        accuracy_df = self.accuracy_test(relative=relative, sqrt=True)
        data = {'accuracy': accuracy_df.values.flatten()}
        verbose = {'accuracy': '(MSE)', 'parameters': f'({self.name})'}

        if hasattr(self, 'data_name'):
            verbose['events'] = f'({self.data_name})'

        if not average:
            verbose.update({'precision_left': f'({abs(st.norm.ppf(quantile_kwargs["quantiles"][0])):.1f}'
                                              + r'$\sigma$)',
                            'precision_right': f'({abs(st.norm.ppf(quantile_kwargs["quantiles"][0])):.1f}'
                                               + r'$\sigma$)'})
            if median:
                left_df, right_df, median_df = self.precision_test(relative, average, median, **quantile_kwargs)
                data.update({'median': median_df.values.flatten(),
                             'precision_left': left_df.values.flatten(),
                             'precision_right': right_df.values.flatten()})
            else:
                left_df, right_df = self.precision_test(relative, average, median, **quantile_kwargs)
                data.update({'precision_left': left_df.values.flatten(),
                             'precision_right': right_df.values.flatten()})
        else:
            verbose.update({'precision': f'''({((abs(st.norm.ppf(quantile_kwargs["quantiles"][0])) +
                                                 abs(st.norm.ppf(quantile_kwargs["quantiles"][1]))) / 2):.1f}'''
                                         + r'$\sigma$)'})
            if median:
                precision_df, median_df = self.precision_test(relative, average, median, **quantile_kwargs)
                data.update({'median': median_df.values.flatten(),
                             'precision': precision_df.values.flatten()})
            else:
                precision_df = self.precision_test(relative, average, median, **quantile_kwargs)
                data.update({'precision': precision_df.values.flatten()})

        if truth:
            data.update({'truth': [sampledict.truth[param]
                                   for sampledict in self.values() for param in self.parameters]})

        # indexes = [
        #     np.array(accuracy_df.index.repeat(len(self.parameters))),
        #     np.array(self.parameters * accuracy_df.index.size)
        #
        # ]
        if not verbose_bool:
            verbose = None
        indexes = pd.MultiIndex.from_product([accuracy_df.index, self.parameters], names=['events', 'parameters'])
        df = self._dataframe_class(data=pd.DataFrame(data=data, index=indexes),
                                   name=f'deviations from {self.name}', sqrt=True,
                                   relative=relative, verbose=verbose)
        # print(df.verbose)
        cols_to_move = []  # Update later for truth possibility
        if median:
            cols_to_move.append('median')
        if truth:
            cols_to_move.append('truth')
        df._df = df._df[cols_to_move + [col for col in df.columns if col not in cols_to_move]]
        return df

    def __setitem__(self, key, value):
        """ Overrides the SampleDict __setitem__ """
        dict.__setitem__(self, key, value)


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
        fig = super().plot(type=type, *args, **kwargs)

        if type == 'corner' and medians is not None:
            import corner
            from ._pesummary_dependencies.configuration import colorcycle

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

                if colors[n] == 'r':
                    color = 'darkred'
                else:
                    color = colors[n]

                corner.overplot_lines(fig, median_list, color=color)
                corner.overplot_points(fig, median_list[None], color=color,
                                       marker=kwargs.get('median_marker', "s"))
        return fig


# Could be moved to a config object later
def set_hist_style(style: str = 'bilby') -> dict:
    available_styles = ['bilby', ]
    if style == 'bilby':
        return {
            # For plt.hist
            'bins': 50,
            'histtype': 'step',
            'density': True,
            'color': 'tab:blue',

            # For various axvlines
            'truth_color': 'orange',
            'truth_format': '-',
            'average_color': 'C0',
            'average_format': '-',
            'errorbar_color': 'C0',
            'errorbar_format': '--'

        }
    elif style == 'deserted':
        return {
            # For plt.hist
            'bins': 50,
            'histtype': 'step',
            'density': True,
            'color': 'khaki',

            # For various axvlines
            'truth_color': 'orange',
            'truth_format': '-',
            'average_color': 'gold',
            'average_format': '-',
            'errorbar_color': 'gold',
            'errorbar_format': '--'

        }
    else:
        raise ValueError(f'style \'{style}\' not understood. This are the available styles: {available_styles}')
