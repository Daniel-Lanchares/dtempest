"""
Classes and utilities for GW-adapted sampling.
"""
import numpy as np
from pathlib import Path
from functools import partialmethod


from dtempest.core._pesummary_dependencies.samples_dict import MultiAnalysisSamplesDict
from dtempest.core.sample_utils import SampleDict, SampleSet, MSEDataFrame, MSESeries, ComparisonSampleDict
from .config import cbc_jargon



class CBCMSESeries(MSESeries):
    __init__ = partialmethod(MSESeries.__init__, jargon=cbc_jargon)


class CBCMSEDataFrame(MSEDataFrame):
    __init__ = partialmethod(MSEDataFrame.__init__, jargon=cbc_jargon, _series_class=CBCMSESeries)


class CBCSampleDict(SampleDict):
    __init__ = partialmethod(SampleDict.__init__,
                             jargon=cbc_jargon, _series_class=CBCMSESeries, _dataframe_class=CBCMSEDataFrame)

    @classmethod
    def from_file(cls, filename: Path | str, **kwargs):
        """Initialize the SamplesDict class with the contents of a (GW) result file

        Parameters
        ----------
        filename: str
            path to the result file you wish to load.
        **kwargs: dict
            all kwargs are passed to the pesummary.io.read function
        """
        from pesummary.io import read

        return read(filename, **kwargs).samples_dict

    def write(self, **kwargs):
        """Save the stored posterior samples to file

        Parameters
        ----------
        **kwargs: dict, optional
            all additional kwargs passed to the pesummary.io.write function
        """
        from pesummary.io import write
        write(self.parameters, self.samples.T, **kwargs)

    from_samplesdict = partialmethod(SampleDict.from_samplesdict,
                                     jargon=cbc_jargon, _series_class=CBCMSESeries, _dataframe_class=CBCMSEDataFrame)

    def default_bounds(self, parameters=None, comparison=False):
        if parameters is None:
            parameters = self.parameters
        return _default_bounds(self, parameters, comparison)

    @property
    def plotting_map(self):
        existing = super(CBCSampleDict, self).plotting_map
        modified = existing.copy()
        modified.update(
            {
                "marginalized_posterior": self._marginalized_posterior,
                "skymap": self._skymap,
                # "bilby_skymap": self._bilby_skymap,
                "hist": self._marginalized_posterior,
                "corner": self._corner,
                # "spin_disk": self._spin_disk,
                "2d_kde": self._2d_kde,
                "triangle": self._triangle,
                "reverse_triangle": self._reverse_triangle,
            }
        )
        return modified

    def generate_all_posterior_samples(self, function=None, **kwargs):
        """Convert samples stored in the SamplesDict according to a conversion
        function

        Parameters
        ----------
        function: func, optional
            function to use when converting posterior samples. Must take a
            dictionary as input and return a dictionary of converted posterior
            samples. Default `pesummary.gw.conversions.convert
        **kwargs: dict, optional
            All additional kwargs passed to function
        """
        if function is None:
            from pesummary.gw.conversions import convert
            function = convert
        _samples = self.copy()
        _keys = list(_samples.keys())
        kwargs.update({"return_dict": True})
        out = function(_samples, **kwargs)
        if kwargs.get("return_kwargs", False):
            converted_samples, extra_kwargs = out
        else:
            converted_samples, extra_kwargs = out, None
        for key, item in converted_samples.items():
            if key not in _keys:
                self[key] = item
        return extra_kwargs

    def _skymap(self, **kwargs):
        """Wrapper for the `pesummary.gw.plots.plot._ligo_skymap_plot`
        function

        Parameters
        ----------
        **kwargs: dict
            All kwargs are passed to the `_ligo_skymap_plot` function
        """
        from pesummary.gw.plots.plot import _ligo_skymap_plot

        if "luminosity_distance" in self.keys():
            dist = self["luminosity_distance"]
        else:
            dist = None

        return _ligo_skymap_plot(self["ra"], self["dec"], dist=dist, **kwargs)


    def _spin_disk(self, **kwargs):
        """Wrapper for the `pesummary.gw.plots.publication.spin_distribution_plots`
        function
        """
        from pesummary.gw.plots.publication import spin_distribution_plots

        required = ["a_1", "a_2", "cos_tilt_1", "cos_tilt_2"]
        if not all(param in self.keys() for param in required):
            raise ValueError(
                "The spin disk plot requires samples for the following "
                "parameters: {}".format(", ".join(required))
            )
        samples = [self[param] for param in required]
        return spin_distribution_plots(required, samples, None, **kwargs)

    def _2d_kde(self, parameters: list, module="gw", **kwargs):
        """Wrapper for the `pesummary.gw.plots.publication.twod_contour_plot` or
        `pesummary.core.plots.publication.twod_contour_plot` function

        Parameters
        ----------
        parameters: list
            list of length 2 giving the parameters you wish to plot
        module: str, optional
            module you wish to use for the plotting
        **kwargs: dict, optional
            all additional kwargs are passed to the `twod_contour_plot` function
        """
        from pesummary.gw.plots.publication import twod_contour_plots
        if module == "gw":
            return twod_contour_plots(
                parameters, [[self[parameters[0]], self[parameters[1]]]],
                [None], {
                    parameters[0]: self.latex_labels[parameters[0]],
                    parameters[1]: self.latex_labels[parameters[1]]
                }, **kwargs
            )
        else:
            raise NotImplementedError
        # return getattr(_module, "twod_contour_plot")(
        #     self[parameters[0]], self[parameters[1]],
        #     xlabel=self.latex_labels[parameters[0]],
        #     ylabel=self.latex_labels[parameters[1]], **kwargs
        # )

    def _triangle(self, parameters: list, module="gw", **kwargs):
        """Wrapper for the `pesummary.core.plots.publication.triangle_plot`
        function

        Parameters
        ----------
        parameters: list
            list of parameters they wish to study
        **kwargs: dict
            all additional kwargs are passed to the `triangle_plot` function
        """
        from pesummary.gw.plots.publication import triangle_plot
        if module == "gw":
            kwargs["parameters"] = parameters
        else:
            raise NotImplementedError
        return triangle_plot(
            (self[parameters[0]]), (self[parameters[1]]),
            xlabel=self.latex_labels[parameters[0]],
            ylabel=self.latex_labels[parameters[1]], **kwargs
        )


class CBCSampleSet(SampleSet):
    __init__ = partialmethod(SampleSet.__init__,
                             _series_class=CBCMSESeries, _dataframe_class=CBCMSEDataFrame, jargon=cbc_jargon)


class CBCComparisonSampleDict(ComparisonSampleDict):  # Subclassing two classes is problematic. Careful

    @classmethod
    def from_files(cls, filenames, **kwargs):
        """Initialize the MultiAnalysisSamplesDict class with the contents of
        multiple result files

        Parameters
        ----------
        filenames: dict
            dictionary containing the path to the result file you wish to load
            as the item and a label associated with each result file as the key.
            If you are providing one or more PESummary metafiles, the key
            is ignored and labels stored in the metafile are used.
        **kwargs: dict
            all kwargs are passed to the pesummary.io.read function
        """
        from pesummary.io import read

        samples = {}
        for label, filename in filenames.items():
            _kwargs = kwargs
            if label in kwargs.keys():
                _kwargs = kwargs[label]
            _file = read(filename, **_kwargs)
            _samples = _file.samples_dict
            if isinstance(_samples, MultiAnalysisSamplesDict):
                _stored_labels = _samples.keys()
                cond1 = any(
                    _label in filenames.keys() for _label in _stored_labels if
                    _label != label
                )
                cond2 = any(
                    _label in samples.keys() for _label in _stored_labels
                )
                if cond1 or cond2:
                    raise ValueError(
                        "The file '{}' contains the labels: {}. The "
                        "dictionary already contains the labels: {}. Please "
                        "provide unique labels for each dataset".format(
                            filename, ", ".join(_stored_labels),
                            ", ".join(samples.keys())
                        )
                    )
                samples.update(_samples)
            else:
                if label in samples.keys():
                    raise ValueError(
                        "The label '{}' has alreadt been used. Please select "
                        "another label".format(label)
                    )
                samples[label] = _samples
        return cls(samples)

    def write(self, labels=None, **kwargs):
        """Save the stored posterior samples to file

        Parameters
        ----------
        labels: list, optional
            list of analyses that you wish to save to file. Default save all
            analyses to file
        **kwargs: dict, optional
            all additional kwargs passed to the pesummary.io.write function
        """
        if labels is None:
            labels = self.labels
        elif not all(label in self.labels for label in labels):
            for label in labels:
                if label not in self.labels:
                    raise ValueError(
                        "Unable to find analysis: '{}'. The list of "
                        "available analyses are: {}".format(
                            label, ", ".join(self.labels)
                        )
                    )
        for label in labels:
            self[label].write(**kwargs)

    def default_bounds(self, parameters=None, analysis: str = None, comparison=False):
        from pesummary.gw.plots.plot import _return_bounds
        from collections import OrderedDict

        _samples = {label: self[label] for label in self.labels}
        _parameters = parameters
        if _parameters is not None:
            parameters = [
                param for param in _parameters if all(
                    param in posterior for posterior in _samples.values()
                )
            ]
            if not len(parameters):
                raise ValueError(
                    "None of the chosen parameters are in all of the posterior "
                    "samples tables. Please specify analysis or choose other parameters to bound"
                )

        else:
            _parameters = [list(_samples.keys()) for _samples in _samples.values()]
            parameters = [
                i for i in _parameters[0] if all(i in _params for _params in _parameters)
            ]

        if analysis is not None:
            return _default_bounds(self[analysis], parameters, comparison)

        else:  # Take minimum low and maximum high of all analysis
            bounds = OrderedDict()
            for param in parameters:
                lows, highs = [], []
                for label in self.labels:
                    low, high = _return_bounds(param, self[label][param], comparison)
                    lows.append(low), highs.append(high)
                low = nan_check(lows, 'min')
                high = nan_check(highs, 'max')

                bounds[param] = {'xlow': low, 'xhigh': high}
            # from pprint import pprint
            # pprint(bounds)
            return bounds


def _default_bounds(samples, parameters, comparison=False):
    from pesummary.gw.plots.plot import _return_bounds
    from collections import OrderedDict
    bounds = OrderedDict()
    for param in parameters:
        low, high = _return_bounds(param, samples[param], comparison)
        bounds[param] = {'xlow': low, 'xhigh': high}
    return bounds


def nan_check(vector, f_type: str = 'max'):
    """

    Parameters
    ----------
    vector : list of values to filter
    f_type : take minimum or maximum

    Returns Max/min of list, or None if not available
    -------

    """
    mapping = {'min': np.nanmin, 'max': np.nanmax}

    if all(v is None for v in vector):
        extreme = None
    else:
        extreme = mapping[f_type](np.array(vector, dtype=np.float64))
    return extreme
