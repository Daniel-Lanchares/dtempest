"""This module adds catalog related functionality to test models on real data"""
import h5py
import numpy as np
from collections.abc import Callable, Iterable

from gwosc.api.v2 import fetch_event_version
from gwpy.timeseries import TimeSeries
from bilby.gw.detector import InterferometerList

from dtempest.gw.generation.utils import ifo_q_transform

full_names = {
    'GW190412': 'GW190412_053044',
    'GW190425': 'GW190425_081805',
    'GW190521': 'GW190521_030229',
    'GW190814': 'GW190814_211039'
}

EventNames = Iterable[str]
Message = Callable[[str], str]
BarredEventDict = dict[EventNames, Message]

def check_barred_events(name: str, exceptions: BarredEventDict = None) -> bool:
    """Checks for barred events, printing a message based on the exceptions dict"""
    if exceptions is None:
        exceptions = {}
    for barred_events, message in exceptions.items():
        if name in barred_events:
            print(message(name))
            return True
    else:
        return False

def fetch_ifo_data(ifo, start, duration, buffer_time, **kwargs) -> TimeSeries:
    """Thin wrapper around the gwpy.timeseries.TimeSeries.fetch_open_data to suit our needs"""
    end = start + duration + buffer_time
    start -= buffer_time
    return TimeSeries.fetch_open_data(ifo, start, end, **kwargs)

class Event:
    def __init__(self,
                 name: str,
                duration: float,
                img_res: tuple[int, int],
                sampling_frequency: float,
                qtrans_kwargs: dict,
                ifolist: list[str],

                catalog: str = None,
                version: str =None,
                **unused_kwargs
                ):
        """
        Real life event recorded in a catalog

        Parameters
        ----------
        name : name of the event
        duration : duration of the data to fetch
        img_res : resolution of the resulting image
        sampling_frequency : sampling frequency of the data
        qtrans_kwargs : kwargs for the QTransform
        ifolist : list of interferometers
        catalog : catalog to search in
        version : version of the catalog to use
        unused_kwargs : allows the user to fetch an event by unpacking metadata dictionaries without KeyErrors
        """
        
        # Type conversions avoid memory segmentation issues (C doing C things)
        self.duration = float(duration)
        self.image_window = tuple(qtrans_kwargs.pop("outseg"))
        self.img_res = tuple(img_res)
        self.frange = tuple(qtrans_kwargs.pop("frange"))
        self.sample_rate = float(sampling_frequency)
        self.extra_qtrans_kwargs = qtrans_kwargs


        self.event = fetch_event_version(name, catalog, version)

        assert set(ifolist) == set(self.event["detectors"]), \
            f"This event has a network configuration {self.event['detectors']} for which this model was not trained {ifolist}."
        
        self.ifos = InterferometerList(
            [ifo for ifo in ifolist if ifo in self.event["detectors"]]
             )
        
        self.start_time = self.event["gps"] - self.duration/2
        for ifo in self.ifos:
            ifo.strain_data.set_from_gwpy_timeseries(fetch_ifo_data(ifo.name, self.start_time, self.duration, buffer_time=0))

        self.data = np.array([ifo_q_transform(np.fft.irfft(ifo.whitened_frequency_domain_strain),
                                            resol=self.img_res,
                                            duration=self.duration,
                                            sampling_frequency=self.sample_rate,
                                            outseg=self.image_window,
                                            frange=self.frange,
                                            **self.extra_qtrans_kwargs) for ifo in self.ifos])

class Injection:
    def __init__(self, n, dataset_path, kind):
        """
        Injection class with similar behaviour to Event

        Parameters
        ----------
        n : number of the injection within the dataset
        dataset_path : path to the dataset
        kind : type of dataset (mainly training/validation)
        """
        self.dataset_path = dataset_path
        self.kind = kind
        self.n = n
        with h5py.File(self.dataset_path, 'r') as h_file:
            self.data, self.labels = h_file[self.kind]['images'][self.n], h_file[self.kind]['labels'][self.n]
