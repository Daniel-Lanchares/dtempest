from functools import partialmethod

from dtempest.core import Estimator

from .config import cbc_jargon
from .sample_utils import CBCSampleSet, CBCSampleDict, CBCMSEDataFrame, CBCMSESeries

class_dict = {
    'sample_set': CBCSampleSet,
    'sample_dict': CBCSampleDict,
    'data_frame': CBCMSEDataFrame,
    'series': CBCMSESeries
}


class CBCEstimator(Estimator):

    __init__ = partialmethod(Estimator.__init__, jargon=cbc_jargon)
    sample_dict = partialmethod(Estimator.sample_dict, _class_dict=class_dict)
    sample_set = partialmethod(Estimator.sample_set, _class_dict=class_dict)
