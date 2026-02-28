from functools import partialmethod

from dtempest.core import Estimator

from .config import cbc_jargon
from .sampling import CBCSampleDict

class_dict = {
    'sample_dict': CBCSampleDict,
}


class CBCEstimator(Estimator):

    __init__ = partialmethod(Estimator.__init__, jargon=cbc_jargon)
    sample_dict = partialmethod(Estimator.sample_dict, _class_dict=class_dict)
