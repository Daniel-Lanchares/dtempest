"""
Configuration and jargon for (current) gravitational wave analysis.
"""
from dtempest.core.config import Jargon
from .parameters import redef_dict
from pesummary.gw.plots.latex_labels import GWlatex_labels



def gw_title_maker(data):
    return f'{data["id"]} Q-Transform image\n(RGB = (L1, H1, V1))'


cbc_jargon = Jargon({
    'parameters': 'parameters',
    'image': 'q-transforms',
    'R': 'L1',
    'G': 'H1',
    'B': 'V1',

    'param_pool': redef_dict,
    'labels': GWlatex_labels,  # label format: $alias [unit]$

    'default_title_maker': gw_title_maker

})

