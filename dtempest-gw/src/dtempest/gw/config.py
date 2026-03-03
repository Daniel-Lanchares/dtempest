from dtempest.core.config import Jargon
from .parameters import redef_dict
from pesummary.gw.plots.latex_labels import GWlatex_labels

"""
Configuration of the gw module. Explicit jargon declarations.
"""


def gw_title_maker(event):
    return f'{event["name"]} Q-Transform image\n(RGB = (L1, H1, V1))'

cbc_jargon = Jargon(parameters="parameters",
                    image="q-transforms",
                    R="L1",
                    G="H1",
                    B="V1",
                    param_pool=redef_dict,
                    labels=GWlatex_labels,
                    default_title_maker=gw_title_maker)