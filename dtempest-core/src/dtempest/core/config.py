# from .model import Estimator

def default_title_maker(data):
    return 'RGB image'


_no_jargon = {
    'parameters': 'parameters',
    'image': 'image',
    'R': 'R',
    'G': 'G',
    'B': 'B',

    'param_pool': None,
    'labels': None,  # label format: $alias [unit]$

    'default_title_maker': default_title_maker
}


class Jargon(dict):  # TODO: add key enforcement.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self:
            super().__init__(_no_jargon)


no_jargon = Jargon(_no_jargon)
