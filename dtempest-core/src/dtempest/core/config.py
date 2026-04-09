from typing import TypedDict, Callable, Any
Title_func = Callable[[Any], str]
Pool_map = dict[str, Callable[[float,...], float]]
Label_map = dict[str, str]

"""
Configuration of the core module. Explicit jargon declarations.
"""

def default_title_maker(data)->str:
    """
    Convenience function for purpose-agnostic titles of images. Meant only as placeholder.
    """
    return 'RGB image'

Jargon = TypedDict('Jargon', {'parameters': str, 'image': str,
                              'R': str, 'G': str, 'B': str,
                              'param_pool': Pool_map, 'labels': Label_map, 'default_title_maker': Title_func})

no_jargon = Jargon(parameters='parameters',
                   image='image',
                   R='R',
                   G='G',
                   B='B',
                   param_pool={},
                   labels={},
                   default_title_maker=default_title_maker)
