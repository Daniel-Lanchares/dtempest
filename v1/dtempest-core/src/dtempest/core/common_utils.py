"""
Miscellaneous utilities.
"""
import pandas as pd


class PrintStyle:
    black = '\033[30m'
    red = '\033[31m'
    green = '\033[32m'
    orange = '\033[33m'
    blue = '\033[34m'
    purple = '\033[35m'
    cyan = '\033[36m'
    lightgrey = '\033[37m'
    darkgrey = '\033[90m'
    lightred = '\033[91m'
    lightgreen = '\033[92m'
    yellow = '\033[93m'
    lightblue = '\033[94m'
    pink = '\033[95m'
    lightcyan = '\033[96m'

    reset = '\033[0m'
    bold = '\033[01m'
    disable = '\033[02m'
    underline = '\033[04m'
    reverse = '\033[07m'
    strikethrough = '\033[09m'
    invisible = '\033[08m'


def identity(x):
    """
    Naive identity function. Used as default pre-process function.

    Parameters
    ----------
    x :
        Any possible argument

    Returns
    -------
    out :
        The passed argument itself

    """
    return x


def get_extractor(name: str):
    """
    Helper function to obtain necessary objects to construct a pre-trained feature extractor from
    the 'torchvision.models' module.

    Parameters
    ----------
    name :
        Name of the function implementing the architecture.

    Returns
    -------
    out :
        (model, pre-trained weights, pre-processing function)

    """
    import torchvision.models as models
    model = getattr(models, name)
    weights = getattr(getattr(models, f'{models_dict[name]}_Weights'), 'DEFAULT')
    pre_process = weights.transforms(antialias=True)  # True for internal compatibility reasons

    return model, weights, pre_process


models_dict = {
    'resnet18': 'ResNet18'
}


def handle_multi_index_format(temp_df: pd.DataFrame,
                              # mask_type: str = 'events',
                              show_reset_index: bool = False,
                              **format_kwargs) -> tuple[pd.DataFrame, dict]:
    """
    Deals with duplication of event names when printing Multi-indexed Dataframes (or potentially Series).

    Updated and adapted from answer at
    https://stackoverflow.com/questions/75575084/transform-a-pandas-multiindex-to-a-single-index-using-indention

    Parameters
    ----------
    temp_df :
        Dataframe to use for format. SHOULD BE AN OBJECT MEANT FOR FORMATTING ONLY, copied from original.
    # mask_type :
        Whether to mask repeating events or parameters.
    show_reset_index :
        Whether to keep or discard default indexes.
    format_kwargs :
        kwargs to pass to format function (to_markdown, for instance). Either returns them unchanged or
        with index value overridden.

    Returns
    -------
    out :
        (DataFrame_like, format kwargs)

    """
    # we reset the index as mentioned above
    temp_df.reset_index(inplace=True)

    # then find all the duplicates in the zeroth level
    mask = temp_df.events.duplicated().values
    print(mask)

    # and we remove the duplicates
    temp_df.loc[mask, ('events',)] = ''
    if not mask.any():
        mask = temp_df.parameters.duplicated().values
        print(mask)
        temp_df.loc[mask, ('parameters',)] = ''

    if not show_reset_index:
        # Override index specification to avoid showing index 0,1,2...
        format_kwargs['index'] = False

    return temp_df, format_kwargs


def merge_headers(string_table: str):
    """
    Final stop of latex table custom formatting

    Parameters
    ----------
    string_table :
        Latex table as a continuous string.

    Returns
    -------
    out :
        The string with merged headers (index name should now show next to column names)

    """
    rows = string_table.split(r' \\')
    problem_name = rows.pop(1).split('&')[0]
    rows[0] = rows[0].replace(r'\toprule', r'\toprule' + problem_name)

    return r' \\'.join(rows)


def is_documented_by(original):
  def wrapper(target):
    target.__doc__ = original.__doc__
    return target
  return wrapper