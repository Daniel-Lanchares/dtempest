from functools import partial

from .config import cbc_jargon
import dtempest.core.conversion_utils as core

make_image = partial(core.make_image, jargon=cbc_jargon)
extract_parameters = partial(core.extract_parameters, jargon=cbc_jargon)
plot_image = partial(core.plot_image, jargon=cbc_jargon)
plot_images = partial(core.plot_images, jargon=cbc_jargon)