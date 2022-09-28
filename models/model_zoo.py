"""Model store which handles pretrained models """

from .pspnet import *
from .deeplabv3 import *
from .deeplabv3_mobile import *
from .psp_mobile import *
from .segformer import *


__all__ = ['get_segmentation_model']


def get_segmentation_model(model, **kwargs):
    models = {
        'psp': get_psp,
        'deeplabv3': get_deeplabv3,
        'deeplab_mobile': get_deeplabv3_mobile,
        'psp_mobile': get_psp_mobile,
        'segformer': get_segformer,
    }
    return models[model](**kwargs)
