from .VesNet import *
from .UNet import *


def get_network(name, in_channels=2,feature_scale=4,
                nonlocal_mode='embedded_gaussian', attention_dsample=(2, 2, 2),
                aggregation_mode='concat'):
    model = _get_model_instance(name)

    if name in ['unet']:
        model = model()
    elif name in ['vesnet']:
        model = model(in_channels=in_channels, feature_scale=feature_scale,)
    else:
        raise 'Model {} not available'.format(name)

    return model


def _get_model_instance(name):
    return {
        'vesnet': VesNet,
        'unet': UNet,

    }[name]
