from .VesNet import *


def get_network(name, n_classes, in_channels=3, feature_scale=4, tensor_dim='2D',
                nonlocal_mode='embedded_gaussian', attention_dsample=(2, 2, 2),
                aggregation_mode='concat'):
    model = _get_model_instance(name, tensor_dim)

    if name in ['vesnet']:
        model = model()
    else:
        raise 'Model {} not available'.format(name)

    return model


def _get_model_instance(name, tensor_dim):
    return {
        'vesnet': {'2D': VesNet},
    }[name][tensor_dim]
