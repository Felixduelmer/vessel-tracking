from .VesNet import *
from .UNet import *
from .ButterflyNet import *
from .convgru_doppler import *
from .ButterflyNet_Simple import *
from .ButterflyNet_Simple_RNN import *
from .ButterflyNetRNNDoubleEncoder import *
from .UNet_RNN import *
from .ButterflyNet_Duplex_RNN import *


def get_network(name, in_channels=2, feature_scale=4,
                nonlocal_mode='embedded_gaussian', attention_dsample=(2, 2, 2),
                aggregation_mode='concat'):
    model = _get_model_instance(name)

    if name in ['unet']:
        model = model(in_channels=in_channels, feature_scale=feature_scale, )
    elif name in ['vesnet']:
        model = model(in_channels=in_channels, feature_scale=feature_scale, )
    elif name in ['butterflynet']:
        model = model(in_channels=in_channels, feature_scale=feature_scale, )
    elif name in ['convgrunet']:
        model = model(in_channels=in_channels, feature_scale=feature_scale, )
    elif name in ['butterflynetsimple']:
        model = model(in_channels=in_channels, feature_scale=feature_scale, )
    elif name in ['butterflynetsimplernn']:
        model = model(in_channels=in_channels, feature_scale=feature_scale, )
    elif name in ['butterflynernndoubleencoder']:
        model = model(in_channels=in_channels, feature_scale=feature_scale, )
    elif name in ['unetrnn']:
        model = model(in_channels=in_channels, feature_scale=feature_scale, )
    elif name in ['butterflynetduplexrnn']:
        model = model(in_channels=in_channels, feature_scale=feature_scale, )
    else:
        raise 'Model {} not available'.format(name)

    return model


def _get_model_instance(name):
    return {
        'vesnet': VesNet,
        'unet': UNet,
        'butterflynet': ButterflyNet,
        'convgrunet': ConvGruNet,
        'butterflynetsimple': ButterflyNetSimple,
        'butterflynetsimplernn': ButterflyNetSimpleRNN,
        'butterflynernndoubleencoder': ButterflyNetRNNDoubleEncoder,
        'unetrnn': UNetRNN,
        'butterflynetduplexrnn': ButterflyNetDuplexRNN

    }[name]
