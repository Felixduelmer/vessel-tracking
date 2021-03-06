import math
import torch
import torch.nn as nn
from .utils import unetConv2, unetUp2Cat
import torch.nn.functional as F
from models.networks_other import init_weights
from models.modules.convgru import ConvGRUCell
from models.networks.utils import GridAttentionBlock2D
from torch.autograd import Variable
import numpy as np


class ButterflyNetRNNDoubleEncoder(nn.Module):

    def __init__(self, feature_scale=8, n_classes=1, is_deconv=True, in_channels=1, is_batchnorm=True):
        super(ButterflyNetRNNDoubleEncoder, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        self.filters = filters

        # downsampling bmode
        self.conv1_bmode = unetConv2(2, filters[0], self.is_batchnorm)
        self.maxpool1_bmode = nn.MaxPool2d(kernel_size=2)

        self.conv2_bmode = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2_bmode = nn.MaxPool2d(kernel_size=2)

        self.conv3_bmode = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3_bmode = nn.MaxPool2d(kernel_size=2)

        self.conv4_bmode = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4_bmode = nn.MaxPool2d(kernel_size=2)

        self.conv5_bmode = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # downsampling doppler
        self.conv1_doppler = unetConv2(1, filters[0], self.is_batchnorm)
        self.maxpool1_doppler = nn.MaxPool2d(kernel_size=2)

        self.conv2_doppler = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2_doppler = nn.MaxPool2d(kernel_size=2)

        self.conv3_doppler = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3_doppler = nn.MaxPool2d(kernel_size=2)

        self.conv4_doppler = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4_doppler = nn.MaxPool2d(kernel_size=2)

        self.conv5_doppler = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # conv gated recurrent units
        self.convGRU_bmode = ConvGRUCell(filters[4], filters[4], 3, False)
        self.convGRU_doppler = ConvGRUCell(filters[4], filters[4], 3, False)
        self.center = unetConv2((filters[4] + filters[4]), filters[4], self.is_batchnorm)

        # attention units

        self.attention4_doppler = GridAttentionBlock2D(filters[3], filters[4])
        self.attention4_bmode = GridAttentionBlock2D(filters[3], filters[4])

        self.attention3_doppler = GridAttentionBlock2D(filters[2], filters[3])
        self.attention3_bmode = GridAttentionBlock2D(filters[2], filters[3])

        self.attention2_doppler = GridAttentionBlock2D(filters[1], filters[2])
        self.attention2_bmode = GridAttentionBlock2D(filters[1], filters[2])

        self.attention1_doppler = GridAttentionBlock2D(filters[0], filters[1])
        self.attention1_bmode = GridAttentionBlock2D(filters[0], filters[1])

        # upsampling
        self.up_concat4 = unetUp2Cat(filters[4], filters[3], self.is_deconv, self.is_batchnorm)
        self.up_concat3 = unetUp2Cat(filters[3], filters[2], self.is_deconv, self.is_batchnorm)
        self.up_concat2 = unetUp2Cat(filters[2], filters[1], self.is_deconv, self.is_batchnorm)
        self.up_concat1 = unetUp2Cat(filters[1], filters[0], self.is_deconv, self.is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs, hidden):
        new_hidden = [None] * 2
        # bmode
        conv1_bmode = self.conv1_bmode(inputs[:, [0, 1], :, :])
        maxpool1_bmode = self.maxpool1_bmode(conv1_bmode)

        conv2_bmode = self.conv2_bmode(maxpool1_bmode)
        maxpool2_bmode = self.maxpool2_bmode(conv2_bmode)

        conv3_bmode = self.conv3_bmode(maxpool2_bmode)
        maxpool3_bmode = self.maxpool3_bmode(conv3_bmode)

        conv4_bmode = self.conv4_bmode(maxpool3_bmode)
        maxpool4_bmode = self.maxpool4_bmode(conv4_bmode)

        conv5_bmode = self.conv5_bmode(maxpool4_bmode)
        center_bmode = self.convGRU_bmode(conv5_bmode, hidden[0])
        new_hidden[0] = center_bmode
        # doppler
        conv1_doppler = self.conv1_doppler(inputs[:, [1], :, :])
        maxpool1_doppler = self.maxpool1_doppler(conv1_doppler)

        conv2_doppler = self.conv2_doppler(maxpool1_doppler)
        maxpool2_doppler = self.maxpool2_doppler(conv2_doppler)

        conv3_doppler = self.conv3_doppler(maxpool2_doppler)
        maxpool3_doppler = self.maxpool3_doppler(conv3_doppler)

        conv4_doppler = self.conv4_doppler(maxpool3_doppler)
        maxpool4_doppler = self.maxpool4_doppler(conv4_doppler)

        conv5_doppler = self.conv5_doppler(maxpool4_doppler)
        center_doppler = self.convGRU_doppler(conv5_doppler, hidden[1])
        new_hidden[1] = center_doppler
        # center
        center = self.center(torch.cat([center_bmode, center_doppler], 1))

        # attention units

        attention4_doppler = self.attention4_doppler(conv4_doppler, center)[0]
        attention4_bmode = self.attention4_bmode(conv4_bmode, center)[0]
        up4 = self.up_concat4(torch.cat([attention4_doppler, attention4_bmode], 1), center)

        attention3_doppler = self.attention3_doppler(conv3_doppler, up4)[0]
        attention3_bmode = self.attention3_bmode(conv3_bmode, up4)[0]
        up3 = self.up_concat3(torch.cat([attention3_doppler, attention3_bmode], 1), up4)

        attention2_doppler = self.attention2_doppler(conv2_doppler, up3)[0]
        attention2_bmode = self.attention2_bmode(conv2_bmode, up3)[0]
        up2 = self.up_concat2(torch.cat([attention2_doppler, attention2_bmode], 1), up3)

        attention1_doppler = self.attention1_doppler(conv1_doppler, up2)[0]
        attention1_bmode = self.attention1_bmode(conv1_bmode, up2)[0]
        up1 = self.up_concat1(torch.cat([attention1_doppler, attention1_bmode], 1), up2)
        final = self.final(up1)

        return final, new_hidden

    @staticmethod
    def apply_sigmoid(pred):
        log_p = torch.sigmoid(pred)

        return log_p

    def init_hidden(self, batch_size, input_size):
        hidden_states = []
        for i in range(2):
            # number of hidden layers comes up first
            hidden_states.append(Variable(torch.zeros(batch_size, self.filters[4], 20, 20, requires_grad=True,
                                                      device=torch.device('cuda'))))
        return hidden_states
