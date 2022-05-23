import math
import torch
import torch.nn as nn
from .utils import unetConv2, unetUp2
import torch.nn.functional as F
from models.networks_other import init_weights
from models.modules.convgru import ConvGRUCell
from models.networks.utils import GridAttentionBlock2D
from torch.autograd import Variable
import numpy as np


class ButterflyNetSimple(nn.Module):

    def __init__(self, feature_scale=8, n_classes=1, is_deconv=True, in_channels=1, is_batchnorm=True):
        super(ButterflyNetSimple, self).__init__()
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

        self.center_concat = unetConv2(filters[4]*2, filters[4], self.is_batchnorm)
        # conv gated recurrent units
        # self.convGRU_bmode = ConvGRUCell(filters[3], filters[3], 3, False)
        # self.convGRU_doppler = ConvGRUCell(filters[3], filters[3], 3, False)
        # self.center = nn.Linear((filters[3] + filters[3]) * 20 * 20, filters[4] * 20 * 20)

        # attention units

        # # self.attention4_doppler = GridAttentionBlock2D(filters[3], filters[4])
        # self.attention4_bmode = GridAttentionBlock2D(filters[3], filters[4])
        #
        # # self.attention3_doppler = GridAttentionBlock2D(filters[2], filters[3])
        # self.attention3_bmode = GridAttentionBlock2D(filters[2], filters[3])
        #
        # # self.attention2_doppler = GridAttentionBlock2D(filters[1], filters[2])
        # self.attention2_bmode = GridAttentionBlock2D(filters[1], filters[2])
        #
        # # self.attention1_doppler = GridAttentionBlock2D(filters[0], filters[1])
        # self.attention1_bmode = GridAttentionBlock2D(filters[0], filters[1])

        # upsampling

        # self.conv4 = unetConv2(filters[3], filters[4], self.is_batchnorm, self.is_deconv)
        # self.conv3 = unetConv2(filters[2], filters[2], self.is_batchnorm, self.is_deconv)
        # self.conv2 = unetConv2(filters[1], filters[1], self.is_batchnorm, self.is_deconv)
        # self.conv1 = unetConv2(filters[0], filters[0], self.is_batchnorm, self.is_deconv)

        self.up_concat4 = unetUp2(filters[4], filters[3], self.is_deconv, self.is_batchnorm)
        self.up_concat3 = unetUp2(filters[3], filters[2], self.is_deconv, self.is_batchnorm)
        self.up_concat2 = unetUp2(filters[2], filters[1], self.is_deconv, self.is_batchnorm)
        self.up_concat1 = unetUp2(filters[1], filters[0], self.is_deconv, self.is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
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

        center_bmode = self.conv5_bmode(maxpool4_bmode)

        conv1_doppler = self.conv1_doppler(inputs[:, [1], :, :])
        maxpool1_doppler = self.maxpool1_doppler(conv1_doppler)

        conv2_doppler = self.conv2_doppler(maxpool1_doppler)
        maxpool2_doppler = self.maxpool2_doppler(conv2_doppler)

        conv3_doppler = self.conv3_doppler(maxpool2_doppler)
        maxpool3_doppler = self.maxpool3_doppler(conv3_doppler)

        conv4_doppler = self.conv4_doppler(maxpool3_doppler)
        maxpool4_doppler = self.maxpool4_doppler(conv4_doppler)
        #
        center_doppler = self.conv5_doppler(maxpool4_doppler)

        center = self.center_concat(torch.cat([center_doppler, center_bmode], 1))

        # attention units
        # attention4_bmode = self.attention4_bmode(conv4_bmode, center_doppler)[0]
        up4 = self.up_concat4(conv4_bmode, center)
        # attention4_doppler = self.attention4_doppler(up4_first, center_doppler)[0]
        # up4_second = self.conv4(attention4_doppler)

        # attention3_bmode = self.attention3_bmode(conv3_bmode, conv4_doppler)[0]
        up3 = self.up_concat3(conv3_bmode, up4)
        # attention3_doppler = self.attention3_doppler(up3_first, conv4_doppler)[0]
        # up3_second = self.conv3(attention3_doppler)

        # attention2_bmode = self.attention2_bmode(conv2_bmode, conv3_doppler)[0]
        up2 = self.up_concat2(conv2_bmode, up3)
        # attention2_doppler = self.attention2_doppler(up2_first, conv3_doppler)[0]
        # up2_second = self.conv2(attention2_doppler)

        # attention1_bmode = self.attention1_bmode(conv1_bmode, conv2_doppler)[0]
        up1 = self.up_concat1(conv1_bmode, up2)
        # attention1_doppler = self.attention1_doppler(up1_first, conv2_doppler)[0]
        # up1_second = self.conv1(attention1_doppler)

        final = self.final(up1)

        return final

    @staticmethod
    def apply_sigmoid(pred):
        log_p = torch.sigmoid(pred)

        return log_p

    def init_hidden(self, batch_size):
        hidden_states = []
        for i in range(2):
            # number of hidden layers comes up first
            hidden_states.append(Variable(torch.zeros(batch_size, self.filters[4], 20, 20, requires_grad=True,
                                                      device=torch.device('cuda'))))
        return hidden_states
