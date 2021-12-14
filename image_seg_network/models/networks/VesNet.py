import torch
import torch.nn as nn
from image_seg_network.models.modules.convgru import ConvGRU
import image_seg_network.models.modules.attention_block as attention_block
from image_seg_network.models.utils import MultiAttentionBlock
from image_seg_network.models.modules.cbam import ChannelGate

'''Define the number of flters you want to normalize to'''


def BatchNormAndPReLU(filters):
    return nn.Sequential(
        nn.BatchNorm2d(num_features=filters),
        nn.PReLU()
    )


'''
Define the number of input channels and output channels
'''


def ConvBatchNormPreLU(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=5, padding=2),
        nn.modules.BatchNorm2d(num_features=out_channels),
        nn.PReLU()
    )


def Conv2x2Stride2x2Prelu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=2, stride=2),
        nn.PReLU()
    )


'''
For simplicity let's start with same amount of input dimensions as hidden dimensions
'''


def ConvGru(input_size, dimensions, dtype):
    return ConvGRU(
        input_size=input_size,
        input_dim=dimensions,
        hidden_dim=dimensions,
        kernel_size=(3, 3),
        num_layers=1,
        dtype=dtype,
    )


class SpatialChannelAttentionModule(nn.Module):
    def __init__(self, filters):
        super(SpatialChannelAttentionModule, self).__init__()
        self.filters = filters
        self.attentionMode = 'concatenation'
        self.attention_dsample = (2, 2, 2)
        self.selfSpatialAttentionHigherResolutionSpace = ChannelGate(
            gate_channels=self.filters[0])
        self.multiScaleSpatialAttention = MultiAttentionBlock(in_size=self.filters[0], gate_size=self.filters[1], inter_size=self.filters[0],
                                                              nonlocal_mode=self.attentionMode, sub_sample_factor=self.attention_dsample)
        self.selfSpatialAttentionLowerResolutionSpace = ChannelGate(
            gate_channels=self.filters[1])

    def forward(self, input, gating_signal):
        x_A = self.selfSpatialAttentionHigherResolutionSpace(input)
        g_A = self.selfSpatialAttentionLowerResolutionSpace(gating_signal)
        return self.multiScaleSpatialAttention(x_A, g_A)


class Encoder(nn.Module):

    def __init__(self, filters):
        super(Encoder, self).__init__()
        self.filters = filters
        self.encoder_1 = ConvBatchNormPreLU(self.filters, self.filters)
        self.encoder_2 = nn.Conv2d(
            self.filters, self.filters, kernel_size=5, padding=2)
        self.encoder_3 = BatchNormAndPReLU(self.filters)

    def forward(self, x):
        x1 = self.encoder_1(x)
        x2 = self.encoder_2(x1)
        # is this summation after or before the BatchNormalization and PreLU?
        return self.encoder_3(torch.add(x, x2))


class Decoder(nn.Module):

    def __init__(self, filters):
        super(Decoder, self).__init__()
        self.filters = filters
        self.conv1 = nn.Conv2d(
            self.filters*2, self.filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.filters, self.filters,
                               kernel_size=3, padding=1)
        self.batchNormAndPrelu = BatchNormAndPReLU(self.filters)

    def forward(self, input_skipConnection, input_UpSampling):
        x1 = self.conv1(
            torch.cat([input_skipConnection, input_UpSampling], dim=1))
        x2 = self.conv2(x1)
        # is this summation after or before the BatchNormalization and PreLU?
        return self.batchNormAndPrelu(torch.add(input_UpSampling, x2))


'''
This is less computational expensive (?) and removes checker artefacts

Do we use Bilinear, Trilinear or Nearest Neighbors ???
'''


class ResizeUpConvolution(nn.Module):

    def __init__(self, filters):
        super(ResizeUpConvolution, self).__init__()
        self.filters = filters
        self.resizeUp = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(
            in_channels=self.filters[1], out_channels=self.filters[0], kernel_size=3, padding=1)

    def forward(self, x):
        x = self.resizeUp(x)
        return self.conv(x)


class VesNet(nn.Module):

    def __init__(self, dtype, in_channels=2, out_channels=1, feature_scale=16, nonlocal_mode='concatenation',
                 attention_dsample=(2, 2, 2)):
        super(VesNet, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.nonlocal_mode = nonlocal_mode
        self.attention_dsample = attention_dsample
        self.dtype = dtype

        input_size = [320, 160, 80, 40]
        filters = [64, 128, 256, 512]
        filters = [int(x / self.feature_scale) for x in filters]

        self.imagePrep = ConvBatchNormPreLU(self.in_channels, filters[0])
        self.encoder1 = Encoder(filters[0])
        self.pool1 = Conv2x2Stride2x2Prelu(filters[0], filters[1])

        self.encoder2 = Encoder(filters[1])
        self.pool2 = Conv2x2Stride2x2Prelu(filters[1], filters[2])

        self.encoder3 = Encoder(filters[2])
        self.pool3 = Conv2x2Stride2x2Prelu(filters[2], filters[3])

        self.encoder4 = Encoder(filters[3])

        # skip connections with Conv GRU
        self.convGru4 = ConvGru(input_size=(
            input_size[0], input_size[0]), dimensions=filters[3], dtype=self.dtype)
        self.convGru3 = ConvGru(input_size=(
            input_size[1], input_size[1]), dimensions=filters[2], dtype=self.dtype)
        self.spatialChannelAttention3 = SpatialChannelAttentionModule(
            filters=filters[2:4])
        self.convGru2 = ConvGru(input_size=(
            input_size[2], input_size[2]), dimensions=filters[1], dtype=self.dtype)
        self.spatialChannelAttention2 = SpatialChannelAttentionModule(
            filters=filters[1:3])
        self.convGru1 = ConvGru(input_size=(
            input_size[3], input_size[3]), dimensions=filters[0], dtype=self.dtype)
        self.spatialChannelAttention1 = SpatialChannelAttentionModule(
            filters=filters[:2])

        self.resizeUp4 = ResizeUpConvolution(filters=filters[2:4])

        self.decoder3 = Decoder(filters[2])
        self.resizeUp3 = ResizeUpConvolution(filters=filters[1:3])

        self.decoder2 = Decoder(filters[1])
        self.resizeUp2 = ResizeUpConvolution(filters=filters[:2])

        self.decoder1 = Decoder(filters[0])

        self.conv_out = nn.Conv2d(filters[0], 1, 1)

    def forward(self, input):

        # encoding path
        resImagePrep = self.imagePrep(input)

        resEncoder1 = self.encoder1(resImagePrep)
        resPool1 = self.pool1(resEncoder1)

        resEncoder2 = self.encoder2(resPool1)
        resPool2 = self.pool2(resEncoder2)

        resEncoder3 = self.encoder3(resPool2)
        resPool3 = self.pool3(resEncoder3)

        resEncoder4 = self.encoder4(resPool3)

        # intermediate Steps
        # temporal attention units
        resConvGru4 = self.convGru4(resEncoder4)
        resConvGru3 = self.convGru3(resEncoder3)
        resConvGru2 = self.convGru2(resEncoder2)
        resConvGru1 = self.convGru1(resEncoder1)

        # decoder
        resSpatialChAtt3 = self.spatialChannelAttention3(
            resConvGru3, resConvGru4)
        resUp4 = self.resizeUp4(resConvGru4)
        resDecoder3 = self.decoder3(resSpatialChAtt3, resUp4)

        resSpatialChAtt2 = self.spatialChannelAttention2(
            resConvGru2, resDecoder3)
        resUp3 = self.resizeUp3(resConvGru3)
        resDecoder2 = self.decoder2(resSpatialChAtt2, resUp3)

        resSpatialChAtt1 = self.spatialChannelAttention1(
            resConvGru1, resDecoder2)
        resUp2 = self.resizeUp2(resConvGru2)
        resDecoder1 = self.decoder1(resSpatialChAtt1, resUp2)
        print(resDecoder1.size())

        # output = torch.sigmoid(self.conv_out(dec1))


if __name__ == '__main__':
    # detect if CUDA is available or not
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        dtype = torch.cuda.FloatTensor  # computation in GPU
    else:
        dtype = torch.FloatTensor

    # image, depth (how many previous images are we putting in), channels, width, height
    image = torch.rand((1, 5, 2, 320, 320))

    model = VesNet(dtype=dtype)

    model(image)
