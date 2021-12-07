import torch
import torch.nn as nn
from convgru import ConvGRU
import attention_block
from network.modules.utils import MultiAttentionBlock
from cbam import ChannelGate

'''Define the number of features you want to normalize to'''
def BatchNormAndPReLU(features):
    return nn.Sequential(
            nn.BatchNorm2d(num_features = features),
            nn.PReLU()
        )

'''
Define the number of input channels and output channels
'''
def ConvBatchNormPreLU(in_channels, features):
    return nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = features, kernel_size=5, padding=2),
            nn.modules.BatchNorm2d(num_features = features),
            nn.PReLU()
        )

def Conv2x2Stride2x2Prelu(in_channels, features):
    return nn.Sequential(
        nn.Conv2d(in_channels = in_channels, out_channels = features, kernel_size=2, stride=2),
        nn.PReLU()
    )

'''
For simplicity let's start with same amount of input dimensions as hidden dimensions
'''
def ConvGru(dimensions):
    return ConvGRU(
            input_dim=dimensions, 
            hidden_dim=dimensions,
            kernel_size=(3, 3),
            num_layers=1,
            dtype=dtype, 
            )

class SpatialChannelAttentionModule(nn.Module):
    def __init__(self, filters, attentionMode, attention_dsample):
        super().__init__(SpatialChannelAttentionModule, self).__init__()
        self.filters = filters
        self.attentionMode = attentionMode
        self.selfSpatialAttentionHigherResolutionSpace = ChannelGate(gate_channels = self.filters[0])
        self.multiScaleSpatialAttention = MultiAttentionBlock(in_size=self.filters[0], gate_size=self.filters[1], inter_size=self.filters[0],
                                                   nonlocal_mode=self.attentionMode, sub_sample_factor= attention_dsample)
        self.selfSpatialAttentionLowerResolutionSpace = ChannelGate(gate_channels = self.filters[1])

    def forward(self, input, gating_signal):
        x_A = self.selfSpatialAttentionHigherResolutionSpace(input)
        g_A = self.selfSpatialAttentionLowerResolutionSpace(gating_signal)
        return self.multiScaleSpatialAttention(x_A, g_A)



class DoubleConvolutionAndNormalization(nn.Module):

    def __init__(self, features):
        super(DoubleConvolutionAndNormalization, self).__init__()
        self.features = features
        self.encoder_1 = ConvBatchNormPreLU(self.features, self.features)
        self.encoder_2 = nn.Conv2d(self.features, self.features, kernel_size=5, padding=2)
        self.encoder_3 = BatchNormAndPReLU(self.features)

    def forward(self, x):
        x1 = self.encoder_1(x)
        x2 = self.encoder_2(x1)
        return self.encoder_3(torch.add(x, x2))


class VesNet(nn.Module):


    def __init__(self, dtype, in_channels=2, out_channels=1, feature_scale=16, nonlocal_mode='concatenation', 
                attention_dsample=(2,2,2)):
        super(VesNet, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.nonlocal_mode = nonlocal_mode
        self.attention_dsample = attention_dsample


        filters = [64, 128, 256, 512]
        filters = [int(x / self.feature_scale) for x in filters]

        self.imagePrep = ConvBatchNormPreLU(self.in_channels, filters[0])
        self.encoder1 = DoubleConvolutionAndNormalization(filters[0])
        self.pool1 = Conv2x2Stride2x2Prelu(filters[0], filters[1])

        self.encoder2 = DoubleConvolutionAndNormalization(filters[1])
        self.pool2 = Conv2x2Stride2x2Prelu(filters[1], filters[2])

        self.encoder3 = DoubleConvolutionAndNormalization(filters[2])
        self.pool3 = Conv2x2Stride2x2Prelu(filters[2], filters[3])

        self.encoder4 = DoubleConvolutionAndNormalization(filters[3])


        # skip connections with Conv GRU
        self.convGru4 = ConvGru(dimensions=filters[3])
        self.convGru3 = ConvGru(dimensions=filters[2])
        self.spatialChannelAttention3 = SpatialChannelAttentionModule(filters=filters[2:3])
        self.convGru2 = ConvGru(dimensions=filters[1])
        self.spatialChannelAttention2 = SpatialChannelAttentionModule(filters=filters[1:2])
        self.convGru1 = ConvGru(dimensions=filters[0])
        self.spatialChannelAttention1 = SpatialChannelAttentionModule(filters=filters[0:1])



        # self.bottleneck = Convolution2D(8*features, 16*features)

        # self.upconv4 = nn.ConvTranspose2d(
        #     16*features, 8*features, kernel_size=2, stride=2)
        # self.decoder4 = Convolution2D(
        #     16*features, 8*features)  # concate, 2*8=16

        # self.upconv3 = nn.ConvTranspose2d(
        #     8*features, 4*features, kernel_size=2, stride=2)
        # self.decoder3 = Convolution2D(8*features, 4*features)  # concate, 2*4=8

        # self.upconv2 = nn.ConvTranspose2d(
        #     4*features, 2*features, kernel_size=2, stride=2)
        # self.decoder2 = Convolution2D(4*features, 2*features)  # concate, 2*2=4

        # self.upconv1 = nn.ConvTranspose2d(
        #     2*features, features, kernel_size=2, stride=2)
        # self.decoder1 = Convolution2D(2*features, features)  # concate, 2*1=2

        # self.conv_out = nn.Conv2d(features, 1, 1)

    def forward(self, input):
        
        #encoding path
        resImagePrep = self.imagePrep(input)

        resEncoder1 = self.encoder1(resImagePrep)
        resPool1 = self.pool1(resEncoder1)

        resEncoder2 = self.encoder2(resPool1)
        resPool2 = self.pool2(resEncoder2)

        resEncoder3 = self.encoder3(resPool2)
        resPool3 = self.pool3(resEncoder3)

        resEncoder4 = self.encoder4(resPool3)

        #intermediate Steps
        #temporal attention units
        resConvGru4 = self.convGru4(resEncoder4)
        resConvGru3 = self.convGru3(resEncoder3)
        resConvGru2 = self.convGru2(resEncoder2)
        resConvGru1 = self.convGru1(resEncoder1)

        #spatial Attention units
        resSpatialChAtt3 = self.spatialChannelAttention3(re)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4 = self.up_concat4(g_conv4, center)
        g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3 = self.up_concat3(g_conv3, up4)
        g_conv2, att2 = self.attentionblock2(conv2, up3)
        up2 = self.up_concat2(g_conv2, up3)
        up1 = self.up_concat1(conv1, up2)


        # bottleneck = self.bottleneck(self.pool4(enc4))

        # dec4 = torch.cat([enc4, self.upconv4(bottleneck)], dim=1)
        # dec4 = self.decoder4(dec4)

        # dec3 = torch.cat([enc3, self.upconv3(dec4)], dim=1)
        # dec3 = self.decoder3(dec3)

        # dec2 = torch.cat([enc2, self.upconv2(dec3)], dim=1)
        # dec2 = self.decoder2(dec2)

        # dec1 = torch.cat([enc1, self.upconv1(dec2)], dim=1)
        # dec1 = self.decoder1(dec1)

        # output = torch.sigmoid(self.conv_out(dec1))


if __name__ == '__main__':
    # detect if CUDA is available or not
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        dtype = torch.cuda.FloatTensor  # computation in GPU
    else:
        dtype = torch.FloatTensor

    image = torch.rand((1, 2, 320, 320))  # image, channels, width, height
    model = VesNet()

    print(model(image))
