# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################
import logging
from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# def init_weights(m, init_w='normal'):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#         if init_w == 'normal':
#             m.weight.data.normal_(0, 1e-3)
#         elif init_w == 'kaiming':
#             init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
#         if m.bias is not None:
#             m.bias.data.zero_()
#     elif isinstance(m, nn.ConvTranspose2d):
#         if init_w == 'normal':
#             m.weight.data.normal_(0, 1e-3)
#         elif init_w == 'kaiming':
#             init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
#         if m.bias is not None:
#             m.bias.data.zero_()
#     elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
#         if init_w == 'normal':
#             m.weight.data.fill_(1)
#             m.bias.data.zero_()
#         elif init_w == 'kaiming':
#             init.normal_(m.weight.data, 1.0, 0.02)
#             init.constant_(m.bias.data, 0.0)
#

def activation_fn(inputs, activation, leakage=0.2):
    if activation == 'relu':
        out = F.relu(inputs)
    elif activation == 'leaky_relu':
        out = F.leaky_relu(inputs, leakage)
    else:
        out = inputs

    return out


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput, activation='relu', leakage=0.2):
        super().__init__()

        self.activation = activation
        self.leakage = leakage

        if noutput > ninput:
            self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        else:
            self.conv = None

        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

        # init_weights(self.conv, init_w='kaiming')
        # init_weights(self.bn, init_w='kaiming')

    def forward(self, input):

        if self.conv is not None:
            output = torch.cat([self.conv(input), self.pool(input)], 1)
        else:
            output = self.pool(input)

        output = self.bn(output)
        return activation_fn(output, self.activation, self.leakage)


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated, activation='relu', leakage=0.2):
        super().__init__()

        self.activation = activation
        self.leakage = leakage

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)
        # init_weights(self.conv3x1_1, init_w='kaiming')

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)
        # init_weights(self.conv1x3_1, init_w='kaiming')

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
        # init_weights(self.bn1, init_w='kaiming')

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))
        # init_weights(self.conv3x1_2, init_w='kaiming')

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                   dilation=(1, dilated))
        # init_weights(self.conv1x3_2, init_w='kaiming')

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
        # init_weights(self.bn2, init_w='kaiming')

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = activation_fn(output, self.activation, self.leakage)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = activation_fn(output, self.activation, self.leakage)

        output = self.conv3x1_2(output)
        output = activation_fn(output, self.activation, self.leakage)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return activation_fn(output + input, self.activation, self.leakage)


class Encoder(nn.Module):
    def __init__(self, in_channels, filter_numbers, activation='relu'):
        super().__init__()
        chans = 32 if in_channels > 16 else 16
        self.initial_block = DownsamplerBlock(in_channels, chans, activation=activation)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(chans, 64, activation=activation))

        for x in range(0, 5):
            self.layers.append(non_bottleneck_1d(64, 0.03, 1, activation=activation))

        self.layers.append(DownsamplerBlock(64, 128, activation=activation))

        for x in range(0, 2):
            self.layers.append(non_bottleneck_1d(128, 0.3, 2, activation))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4, activation))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8, activation))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16, activation))

    def forward(self, input):
        p1 = output = self.initial_block(input)

        for i in range(0, 6):
            output = self.layers[i](output)
        p2 = output

        for i in range(6, len(self.layers)):
            output = self.layers[i](output)
        p3 = output

        return p1, p2, p3


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput, activation='relu', leakage=0.2):
        super().__init__()
        self.activation = activation
        self.leakage = leakage

        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

        # init_weights(self.conv, init_w='kaiming')
        # init_weights(self.bn, init_w='kaiming')

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return activation_fn(output, self.activation, self.leakage)


class Decoder(nn.Module):
    def __init__(self, num_classes, activation='relu'):
        super().__init__()

        self.actiation = activation

        self.layer1 = UpsamplerBlock(128, 64, activation)
        self.layer2 = non_bottleneck_1d(64, 0, 1, activation)
        self.layer3 = non_bottleneck_1d(64, 0, 1, activation)  # 64x64x304

        self.layer4 = UpsamplerBlock(64, 32, activation)
        self.layer5 = non_bottleneck_1d(32, 0, 1, activation)
        self.layer6 = non_bottleneck_1d(32, 0, 1, activation)  # 32x128x608

        self.output_conv = nn.ConvTranspose2d(32, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)
        # init_weights(self.output_conv, init_w='kaiming')

    def forward(self, input):
        output = input
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        em2 = output
        output = self.layer4(output)
        output = self.layer5(output)
        output = self.layer6(output)
        em1 = output

        output = self.output_conv(output)

        return output, em1, em2


class ERFNet(nn.Module):
    def __init__(self, erf_config):  # use encoder to pass pretrained encoder
        super().__init__()
        in_channels = erf_config.input_features
        activation = erf_config.activation.name
        filter_numbers = erf_config.filter_numbers

        self.encoder = Encoder(in_channels, filter_numbers, activation)

        if erf_config.pretrained is not None or erf_config.pretrained:
            print('Load pretrained model')
            target_state = self.state_dict()
            check = torch.load(erf_config.pretrained)
            for name, val in check.items():
                mono_name = name[7:]
                if mono_name not in target_state:
                    continue
                try:
                    target_state[mono_name].copy_(val)
                except RuntimeError as e:
                    logging.warning(f'Error occured during loading the layer: {name} of the pretrained erfnet model: {e}')
                    continue
            print('Pretrained model loaded')

    def forward(self, input):
        p1, p2, p3 = self.encoder.forward(input)
        return input, p1, p2, p3
