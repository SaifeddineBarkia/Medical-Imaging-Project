# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:19:23 2021

@author: Saif
"""
import torch
import torch.nn as nn
import torch.nn.functional as F



class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super(block, self).__init__()
        self.expansion = 1
        self.conv1 = nn.Conv3d(
            in_channels, intermediate_channels, kernel_size=3, stride=stride, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm3d(intermediate_channels)
        
        self.conv2 = nn.Conv3d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(intermediate_channels)
        
        """self.conv3 = nn.Conv3d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm3d(intermediate_channels * self.expansion)"""
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()
        
        print("before", identity.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        print("x",x.shape,"id",identity.shape)
        
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, conv1_t_size=7, conv1_t_stride=1):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv3d(image_channels, 64, kernel_size=(conv1_t_size,7,7), stride=(conv1_t_stride, 2, 2), padding=(conv1_t_size // 2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        out_1 = self.layer1(x)
        out_2 = self.layer2(x)
        out_3 = self.layer3(x)
        out_4 = self.layer4(x)

        out_4 = self.avgpool(out_4)

        return out_1 , out_2, out_3, out_4

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels :
            identity_downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_channels,
                    intermediate_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm3d(intermediate_channels ),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


def ResNet34(img_channel=3):
    return ResNet(block, [3, 4, 6, 3], img_channel, )


def test():
    net = ResNet34(img_channel=3)
    y1,y2,y3,y4 = net(torch.randn(1, 3, 32, 32, 32)).to("cuda")
    print(y1.size())


test()


    