#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import h5py
from datetime import datetime
import time
import os
import code
import math
from torchsummary import summary
torch.cuda.set_device(1)

# -----------------------------------------------------------------------------------------------------------------------
# 卷积层的封装
class BConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(BConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_ch),  # 添加了BN层
            nn.ReLU(inplace=True))

    def forward(self, input):
        return self.conv(input)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.PReLU(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.PReLU(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class depthwise_conv(nn.Module):
    def __init__(self, nin, kernels_per_layer):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin)

    def forward(self, x):
        out = self.depthwise(x)
        return out


class pointwise_conv(nn.Module):
    def __init__(self, nin, nout):
        super(pointwise_conv, self).__init__()
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.pointwise(x)
        return out



class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = 1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class DoubleDSConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleDSConv, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False),
            depthwise_separable_conv(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.PReLU(out_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            depthwise_separable_conv(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.PReLU(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


# 加深unet 网络结构定义
class Deep_UNet(nn.Module):
    def __init__(self, channels):
        super(Deep_UNet, self).__init__()


        # self.conv1 = BConv(channels, 8)  # 550x800
        # self.conv1 = DoubleConv(channels, 8)  # 550x800  8.7
        self.conv1 = DoubleDSConv(channels, 8)  # 550x800
        self.pool1 = nn.MaxPool2d(2, stride=2)  # 275x400
        # self.conv2 = BConv(8, 16)
        # self.conv2 = DoubleConv(8, 16)  # 8.7
        self.conv2 = DoubleDSConv(8, 16)
        self.pool2 = nn.MaxPool2d(2, stride=2)  # 137x200
        # self.conv3 = BConv(16, 32)
        # self.conv3 = DoubleConv(16, 32)  # 8.7
        self.conv3 = DoubleDSConv(16, 32)
        self.pool3 = nn.MaxPool2d(2, stride=2)  # 68x100
        # self.conv4 = BConv(32, 64)
        # self.conv4 = DoubleConv(32, 64)  # 8.7
        self.conv4 = DoubleDSConv(32, 64)
        self.pool4 = nn.MaxPool2d(2, stride=2)  # 34x50

        # self.conv5 = BConv(64, 128)
        # self.conv5 = DoubleConv(64, 128)  # 8.7
        self.conv5 = DoubleDSConv(64, 128)
        self.up6 = nn.Upsample(size=(30, 35))
        # self.conv6 = BConv(192, 64)
        # self.conv6 = DoubleConv(192, 64)  # 8.7
        self.conv6 = DoubleDSConv(192, 64)
        self.up7 = nn.Upsample(size=(60, 70))
        # self.conv7 = BConv(96, 32)
        # self.conv7 = DoubleConv(96, 32)  # 8.7
        self.conv7 = DoubleDSConv(96, 32)
        self.up8 = nn.Upsample(size=(120, 140))
        # self.conv8 = BConv(48, 16)
        # self.conv8 = DoubleConv(48, 16)  # 8.7
        self.conv8 = DoubleDSConv(48, 16)
        self.up9 = nn.Upsample(size=(241, 281))  # 241, 281
        # self.conv9 = BConv(24, 8)
        # self.conv9 = DoubleConv(24, 8)  # 8.7
        self.conv9 = DoubleDSConv(24, 8)  # 8.7
        self.conv10 = nn.Conv2d(8, 1, 1, stride=1)

    def forward(self, x):
        # print('x', x.shape)
        x1 = self.conv1(x)
        # print('x1', x1.shape)
        x2 = self.pool1(x1)
        x3 = self.conv2(x2)
        # print('x3', x3.shape)
        x4 = self.pool2(x3)
        x5 = self.conv3(x4)
        # print('x5', x5.shape)
        x6 = self.pool3(x5)
        x7 = self.conv4(x6)
        # print('x7', x7.shape)
        x8 = self.pool4(x7)

        x9 = self.conv5(x8)
        # print('x9', x9.shape)
        x10 = self.up6(x9)
        merge0 = torch.cat([x10, x7], dim=1)
        x11 = self.conv6(merge0)
        # print('x11', x11.shape)
        x12 = self.up7(x11)
        # code.interact(local=locals())
        merge1 = torch.cat([x12, x5], dim=1)
        x13 = self.conv7(merge1)
        # print('x13', x13.shape)
        x14 = self.up8(x13)
        merge2 = torch.cat([x14, x3], dim=1)
        x15 = self.conv8(merge2)
        # print('x15', x15.shape)
        x16 = self.up9(x15)
        merge3 = torch.cat([x16, x1], dim=1)
        x17 = self.conv9(merge3)
        # print('x17', x17.shape)
        x18 = self.conv10(x17)
        # print('x18', x18.shape)
        return x18


# 加深unet 网络结构定义
class Deep_UNet_GRB(nn.Module):
    """
    网络结构
    """
    def __init__(self, channels):
        super(Deep_UNet_GRB, self).__init__()
        # self.conv1 = BConv(channels, 8)  # 550x800
        self.conv1 = DoubleConv(channels, 8)  # 550x800  8.7
        self.pool1 = nn.MaxPool2d(2, stride=2)  # 275x400
        # self.conv2 = BConv(8, 16)
        self.conv2 = DoubleConv(8, 16)  # 8.7
        self.pool2 = nn.MaxPool2d(2, stride=2)  # 137x200
        # self.conv3 = BConv(16, 32)
        self.conv3 = DoubleConv(16, 32)  # 8.7
        self.pool3 = nn.MaxPool2d(2, stride=2)  # 68x100
        # self.conv4 = BConv(32, 64)
        self.conv4 = DoubleConv(32, 64)  # 8.7
        self.pool4 = nn.MaxPool2d(2, stride=2)  # 34x50

        # self.conv5 = BConv(64, 128)
        self.conv5 = DoubleConv(64, 128)  # 8.7
        self.up6 = nn.Upsample(size=(13, 20))
        # self.conv6 = BConv(192, 64)
        self.conv6 = DoubleConv(192, 64)  # 8.7
        self.up7 = nn.Upsample(size=(27, 40))
        # self.conv7 = BConv(96, 32)
        self.conv7 = DoubleConv(96, 32)  # 8.7
        self.up8 = nn.Upsample(size=(55, 80))
        # self.conv8 = BConv(48, 16)
        self.conv8 = DoubleConv(48, 16)  # 8.7
        self.up9 = nn.Upsample(size=(110, 160))
        # self.conv9 = BConv(24, 8)
        self.conv9 = DoubleConv(24, 8)  # 8.7
        self.conv10 = nn.Conv2d(8, 1, 1, stride=1)

    def forward(self, x):
        # print('x', x.shape)
        x1 = self.conv1(x)
        # print('x1', x1.shape)
        x2 = self.pool1(x1)
        x3 = self.conv2(x2)
        # print('x3', x3.shape)
        x4 = self.pool2(x3)
        x5 = self.conv3(x4)
        # print('x5', x5.shape)
        x6 = self.pool3(x5)
        x7 = self.conv4(x6)
        # print('x7', x7.shape)
        x8 = self.pool4(x7)

        x9 = self.conv5(x8)
        # print('x9', x9.shape)
        x10 = self.up6(x9)
        merge0 = torch.cat([x10, x7], dim=1)
        x11 = self.conv6(merge0)
        # print('x11', x11.shape)
        x12 = self.up7(x11)
        # code.interact(local=locals())
        merge1 = torch.cat([x12, x5], dim=1)
        x13 = self.conv7(merge1)
        # print('x13', x13.shape)
        x14 = self.up8(x13)
        merge2 = torch.cat([x14, x3], dim=1)
        x15 = self.conv8(merge2)
        # print('x15', x15.shape)
        x16 = self.up9(x15)
        merge3 = torch.cat([x16, x1], dim=1)
        x17 = self.conv9(merge3)
        # print('x17', x17.shape)
        x18 = self.conv10(x17)
        # print('x18', x18.shape)
        return x18


# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# model = Deep_UNet(1)
# model.to(device)
# # print(model)
# # summary(model, (1, 110, 160))
# summary(model, (1, 241, 281))
