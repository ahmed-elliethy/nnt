# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Copyright © 2022 Ahmed Elliethy.
#
# All rights reserved.
#
# This software should be used, reproduced and modified only for informational and nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package)
#
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, input, output, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.input, self.output, self.kernel_size, self.stride, self.padding = input, output, kernel_size, stride, padding
        self.conv = nn.Conv2d(input, output, kernel_size, stride=self.stride, padding=self.padding)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        return x


class DeConvBlock(nn.Module):
    def __init__(self, input, output, kernel_size=3, stride=1, padding=1):
        super(DeConvBlock, self).__init__()
        self.input, self.output, self.kernel_size, self.stride, self.padding = input, output, kernel_size, stride, padding
        self.deconv = nn.ConvTranspose2d(input, output, kernel_size, stride=self.stride, padding=self.padding)

    def forward(self, x, skip=None, rel=None):
        x = self.deconv(x)
        if skip is not None:
            x = x + skip
        if rel is None:
            x = F.relu(x)
        return x


class NoiseInjectionModel(nn.Module):
    def __init__(self):
        super(NoiseInjectionModel, self).__init__()

        self.conv2 = ConvBlock(input=128, output=128)  # skip 2
        self.conv3 = ConvBlock(input=128, output=128)
        self.conv4 = ConvBlock(input=128, output=128)  # skip 4
        self.conv5 = ConvBlock(input=128, output=128)
        self.conv6 = ConvBlock(input=128, output=128)  # not used
        self.conv7 = ConvBlock(input=128, output=128)  # not used
        self.conv8 = ConvBlock(input=128, output=128)  # not used
        self.conv9 = ConvBlock(input=128, output=128)  # not used
        self.conv10 = ConvBlock(input=128, output=128)  # not used
        self.conv11 = ConvBlock(input=128, output=128)  # not used
        self.conv12 = ConvBlock(input=128, output=128)  # not used
        self.conv13 = ConvBlock(input=128, output=128)  # not used
        self.conv14 = ConvBlock(input=128, output=128)  # not used
        self.conv15 = ConvBlock(input=128, output=128)  # not used

        self.deconv2 = DeConvBlock(input=128, output=128)  # not used
        self.deconv3 = DeConvBlock(input=128, output=128)  # not used
        self.deconv4 = DeConvBlock(input=128, output=128)  # not used
        self.deconv5 = DeConvBlock(input=128, output=128)  # not used
        self.deconv6 = DeConvBlock(input=128, output=128)  # not used
        self.deconv7 = DeConvBlock(input=128, output=128)  # not used
        self.deconv8 = DeConvBlock(input=128, output=128)  # not used
        self.deconv9 = DeConvBlock(input=128, output=128)  # not used
        self.deconv10 = DeConvBlock(input=128, output=128)
        self.deconv11 = DeConvBlock(input=128, output=128)
        self.deconv12 = DeConvBlock(input=128, output=128)
        self.deconv13 = DeConvBlock(input=128, output=128)
        self.deconv14 = DeConvBlock(input=128, output=128)  # not used
        self.deconv15 = DeConvBlock(input=128, output=1)

    def forward(self, first, last):
        # encoder
        x = torch.cat((first, last), 1)
        skip2 = self.conv2.forward(x)
        x = self.conv3.forward(skip2)
        skip4 = self.conv4.forward(x)
        x = self.conv5.forward(skip4)

        # decoder
        x = self.deconv10.forward(x)
        x = self.deconv11.forward(x, skip=skip4)
        x = self.deconv12.forward(x)
        x = self.deconv13.forward(x, skip=skip2)
        x = self.deconv15.forward(x)

        return x
