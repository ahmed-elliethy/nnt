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

import numpy as np
import torch
import torch.nn as nn


class Bias(torch.nn.Module):
    def __init__(self, x):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(x))

    def forward(self, x):
        return self.bias[:, np.newaxis, np.newaxis] + x


class DnCnnNp(nn.Module):
    def __init__(self, images, num_levels=17, padding='same'):
        super(DnCnnNp, self).__init__()

        self._num_levels = num_levels
        self._actfun = [nn.ReLU(inplace=False), ] * (self._num_levels - 1) + [nn.Identity(), ]
        self._f_size = [3, ] * self._num_levels
        self._f_num = [64, ] * (self._num_levels - 1) + [1, ]
        self._f_stride = [1, ] * self._num_levels
        self._bnorm = [False, ] + [True, ] * (self._num_levels - 2) + [False, ]
        self._res = [0, ] * self._num_levels
        self._bnorm_init_var = 1e-4
        self._bnorm_init_gamma = np.sqrt(2.0 / (9.0 * 64.0))
        self._bnorm_epsilon = 1e-5

        self.level = [None, ] * self._num_levels
        self.input = images
        self.extra_train = []
        self.variables_list = []
        self.trainable_list = []
        self.decay_list = []
        self.padding = padding

        x = self.input
        self.layers = []
        for i in range(self._num_levels):
            self.layers.append(self._conv(x, self._f_size[i], self._f_num[i], self._f_stride[i]))
            if self._bnorm[i]:
                self.layers.append(self._batch_norm(self._f_num[i]))
            self.layers.append(self._bias(self._f_num[i]))
            if i != self._num_levels - 1:
                self.layers.append(self._actfun[i])
            x = self._f_num[i]

        self.dncnn = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.dncnn(x)

    def _batch_norm(self, x):
        """Batch normalization."""
        return nn.BatchNorm2d(x)

    def _bias(self, x):
        """Bias term."""
        return Bias(x)

    def _conv(self, x, filter_size, out_filters, stride):
        """Convolution."""
        return nn.Conv2d(in_channels=x, out_channels=out_filters, kernel_size=filter_size, padding=self.padding,
                         stride=stride, bias=False)
