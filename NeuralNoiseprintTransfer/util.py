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
import cv2
import numpy as np
import torch
from skimage.color import rgb2ycbcr, ycbcr2rgb
from skimage.exposure import match_histograms


def image_loader(org, device):
    img = org
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 1)
    i_source = torch.Tensor(img)
    i_source = i_source.to(device)
    return i_source


def transfer_color_to_gray(ref_color_file_name, gray_img):
    img_color = cv2.imread(ref_color_file_name)
    img_color_ycbcr = rgb2ycbcr(img_color)
    generated_img_color_ycbcr = 0 * img_color_ycbcr
    generated_img_color_ycbcr[:, :, 0] = gray_img * 255
    generated_img_color_ycbcr[:, :, 1] = img_color_ycbcr[:, :, 1]
    generated_img_color_ycbcr[:, :, 2] = img_color_ycbcr[:, :, 2]
    generated_img_color = ycbcr2rgb(generated_img_color_ycbcr)
    generated_img_color2 = match_histograms(generated_img_color, img_color / 255.0, multichannel=True)
    return generated_img_color2
